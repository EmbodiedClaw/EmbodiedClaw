"""
Unified evaluation runner: single-process LLM agent + simulation.

Iterates through evaluation episodes, for each one:
1. Spawns objects at pre-computed positions
2. Runs a ReAct loop: LLM observes → thinks → calls tools → gets results
3. Saves per-step artifacts (responses, tool calls, images, debug info)

Uses OpenAI-compatible API with function-calling.
No MCP/SSE dependency.
"""

import asyncio
import json
import os
import base64
import time
import shutil
from copy import deepcopy
from pathlib import Path
from typing import List, Dict

import openai
from PIL import Image as PILImage
import io

# Non-qwen eval path: disable precomputed nav positions by default
os.environ.setdefault('IS_TEST', '0')

from eval_server.mcp_env import (
    env,
    camera,
    processed_eval_episodes,
    spawn_objects_by_world_graph,
    TRAJ_PATH,
    TARGET_SCENE_ID,
)
from eval_server.perception_utils import init_annotators
from eval_server.tools import OPENAI_TOOLS
from eval_server.actions import (
    EvalState,
    dispatch_action,
    step_simulation,
    get_debug_info,
)

# =============================================================================
# Configuration
# =============================================================================

MODEL_NAME = os.environ.get('MODEL_NAME', 'gpt-4o')
MAX_STEP = int(os.environ.get('MAX_STEP', 20))
EVAL_OUTPUT_PATH = Path(os.environ.get(
    'EVAL_OUTPUT_PATH',
    str(TRAJ_PATH / TARGET_SCENE_ID)
))

SYSTEM_PROMPT = (Path(__file__).parent / 'baseline_prompt.txt').read_text()

# OpenAI client
client = openai.AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE_URL"),
)

# Filter out 'ask' tool for automated eval (no user to respond)
EVAL_TOOLS = [t for t in OPENAI_TOOLS if t['function']['name'] != 'ask']

# Initialize perception annotators
init_annotators(camera, resolution=(640, 480))

# Global state
state = EvalState()

# Sanitize model name for filesystem
MODEL_DIR_NAME = MODEL_NAME.replace('-', '_').replace('/', '_')


# =============================================================================
# Per-step saving
# =============================================================================

def save_step_artifacts(
    traj_dir: Path,
    step_count: int,
    message,
    tool_call,
    result_type: str,
    result_data: str,
    debug_info: dict,
    infer_time: float,
    total_time: float,
):
    """Save all artifacts for a single evaluation step."""
    step_dir = traj_dir / f"step_{step_count:03d}"
    os.makedirs(step_dir, exist_ok=True)

    # Raw LLM response
    with open(step_dir / "raw_response.json", 'w') as f:
        json.dump({
            "role": message.role,
            "content": message.content,
            "tool_calls": [{
                "id": tc.id,
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                }
            } for tc in (message.tool_calls or [])],
        }, f, indent=2)

    # Tool call info
    with open(step_dir / "tool_call.json", 'w') as f:
        json.dump({
            "tool_name": tool_call.function.name,
            "args": json.loads(tool_call.function.arguments),
        }, f, indent=2)

    # Timing
    with open(step_dir / "inference_time.json", 'w') as f:
        json.dump({
            "infer_time": infer_time,
            "total_time_till_now": total_time,
        }, f, indent=2)

    # Observation
    if result_type == "text":
        with open(step_dir / "textual_observation.txt", 'w') as f:
            f.write(result_data)
    elif result_type == "image":
        image_data = base64.b64decode(result_data)
        image = PILImage.open(io.BytesIO(image_data))
        image.save(step_dir / "image_observation.png")

    # Debug info
    with open(step_dir / "debug_info.json", 'w') as f:
        json.dump(debug_info, f, indent=2, default=str)


# =============================================================================
# Episode runner
# =============================================================================

async def run_episode(episode_idx: int, episode: dict) -> dict:
    """
    Run one evaluation episode with LLM ReAct loop.

    Returns:
        dict with episode results (task_id, steps, success, etc.)
    """
    global state

    task_id = episode['task_id']
    traj_dir = EVAL_OUTPUT_PATH / task_id / MODEL_DIR_NAME

    # Skip if already processed
    if traj_dir.exists() and len(os.listdir(str(traj_dir))) > 0:
        print(f"[SKIP] Episode {episode_idx} ({task_id}) already processed.")
        return {"task_id": task_id, "status": "skipped"}

    os.makedirs(traj_dir, exist_ok=True)

    # 1. Spawn objects
    try:
        state.current_extra_assets = spawn_objects_by_world_graph(
            env, episode, state.current_extra_assets
        )
    except Exception as e:
        print(f"[ERROR] Failed to spawn objects for episode {episode_idx}: {e}")
        with open(traj_dir / "spawn_error.txt", 'w') as f:
            import traceback
            f.write(traceback.format_exc())
        return {"task_id": task_id, "status": "spawn_error", "error": str(e)}

    # 2. Settle simulation
    state.world_graph = deepcopy(episode['initial_world_graph'])
    step_simulation(env, 500)

    # 3. Reset agent state
    state.current_obs_dict = None
    state.current_marker_map = None
    state.current_landmark = None
    state.current_inv = None
    state.current_pos = None
    state.camera_orientation = None

    # 4. Setup conversation
    instruction = episode['task_description']
    messages: List[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Your task is: {instruction}"},
    ]

    # Save task metadata
    with open(traj_dir / "task_meta.json", 'w') as f:
        json.dump({
            "task_id": task_id,
            "episode_idx": episode_idx,
            "task_description": instruction,
            "model": MODEL_NAME,
        }, f, indent=2)

    # 5. ReAct loop
    total_time = 0.0
    step_count = 0
    final_action = None

    for _ in range(MAX_STEP):
        # Call LLM
        start_time = time.time()
        try:
            response = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                tools=EVAL_TOOLS,
                tool_choice="auto",
                max_tokens=1000,
            )
        except Exception as e:
            print(f"[ERROR] LLM API error at step {step_count}: {e}")
            break
        infer_time = time.time() - start_time
        total_time += infer_time

        if not response.choices:
            print(f"[WARN] No response from model at step {step_count}.")
            break

        message = response.choices[0].message
        messages.append(message)

        # If model doesn't call a tool, nudge it
        if not message.tool_calls:
            messages.append({
                "role": "user",
                "content": "Please use the available tools to proceed with the task.",
            })
            continue

        step_count += 1
        tool_call = message.tool_calls[0]
        func_name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)
        print(f"  Step {step_count}: {func_name}({args})")

        # Check for finish
        if func_name == "finish":
            save_step_artifacts(
                traj_dir, step_count, message, tool_call,
                "text", "Episode finished by agent.",
                get_debug_info(state), infer_time, total_time,
            )
            final_action = "finish"
            break

        # Execute action directly (no MCP)
        result_type, result_data, debug_info = dispatch_action(
            func_name, args, state, env,
        )

        # Save step artifacts
        save_step_artifacts(
            traj_dir, step_count, message, tool_call,
            result_type, result_data, debug_info,
            infer_time, total_time,
        )

        # Feed result back to LLM
        if result_type == "text":
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result_data,
            })
        elif result_type == "image":
            # Tool response (required by OpenAI API for tool calls)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": "Image observation returned. See the image below.",
            })
            # Image as user message
            messages.append({
                "role": "user",
                "content": [{
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{result_data}",
                    },
                }],
            })

    # Save episode summary
    summary = {
        "task_id": task_id,
        "episode_idx": episode_idx,
        "total_steps": step_count,
        "total_inference_time": total_time,
        "final_action": final_action,
        "status": "finished" if final_action == "finish" else "max_steps_reached",
    }
    with open(traj_dir / "episode_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"  Episode {episode_idx} ({task_id}): {summary['status']} in {step_count} steps")

    return summary


# =============================================================================
# Main
# =============================================================================

def main():
    """Main entry point for batch evaluation."""
    os.makedirs(EVAL_OUTPUT_PATH, exist_ok=True)

    results = []
    for idx, episode in enumerate(processed_eval_episodes):
        print(f"\n[Episode {idx}/{len(processed_eval_episodes)}] "
              f"task_id={episode['task_id']}")
        result = asyncio.run(run_episode(idx, episode))
        results.append(result)

    # Save overall results
    finished = sum(1 for r in results if r.get('status') == 'finished')
    skipped = sum(1 for r in results if r.get('status') == 'skipped')
    errors = sum(1 for r in results if 'error' in r.get('status', ''))

    print(f"\n{'='*60}")
    print(f"Evaluation complete: {len(results)} episodes")
    print(f"  Finished: {finished}")
    print(f"  Skipped:  {skipped}")
    print(f"  Errors:   {errors}")
    print(f"{'='*60}")

    with open(EVAL_OUTPUT_PATH / "eval_results.json", 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
