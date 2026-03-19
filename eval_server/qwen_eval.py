"""
Qwen VLM checkpoint evaluation runner.

Single-process: local Qwen inference + Grutopia simulation on the same GPU.
Based on agent_eval.py, replacing OpenAI API with local transformers inference.

Usage:
    MODEL_PATH=/path/to/qwen/checkpoint \
    TARGET_SCENE_ID=MVUCSQAKTKJ5EAABAAAAABY8 \
    TASK_SOURCE_PATH=metadata/benchmarks/verify_results/visual_pnp/MVUCSQAKTKJ5EAABAAAAABY8/valid_tasks.json \
    ORIGINAL_TASK_PATH=metadata/benchmarks/visual_pnp.json \
    TRAJ_PATH=eval_output \
    START_IDX=0 END_IDX=114 \
    python -m eval_server.qwen_eval
"""

import json
import os
import base64
import time
import traceback
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Any, Optional

from PIL import Image as PILImage
import io

# qwen_eval uses precomputed nav positions in actions.nav_to
os.environ['IS_TEST'] = '1'

# =============================================================================
# Simulation imports (initialises env at import time)
# =============================================================================

from eval_server.mcp_env import (
    env,
    camera,
    processed_eval_episodes,
    spawn_objects_by_world_graph,
    TRAJ_PATH,
    TARGET_SCENE_ID,
    object_per_room,
)
from eval_server.perception_utils import init_annotators
from eval_server.actions import (
    EvalState,
    dispatch_action,
    step_simulation,
    get_debug_info,
    get_rgb_observation,
    rgb_array_to_base64,
)

# =============================================================================
# Model imports & loading
# =============================================================================

from transformers import AutoModelForImageTextToText, AutoProcessor
from eval_server.history_manager import HistoryManager

MODEL_PATH = os.environ['MODEL_PATH']
MODEL_NAME = MODEL_PATH.rstrip('/').split('/')[-1]
MODEL_DIR_NAME = MODEL_NAME.replace('-', '_').replace('/', '_')

print(f"Loading Qwen model from: {MODEL_PATH}")
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_PATH, dtype="auto", device_map="auto", local_files_only=True,
)
processor = AutoProcessor.from_pretrained(MODEL_PATH)
print("Model loaded successfully!")

# =============================================================================
# Configuration
# =============================================================================

MAX_STEP = int(os.environ.get('MAX_STEP', '30'))
START_IDX = int(os.environ.get('START_IDX', '0'))
END_IDX = int(os.environ.get('END_IDX', str(len(processed_eval_episodes))))
EVAL_OUTPUT_PATH = Path(os.environ.get(
    'EVAL_OUTPUT_PATH',
    str(TRAJ_PATH / TARGET_SCENE_ID),
))

# Initialise perception annotators
init_annotators(camera, resolution=(640, 480))

# Global state
state = EvalState()


# =============================================================================
# Prompt construction
# =============================================================================

def build_scene_description() -> str:
    """Build scene description from object_per_room, converting room/N → room_N."""
    lines = []
    for idx, (room, furnitures) in enumerate(sorted(object_per_room.items()), 1):
        room_display = room.replace('/', '_').replace(' ', '_')
        furniture_str = ', '.join(sorted(furnitures))
        lines.append(f"{idx}) In {room_display}: {furniture_str}.")
    return '\n'.join(lines)


SCENE_DESCRIPTION = build_scene_description()

TOOL_LIST = """\
- {"name": "nav_to", "description": "navigate to a receptacle", "parameters": {"type": "object", "required": ["receptacle_name"], "properties": {"receptacle_name": {"type": "string", "description": "the name of the receptacle to perform navigation"}}}}
- {"name": "walk_around", "description": "walk around the current receptacle to get all objects on top of it.", "parameters": {}}
- {"name": "gaze_at", "description": "gaze at and approach an object for manipulation.", "parameters": {"type": "object", "required": ["marker_id"], "properties": {"marker_id": {"type": "string", "description": "the marker id of the object to gaze"}}}}
- {"name": "show_object_by_category", "description": "detect objects of a category in your view and show their markers.", "parameters": {"type": "object", "required": ["target_category"], "properties": {"target_category": {"type": "string", "description": "the name of the object to gaze"}}}}
- {"name": "show_receptacles", "description": "highlight all receptacle objects in your view and show their markers.", "parameters": {}}
- {"name": "pick", "description": "pick up a specific object in your view by marker id, the object will become your inventory", "parameters": {"type": "object", "required": ["marker_id"], "properties": {"marker_id": {"type": "string", "description": "id of the marker on the object to pick up"}}}}
- {"name": "place", "description": "place your inventory on top of a receptacle surface.", "parameters": {"type": "object", "required": ["marker_id"], "properties": {"marker_id": {"type": "string", "description": "id of the marker on the receptacle surface to place your inventory"}}}}
- {"name": "open", "description": "open the door of an articulated object", "parameters": {"type": "object", "required": ["marker_id"], "properties": {"marker_id": {"type": "string", "description": "marker id of the receptacle door to open"}}}}
- {"name": "close", "description": "close the door of an articulated object", "parameters": {"type": "object", "required": ["marker_id"], "properties": {"marker_id": {"type": "string", "description": "marker id of the receptacle door to close"}}}}
- {"name": "finish", "description": "you must call this tool to finish the episode when you think you have completed all the instructions.", "parameters": {}}
- {"name": "ask", "description": "ask the user a question to get more information about the task.", "parameters": {"type": "object", "required": ["question"], "properties": {"question": {"type": "string", "description": "the question to ask the user"}}}}"""

PROMPT_TEMPLATE = f"""\
You are a robot operating in a home. Given a task, you must accomplish the task using a defined
set of actions to achieve the desired outcome.

## Scene Description
The scene has the following rooms and receptacles:

{SCENE_DESCRIPTION}
 ###

## Tool list

You MUST call the following tools to accomplish the task

{TOOL_LIST}

## Output Format Requirements
1. Think step-by-step and reflect on the current state, what has been done, what needs to be done next, and why. Be specific about what needs to be accomplished. Essentially, narrate your thought process: e.g., analyzing the goal, considering which tools to use, verifying if previous actions succeeded, and how you plan to proceed.

2. Summary: This field is an object (dictionary) that provides a concise summary of the situation and plan. It has three sub-fields:

- History: A string summarizing the recent history of actions taken and key observations/results. For example, it might list actions that have been executed so far and whether they succeeded, or what was observed . Grasp/hold state rule: If the agent executed a Pick action SUCCESSFULLY, the robot is considered to be holding that object continuously until a SUCCESSFUL Place action is completed.

- New Schedule: Based on the history, develop a new advanced plan or an upcoming sequence of actions (3-4 high-level subtasks). Focus on high-level goals and descriptive references rather than exact object identifiers. Use this to list any updated high-level plan or sequence of upcoming actions if the original plan has changed, or to indicate a re-planning.

- Current subtask: A short string describing what the robot needs to do NEXT based on the current state (after all History actions have been completed).

3. Then output your action decision in JSON format.

## Important Notes
1)The Marker_id is ONLY used to refer to specific objects AT CURRENT STEP. NEVER use the previous marker_id for the current step, as the object positions may have changed. Instead, you MUST REFER TO THE LATEST OBSERVATIONS TO GET THE CORRECT marker_id FOR THE CURRENT STEP.
 2)**Prevent Repeating Action Sequences**: Do not repeatedly execute the same action or sequence of actions.
 3)**Multiple Instances**: There may be multiple instances of the same object, distinguished by an index following their names, e.g., Cabinet_2, Cabinet_3.
3) The open/close action is only valid when you don't have any inventory.

Your Task is:
{{TASK}}

Current Task Progress:
{{TASK_PROGRESS}}

Recent Interaction History:
Last Action: {{LAST_ACTION}} {{LAST_OBS}}"""


# =============================================================================
# Qwen inference
# =============================================================================

def generate_qwen_response(
    task: str,
    accumulated_history: Optional[str],
    last_action: dict,
    obs_text: str,
    obs_images: List[str],
    step_count: int,
    traj_dir: Path,
) -> Dict[str, Any]:
    """
    Build prompt, run Qwen inference, parse response.

    Args:
        task: task description string
        accumulated_history: framework-maintained history string (None for step 0)
        last_action: action dict from previous step (or {} for step 0)
        obs_text: textual observation string (may include <image> placeholders)
        obs_images: list of image file paths for the current observation
        step_count: current step index
        traj_dir: directory to save artifacts

    Returns:
        {"gpt_history": str, "action": {"tool_name": str, "args": dict}}
    """
    step_dir = traj_dir / f"step_{step_count:03d}"
    os.makedirs(step_dir, exist_ok=True)

    # Format task_progress field to match training data format:
    # step 0 → '{}'; step 1+ → "{'History': 'n) ...\nn+1) ...'}"
    if accumulated_history is None:
        task_progress_str = '{}'
    else:
        task_progress_str = f"{{'History': '{accumulated_history}'}}"

    # Build user message text
    user_text = (
        PROMPT_TEMPLATE
        .replace('{TASK}', task)
        .replace('{TASK_PROGRESS}', task_progress_str)
        .replace('{LAST_ACTION}', json.dumps(last_action))
        .replace('{LAST_OBS}', obs_text)
    )

    with open(step_dir / "user_message.txt", 'w') as f:
        f.write(user_text)

    # Build message content
    content = [{"type": "text", "text": user_text}]
    for img_path in obs_images:
        content.append({"type": "image", "image": img_path})

    messages = [{"role": "user", "content": content}]

    # Tokenize
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    # Generate
    generated_ids = model.generate(**inputs, max_new_tokens=4096)
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    raw_text = output_text[0]

    with open(step_dir / "raw_response.txt", 'w') as f:
        f.write(raw_text)

    # Parse: strip <think>...</think>, then JSON
    final_output = raw_text.strip()
    if '</think>' in final_output:
        final_output = final_output.split('</think>')[-1].strip()

    formatted_output = json.loads(final_output)
    gpt_history = formatted_output['summary']['History']
    action = formatted_output['next action']

    return {
        "gpt_history": gpt_history,
        "action": action,
    }


# =============================================================================
# Metric
# =============================================================================

def evaluate_world_graph(state: EvalState, episode: dict) -> bool:
    """
    Check if current world graph matches target.

    Target = initial_world_graph with target object moved from src to dest.
    """
    target_wg = deepcopy(episode['initial_world_graph'])
    target_obj_name = episode['target_object_name']
    src = episode['src']
    dest = episode['dest']

    # Move target object from src to dest
    if target_obj_name in target_wg.get(src, {}).get('content', []):
        target_wg[src]['content'].remove(target_obj_name)
    if dest in target_wg:
        target_wg[dest]['content'].append(target_obj_name)

    # Compare content lists for all furniture
    for furniture in target_wg:
        target_content = sorted(target_wg[furniture].get('content', []))
        actual_content = sorted(state.world_graph.get(furniture, {}).get('content', []))
        if target_content != actual_content:
            return False
    return True


# =============================================================================
# Per-step saving
# =============================================================================

def save_step_artifacts(
    traj_dir: Path,
    step_count: int,
    raw_response: str,
    tool_name: str,
    tool_args: dict,
    result_type: str,
    result_data: str,
    debug_info: dict,
    infer_time: float,
    total_time: float,
):
    """Save all artifacts for a single evaluation step."""
    step_dir = traj_dir / f"step_{step_count:03d}"
    os.makedirs(step_dir, exist_ok=True)

    # Tool call info
    with open(step_dir / "tool_call.json", 'w') as f:
        json.dump({"tool_name": tool_name, "args": tool_args}, f, indent=2)

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

def run_episode(episode_idx: int, episode: dict) -> dict:
    """Run one evaluation episode with Qwen ReAct loop."""
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

    # 4. Save task metadata
    instruction = episode['task_description']
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
    success = False

    history_manager = HistoryManager()
    accumulated_history: Optional[str] = None  # None → step-0 format (empty history)
    last_action: dict = {}
    obs_text: str = ''
    obs_images: List[str] = []

    for _ in range(MAX_STEP):
        step_dir = traj_dir / f"step_{step_count:03d}"
        os.makedirs(step_dir, exist_ok=True)

        # Call Qwen
        start_time = time.time()
        try:
            res = generate_qwen_response(
                task=instruction,
                accumulated_history=accumulated_history,
                last_action=last_action,
                obs_text=obs_text,
                obs_images=obs_images,
                step_count=step_count,
                traj_dir=traj_dir,
            )
        except Exception as e:
            print(f"[ERROR] Qwen inference error at step {step_count}: {e}")
            traceback.print_exc()
            with open(step_dir / "inference_error.txt", 'w') as f:
                f.write(traceback.format_exc())
            break
        infer_time = time.time() - start_time
        total_time += infer_time

        gpt_history = res['gpt_history']
        action = res['action']
        func_name = action.get('tool_name', '')
        args = action.get('args', {})

        # Update HistoryManager with the new step extracted from model output
        new_step = history_manager.extract_new_step_from_gpt_output(gpt_history)
        if step_count == 0:
            # Step 0: initialize with model's first history entry
            history_manager.accumulated_history = [f"0) {new_step}"]
            history_manager.step_num = 0
        else:
            history_manager.add_step(new_step)
        accumulated_history = history_manager.get_formatted_history()

        # Tool name mapping
        if func_name == 'place_on_top':
            func_name = 'place'

        print(f"  Step {step_count}: {func_name}({args})  [{infer_time:.1f}s]")

        # Check for finish
        if func_name == "finish":
            success = evaluate_world_graph(state, episode)
            save_step_artifacts(
                traj_dir, step_count, '',
                func_name, args,
                "text", f"Episode finished. Success: {success}",
                get_debug_info(state), infer_time, total_time,
            )
            final_action = "finish"
            step_count += 1
            break

        # Execute action
        try:
            result_type, result_data, debug_info = dispatch_action(
                func_name, args, state, env,
            )
        except Exception as e:
            print(f"[ERROR] Action execution error at step {step_count}: {e}")
            traceback.print_exc()
            with open(step_dir / "action_error.txt", 'w') as f:
                f.write(traceback.format_exc())
            break

        # Save step artifacts
        save_step_artifacts(
            traj_dir, step_count, '',
            func_name, args,
            result_type, result_data,
            debug_info, infer_time, total_time,
        )

        # Build observation for next step
        last_action = {"tool_name": func_name, "args": args}
        obs_text = ''
        obs_images = []

        if result_type == "text":
            obs_text = f"\nAfter performing the last action, you observed the text: {result_data}"
            # Also capture an RGB observation image
            rgb_obs = get_rgb_observation()
            rgb_b64 = rgb_array_to_base64(rgb_obs)
            img_path = step_dir / "obs_image.png"
            img_data = base64.b64decode(rgb_b64)
            PILImage.open(io.BytesIO(img_data)).save(img_path)
            obs_text += "\nYou also observed the following image: <image>"
            obs_images.append(str(img_path))
        elif result_type == "image":
            obs_text = "\nAfter performing the last action, you observed the image <image>"
            img_path = step_dir / "image_observation.png"
            # image_observation.png already saved by save_step_artifacts
            obs_images.append(str(img_path))

        step_count += 1

    # Save episode summary
    summary = {
        "task_id": task_id,
        "episode_idx": episode_idx,
        "total_steps": step_count,
        "total_inference_time": total_time,
        "final_action": final_action,
        "success": success,
        "status": "finished" if final_action == "finish" else "max_steps_reached",
    }
    with open(traj_dir / "episode_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"  Episode {episode_idx} ({task_id}): {summary['status']} "
          f"success={success} in {step_count} steps")

    return summary


# =============================================================================
# Main
# =============================================================================

def main():
    """Main entry point for Qwen batch evaluation."""
    episodes = processed_eval_episodes[START_IDX:END_IDX]
    print(f"\nQwen Evaluation: {MODEL_NAME}")
    print(f"Episodes: [{START_IDX}, {END_IDX}) = {len(episodes)} episodes")
    print(f"Output: {EVAL_OUTPUT_PATH}")
    print(f"Max steps per episode: {MAX_STEP}\n")

    os.makedirs(EVAL_OUTPUT_PATH, exist_ok=True)

    results = []
    for i, episode in enumerate(episodes):
        global_idx = START_IDX + i
        print(f"\n[Episode {global_idx}/{END_IDX}] task_id={episode['task_id']}")
        result = run_episode(global_idx, episode)
        results.append(result)

    # Aggregate results
    finished = sum(1 for r in results if r.get('status') == 'finished')
    skipped = sum(1 for r in results if r.get('status') == 'skipped')
    errors = sum(1 for r in results if 'error' in r.get('status', ''))
    successes = sum(1 for r in results if r.get('success', False))

    print(f"\n{'=' * 60}")
    print(f"Evaluation complete: {len(results)} episodes")
    print(f"  Finished:  {finished}")
    print(f"  Successes: {successes}")
    print(f"  Skipped:   {skipped}")
    print(f"  Errors:    {errors}")
    if finished > 0:
        print(f"  Success rate: {successes / finished * 100:.1f}%")
    print(f"{'=' * 60}")

    with open(EVAL_OUTPUT_PATH / f"eval_results_{START_IDX}_{END_IDX}.json", 'w') as f:
        json.dump({
            "model": MODEL_NAME,
            "model_path": MODEL_PATH,
            "scene_id": TARGET_SCENE_ID,
            "start_idx": START_IDX,
            "end_idx": END_IDX,
            "total": len(results),
            "finished": finished,
            "successes": successes,
            "skipped": skipped,
            "errors": errors,
            "success_rate": successes / finished if finished > 0 else 0,
            "episodes": results,
        }, f, indent=2)


if __name__ == "__main__":
    main()
