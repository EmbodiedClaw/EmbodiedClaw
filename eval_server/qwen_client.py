"""
Qwen MCP client for evaluation.

Runs local Qwen inference as MCP client, while simulation runs in a separate
process via eval_server.mcp_server.

Usage:
    # terminal 1: start server
    PORT=8080 TARGET_SCENE_ID=... TASK_SOURCE_PATH=... ORIGINAL_TASK_PATH=... TRAJ_PATH=... \
    python eval_server/mcp_server.py

    # terminal 2: start client
    MODEL_PATH=/path/to/qwen/checkpoint \
    MCP_SERVER_URL=http://localhost:8080/sse \
    TRAJ_PATH=eval_output \
    START_IDX=0 END_IDX=999999 \
    python -m eval_server.qwen_client
"""

import asyncio
import ast
import base64
import io
import json
import os
import time
import traceback
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image as PILImage
from mcp import ClientSession
from mcp.client.sse import sse_client
from transformers import AutoModelForImageTextToText, AutoProcessor


# =============================================================================
# Configuration
# =============================================================================

MODEL_PATH = os.environ["MODEL_PATH"]
MODEL_NAME = MODEL_PATH.rstrip("/").split("/")[-1]
MODEL_DIR_NAME = MODEL_NAME.replace("-", "_").replace("/", "_")

MCP_SERVER_URL = os.environ["MCP_SERVER_URL"]
MAX_STEP = int(os.environ.get("MAX_STEP", "30"))
START_IDX = int(os.environ.get("START_IDX", "0"))
END_IDX = int(os.environ.get("END_IDX", "999999"))

OUTPUT_BASE = Path(os.environ.get("EVAL_OUTPUT_PATH", os.environ.get("TRAJ_PATH", "eval_output")))

print(f"Loading Qwen model from: {MODEL_PATH}")
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_PATH,
    dtype="auto",
    device_map="auto",
    local_files_only=True,
)
processor = AutoProcessor.from_pretrained(MODEL_PATH)

LORA_PATH = os.environ.get("LORA_PATH", None)
if LORA_PATH:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, LORA_PATH)
print("Model loaded successfully!")


# =============================================================================
# Prompt
# =============================================================================

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


def build_scene_description(rooms_and_furniture: Dict[str, List[str]]) -> str:
    lines = []
    for idx, (room, furnitures) in enumerate(sorted(rooms_and_furniture.items()), 1):
        room_display = room.replace("/", "_").replace(" ", "_")
        furniture_str = ", ".join(sorted(furnitures))
        lines.append(f"{idx}) In {room_display}: {furniture_str}.")
    return "\n".join(lines)


def build_prompt_template(scene_description: str) -> str:
    return f"""\
You are a robot operating in a home. Given a task, you must accomplish the task using a defined
set of actions to achieve the desired outcome.

## Scene Description
The scene has the following rooms and receptacles:

{scene_description}
 ###

## Tool list

You MUST call the following tools to accomplish the task

{TOOL_LIST}

## Output Format Requirements
1. Think step-by-step and reflect on the current state, what has been done, what needs to be done next, and why. Be specific about what needs to be accomplished. Essentially, narrate your thought process: e.g., analyzing the goal, considering which tools to use, verifying if previous actions succeeded, and how you plan to proceed.

2. Summary: This field is an object (dictionary) that provides a concise summary of the situation and plan. It has three sub-fields:

- History: A string summarizing the recent history of actions taken and key observations/results. For example, it might list actions that have been executed so far and whether they succeeded, or what was observed. Grasp/hold state rule: If the agent executed a Pick action SUCCESSFULLY, the robot is considered to be holding that object continuously until a SUCCESSFUL Place action is completed.

- New Schedule: Based on the history, develop a new advanced plan or an upcoming sequence of actions (3-4 high-level subtasks). Focus on high-level goals and descriptive references rather than exact object identifiers. Use this to list any updated high-level plan or sequence of upcoming actions if the original plan has changed, or to indicate a re-planning.

- Current subtask: A short string describing what the robot needs to do NEXT based on the current state (after all History actions have been completed).

3. Then output your action decision in JSON format.

## Important Notes
1)The Marker_id is ONLY used to refer to specific objects AT CURRENT STEP. NEVER use the previous marker_id for the current step, as the object positions may have changed. Instead, you MUST REFER TO THE LATEST OBSERVATIONS TO GET THE CORRECT marker_id FOR THE CURRENT STEP.
2)**Prevent Repeating Action Sequences**: Do not repeatedly execute the same action or sequence of actions.
3)**Multiple Instances**: There may be multiple instances of the same object, distinguished by an index following their names, e.g., Cabinet_2, Cabinet_3.
4)The open/close action is only valid when you don't have any inventory.

Your Task is:
{{TASK}}

Current Task Progress:
{{TASK_PROGRESS}}

Recent Interaction History:
Last Action: {{LAST_ACTION}} {{LAST_OBS}}"""


# =============================================================================
# Helpers
# =============================================================================


def parse_json_loose(text: str) -> Dict[str, Any]:
    text = text.strip()
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return json.loads(text[start:end + 1])

    raise ValueError("Cannot parse model output as JSON")


def decode_base64_image(image_b64: str, out_path: Path) -> None:
    img_bytes = base64.b64decode(image_b64)
    img = PILImage.open(io.BytesIO(img_bytes))
    img.save(out_path)


def evaluate_world_graph(actual: Dict[str, Any], target: Dict[str, Any]) -> bool:
    if not isinstance(actual, dict) or not isinstance(target, dict):
        return False

    for furniture in target:
        target_content = sorted(target.get(furniture, {}).get("content", []))
        actual_content = sorted(actual.get(furniture, {}).get("content", []))
        if target_content != actual_content:
            return False
    return True


async def fetch_next_task(session: ClientSession) -> Optional[Dict[str, Any]]:
    result = await session.call_tool("finish", arguments={})

    if not result.content:
        return None

    for c in result.content:
        if getattr(c, "type", None) != "text":
            continue

        text = c.text.strip()
        if text == "All evaluation episodes completed.":
            return None

        try:
            return json.loads(text)
        except Exception:
            print(f"[WARN] Cannot parse finish response as JSON: {text[:200]}")
            return None

    return None


def parse_tool_result(result) -> Tuple[str, str, Dict[str, Any], List[str]]:
    texts: List[str] = []
    images_b64: List[str] = []
    debug_info: Dict[str, Any] = {}

    for content in result.content:
        ctype = getattr(content, "type", "")
        if ctype == "text":
            txt = content.text
            if txt.startswith("Debug Info:\n"):
                raw_debug = txt[len("Debug Info:\n"):]
                try:
                    debug_info = json.loads(raw_debug)
                except Exception:
                    debug_info = {"raw": raw_debug}
            else:
                texts.append(txt)
        elif ctype == "image":
            images_b64.append(content.data)

    if images_b64 and not texts:
        return "image", images_b64[0], debug_info, images_b64

    return "text", "\n".join(texts).strip(), debug_info, images_b64


def extract_world_graph_from_debug(debug_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Extract machine-readable world graph from debug payload.

    Compatible with both new format ("world_graph": dict) and legacy format
    ("WORLD_GRAPH": "fur: [obj1, ...]" summary string).
    """
    if not isinstance(debug_info, dict):
        return None

    wg = debug_info.get("world_graph")
    if isinstance(wg, dict):
        return wg

    # Fallback compatibility: some versions may return uppercase key.
    wg_upper = debug_info.get("WORLD_GRAPH")
    if isinstance(wg_upper, dict):
        return wg_upper

    if not isinstance(wg_upper, str):
        return None

    parsed: Dict[str, Any] = {}
    for raw_line in wg_upper.splitlines():
        line = raw_line.strip()
        if not line or ":" not in line:
            continue

        furniture, content_str = line.split(":", 1)
        furniture = furniture.strip()
        content_str = content_str.strip()

        try:
            content = ast.literal_eval(content_str)
            if not isinstance(content, list):
                content = []
        except Exception:
            content = []

        parsed[furniture] = {"content": content}

    return parsed or None


def save_step_artifacts(
    traj_dir: Path,
    step_count: int,
    tool_name: str,
    tool_args: dict,
    result_type: str,
    result_data: str,
    debug_info: dict,
    infer_time: float,
    total_time: float,
):
    step_dir = traj_dir / f"step_{step_count:03d}"
    os.makedirs(step_dir, exist_ok=True)

    with open(step_dir / "tool_call.json", "w") as f:
        json.dump({"tool_name": tool_name, "args": tool_args}, f, indent=2)

    with open(step_dir / "inference_time.json", "w") as f:
        json.dump(
            {
                "infer_time": infer_time,
                "total_time_till_now": total_time,
            },
            f,
            indent=2,
        )

    if result_type == "text":
        with open(step_dir / "textual_observation.txt", "w") as f:
            f.write(result_data)
    elif result_type == "image":
        decode_base64_image(result_data, step_dir / "image_observation.png")

    with open(step_dir / "debug_info.json", "w") as f:
        json.dump(debug_info, f, indent=2, default=str)


# =============================================================================
# Qwen inference
# =============================================================================


def generate_qwen_response(
    prompt_template: str,
    task: str,
    task_progress: dict,
    last_action: dict,
    obs_text: str,
    obs_images: List[str],
    step_count: int,
    traj_dir: Path,
) -> Dict[str, Any]:
    step_dir = traj_dir / f"step_{step_count:03d}"
    os.makedirs(step_dir, exist_ok=True)

    user_text = (
        prompt_template
        .replace("{TASK}", task)
        .replace("{TASK_PROGRESS}", json.dumps(task_progress, ensure_ascii=False))
        .replace("{LAST_ACTION}", json.dumps(last_action, ensure_ascii=False))
        .replace("{LAST_OBS}", obs_text)
    )

    with open(step_dir / "user_message.txt", "w") as f:
        f.write(user_text)

    content = [{"type": "text", "text": user_text}]
    for img_path in obs_images:
        content.append({"type": "image", "image": img_path})

    messages = [{"role": "user", "content": content}]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

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

    with open(step_dir / "raw_response.txt", "w") as f:
        f.write(raw_text)

    parsed = parse_json_loose(raw_text)
    task_progress_out = parsed["summary"]
    action = parsed["next action"]

    return {
        "task_progress": task_progress_out,
        "action": action,
    }


# =============================================================================
# Episode loop
# =============================================================================


async def run_episode(session: ClientSession, task_info: Dict[str, Any]) -> Tuple[dict, Optional[dict]]:
    task_id = task_info["task_id"]
    scene_id = task_info.get("scene_id", "unknown_scene")
    episode_idx = int(task_info.get("episode_idx", -1))

    traj_dir = OUTPUT_BASE / scene_id / task_id / MODEL_DIR_NAME

    if traj_dir.exists() and len(os.listdir(str(traj_dir))) > 0:
        print(f"[SKIP] Episode {episode_idx} ({task_id}) already processed.")
        next_task = await fetch_next_task(session)
        return {"task_id": task_id, "status": "skipped"}, next_task

    os.makedirs(traj_dir, exist_ok=True)

    instruction = task_info["task_description"]
    rooms_and_furniture = task_info.get("rooms_and_furniture", {})
    scene_description = build_scene_description(rooms_and_furniture)
    prompt_template = build_prompt_template(scene_description)

    with open(traj_dir / "task_meta.json", "w") as f:
        json.dump(
            {
                "task_id": task_id,
                "episode_idx": episode_idx,
                "task_description": instruction,
                "model": MODEL_NAME,
                "scene_id": scene_id,
                "source_furniture": task_info.get("source_furniture"),
                "destination_furniture": task_info.get("destination_furniture"),
                "target_object": task_info.get("target_object"),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    total_time = 0.0
    step_count = 0
    final_action = None
    success = False

    task_progress: dict = {}
    last_action: dict = {}
    obs_text: str = ""
    obs_images: List[str] = []

    latest_world_graph = task_info.get("initial_world_graph", {})
    next_task: Optional[Dict[str, Any]] = None

    for _ in range(MAX_STEP):
        step_dir = traj_dir / f"step_{step_count:03d}"
        os.makedirs(step_dir, exist_ok=True)

        start_time = time.time()
        try:
            llm_res = generate_qwen_response(
                prompt_template=prompt_template,
                task=instruction,
                task_progress=task_progress,
                last_action=last_action,
                obs_text=obs_text,
                obs_images=obs_images,
                step_count=step_count,
                traj_dir=traj_dir,
            )
        except Exception as e:
            print(f"[ERROR] Qwen inference error at step {step_count}: {e}")
            traceback.print_exc()
            with open(step_dir / "inference_error.txt", "w") as f:
                f.write(traceback.format_exc())
            break

        infer_time = time.time() - start_time
        total_time += infer_time

        task_progress = llm_res["task_progress"]
        action = llm_res["action"]
        func_name = action.get("tool_name", "")
        args = action.get("args", {})

        if func_name == "place_on_top":
            func_name = "place"

        print(f"  Step {step_count}: {func_name}({args}) [{infer_time:.1f}s]")

        if func_name == "finish":
            success = evaluate_world_graph(
                latest_world_graph,
                task_info.get("target_world_graph", {}),
            )
            save_step_artifacts(
                traj_dir=traj_dir,
                step_count=step_count,
                tool_name=func_name,
                tool_args=args,
                result_type="text",
                result_data=f"Episode finished. Success: {success}",
                debug_info={"world_graph": latest_world_graph},
                infer_time=infer_time,
                total_time=total_time,
            )
            final_action = "finish"
            step_count += 1
            next_task = await fetch_next_task(session)
            break

        try:
            mcp_result = await session.call_tool(func_name, arguments=args)
            result_type, result_data, debug_info, images_b64 = parse_tool_result(mcp_result)
        except Exception as e:
            print(f"[ERROR] Action execution error at step {step_count}: {e}")
            traceback.print_exc()
            with open(step_dir / "action_error.txt", "w") as f:
                f.write(traceback.format_exc())
            break

        parsed_wg = extract_world_graph_from_debug(debug_info)
        if parsed_wg is not None:
            latest_world_graph = parsed_wg

        save_step_artifacts(
            traj_dir=traj_dir,
            step_count=step_count,
            tool_name=func_name,
            tool_args=args,
            result_type=result_type,
            result_data=result_data,
            debug_info=debug_info,
            infer_time=infer_time,
            total_time=total_time,
        )

        last_action = {"tool_name": func_name, "args": args}
        obs_text = ""
        obs_images = []

        if result_type == "text":
            if result_data:
                obs_text = f"\nAfter performing the last action, you observed the text: {result_data}"
            if images_b64:
                obs_paths = []
                for i, img_b64 in enumerate(images_b64):
                    img_path = step_dir / f"obs_image_{i}.png"
                    decode_base64_image(img_b64, img_path)
                    obs_paths.append(str(img_path))
                obs_images.extend(obs_paths)
                obs_text += "\nYou also observed image(s): " + " ".join(["<image>" for _ in obs_paths])
        elif result_type == "image":
            obs_paths = []
            for i, img_b64 in enumerate(images_b64):
                img_path = step_dir / f"image_observation_{i}.png"
                decode_base64_image(img_b64, img_path)
                obs_paths.append(str(img_path))
            obs_images.extend(obs_paths)
            obs_text = "\nAfter performing the last action, you observed image(s): " + " ".join(
                ["<image>" for _ in obs_paths]
            )

        step_count += 1
    else:
        next_task = None

    summary = {
        "task_id": task_id,
        "episode_idx": episode_idx,
        "total_steps": step_count,
        "total_inference_time": total_time,
        "final_action": final_action,
        "success": success,
        "status": "finished" if final_action == "finish" else "max_steps_reached",
    }

    with open(traj_dir / "episode_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(
        f"  Episode {episode_idx} ({task_id}): {summary['status']} "
        f"success={success} in {step_count} steps"
    )

    return summary, next_task


# =============================================================================
# Main
# =============================================================================


async def main_async():
    print(f"\nQwen MCP Evaluation: {MODEL_NAME}")
    print(f"MCP server: {MCP_SERVER_URL}")
    print(f"Episode range: [{START_IDX}, {END_IDX})")
    print(f"Output: {OUTPUT_BASE}")
    print(f"Max steps per episode: {MAX_STEP}\n")

    os.makedirs(OUTPUT_BASE, exist_ok=True)

    results: List[dict] = []

    async with AsyncExitStack() as stack:
        read_stream, write_stream = await stack.enter_async_context(sse_client(MCP_SERVER_URL))
        session = await stack.enter_async_context(ClientSession(read_stream, write_stream))
        await session.initialize()

        tools = await session.list_tools()
        print(f"Connected to MCP server, tools={len(tools.tools)}")

        task_info = await fetch_next_task(session)

        while task_info is not None:
            episode_idx = int(task_info.get("episode_idx", -1))

            if episode_idx < START_IDX:
                task_info = await fetch_next_task(session)
                continue

            if episode_idx >= END_IDX:
                break

            print(f"\n[Episode {episode_idx}] task_id={task_info.get('task_id')}")
            result, next_task = await run_episode(session, task_info)
            results.append(result)
            task_info = next_task

    finished = sum(1 for r in results if r.get("status") == "finished")
    skipped = sum(1 for r in results if r.get("status") == "skipped")
    errors = sum(1 for r in results if "error" in r.get("status", ""))
    successes = sum(1 for r in results if r.get("success", False))

    print(f"\n{'=' * 60}")
    print(f"Evaluation complete: {len(results)} episodes")
    print(f"  Finished:  {finished}")
    print(f"  Successes: {successes}")
    print(f"  Skipped:   {skipped}")
    print(f"  Errors:    {errors}")
    if finished > 0:
        print(f"  Success rate: {successes / finished * 100:.1f}%")
    print(f"{'=' * 60}")

    out_path = OUTPUT_BASE / f"qwen_client_results_{START_IDX}_{END_IDX}.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "mode": "mcp_client",
                "model": MODEL_NAME,
                "model_path": MODEL_PATH,
                "mcp_server_url": MCP_SERVER_URL,
                "start_idx": START_IDX,
                "end_idx": END_IDX,
                "total": len(results),
                "finished": finished,
                "successes": successes,
                "skipped": skipped,
                "errors": errors,
                "success_rate": successes / finished if finished > 0 else 0,
                "episodes": results,
            },
            f,
            indent=2,
        )


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
