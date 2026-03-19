"""
MCP Server for evaluation environment.

Provides MCP interface for agent interaction with simulation environment.
Uses SSE transport over HTTP, no built-in LLM inference.
"""

import asyncio
import base64
import json
import os
from io import BytesIO
from copy import deepcopy
from typing import List
from dataclasses import dataclass

from PIL import Image
from mcp import types
from mcp.server.lowlevel import Server
from mcp.server.sse import SseServerTransport
from starlette.applications import Starlette
from starlette.responses import Response
from starlette.routing import Mount, Route
import uvicorn

# Non-qwen eval path: disable precomputed nav positions by default
os.environ.setdefault('IS_TEST', '0')

from eval_server.mcp_env import (
    env,
    camera,
    processed_eval_episodes,
    spawn_objects_by_world_graph,
    TARGET_SCENE_ID,
    object_per_room,
)
from eval_server.perception_utils import init_annotators
from eval_server.tools import MCP_TOOLS
from eval_server.actions import (
    EvalState,
    dispatch_action,
    step_simulation,
)


# =============================================================================
# MCP Server Setup
# =============================================================================

sse = SseServerTransport("/messages/")
app = Server('eval-server')


@app.list_tools()
async def list_tools(*args) -> list[types.Tool]:
    """List all available MCP tools."""
    return MCP_TOOLS


# =============================================================================
# Task Manager
# =============================================================================

@dataclass
class Task:
    """Represents a pending action request."""
    action_name: str
    action_args: dict
    future: asyncio.Future


class TaskManager:
    """Manages async action requests bridging MCP calls to simulation."""

    def __init__(self):
        self.task = None

    def register(self, action: str, args: dict) -> asyncio.Future:
        """Register a new action request."""
        future = asyncio.get_running_loop().create_future()
        self.task = Task(
            action_name=action,
            action_args=args,
            future=future
        )
        return future

    def return_result(self, result: list):
        """Return result to the waiting MCP client."""
        if self.task is not None:
            self.task.future.set_result(result)
            self.task = None

    def has_task(self) -> bool:
        """Check if there's a pending task."""
        return self.task is not None


# =============================================================================
# Global State
# =============================================================================

state = EvalState()
manager = TaskManager()
current_episode_idx = -1

# Optional episode slicing for parallel evaluation
EPISODE_START_IDX = int(os.getenv("START_IDX", "0"))
EPISODE_END_IDX = int(os.getenv("END_IDX", str(len(processed_eval_episodes))))
EPISODE_END_IDX = min(EPISODE_END_IDX, len(processed_eval_episodes))

if EPISODE_START_IDX < 0:
    EPISODE_START_IDX = 0
if EPISODE_START_IDX > EPISODE_END_IDX:
    EPISODE_START_IDX = EPISODE_END_IDX

eval_episodes = processed_eval_episodes[EPISODE_START_IDX:EPISODE_END_IDX]

# Initialize perception
init_annotators(camera, resolution=(640, 480))


# =============================================================================
# MCP Handler
# =============================================================================

@app.call_tool()
async def wrapped_handler(
    action_name: str, arguments: dict
) -> List[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    """Handle MCP tool calls."""
    future = manager.register(action_name, arguments)
    result = await future
    return result


# =============================================================================
# HTTP Routes
# =============================================================================

async def handle_sse(request):
    """Handle SSE connection."""
    async with sse.connect_sse(
        request.scope, request.receive, request._send
    ) as streams:
        await app.run(
            streams[0], streams[1], app.create_initialization_options()
        )
    return Response()


starlette_app = Starlette(
    debug=True,
    routes=[
        Route("/sse", endpoint=handle_sse, methods=["GET"]),
        Mount("/messages/", app=sse.handle_post_message),
    ],
)


# =============================================================================
# Action Handlers
# =============================================================================

def handle_finish() -> list:
    """Load next evaluation episode and return full task information."""
    global current_episode_idx, state

    current_episode_idx += 1

    if current_episode_idx >= len(eval_episodes):
        return [types.TextContent(
            type='text',
            text="All evaluation episodes completed."
        )]

    episode = eval_episodes[current_episode_idx]
    global_episode_idx = EPISODE_START_IDX + current_episode_idx

    try:
        state.current_extra_assets = spawn_objects_by_world_graph(
            env, episode, state.current_extra_assets
        )
    except Exception as e:
        import traceback
        return [types.TextContent(
            type='text',
            text=f"Error spawning episode {current_episode_idx}: {traceback.format_exc()}"
        )]

    # Settle simulation
    state.world_graph = deepcopy(episode['initial_world_graph'])
    step_simulation(env, 500)

    # Reset agent state
    state.current_obs_dict = None
    state.current_marker_map = None
    state.current_landmark = None
    state.current_inv = None
    state.current_pos = None
    state.camera_orientation = None

    # Build target world graph (move target object from src to dest)
    target_world_graph = deepcopy(episode['initial_world_graph'])
    src = episode['src']
    dest = episode['dest']
    target_obj_name = episode['target_object_name']

    # Remove target from src
    if src in target_world_graph and target_obj_name in target_world_graph[src]['content']:
        target_world_graph[src]['content'].remove(target_obj_name)

    # Add target to dest
    if dest in target_world_graph:
        if target_obj_name not in target_world_graph[dest]['content']:
            target_world_graph[dest]['content'].append(target_obj_name)

    # Compile full task information
    task_info = {
        # Basic info
        'task_id': episode['task_id'],
        'episode_idx': global_episode_idx,
        'episode_local_idx': current_episode_idx,
        'episode_range': [EPISODE_START_IDX, EPISODE_END_IDX],
        'task_description': episode['task_description'],

        # Scene info
        'scene_id': TARGET_SCENE_ID,
        'rooms_and_furniture': dict(object_per_room),

        # World graphs
        'initial_world_graph': episode['initial_world_graph'],
        'target_world_graph': target_world_graph,

        # Task specifics
        'target_object': {
            'id': episode['target_object_id'],
            'name': episode['target_object_name'],
            'category': episode['target_category'],
        },
        'source_furniture': src,
        'destination_furniture': dest,

        # Distractors
        'source_distractors': episode.get('src_distractors', []),
        'destination_distractors': episode.get('dest_distractors', []),
        'object_distractors': episode.get('obj_distractors', []),
        'distractor_metadata': episode.get('obj_distractor_meta', {}),

        # Execution plan (for reference)
        'execution_plan': episode.get('execution_plan', []),
    }

    return [types.TextContent(
        type='text',
        text=json.dumps(task_info, indent=2)
    )]


def execute_action(action_name: str, arguments: dict) -> list:
    """Execute simulation action and return result."""
    global state

    # Special handling for finish action
    if action_name == "finish":
        return handle_finish()

    # Execute action using dispatch_action
    result_type, result_data, debug_info = dispatch_action(
        action_name, arguments, state, env
    )

    # Step simulation to settle physics
    step_simulation(env, 200)

    # Get current observation
    obs, *_ = env.step([{}])
    state.current_obs_dict = obs

    # Build result with debug info
    result = []

    if result_type == "text":
        result.append(types.TextContent(type='text', text=result_data))
    elif result_type == "image":
        result.append(types.ImageContent(
            type='image',
            data=result_data,
            mimeType="image/png"
        ))

    # Add observation image (except for perception actions that already return images)
    if result_type != "image" and state.current_obs_dict is not None:
        obs_data = state.current_obs_dict[0]['person']['sensors']['floating']
        rgb = Image.fromarray(obs_data['rgba'][..., :3])
        buffered = BytesIO()
        rgb.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        result.append(types.ImageContent(
            type='image',
            data=img_str,
            mimeType="image/png"
        ))

    # Add debug info (from dispatch_action)
    debug_str = json.dumps(debug_info, indent=2, default=str)
    result.append(types.TextContent(type='text', text=f"Debug Info:\n{debug_str}"))

    return result


# =============================================================================
# Main Loop
# =============================================================================

def run_api():
    """Run the HTTP API server."""
    port = int(os.getenv("PORT", 8080))
    config = uvicorn.Config(
        starlette_app,
        host="0.0.0.0",
        port=port,
        lifespan="on"
    )
    server = uvicorn.Server(config)
    server.run()


def main():
    """Main entry point."""
    import threading

    # Start HTTP server in background thread
    api_thread = threading.Thread(
        target=run_api,
        name="HTTP-Thread",
        daemon=True
    )
    api_thread.start()

    print(f"MCP Server running on port {os.getenv('PORT', 8080)}")
    print(f"Scene: {TARGET_SCENE_ID}")
    print(
        f"Episodes: {len(eval_episodes)} "
        f"(range [{EPISODE_START_IDX}, {EPISODE_END_IDX}))"
    )

    # Main simulation loop
    while env.simulation_app.is_running():
        if not manager.has_task():
            # No pending task, just step simulation
            env.step([{}])
            continue

        # Execute pending task
        action_name = manager.task.action_name
        arguments = manager.task.action_args

        try:
            result = execute_action(action_name, arguments)
            manager.return_result(result)
        except Exception as e:
            import traceback
            error_msg = f"Error executing {action_name}: {traceback.format_exc()}"
            print(error_msg)
            manager.return_result([
                types.TextContent(type='text', text=error_msg)
            ])

    env.simulation_app.close()


if __name__ == "__main__":
    main()
