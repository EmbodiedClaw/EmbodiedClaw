#!/bin/bash

# Debug MCP Server — non-headless (opens Omniverse GUI window)
# For interactive debugging with visual inspection.
#
# Usage:
#   ./scripts/eval/run_mcp_server_debug.sh
#
# Optional overrides:
#   TARGET_SCENE_ID=... TASK_SOURCE_PATH=... ORIGINAL_TASK_PATH=... ./scripts/eval/run_mcp_server_debug.sh
export TARGET_SCENE_ID=MVUCSQAKTKJ5EAABAAAAABY8
export TRAJ_PATH=eval_output_debug
export TASK_SOURCE_PATH=/cpfs/user/miboyu/replay/data/benchmarks/verify_results/visual_pnp/MVUCSQAKTKJ5EAABAAAAABY8/physics_passed_tasks_71.json
export ORIGINAL_TASK_PATH=/cpfs/user/miboyu/replay/data/benchmarks/verify_results/visual_pnp/MVUCSQAKTKJ5EAABAAAAABY8/original_tasks.json
export TRAJ_PATH=${TRAJ_PATH:-eval_output_debug}
export PORT=${PORT:-8080}
export IS_TEST=${IS_TEST:-0}

echo "Starting MCP Server (DEBUG / non-headless)..."
echo "Scene:    ${TARGET_SCENE_ID}"
echo "Tasks:    ${TASK_SOURCE_PATH}"
echo "Port:     ${PORT}"
echo "IS_TEST:  ${IS_TEST}"
echo ""

python -m eval_server.mcp_server_debug
