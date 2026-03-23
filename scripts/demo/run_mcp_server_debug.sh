#!/bin/bash

# Debug MCP Server — non-headless (opens Omniverse GUI window)
# For interactive debugging with visual inspection.
#
# Usage:
#   ./run_mcp_server_debug.sh
#
# Optional overrides:
#   DEMO_TASK_CONFIG=configs/demo_task.yaml  — task config (default shown)
#   TARGET_SCENE_ID=...                      — override scene_id in config
#   PORT=8080                                — server port (default 8080)
#   IS_TEST=0                                — test mode flag
#   TRAJ_PATH=eval_output_debug              — output directory
#   USE_LIFT_ROBOT=0  LIFT_USD_PATH=...      — enable lift robot
#   USE_EMPTY_SCENE=0 EMPTY_USD_PATH=...     — use empty scene instead

export DEMO_TASK_CONFIG=${DEMO_TASK_CONFIG:-configs/demo_task.yaml}
export TRAJ_PATH=${TRAJ_PATH:-eval_output_debug}
export PORT=${PORT:-8080}
export IS_TEST=${IS_TEST:-0}

echo "Starting MCP Server (DEBUG / non-headless)..."
echo "Config:   ${DEMO_TASK_CONFIG}"
echo "Port:     ${PORT}"
echo "IS_TEST:  ${IS_TEST}"
echo ""

python -m mcp_server.mcp_server_debug
