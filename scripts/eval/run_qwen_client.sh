#!/bin/bash
# Qwen MCP client startup script
# Requires eval_server/mcp_server.py already running.

if [ -z "$MODEL_PATH" ]; then
    echo "ERROR: MODEL_PATH is not set."
    echo "Usage: MODEL_PATH=/path/to/checkpoint ./scripts/eval/run_qwen_client.sh"
    exit 1
fi

export MCP_SERVER_URL=${MCP_SERVER_URL:-http://localhost:8080/sse}
export TRAJ_PATH=${TRAJ_PATH:-eval_output}
export START_IDX=${START_IDX:-0}
export END_IDX=${END_IDX:-999999}
export MAX_STEP=${MAX_STEP:-30}

echo "Starting Qwen MCP client..."
echo "Model: ${MODEL_PATH}"
echo "MCP server: ${MCP_SERVER_URL}"
echo "Episode range: [${START_IDX}, ${END_IDX})"
echo "Output: ${TRAJ_PATH}"
echo ""

python -m eval_server.qwen_client
