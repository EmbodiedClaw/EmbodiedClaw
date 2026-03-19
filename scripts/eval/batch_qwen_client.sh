#!/bin/bash
# Batch launch 4 Qwen MCP clients in tmux
# Connects to servers on ports 8000-8003

# --- Configuration ---
SESSION_NAME="qwen_eval_clients"
GPU_COUNT=4
PORT_BASE=8000

if [ -z "$MODEL_PATH" ]; then
    echo "ERROR: MODEL_PATH environment variable is not set."
    echo "Usage: MODEL_PATH=/path/to/checkpoint ./scripts/eval/batch_qwen_client.sh"
    exit 1
fi

TRAJ_PATH="${TRAJ_PATH:-eval_output}"
MAX_STEP=${MAX_STEP:-30}
MCP_SERVER_HOST="${MCP_SERVER_HOST:-http://localhost}"

# --- Clean up old session ---
tmux kill-session -t $SESSION_NAME 2>/dev/null

# --- Create windows ---
for i in $(seq 0 $((GPU_COUNT - 1))); do
    GPU_ID=$i
    PORT=$((PORT_BASE + i))
    WIN_NAME="client_${PORT}"

    CMD="export CUDA_VISIBLE_DEVICES=$GPU_ID; \
export MODEL_PATH=$MODEL_PATH; \
export MCP_SERVER_URL=${MCP_SERVER_HOST}:${PORT}/sse; \
export TRAJ_PATH=$TRAJ_PATH; \
export MAX_STEP=$MAX_STEP; \
echo '-> Starting Qwen client on GPU: $GPU_ID, MCP: ${MCP_SERVER_HOST}:${PORT}/sse'; \
python -m eval_server.qwen_client; \
exec bash"

    if [ "$i" -eq 0 ]; then
        tmux new-session -d -s "$SESSION_NAME" -n "$WIN_NAME"
    else
        tmux new-window -t "$SESSION_NAME" -n "$WIN_NAME"
    fi

    sleep 0.2
    tmux send-keys -t "${SESSION_NAME}:${WIN_NAME}" "$CMD" Enter
    echo "[OK] Window: $WIN_NAME (GPU: $GPU_ID, Port: $PORT)"
done

echo ""
echo "All Qwen clients launched."
echo ""
tmux select-window -t "${SESSION_NAME}:client_${PORT_BASE}"
tmux attach-session -t $SESSION_NAME
