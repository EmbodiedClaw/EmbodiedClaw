#!/bin/bash
# Batch launch 4 MCP Servers in tmux (one GPU per window, ports 8000-8003)
# Usage: ./scripts/eval/batch_mcp_server.sh

# --- Configuration ---
SESSION_NAME="mcp_servers"
GPU_COUNT=4
PORT_BASE=8000

TARGET_ID="${TARGET_SCENE_ID:-MVUCSQAKTKJ5EAABAAAAAAI8}"
TASK_SOURCE="${TASK_SOURCE_PATH:-metadata/tasks/verify_results/obj_distractor/${TARGET_ID}/physics_passed_tasks.json}"
ORIGINAL_TASK="${ORIGINAL_TASK_PATH:-metadata/tasks/obj_distractor.json}"
TRAJ="${TRAJ_PATH:-eval_output}"

# --- Clean up old session ---
tmux kill-session -t $SESSION_NAME 2>/dev/null

# --- Create windows ---
for i in $(seq 0 $((GPU_COUNT - 1))); do
    GPU_ID=$i
    PORT=$((PORT_BASE + i))
    WIN_NAME="server_${PORT}"

    CMD="export CUDA_VISIBLE_DEVICES=$GPU_ID; \
export TARGET_SCENE_ID=$TARGET_ID; \
export TASK_SOURCE_PATH=$TASK_SOURCE; \
export ORIGINAL_TASK_PATH=$ORIGINAL_TASK; \
export TRAJ_PATH=$TRAJ; \
export PORT=$PORT; \
echo '-> Starting MCP Server on GPU: $GPU_ID, Port: $PORT'; \
echo '   Scene: $TARGET_ID'; \
python eval_server/mcp_server.py"

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
echo "All MCP servers launched (ports ${PORT_BASE}-$((PORT_BASE + GPU_COUNT - 1)))."
echo ""
tmux select-window -t "${SESSION_NAME}:server_${PORT_BASE}"
tmux attach-session -t $SESSION_NAME
