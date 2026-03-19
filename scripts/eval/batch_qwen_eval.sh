#!/bin/bash
# Batch launch 4 Qwen eval servers in tmux (single-process mode)
# Each window: GPU + simulator + Qwen inference

# --- Configuration ---
SESSION_NAME="qwen_eval_servers"
GPU_COUNT=4

TOTAL_EPISODES=${TOTAL_EPISODES:-456}
CHUNK=$(( (TOTAL_EPISODES + GPU_COUNT - 1) / GPU_COUNT ))

TARGET_SCENE_ID="${TARGET_SCENE_ID:-MVUCSQAKTKJ5EAABAAAAABY8}"
TASK_SOURCE="${TASK_SOURCE_PATH:-metadata/benchmarks/verify_results/visual_pnp/${TARGET_SCENE_ID}/valid_tasks.json}"
ORIGINAL_TASK="${ORIGINAL_TASK_PATH:-metadata/benchmarks/visual_pnp.json}"
TRAJ_PATH="${TRAJ_PATH:-eval_output}"

if [ -z "$MODEL_PATH" ]; then
    echo "ERROR: MODEL_PATH environment variable is not set."
    echo "Usage: MODEL_PATH=/path/to/checkpoint ./scripts/eval/batch_qwen_eval.sh"
    exit 1
fi

# --- Clean up old session ---
tmux kill-session -t $SESSION_NAME 2>/dev/null

# --- Create windows ---
for i in $(seq 0 $((GPU_COUNT - 1))); do
    GPU_ID=$i
    START_IDX=$((i * CHUNK))
    END_IDX=$(( (i + 1) * CHUNK ))
    if [ $END_IDX -gt $TOTAL_EPISODES ]; then
        END_IDX=$TOTAL_EPISODES
    fi
    WIN_NAME="eval_${GPU_ID}"

    CMD="export CUDA_VISIBLE_DEVICES=$GPU_ID; \
export TARGET_SCENE_ID=$TARGET_SCENE_ID; \
export TASK_SOURCE_PATH=$TASK_SOURCE; \
export ORIGINAL_TASK_PATH=$ORIGINAL_TASK; \
export TRAJ_PATH=$TRAJ_PATH; \
export MODEL_PATH=$MODEL_PATH; \
export START_IDX=$START_IDX; \
export END_IDX=$END_IDX; \
export IS_TEST=1; \
echo '-> Qwen eval on GPU: $GPU_ID, Episodes: [$START_IDX, $END_IDX)'; \
python -m eval_server.qwen_eval; \
exec bash"

    if [ "$i" -eq 0 ]; then
        tmux new-session -d -s "$SESSION_NAME" -n "$WIN_NAME"
    else
        tmux new-window -t "$SESSION_NAME" -n "$WIN_NAME"
    fi

    sleep 0.2
    tmux send-keys -t "${SESSION_NAME}:${WIN_NAME}" "$CMD" Enter
    echo "[OK] Window: $WIN_NAME (GPU: $GPU_ID, Episodes: [$START_IDX, $END_IDX))"
done

echo ""
echo "All Qwen eval servers launched."
echo ""
tmux select-window -t "${SESSION_NAME}:eval_0"
tmux attach-session -t $SESSION_NAME
