#!/bin/bash

# MCP Server startup script for evaluation

export TARGET_SCENE_ID=${TARGET_SCENE_ID:-MVUCSQAKTKJ5EAABAAAAAAI8}
export TASK_SOURCE_PATH=${TASK_SOURCE_PATH:-metadata/tasks/verify_results/obj_distractor/${TARGET_SCENE_ID}/physics_passed_tasks.json}
export ORIGINAL_TASK_PATH=${ORIGINAL_TASK_PATH:-metadata/tasks/obj_distractor.json}
export TRAJ_PATH=${TRAJ_PATH:-eval_output}
export PORT=${PORT:-8080}
export IS_TEST=${IS_TEST:-0}

echo "Starting MCP Server..."
echo "Scene: ${TARGET_SCENE_ID}"
echo "Tasks: ${TASK_SOURCE_PATH}"
echo "Port: ${PORT}"
echo "IS_TEST: ${IS_TEST}"
echo ""

python eval_server/mcp_server.py
