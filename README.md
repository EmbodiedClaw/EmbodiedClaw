# EmbodiedClaw

MCP-based evaluation system for embodied AI agents in simulated home environments. Agents interact with the simulation through the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/), performing pick-and-place tasks with visual observations.

## Setup

### 1. Clone with submodules

```bash
git clone --recurse-submodules <repo-url>
cd EmbodiedClaw
```

### 2. Install the simulation engine

```bash
cd internutopia
pip install -e .
cd ..
```

### 3. Install eval dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure paths

Edit `config.yaml` to point to your local asset paths, or override via environment variables:

```bash
export SCENE_USD_ROOT=/path/to/scenes
export OCC_MAP_ROOT=/path/to/occ_maps
export ROBOT_USD_PATH=/path/to/robot.usd
export SCENE_ANNO_ROOT=/path/to/scene_annotations
export NAV_POSITION_PATH=/path/to/nav_position.jsonl
```

## Usage

### Running the MCP Server

```bash
TARGET_SCENE_ID=MVUCSQAKTKJ5EAABAAAAABY8 \
TASK_SOURCE_PATH=path/to/physics_passed_tasks.json \
ORIGINAL_TASK_PATH=path/to/tasks.json \
TRAJ_PATH=eval_output \
python eval_server/mcp_server.py
```

Or use the convenience script:

```bash
./scripts/eval/run_mcp_server.sh
```

### Evaluating with OpenAI API

```bash
OPENAI_API_KEY=your-key \
MODEL_NAME=gpt-4o \
TARGET_SCENE_ID=MVUCSQAKTKJ5EAABAAAAABY8 \
TASK_SOURCE_PATH=path/to/physics_passed_tasks.json \
ORIGINAL_TASK_PATH=path/to/tasks.json \
TRAJ_PATH=eval_output \
python -m eval_server.agent_eval
```

### Evaluating with Local Qwen Model

**Single-process mode** (model + simulator on same GPU):

```bash
MODEL_PATH=/path/to/qwen/checkpoint \
TARGET_SCENE_ID=MVUCSQAKTKJ5EAABAAAAABY8 \
TASK_SOURCE_PATH=path/to/physics_passed_tasks.json \
ORIGINAL_TASK_PATH=path/to/tasks.json \
TRAJ_PATH=eval_output \
python -m eval_server.qwen_eval
```

**Client-server mode** (separate processes):

```bash
# Terminal 1: Start MCP server
./scripts/eval/run_mcp_server.sh

# Terminal 2: Start Qwen client
MODEL_PATH=/path/to/checkpoint ./scripts/eval/run_qwen_client.sh
```

### Batch Evaluation (4 GPUs)

```bash
# Single-process mode
MODEL_PATH=/path/to/checkpoint ./scripts/eval/batch_qwen_eval.sh

# Client-server mode
./scripts/eval/batch_mcp_server.sh        # Terminal 1
MODEL_PATH=/path/to/checkpoint ./scripts/eval/batch_qwen_client.sh  # Terminal 2
```

## Available MCP Tools

| Tool | Description |
|------|-------------|
| `finish` | Complete current episode, load next task |
| `check_scene_objects` | List all receptacles by room |
| `nav_to` | Navigate camera to a furniture receptacle |
| `walk_around` | List objects on current furniture |
| `show_object_by_category` | Detect and highlight objects by category |
| `show_receptacles` | Highlight visible receptacles |
| `gaze_at` | Focus camera on a specific object |
| `pick` | Pick up an object by marker ID |
| `place` | Place held object on a receptacle |
| `open` / `close` | Operate articulated doors |

## Metric

Success is measured by exact match between the current world graph and the target world graph when the agent calls `finish`.

## License

MIT
