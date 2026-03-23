# EmbodiedClaw

**EmbodiedClaw** is an open-source framework for training and evaluating embodied AI agents on household manipulation tasks. Agents interact with a physics-based simulation through [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) tool calls, receiving multimodal observations (RGB images + structured text) in return.

## Overview

This repository provides the infrastructure for agents to perceive and act in simulated home environments. The current release includes:

- **InternUtopia simulation engine** — physics-based 3D simulation built on [InternUtopia](https://github.com/InternRobotics/InternUtopia) (Isaac Sim)
- **nanobot communication interface** — lightweight messaging layer bridging the simulation with external processes
- **MCP server** — exposes action primitives as MCP tools over SSE/HTTP, enabling any MCP-compatible agent to interact with the simulation

## Roadmap

| Stage | Status | Description |
|-------|--------|-------------|
| Phase 1 | **Released** | Simulation engine (InternUtopia), nanobot interface, MCP server |
| Phase 2 | Coming soon | Task & trajectory generation pipeline (GRScenes + Mesatask) |
| Phase 3 | Coming soon | Agent implementation, SFT & RL training pipeline |
| Phase 4 | Coming soon | Technical report |

## Architecture

```
Agent (any MCP client)
        │  MCP tool calls (SSE/HTTP)
        ▼
   MCP Server  ──────────────────────────────────────────────────────┐
        │                                                             │
        │  action dispatch                                            │
        ▼                                                             │
  Simulation (InternUtopia / Isaac Sim)                              │
        │                                                             │
        │  RGB images + structured observations                       │
        └─────────────────────────────────────────────────────────────┘
```

The agent calls MCP tools → the server dispatches them to the simulator → the simulator returns multimodal observations (image + text) back through MCP.

## Available MCP Tools

| Tool | Description |
|------|-------------|
| `check_scene_objects` | List all receptacles by room |
| `nav_to` | Navigate to a furniture receptacle |
| `walk_around` | Survey all objects on the current receptacle |
| `gaze_at` | Focus camera on a specific object by marker ID |
| `show_object_by_category` | Detect and annotate objects by category in view |
| `show_receptacles` | Highlight all visible receptacle surfaces |
| `pick` | Pick up an object by marker ID |
| `place` | Place held object onto a receptacle surface |
| `open` / `close` | Operate articulated doors |
| `ask` | Request clarification from the user |
| `finish` | Signal task completion and load the next episode |

Each tool call returns an RGB observation image and structured text feedback from the simulation.

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

### 4. Configure asset paths

Edit `config.yaml` or set environment variables:

```bash
export SCENE_USD_ROOT=/path/to/scenes
export OCC_MAP_ROOT=/path/to/occ_maps
export ROBOT_USD_PATH=/path/to/robot.usd
export SCENE_ANNO_ROOT=/path/to/scene_annotations
export NAV_POSITION_PATH=/path/to/nav_position.jsonl
```

## Usage

### Start the MCP server

```bash
TARGET_SCENE_ID=<scene_id> \
TASK_SOURCE_PATH=path/to/physics_passed_tasks.json \
ORIGINAL_TASK_PATH=path/to/tasks.json \
TRAJ_PATH=eval_output \
python eval_server/mcp_server.py
```

The server listens on port `8080` (override with `PORT=<n>`). Connect any MCP-compatible agent to `http://localhost:8080/sse`.

### Evaluate with an OpenAI-compatible model

```bash
OPENAI_API_KEY=your-key \
MODEL_NAME=gpt-4o \
TARGET_SCENE_ID=<scene_id> \
TASK_SOURCE_PATH=path/to/physics_passed_tasks.json \
ORIGINAL_TASK_PATH=path/to/tasks.json \
TRAJ_PATH=eval_output \
python -m eval_server.agent_eval
```

### Evaluate with a local Qwen model

**Single-process** (model + simulator share GPU):

```bash
MODEL_PATH=/path/to/qwen/checkpoint \
TARGET_SCENE_ID=<scene_id> \
TASK_SOURCE_PATH=path/to/physics_passed_tasks.json \
ORIGINAL_TASK_PATH=path/to/tasks.json \
TRAJ_PATH=eval_output \
python -m eval_server.qwen_eval
```

**Client-server** (separate processes):

```bash
# Terminal 1 — simulation server
./scripts/eval/run_mcp_server.sh

# Terminal 2 — inference client
MODEL_PATH=/path/to/checkpoint ./scripts/eval/run_qwen_client.sh
```

### Batch evaluation (multi-GPU)

```bash
# Single-process
MODEL_PATH=/path/to/checkpoint ./scripts/eval/batch_qwen_eval.sh

# Client-server
./scripts/eval/batch_mcp_server.sh
MODEL_PATH=/path/to/checkpoint ./scripts/eval/batch_qwen_client.sh
```

## Evaluation Metric

Success is measured by world-graph exact match: the agent calls `finish`, and its resulting object placement is compared against the target world graph.

## License

MIT
