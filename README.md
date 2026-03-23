<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&height=300&color=gradient&text=EmbodiedClaw&animation=fadeIn&desc=Bridging%20Agents%20to%20Interactive%20Physical%20Environments&descSize=18&descAlignY=65" alt="EmbodiedClaw banner" />
</p>

<h1 align="center">EmbodiedClaw: Open Framework for Embodied Manipulation Training & Evaluation</h1>

<p align="center">
  <a href="https://github.com/InternRobotics/EmbodiedClaw"><img src="https://img.shields.io/badge/GitHub-EmbodiedClaw-black?style=flat-square&logo=github&logoColor=white" alt="GitHub" /></a>
  <img src="https://img.shields.io/badge/Phase_1-Released-10b981?style=flat-square" alt="Phase 1 Released" />
  <img src="https://img.shields.io/badge/Phase_2-Coming_Soon-f97316?style=flat-square" alt="Phase 2 Coming Soon" />
  <img src="https://img.shields.io/badge/License-MIT-6366f1?style=flat-square" alt="MIT License" />
</p>

---

## 📰 News

* **[2026.03.23]** 🔥🔥 **Phase 1 Released** — Simulation engine (InternUtopia), nanobot interface, and MCP server are now open-source!
* **[Coming Soon]** 🔥 — Task & trajectory generation pipeline (GRScenes + Mesatask)
* **[Coming Soon]** 🔥 — Agent implementation, SFT & RL training pipeline
* **[Coming Soon]** 🔥 — Technical report

---

## Introduction

**EmbodiedClaw** is an open-source framework for training and evaluating embodied AI agents on household manipulation tasks. Agents interact with a physics-based simulation through [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) tool calls, receiving multimodal observations (RGB images + structured text) in return.

### 🌟 Key Highlights

* **MCP-native interface**: Exposes action primitives as standard MCP tools over SSE/HTTP — any MCP-compatible agent (Claude, GPT-4o, local models) can plug in without modification.
* **Multimodal observations**: Every tool call returns an RGB image plus structured text feedback, enabling vision-language agents to reason about the scene.
* **Physics-based simulation**: Built on [InternUtopia](https://github.com/InternRobotics/InternUtopia) (Isaac Sim), providing realistic object interactions and articulated manipulation.
* **End-to-end pipeline**: From simulation through evaluation to training — a unified framework covering the full embodied AI development cycle.

---

## Roadmap

| Stage | Status | Description |
|-------|--------|-------------|
| Phase 1 | ✅ **Released** | Simulation engine (InternUtopia), nanobot interface, MCP server |
| Phase 2 | 🔜 Coming soon | Task & trajectory generation pipeline (GRScenes + Mesatask) |
| Phase 3 | 🔜 Coming soon | Agent implementation, SFT & RL training pipeline |
| Phase 4 | 🔜 Coming soon | Technical report |

---

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

---

## Available MCP Tools

| Tool | Description |
|------|-------------|
| `list_receptacles` | List all receptacles by room |
| `navigate_to` | Navigate to a furniture receptacle |
| `explore_receptacle` | Survey all objects on the current receptacle |
| `focus_on` | Focus camera on a specific object by marker ID |
| `find_objects` | Find and highlight objects of a given category in view |
| `highlight_receptacles` | Highlight all visible receptacle surfaces |
| `pick` | Pick up an object by marker ID |
| `place` | Place held object onto a receptacle surface |
| `open` / `close` | Operate articulated doors |
| `ask` | Request clarification from the user |
| `finish` | Signal task completion and load the next episode |

Each tool call returns an RGB observation image and structured text feedback from the simulation.

---

## Quick Start

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

```bash
export SCENE_USD_ROOT=/path/to/scenes
export OCC_MAP_ROOT=/path/to/occ_maps
export ROBOT_USD_PATH=/path/to/robot.usd
export SCENE_ANNO_ROOT=/path/to/scene_annotations
export NAV_POSITION_PATH=/path/to/nav_position.jsonl
```

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

---

## Evaluation Metric

Success is measured by **world-graph exact match**: the agent calls `finish`, and its resulting object placement is compared against the target world graph.

---

## 📑 Citation

If you find EmbodiedClaw useful for your research and applications, please cite using this BibTeX:

```bibtex
@misc{embodiedclaw2025,
  title  = {EmbodiedClaw: Open Framework for Embodied Manipulation Training & Evaluation},
  author = {},
  year   = {2025},
  url    = {https://github.com/InternRobotics/EmbodiedClaw}
}
```

---

## Acknowledgement

EmbodiedClaw is built on top of [**InternUtopia**](https://github.com/InternRobotics/InternUtopia) (Isaac Sim). We thank the teams behind [**Model Context Protocol**](https://modelcontextprotocol.io/) and [**NVIDIA Isaac Sim**](https://developer.nvidia.com/isaac-sim) for their foundational work.

---

## License

This project is licensed under the [MIT License](LICENSE).
