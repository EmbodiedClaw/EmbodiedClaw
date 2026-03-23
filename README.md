<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&height=300&color=gradient&text=EmbodiedClaw&animation=fadeIn&desc=Bridging%20Agents%20to%20Interactive%20Physical%20Environments&descSize=18&descAlignY=65" alt="EmbodiedClaw banner" />
</p>

<h1 align="center">Discover, Communicate, Deploy: Visual Embodied Agents for Open-World Interaction</h1>

<p align="center">
  <a href="https://github.com/InternRobotics/EmbodiedClaw"><img src="https://img.shields.io/badge/GitHub-EmbodiedClaw-black?style=flat-square&logo=github&logoColor=white" alt="GitHub" /></a>
  <a href="https://arxiv.org/abs/TODO"><img src="https://img.shields.io/badge/arXiv-paper-b31b1b?style=flat-square&logo=arxiv&logoColor=white" alt="arXiv" /></a>
  <img src="https://img.shields.io/badge/Phase_1-Released-10b981?style=flat-square" alt="Phase 1 Released" />
  <img src="https://img.shields.io/badge/Phase_2-Coming_Soon-f97316?style=flat-square" alt="Phase 2 Coming Soon" />
  <img src="https://img.shields.io/badge/License-MIT-6366f1?style=flat-square" alt="MIT License" />
</p>

---

## 📰 News

* **[2026.03.23]** 🔥🔥 **Phase 1 Released** — Simulation engine (InternUtopia), nanobot interface, and MCP server are now open-source!
* **[Coming Soon]** 🔥 **Phase 2** — Task & trajectory generation pipeline (GRScenes + MesaTask): ~5,800 tasks, ~15,000 CoT-annotated conversation turns
* **[Coming Soon]** 🔥 **Phase 3** — Agent weights (Qwen3-VL-8B fine-tuned via SFT + closed-loop GRPO)
* **[Coming Soon]** 🔥 **Phase 4** — Technical report (ECCV 2026)

---

## Introduction

**EmbodiedClaw** is an open-source framework for building, training, and evaluating visual embodied agents on household manipulation tasks. Unlike prior frameworks that rely on Oracle APIs or privileged simulator states, EmbodiedClaw enforces a **privilege-free** paradigm: agents perceive the world exclusively through ego-centric RGB observations and interact via a toolchain that is fully realizable in the physical world.

The framework introduces two core innovations: a **closed-loop simulation environment** with a State-Driven Simulated User that issues ambiguous instructions and provides dynamic feedback, and an **automated data pipeline** that generates complex interactive trajectories for agent training. Agents are trained on [Qwen3-VL-8B](https://github.com/QwenLM/Qwen3-VL) via a two-stage SFT + RL paradigm and achieve strong **zero-shot Sim2Real transfer** on a Lift2 dual-arm mobile robot.

### 🌟 Key Highlights

* **Privilege-free environment**: No Oracle APIs or global object lists — agents discover objects solely from pixel-level RGB observations, ensuring direct transferability to the real world.
* **Three-tiered active exploration**: Hierarchical toolchain (`nav_to` → `walk_around` → `show_object_by_category` → `gaze_at`) enables multi-level, occlusion-aware scene understanding grounded in 2D visual perception.
* **State-Driven Simulated User**: An LLM-powered user (Gemini) issues deliberately ambiguous instructions and answers agent queries based on shared interaction context, training agents to proactively resolve uncertainty.
* **End-to-end training pipeline**: Automated SFT data synthesis (~5,800 tasks, ~15,000 turns) with CoT annotations, followed by closed-loop GRPO reinforcement learning with world-graph state-differential rewards.
* **Sim2Real transfer**: Agents trained in simulation deploy zero-shot on a real Lift2 dual-arm mobile robot with open-vocabulary spatial perception.

### Comparison with Existing Environments

| Environment | Non-privileged Scene Perception | Active Scene Exploration | Visual-Grounded Action | User-Participated Task | Real-World Deploy |
|---|:---:|:---:|:---:|:---:|:---:|
| EmbodiedBench | | | | | |
| ManiTaskGen | | ✅ | ✅ | | |
| EmMoE | ✅ | ✅ | | | |
| PartNR | ✅ | ✅ | | ✅ | |
| **EmbodiedClaw (Ours)** | ✅ | ✅ | ✅ | ✅ | ✅ |

---

## Roadmap

| Stage | Status | Description |
|-------|--------|-------------|
| Phase 1 | ✅ **Released** | Simulation engine (InternUtopia), nanobot interface, MCP server |
| Phase 2 | 🔜 Coming soon | Task & trajectory generation pipeline (GRScenes + MesaTask) |
| Phase 3 | 🔜 Coming soon | Agent weights (Qwen3-VL-8B, SFT + GRPO) |
| Phase 4 | 🔜 Coming soon | Technical report (ECCV 2026) |

---

## Environment Design

EmbodiedClaw is built on the [GRUtopia](https://github.com/OpenRobotLab/GRUtopia) platform with high-fidelity rendering and rigid-body physics. The environment spans **7 scenes**, **11 room types**, **100 interactive receptacles**, and over **2,500 object instances** across **639 semantic categories** (from [MesaTask](https://github.com/hao2025/mesatask)).

### Privilege-Free Toolchain

The agent's initial observation is a receptacle list derived from a 3D occupancy map — it has **no access** to object distributions or fine-grained scene layouts. All object information must be discovered through active exploration:

| Tier | Tool | Description |
|------|------|-------------|
| Navigation | `nav_to` | Move between receptacle nodes using structural priors |
| Scanning | `walk_around` | Orbit a receptacle with open-vocabulary 2D detection to overcome occlusions |
| Filtering | `show_object_by_category` | Actively detect objects of a user-specified category in the current view |
| Alignment | `gaze_at` | Approach and center a specific object (by visual ID) in the camera frame |
| Manipulation | `pick` / `place` | Pick up or place an object by visual marker ID |
| Articulation | `open` / `close` | Operate articulated furniture doors |
| Communication | `ask` | Query the simulated user to resolve instruction ambiguity |
| Completion | `finish` | Signal task done and load the next episode |

Detected objects are back-projected from 2D segmentation masks into 3D space, constructing a spatial memory map. Objects are annotated with **visual IDs** in the RGB observation, enabling unambiguous reference for downstream manipulation without privileged 3D coordinates.

### State-Driven Simulated User

An LLM-powered simulated user (Gemini) is embedded directly in the environment dynamics. It:
- Issues **deliberately ambiguous** initial instructions (e.g., referring to objects with incomplete descriptions)
- Maintains a **task-specific profile** to prevent hallucination of non-existent items
- Stays **synchronized** with the agent's full interaction history and real-time observations
- Answers `ask` queries with context-aware natural language to progressively resolve ambiguity

---

## Agent Architecture

The agent is built on **Qwen3-VL-8B** and operates in a closed-loop ReAct inference cycle:

```
Prompt Assembly (scene description + tool schema + task + history + last obs)
        │
        ▼
  Qwen3-VL-8B  →  <think>...</think>  +  {summary, next_action}
        │
        ▼
  Tool Executor  →  MCP/SSE  →  Isaac Sim
        │
        ▼
  RGB observation + text feedback  →  HistoryManager  →  next step
```

**HistoryManager** prevents *history hallucination* by maintaining a ground-truth append-only buffer: only the final line of the model's self-reported history is extracted and appended at each step, ensuring the injected history is always consistent and monotonically growing.

Episodes terminate on `finish`, a maximum of **30 steps**, or an execution error.

### Training Pipeline

| Stage | Method | Data | Details |
|-------|--------|------|---------|
| Stage I | SFT (behavior cloning) | ~15K CoT-annotated turns | Aligns the model to the privilege-free tool interface; vision encoder frozen |
| Stage II | Closed-loop GRPO | Online rollouts | State-differential reward: +1 per object placed at goal location in the world graph |

---

## Architecture Overview

```
Agent (Qwen3-VL-8B or any MCP client)
        │  MCP tool calls (SSE/HTTP)
        ▼
   MCP Server  ──────────────────────────────────────────────────────┐
        │                                                             │
        │  action dispatch                                            │
        ▼                                                             │
  Simulation (GRUtopia / Isaac Sim)          Simulated User (Gemini) │
        │                  ▲                         │               │
        │  RGB + visual IDs│                    ask / answer         │
        └──────────────────┴─────────────────────────────────────────┘
```

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

Success is measured by **world-graph state-differential match**: after the agent calls `finish`, the resulting object placement configuration is compared against the target world graph. A reward of **+1** is granted for each object successfully placed at its goal location.

---

## 📑 Citation

If you find EmbodiedClaw useful for your research, please cite:

```bibtex
@inproceedings{embodiedclaw2026eccv,
  title     = {Discover, Communicate, Deploy: Visual Embodied Agents for Open-World Interaction},
  author    = {},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2026}
}
```

---

## Acknowledgement

EmbodiedClaw is built on top of [**GRUtopia**](https://github.com/OpenRobotLab/GRUtopia) and [**InternUtopia**](https://github.com/InternRobotics/InternUtopia) (Isaac Sim). Our agent is trained from [**Qwen3-VL**](https://github.com/QwenLM/Qwen3-VL). The data pipeline leverages [**MesaTask**](https://github.com/hao2025/mesatask) assets and [**Gemini**](https://deepmind.google/technologies/gemini/) as the simulated user backbone. We thank the teams behind [**Model Context Protocol**](https://modelcontextprotocol.io/) and [**NVIDIA Isaac Sim**](https://developer.nvidia.com/isaac-sim) for their foundational work.

---

## License

This project is licensed under the [MIT License](LICENSE).
