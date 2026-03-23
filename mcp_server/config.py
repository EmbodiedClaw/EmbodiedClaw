"""
Centralized path configuration for EmbodiedClaw.

Loads paths from config.yaml at the project root.
Each value can be overridden by an environment variable of the same name (uppercase).
"""

import os
from pathlib import Path

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_PATH = _PROJECT_ROOT / "config.yaml"

# Load config once at import time
with open(_CONFIG_PATH, "r") as f:
    _raw = yaml.safe_load(f)

_paths = _raw.get("paths", {})


def _get(key: str) -> str:
    """Return env-var override if set, otherwise the YAML value."""
    env_key = key.upper()
    return os.environ.get(env_key, _paths.get(key, ""))


def get_scene_usd_path(scene_id: str) -> str:
    """Full path to the scene USD file."""
    root = _get("scene_usd_root")
    return (
        f"{root}/{scene_id}_usd/"
        f"start_result_interaction_noMDL_move.usd"
    )


def get_occ_map_path(scene_id: str) -> str:
    """Path to the occupancy map directory for a scene."""
    root = _get("occ_map_root")
    return f"{root}/{scene_id}"


def get_robot_usd_path() -> str:
    """Path to the robot USD asset."""
    return _get("robot_usd_path")


def get_scene_anno_path(scene_id: str) -> str:
    """Path to the scene caption annotation JSON."""
    root = _get("scene_anno_root")
    return f"{root}/{scene_id}_all_caption_processed.json"


def get_nav_position_path() -> str:
    """Path to the precomputed nav positions JSONL file."""
    return _get("nav_position_path")


def get_metadata_path(filename: str) -> Path:
    """Path to a bundled metadata file (in repo metadata/ dir)."""
    return _PROJECT_ROOT / "metadata" / filename
