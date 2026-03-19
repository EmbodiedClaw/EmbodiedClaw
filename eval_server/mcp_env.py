"""
Environment setup and configuration for evaluation.
Aligned with replay/mcp_env.py: uses env vars, same asset libraries,
pre-computed positions from physics verification.
"""

import os
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from copy import deepcopy

from eval_server import config

# =============================================================================
# Configuration from environment variables
# =============================================================================

TRAIN_SCENE_IDS = [
    'MVUCSQAKTKJ5EAABAAAAABQ8', 'MVUCSQAKTKJ5EAABAAAAAAQ8',
    'MVUCSQAKTKJ5EAABAAAAABA8', 'MVUCSQAKTKJ5EAABAAAAACA8',
    'MVUCSQAKTKJ5EAABAAAAAAI8', 'MV7J6NIKTKJZ2AABAAAAAEI8',
]
TEST_SCENE_IDS = ['MVUCSQAKTKJ5EAABAAAAABY8']

TARGET_SCENE_ID = os.environ.get('TARGET_SCENE_ID', 'MVUCSQAKTKJ5EAABAAAAABY8')
TASK_SOURCE_PATH = os.environ['TASK_SOURCE_PATH']
ORIGINAL_TASK_PATH = os.environ['ORIGINAL_TASK_PATH']
TRAJ_PATH = Path(os.environ['TRAJ_PATH'])

assert TARGET_SCENE_ID in TRAIN_SCENE_IDS + TEST_SCENE_IDS, \
    f"TARGET_SCENE_ID {TARGET_SCENE_ID} not in predefined scene ids"

SCENE_USD_PATH = config.get_scene_usd_path(TARGET_SCENE_ID)
OCC_MAP_PATH = Path(config.get_occ_map_path(TARGET_SCENE_ID))
assert os.path.exists(SCENE_USD_PATH), f"Scene USD path does not exist: {SCENE_USD_PATH}"

# =============================================================================
# Furniture and scene setup
# =============================================================================

from internutopia_extension.configs.objects import InteractiveObjCfg

scene_anno = json.load(open(config.get_metadata_path('scene_furniture_library.json')))
all_captions = json.load(open(config.get_scene_anno_path(TARGET_SCENE_ID)))

furnitures = []
furniture_names = []
furniture_prims = []
object_per_room = defaultdict(list)

empty_world_graph = {}
for furniture_uid, furniture_info in scene_anno.items():
    if TARGET_SCENE_ID != furniture_info['scene_id']:
        continue
    used_keys = ["name", "prim_path", "components"]
    furniture_data = {k: furniture_info[k] for k in used_keys}
    furniture = InteractiveObjCfg(**furniture_data)
    assert '/Root' not in str(furniture_data)
    furnitures.append(furniture)
    furniture_names.append(furniture_info['name'])
    empty_world_graph[furniture_info['name']] = {'content': []}
    if 'door' in furniture_info['components']:
        empty_world_graph[furniture_info['name']]['door'] = False
    furniture_prims.append(furniture_info['prim_path'])
    room_name = furniture_info['room_name']
    object_per_room[room_name].append(furniture_info['name'])

# =============================================================================
# Simulation setup
# =============================================================================

from internutopia_extension.configs.tasks import (
    FiniteStepTaskCfg,
    FiniteStepTaskEpisodeCfg,
)
from internutopia_extension.configs.robots.human_avatar import (
    HumanAvatarCfg,
    floating_camera_cfg,
)

mocked_robot = HumanAvatarCfg(
    name='person',
    position=(1.4, -0.7, 1.05),
    controllers=[],
    create_robot=True,
    usd_path=config.get_robot_usd_path(),
    scale=(0.001, 0.001, 0.001),
    sensors=[
        floating_camera_cfg.update(
            name='floating', resolution=(640, 480), enable=True, depth=True
        ),
    ],
)

episode_cfg = FiniteStepTaskEpisodeCfg(
    scene_asset_path=SCENE_USD_PATH,
    scene_scale=(0.01, 0.01, 0.01),
    robots=[mocked_robot],
    objects=furnitures,
    extra={},
)

from internutopia.core.config import Config, SimConfig
from internutopia.core.vec_env import Env
from internutopia.core.runtime import SimulatorRuntime

sim_config = Config(
    simulator=SimConfig(physics_dt=1 / 240, rendering_dt=1 / 240, use_fabric=False),
    task_config=FiniteStepTaskCfg(
        env_num=1,
        task_settings={'max_step': 1e10},
        episodes=[episode_cfg],
    ),
)

sim_runtime = SimulatorRuntime(
    config_class=sim_config, headless=True, native=False, webrtc=False
)

# IMPORTANT: import_extensions() must be called AFTER sim_runtime is initialized
# and BEFORE importing any internutopia_extension.eba_actions
from internutopia_extension import import_extensions
import_extensions()

from internutopia.core.scene.object import create_object
env = Env(sim_runtime)
env.reset()

import omni.replicator.core as rep

camera = rep.create.camera(
    name="Camera_0",
    position=(1.4, -0.7, 1.5),
    clipping_range=(0.01, 1000),
)

from internutopia.core.util.omni_usd_util import compute_path_bbox
from internutopia_extension.datagen.occ_map import NavManager
from internutopia.server.demo.camera_utils import (
    track_object,
    calculate_look_at_quaternion_fixed_camera,
)
from omni.isaac.core.prims.xform_prim import XFormPrim

camera_prim = XFormPrim(prim_path="/Replicator/Camera_0_Xform")
nav_manager = NavManager(scene_id=TARGET_SCENE_ID)

# =============================================================================
# Asset library and task processing
# =============================================================================

asset_lib = json.load(open(
    config.get_metadata_path('consolidated_asset_library_with_size.json')
))

# Load tasks - merge physics verified positions with original task metadata
with open(TASK_SOURCE_PATH, 'r') as f:
    physics_tasks = json.load(f)

with open(ORIGINAL_TASK_PATH, 'r') as f:
    original_tasks = json.load(f)


def query_object_category(obj_id):
    """Query object category from asset library."""
    obj_meta = asset_lib[obj_id]
    return obj_meta['category']


def build_world_graph_from_placements(placements):
    """Build world graph from placements dict."""
    world_graph = deepcopy(empty_world_graph)
    for obj_name, placement_info in placements.items():
        furniture_name = placement_info['furniture']
        if furniture_name in world_graph:
            world_graph[furniture_name]['content'].append(obj_name)
    return world_graph


def find_target_object_name(placements, target_obj_id, src_furniture=None):
    """Find the full object name for the target object from placements.

    If duplicated instances with the same original_id exist on multiple
    receptacles, prefer the one placed on src_furniture.
    """
    fallback_name = None
    for obj_name, placement_info in placements.items():
        if placement_info['original_id'] != target_obj_id:
            continue

        if fallback_name is None:
            fallback_name = obj_name

        if src_furniture is not None and placement_info.get('furniture') == src_furniture:
            return obj_name

    return fallback_name


# Process tasks with pre-computed positions merged with original metadata
processed_eval_episodes = []
for task in physics_tasks:
    task_id = task['task_id']
    original_idx = task.get('original_idx', 0)

    original_task = original_tasks[original_idx]

    initial_world_graph = build_world_graph_from_placements(task['placements'])

    obj_positions = {}
    for obj_name, pos_info in task['final_positions'].items():
        obj_positions[obj_name] = tuple(pos_info['final_pos'])

    target_obj_id = list(original_task['obj_meta'].keys())[0]
    target_object_name = find_target_object_name(
        task['placements'],
        target_obj_id,
        src_furniture=original_task['src'],
    )
    target_category = query_object_category(target_obj_id)


    obj_distractor_meta = {
        obj_id: original_task['obj_distractor_meta'][obj_id]
        for obj_id in original_task['obj_distractors']
    }

    processed_episode = {
        'task_id': task_id,
        'original_idx': original_idx,
        'task_description': task['task_description'],
        'task_description_original': original_task['task_description'],
        'initial_world_graph': initial_world_graph,
        'placements': task['placements'],
        'final_positions': task['final_positions'],
        'obj_positions': obj_positions,
        'execution_plan': original_task['execution_plan'],
        'src': original_task['src'],
        'dest': original_task['dest'],
        'src_distractors': original_task['src_distractors'],
        'dest_distractors': original_task['dest_distractors'],
        'obj_distractors': original_task['obj_distractors'],
        'obj_distractor_meta': obj_distractor_meta,
        'target_object_id': target_obj_id,
        'target_object_name': target_object_name,
        'target_category': target_category,
    }
    processed_eval_episodes.append(processed_episode)

print(f"Loaded eval episodes: {len(processed_eval_episodes)}")


# =============================================================================
# Object spawning with pre-computed positions
# =============================================================================

def spawn_objects_by_world_graph(env: Env, episode: dict, current_objects: dict):
    """
    Spawn objects using statically computed positions from placements.

    Uses the pre-computed placement positions (from static verification stage)
    rather than physics-settled final_positions, so objects settle naturally
    through physics simulation after spawning.

    Args:
        env: Simulation environment
        episode: Episode dict containing 'placements', 'task_id'
        current_objects: Dict of currently spawned objects to clean up

    Returns:
        Dict mapping object names to their metadata (category, original_id)
    """
    placements = episode['placements']

    # Clean up existing objects
    for current_obj_name in current_objects.keys():
        current_obj = env.runner.current_tasks[
            env.runner.current_task_name
        ].scene.get_object(current_obj_name)
        if current_obj is not None:
            current_obj.set_world_pose(position=(-100, -100, 0))

    current_objects = {}

    # Spawn objects at statically computed positions (raised slightly for physics settle)
    for obj_name, placement_info in placements.items():
        original_obj_id = placement_info['original_id']
        pos = placement_info['position']
        spawn_pos = (pos[0], pos[1], pos[2] + 0.1)

        obj_meta = asset_lib[original_obj_id]
        category = obj_meta['category']

        obj_cfg = InteractiveObjCfg(
            name=obj_name,
            prim_path=f"/World/env_0/scene/Meshes/{obj_name}",
            usd_path=str(obj_meta['usd_path']),
            position=spawn_pos,
            collider=True,
            components={"graspable": f"/World/env_0/scene/Meshes/{obj_name}"},
            scale=obj_meta['usd_scale'],
        )

        _obj = create_object(obj_cfg)
        _obj.set_up_scene(
            env.runner.current_tasks[env.runner.current_task_name].scene
        )
        env.runner.current_tasks[env.runner.current_task_name].objects[obj_name] = _obj
        current_objects[obj_name] = {
            "category": category,
            "original_id": original_obj_id,
        }

    # Step simulation to settle objects
    for _ in range(20):
        env.step(action=[{}])

    return current_objects
