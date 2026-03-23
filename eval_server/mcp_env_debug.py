"""
Debug version of mcp_env.py: non-headless, opens the Isaac Sim GUI window.

Differences from mcp_env.py:
  - headless=False, native=True  → opens the Omniverse GUI for visual inspection
  - uses local metadata/ paths   → no dependency on /cpfs/user/miboyu/sft-refactored/
  - TASK_SOURCE_PATH / ORIGINAL_TASK_PATH are optional (defaults to empty lists)
  - all_captions load failure is non-fatal

Usage:
    HEADLESS=0 TARGET_SCENE_ID=... TASK_SOURCE_PATH=... ORIGINAL_TASK_PATH=... \\
        python -m eval_server.mcp_server_debug

Or import directly:
    from eval_server.mcp_env_debug import env, camera, processed_eval_episodes, ...
"""

import os
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from copy import deepcopy

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
TRAJ_PATH = Path(os.environ.get('TRAJ_PATH', 'eval_output_debug'))

TASK_SOURCE_PATH = os.environ.get('TASK_SOURCE_PATH', '')
ORIGINAL_TASK_PATH = os.environ.get('ORIGINAL_TASK_PATH', '')

assert TARGET_SCENE_ID in TRAIN_SCENE_IDS + TEST_SCENE_IDS, \
    f"TARGET_SCENE_ID {TARGET_SCENE_ID} not in predefined scene ids"
assert TASK_SOURCE_PATH, "TASK_SOURCE_PATH env var is required"
assert ORIGINAL_TASK_PATH, "ORIGINAL_TASK_PATH env var is required"

EMPTY_USD_PATH = "/cpfs/shared/simulation/liyangzi/grutopia/assets/scenes/empty.usd"
USE_EMPTY_SCENE = os.environ.get('USE_EMPTY_SCENE', '0') == '1'

SCENE_USD_PATH = (
    EMPTY_USD_PATH if USE_EMPTY_SCENE else
    f"/cpfs/shared/simulation/liyangzi/grutopia/assets/scenes/"
    f"GRScenes-100/home_scenes/scenes/{TARGET_SCENE_ID}_usd/"
    f"start_result_interaction_noMDL_move.usd"
)
OCC_MAP_PATH = Path(f"/cpfs/shared/simulation/miboyu/occ_map/{TARGET_SCENE_ID}")
assert os.path.exists(SCENE_USD_PATH), f"Scene USD path does not exist: {SCENE_USD_PATH}"

# =============================================================================
# Data paths — prefer local metadata/, fall back to absolute path
# =============================================================================

_REPO_ROOT = Path(__file__).parent.parent

def _resolve_data_path(local_rel: str, absolute_fallback: str) -> str:
    local = _REPO_ROOT / local_rel
    if local.exists():
        return str(local)
    return absolute_fallback

FURNITURE_LIB_PATH = _resolve_data_path(
    'metadata/scene_furniture_library.json',
    '/cpfs/user/miboyu/sft-refactored/data/scene_furniture_library.json',
)
ASSET_LIB_PATH = _resolve_data_path(
    'metadata/consolidated_asset_library_with_size.json',
    '/cpfs/user/miboyu/sft-refactored/data/consolidated_asset_library_with_size.json',
)

print(f"[mcp_env_debug] FURNITURE_LIB_PATH = {FURNITURE_LIB_PATH}")
print(f"[mcp_env_debug] ASSET_LIB_PATH     = {ASSET_LIB_PATH}")

# =============================================================================
# Furniture and scene setup
# =============================================================================

from internutopia_extension.configs.objects import InteractiveObjCfg, UsdObjCfg

scene_anno = json.load(open(FURNITURE_LIB_PATH))

# Captions are optional in debug mode
_caption_path = f"/cpfs/user/miboyu/sft/data/scene_anno/{TARGET_SCENE_ID}_all_caption_processed.json"
try:
    all_captions = json.load(open(_caption_path))
except FileNotFoundError:
    print(f"[mcp_env_debug] WARNING: captions not found at {_caption_path}, skipping.")
    all_captions = {}

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
# Lift robot definition (optional, enabled by USE_LIFT_ROBOT=1)
# =============================================================================

from typing import Optional as _Optional
from collections import OrderedDict as _OrderedDict
from internutopia.core.config import RobotCfg
from internutopia.core.robot.robot import BaseRobot
from internutopia.core.robot.articulation import IArticulation
from internutopia.core.scene.scene import IScene


class LiftCfg(RobotCfg):
    name: _Optional[str] = 'lift'
    type: _Optional[str] = 'Lift'
    prim_path: _Optional[str] = '/lift'
    usd_path: _Optional[str] = "/cpfs/shared/simulation/miboyu/lift2/lift2_no_occ.usd"


@BaseRobot.register('Lift')
class LiftRobot(BaseRobot):
    def __init__(self, config: LiftCfg, scene: IScene):
        super().__init__(config, scene)
        self.articulation = IArticulation.create(
            prim_path=config.prim_path,
            name=config.name,
            usd_path=config.usd_path,
            position=np.array(config.position),
        )

    def post_reset(self):
        super().post_reset()
        print("[LiftRobot] joints:", self.articulation.dof_names)
        from pxr import UsdPhysics, Usd, PhysxSchema
        from omni.isaac.core.utils.stage import get_current_stage
        stage = get_current_stage()
        root_prim = stage.GetPrimAtPath(self.config.prim_path)
        for prim in Usd.PrimRange(root_prim):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                if not prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
                    PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
                PhysxSchema.PhysxRigidBodyAPI(prim).GetDisableGravityAttr().Set(True)
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Set(False)

    def apply_action(self, action):
        if not isinstance(action, dict):
            return
        for controller_name, controller_action in action.items():
            controller = self.controllers[controller_name]
            control = controller.action_to_control(controller_action)
            self.articulation.apply_action(control)

    def get_obs(self):
        position, orientation = self.articulation.get_pose()
        controllers_obs, sensors_obs = super()._get_controllers_and_sensors_obs()
        obs = {'position': position, 'orientation': orientation,
               'controllers': controllers_obs, 'sensors': sensors_obs}
        obs["joint_velocitis"] = self.articulation.get_joint_velocities()
        obs["joint_positions"] = self.articulation.get_joint_positions()
        return self._make_ordered(obs)


USE_LIFT_ROBOT = os.environ.get('USE_LIFT_ROBOT', '0') == '1'

lift_cfg = LiftCfg(
    position=(1.6, -1.3, 0.0),
    controllers=[],
    sensors=[],
)

# =============================================================================
# Simulation setup — NON-HEADLESS (GUI window)
# =============================================================================

from internutopia_extension.configs.tasks import FiniteStepTaskCfg
from internutopia.core.config import Config, SimConfig
from internutopia.core.gym_env import Env
from internutopia_extension import import_extensions

config = Config(
    simulator=SimConfig(physics_dt=1 / 240, rendering_dt=1 / 240, use_fabric=False, headless=False, webrtc=False),
    task_configs=[
        FiniteStepTaskCfg(
            scene_asset_path=SCENE_USD_PATH,
            scene_scale=(0.01, 0.01, 0.01),
            robots=[lift_cfg] if USE_LIFT_ROBOT else [],
            objects=furnitures,
            max_steps=int(1e10),
        ),
    ],
)

env = Env(config)
import_extensions()
obs, _ = env.reset()
print(f'========INIT OBS{obs}=============')

from internutopia.core.scene.object import create_object

import omni.replicator.core as rep

camera = rep.create.camera(
    name="Camera_0",
    position=(1.4, -0.7, 1.5),
    clipping_range=(0.01, 1000),
)

from internutopia.core.util.omni_usd_util import compute_path_bbox
from internutopia_extension.utils.occ_map import NavManager
from internutopia_extension.utils.camera_utils import (
    track_object,
    calculate_look_at_quaternion_fixed_camera,
)
from omni.isaac.core.prims.xform_prim import XFormPrim

camera_prim = XFormPrim(prim_path="/Replicator/Camera_0_Xform")
nav_manager = NavManager(scene_id=TARGET_SCENE_ID)

# =============================================================================
# Asset library and task processing
# =============================================================================

asset_lib = json.load(open(ASSET_LIB_PATH))

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

print(f"[mcp_env_debug] Loaded eval episodes: {len(processed_eval_episodes)}")


# =============================================================================
# Object spawning with pre-computed positions
# =============================================================================

def spawn_objects_by_world_graph(env: Env, episode: dict, current_objects: dict):
    """
    Spawn objects at their physics-verified final positions (obj_positions),
    so they sit on furniture immediately without in-simulation physics settling.

    Args:
        env: Simulation environment
        episode: Episode dict containing 'placements', 'obj_positions', 'task_id'
        current_objects: Dict of currently spawned objects to clean up

    Returns:
        Dict mapping object names to their metadata (category, original_id)
    """
    placements = episode['placements']

    # Clean up existing objects
    for current_obj_name in current_objects.keys():
        current_obj = env.runner.current_tasks[
            env._current_task_name
        ].objects.get(current_obj_name)
        if current_obj is not None:
            current_obj.prim.set_world_pose(position=(-100, -100, 0))

    current_objects = {}

    # Spawn objects at their physics-verified final positions recorded during
    # offline verification, so they sit on furniture immediately without
    # relying on in-simulation physics settling.
    obj_positions = episode.get('obj_positions', {})
    for obj_name, placement_info in placements.items():
        original_obj_id = placement_info['original_id']
        if obj_name in obj_positions:
            fp = obj_positions[obj_name]
            # Use final_pos XY for accuracy; add a small Z offset so the
            # object approaches from above, avoiding PhysX depenetration
            # pushing it downward through the furniture surface.
            spawn_pos = (fp[0], fp[1], fp[2] + 0.05)
        else:
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
            env.runner.current_tasks[env._current_task_name]._scene
        )
        env.runner.current_tasks[env._current_task_name].objects[obj_name] = _obj
        current_objects[obj_name] = {
            "category": category,
            "original_id": original_obj_id,
        }

    # A few physics steps are required so the engine registers the newly
    # created rigid bodies before the main loop takes over.
    for _ in range(20):
        env.step(action=[{}])

    return current_objects