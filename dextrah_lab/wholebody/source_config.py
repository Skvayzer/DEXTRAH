"""Load the completed touch run, then apply only enumerated body extensions."""
from copy import deepcopy
from pathlib import Path
from .task_contract import TOUCH_RUN, verify_task_contract


def source_task_config(output, num_envs, device='cuda:0', source_run=TOUCH_RUN):
    from isaaclab.utils import replace_strings_with_slices
    from dextrah_lab.g1_adept.touch_continuation import load_yaml
    from dextrah_lab.tasks.g1_revo2_adept.g1_sonic_touch_env import G1SonicTouchEnvCfg
    saved = load_yaml(Path(source_run)/'params/env_resolved.yaml')
    cfg = G1SonicTouchEnvCfg()
    cfg.seed = saved['seed']
    cfg.from_dict(replace_strings_with_slices(deepcopy(saved)))
    cfg.assets.robot_profile = 'g1_brainco'
    cfg.assets.robot_fix_base = False
    cfg.assets.robot_self_collision = False
    cfg.assets.robot_actuate_left_hand = True
    cfg.assets.robot_body_collision_enabled = False
    cfg.scene.num_envs = num_envs
    cfg.sim.device = device
    cfg.bps_artifact_dir = str(Path(output)/'bps')
    # Retain original bank cache; output artifacts never overwrite source run.
    if num_envs <= 1200:
        for key, value in dict(gpu_found_lost_pairs_capacity=2**20,
                gpu_found_lost_aggregate_pairs_capacity=2**21, gpu_total_aggregate_pairs_capacity=2**20,
                gpu_max_rigid_contact_count=2**20, gpu_max_rigid_patch_count=2**18,
                gpu_collision_stack_size=2**26).items():
            setattr(cfg.sim.physx, key, value)
    contract = verify_task_contract(saved, cfg)
    return cfg, contract
