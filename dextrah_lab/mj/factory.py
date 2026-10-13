"""Build the mjlab task with the run-1105 configuration (the Isaac backend stays separate)."""
from pathlib import Path

WORKSPACE = Path('/data1/users/konstantin.smirnov')

# Contact capacity per world (MuJoCo Warp allocates nconmax*nworld contacts and
# njmax constraint rows per world). Measured in scripts/mj_contact_capacity.py.
DEFAULT_NCONMAX = 96
DEFAULT_NJMAX = 600
# MuJoCo solver settings for grasping; swept in the parity tests.
DEFAULT_MUJOCO = dict(cone='elliptic', impratio=10., iterations=20, ls_iterations=20)


def make_mj_env(num_envs, device='cuda:0', *, seed=42, nconmax=DEFAULT_NCONMAX, njmax=DEFAULT_NJMAX,
                mujoco_cfg=None, scene_params=None, velocity_limits=True, run=None):
    from .env import G1SonicTouchMjEnv, load_run_cfg, RUN_1105
    from dextrah_lab.wholebody.sonic import FrozenSonic
    cfg = load_run_cfg(run or RUN_1105)
    cfg.seed = seed
    sonic = FrozenSonic(WORKSPACE/'GRAIL', WORKSPACE/'G1-SONIC-models/checkpoint/SONIC/models/sonic_manipulation_base', device)
    return G1SonicTouchMjEnv(cfg, sonic=sonic, num_envs=num_envs, device=device, nconmax=nconmax, njmax=njmax,
                             mujoco_cfg=DEFAULT_MUJOCO if mujoco_cfg is None else mujoco_cfg,
                             scene_params=scene_params, velocity_limits=velocity_limits, seed=seed)
