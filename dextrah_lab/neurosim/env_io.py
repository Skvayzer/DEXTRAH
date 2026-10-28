"""Carry env plumbing shared by the neural-simulator scripts: build the carry env of a carry expert's run
contract, and snapshot the logged fields (dextrah_lab/neurosim/data.FIELDS) of every world.

The fields match scripts/neurosim/collect_carry_transitions.py: object COM pose / velocity relative to the
palm (palm frame), hand joints and targets, palm velocity and gravity in the palm frame, table relative to the
palm and parked flag, external force on the object, contact forces, support, mass / COM, phase, episode step,
arm reference and change, walking command.
"""
import json
from pathlib import Path


# Per-step domain randomisation of the run config (as scripts/record_wholebody_videos.NO_RANDOMIZATION): observation
# and action delays (each world draws a 0..max-step-old action every step), noised / delayed object state, joint
# velocity noise, random object force / torque. Off makes worlds with identical inputs identical.
NO_RANDOMIZATION = dict(use_obs_delay=False, use_action_delay=False, use_object_state_delay_noise=False,
                        object_state_xyz_noise_std=0., object_state_rotation_noise_degrees=0.,
                        joint_velocity_obs_noise_std=0., force_scale=0., torque_scale=0.)


def build_carry_env(checkpoint, num_envs, *, settings_override=None, holding_state=None, seed=0, nconmax=64, njmax=400,
                    level=2, device='cuda:0', no_randomization=False):
    """(env, policy model, contract, settings) for a carry expert checkpoint; settings_override: CarrySettings
    fields to replace (e.g. no mass / push randomisation); holding_state: another SGS state file (same scene);
    no_randomization: the run config's per-step domain randomisation off (NO_RANDOMIZATION)."""
    import torch  # noqa: F401
    from dextrah_lab.mj.carry_env import G1CarryMjEnv
    from dextrah_lab.mj.env import load_run_cfg
    from dextrah_lab.mj.factory import DEFAULT_MUJOCO, WORKSPACE
    from dextrah_lab.mj.robot import ContactParams
    from dextrah_lab.mj.scene import SceneParams
    from dextrah_lab.sgs.mj_task import SgsSettings
    from dextrah_lab.wholebody.carry_compose import CarrySettings
    from dextrah_lab.wholebody.sonic import FrozenSonic
    from mj_table_to_table import build_policy_dims
    run = Path(checkpoint).resolve().parent.parent
    contract = json.loads((run/'task_contract.json').read_text())
    cm = contract['carry_mjlab']
    raw = {k: tuple(v) if isinstance(v, list) else v for k, v in cm['settings'].items()}
    raw.update(settings_override or {})
    settings = CarrySettings(**raw)
    task_run = Path(contract['task_config_source']).parent.parent
    task_contract = json.loads((task_run/'task_contract.json').read_text())
    cfg = load_run_cfg(task_run)
    cfg.seed = seed
    if no_randomization:
        for key, value in NO_RANDOMIZATION.items():
            if not hasattr(cfg.domain_randomization, key):
                raise KeyError(f'domain_randomization.{key} missing')
            setattr(cfg.domain_randomization, key, value)
    cfg.sonic_body.spawn_on_table = task_contract['body_termination']['spawn_on_table']
    holding = Path(holding_state or cm['holding_state'])
    held_contract = json.loads((holding.parent.parent/'task_contract.json').read_text())
    sonic = FrozenSonic(WORKSPACE/'GRAIL', WORKSPACE/'G1-SONIC-models/checkpoint/SONIC/models/sonic_manipulation_base', device)
    trained = contract.get('contact_model', {})
    params = SceneParams(contact=ContactParams(condim=trained.get('condim', 3), torsional=trained.get('torsional', ContactParams().torsional)))
    env = G1CarryMjEnv(cfg, sonic=sonic, carry=settings, holding_state=holding,
                       held_settings=SgsSettings(**dict(held_contract['sgs']['settings'], eval_pool_size=0)),
                       clips_path=Path(cm['clips']), num_envs=num_envs, device=device, nconmax=nconmax, njmax=njmax,
                       mujoco_cfg=DEFAULT_MUJOCO, seed=seed, scene_params=params)
    env._curriculum.level = level
    env._seq.set_level(level)
    model, _ = build_policy_dims(env, checkpoint, contract, device, env.cfg.observation_space, env.hand_surface_dim)
    return env, model, contract, settings


def hand_joint_ids(env):
    import torch
    if hasattr(env, '_passive_joint_ids'):
        return torch.unique(torch.cat((env._hand_joint_ids, env._passive_joint_ids)))
    return env._hand_joint_ids


def snapshot(env, hand_ids):
    """Logged fields of every world after a step (the collector's layout, (N, d) each)."""
    import torch
    from isaaclab.utils.math import quat_apply_inverse, quat_conjugate, quat_mul
    n, device = env.num_envs, env.device
    _, palm = env._palm()
    pq = palm[:, 3:7]
    obj = env.object.data
    com = env.object.free_qpos[:, :3]
    _, rel_lin, rel_ang = env._hold_kinematics()
    data = env.robot.data
    c = env._last_contacts
    zero = torch.zeros(n, device=device)
    gravity_w = torch.tensor([0., 0., -1.], device=device).expand(n, 3)
    obj_mj = int(env._held.body)
    return dict(
        obj_pos=quat_apply_inverse(pq, com-palm[:, :3]), obj_quat=quat_mul(quat_conjugate(pq), obj.root_quat_w),
        obj_linvel=rel_lin, obj_angvel=rel_ang,
        hand_q=data.joint_pos[:, hand_ids], hand_qd=data.joint_vel[:, hand_ids], hand_target=env._cur_targets[:, env._hand_joint_ids],
        palm_linvel=quat_apply_inverse(pq, palm[:, 7:10]), palm_angvel=quat_apply_inverse(pq, palm[:, 10:13]),
        gravity=quat_apply_inverse(pq, gravity_w),
        table_pos=quat_apply_inverse(pq, env.table.data.root_pos_w-palm[:, :3]), table_parked=env._table_parked.float(),
        object_force=quat_apply_inverse(pq, env.sim.data.xfrc_applied[:, obj_mj, :3]),
        finger_force=c['finger_object_force_l1'], palm_force=c['palm_object_force_l1'],
        table_force=c.get('object_table_force', zero), ground_force=c.get('object_ground_force', zero),
        supported=env._supported.float(), mass=env._object_mass[:, 0], com_offset=env.object.com_offset.clone(),
        phase=env._seq.phase(env._episode_time()).float(), carry_t=env._carry_t.float(),
        arm_now=env._arm_now, arm_D=env._arm_D, command=env._clip_command)
