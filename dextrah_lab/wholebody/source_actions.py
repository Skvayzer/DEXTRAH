"""One body writer, source finger processing, shared SAPG action delay."""
from copy import copy
from types import SimpleNamespace
import torch


def delayed_actions(env, actions):
    dr = env.cfg.domain_randomization
    if not dr.use_action_delay or dr.action_delay_max <= 0:
        return actions
    start = (env.episode_length_buf == 0) & (env._successes == 0)
    env._wholebody_action_queue[start] = actions[start, None]
    env._wholebody_action_queue = torch.roll(env._wholebody_action_queue, 1, 1)
    env._wholebody_action_queue[:, 0] = actions
    indices = torch.randint(0, env._wholebody_action_queue.shape[1], (env.num_envs,), device=env.device)
    return env._wholebody_action_queue[torch.arange(env.num_envs, device=env.device), indices]


def apply_wholebody_action(env, actions, source_pipeline):
    """Reuse the original finger/mimic math without a second arm controller.

    A lightweight proxy disables ONLY the already-applied delay for this call.
    Source arm-delta calculations are discarded. Only the SONIC target writes
    the 29 body joints. Preserve source queue shape for diagnostic consumers.
    """
    if actions.shape != (env.num_envs, 35) or not torch.isfinite(actions).all():
        raise ValueError('Expected finite 29 absolute body + 6 finger actions')
    actions = actions.clamp(-1, 1)
    actions = delayed_actions(env, actions)
    cfg = copy(env.cfg)
    cfg.domain_randomization = copy(cfg.domain_randomization)
    cfg.domain_randomization.use_action_delay = False
    keys = ('device', 'step_dt', '_num_arm_action_joints', '_prev_targets', '_cur_targets',
            '_arm_joint_ids', '_hand_joint_ids', '_arm_lower', '_arm_upper', '_hand_lower',
            '_hand_upper', '_mimic_target_joint_ids', '_mimic_source_hand_action_ids',
            '_mimic_multipliers', '_mimic_offsets', 'robot')
    proxy = SimpleNamespace(cfg=cfg, **{k: getattr(env, k) for k in keys})
    projected = torch.cat((actions.new_zeros(env.num_envs, 7), actions[:, 29:]), -1)
    source_pipeline(proxy, projected)
    targets = env._body_center + env._body_half_range * actions[:, :29]
    targets = targets.clamp(env._body_lower, env._body_upper)
    env._cur_targets[:, env._body_ids] = targets
    env._prev_targets = env._cur_targets.clone()
    env._last_body_action[:] = (targets-env._body_nominal)/env._body_scales
    # Historical source arm-delta entries no longer have a meaningful control
    # interpretation; explicitly zero them, keep real finger queue values.
    env._action_queue[:, :, :7] = 0
    env._action_queue[:, :, 7:] = env._wholebody_action_queue[:, :, 29:]
