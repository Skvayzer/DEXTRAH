"""Distinguish ordinary body falls from finite articulation-speed outliers.

No state sanitization or motor changes. Non-finite simulation states remain a
hard error; finite speed outliers can end only the affected robot's episode.
"""
import torch


def classify_body_state(joint_pos, joint_vel, root_state, gravity, origins,
                        object_state, *, minimum_height=.4, minimum_upright=.5,
                        maximum_joint_speed=1000., numerical_failure_mode='abort'):
    if numerical_failure_mode not in ('abort', 'reset'):
        raise ValueError('numerical_failure_mode must be abort or reset')
    if not maximum_joint_speed > 0:
        raise ValueError('maximum_joint_speed must be positive')
    # Fail before constructing rewards/terminal critic observations from NaNs.
    finite = torch.stack([torch.isfinite(x).all() for x in
                          (joint_pos, joint_vel, root_state, gravity, object_state)]).all()
    if not finite:
        raise RuntimeError('Non-finite simulation state; refusing to feed it to SAPG')
    speed = joint_vel.abs().amax(dim=-1)
    numerical = speed > maximum_joint_speed
    if numerical_failure_mode == 'abort' and numerical.any():
        raise RuntimeError(f'Numerically invalid articulation: maximum joint speed {float(speed.max())} rad/s')
    height = root_state[:, 2]-origins[:, 2]
    fallen = (height < minimum_height) | (-gravity[:, 2] < minimum_upright)
    return fallen, numerical, speed


def combine_terminations(task_terminated, task_truncated, fallen, numerical):
    body_failure = fallen | numerical
    # Body failure is a true terminal, never a time-limit value bootstrap.
    return task_terminated | body_failure, task_truncated & ~body_failure
