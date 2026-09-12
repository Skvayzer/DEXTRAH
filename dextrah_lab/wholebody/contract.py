"""Explicit joint/action ownership and timing. No simulator imports."""
import numpy as np

# Checkpoint's Isaac Lab order, NOT URDF traversal or Unitree motor-ID order.
BODY_JOINTS = (
    'left_hip_pitch_joint', 'right_hip_pitch_joint', 'waist_yaw_joint',
    'left_hip_roll_joint', 'right_hip_roll_joint', 'waist_roll_joint',
    'left_hip_yaw_joint', 'right_hip_yaw_joint', 'waist_pitch_joint',
    'left_knee_joint', 'right_knee_joint',
    'left_shoulder_pitch_joint', 'right_shoulder_pitch_joint',
    'left_ankle_pitch_joint', 'right_ankle_pitch_joint',
    'left_shoulder_roll_joint', 'right_shoulder_roll_joint',
    'left_ankle_roll_joint', 'right_ankle_roll_joint',
    'left_shoulder_yaw_joint', 'right_shoulder_yaw_joint',
    'left_elbow_joint', 'right_elbow_joint',
    'left_wrist_roll_joint', 'right_wrist_roll_joint',
    'left_wrist_pitch_joint', 'right_wrist_pitch_joint',
    'left_wrist_yaw_joint', 'right_wrist_yaw_joint',
)
PHYSICS_DT = 1 / 200
CONTROL_DT = 1 / 50
TACTILE_HZ = 70
LATENT_DIM = 64
LATENT_SCALE = .1


def hand_joints(side):
    if side not in ('left', 'right'):
        raise ValueError('Hand side must be left or right')
    return tuple(side+'_'+name+'_joint' for name in (
        'thumb_metacarpal', 'thumb_proximal', 'index_proximal',
        'middle_proximal', 'ring_proximal', 'pinky_proximal'))


def joint_indices(actual, expected):
    if len(set(actual)) != len(actual) or len(set(expected)) != len(expected):
        raise ValueError('Duplicate joint name')
    missing = set(expected)-set(actual)
    if missing:
        raise ValueError(f'Missing required joints: {sorted(missing)}')
    return [actual.index(name) for name in expected]


def nominal_body_pose():
    result = np.zeros(29, dtype=np.float32)
    for i, name in enumerate(BODY_JOINTS):
        if '_hip_pitch_' in name: result[i] = -.312
        elif '_knee_' in name: result[i] = .669
        elif '_ankle_pitch_' in name: result[i] = -.363
        elif '_elbow_' in name: result[i] = .6
        elif '_shoulder_pitch_' in name: result[i] = .2
        elif name == 'left_shoulder_roll_joint': result[i] = .2
        elif name == 'right_shoulder_roll_joint': result[i] = -.2
    return result


def checkpoint_reference_command(q, qd):
    """Preserve released command packing exactly, including its reshape.

    GRAIL commands.py concatenates ALL future q frames then ALL future qd
    frames. observations.py subsequently reshapes this to (...,10,58).
    This is deliberately NOT per-frame concatenate([q_t, qd_t]). Changing
    that order silently changes the pretrained encoder input.
    """
    q, qd = np.asarray(q), np.asarray(qd)
    if q.shape != qd.shape or q.shape[-2:] != (10, 29):
        raise ValueError('Expected ten reference frames of 29 body joints')
    if not np.isfinite(q).all() or not np.isfinite(qd).all():
        raise ValueError('Nonfinite reference')
    shape = q.shape[:-2]
    return np.concatenate((q.reshape(*shape, 290), qd.reshape(*shape, 290)), -1).reshape(*shape, 10, 58)
