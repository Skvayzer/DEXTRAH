"""Right-hand teacher -> standing whole-body KINEMATIC reference.

This constructs an exact same-robot torso alignment, not whole-body IK. Body
balance, collisions, contact timing and object load transfer remain simulation
validation gates. Do not label this conversion a successful distillation.
"""
import numpy as np
from .contract import BODY_JOINTS, hand_joints, joint_indices, nominal_body_pose
from .kinematics import UrdfKinematics, pose_matrix, matrix_pose
from .reference import retime_segment, _retime_pose

RIGHT_ARM = tuple('right_' + name + '_joint' for name in (
    'shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow',
    'wrist_roll', 'wrist_pitch', 'wrist_yaw'))


def align_teacher_segment(trace, metadata, segment, urdf, target_root=(0., 0., .76, 1., 0., 0., 0.)):
    """Preserve recorded wrist-object transforms under ONE rigid scene transform.

    Reject moving torso sources: a per-frame scene warp would unrealistically
    move the table. Finger positions and applied targets are separate arrays.
    Returned commanded targets are PREVIOUS applied commands, not labels for
    the next policy action. New teacher-policy captures must supply those.
    """
    tree = UrdfKinematics(urdf)
    names, body_names = metadata['joint_names'], metadata['body_names']
    joint_indices(names, (*RIGHT_ARM, *hand_joints('right')))
    a, b = segment.start, segment.stop
    n = b-a
    torso_id, wrist_id = joint_indices(body_names, ('torso_link', 'right_wrist_yaw_link'))
    source_torso = pose_matrix(np.c_[trace['body_pos'][a:b, torso_id], trace['body_quat'][a:b, torso_id]])
    if not np.allclose(source_torso, source_torso[:1], atol=2e-4):
        raise ValueError('Teacher torso moves: constant scene alignment is not valid')
    source_positions = dict(metadata['visual_static_joint_pos'])
    source_positions.update({name:trace['joint_pos'][a:b, i] for i, name in enumerate(names)})
    measured_wrist = pose_matrix(np.c_[trace['body_pos'][a:b, wrist_id], trace['body_quat'][a:b, wrist_id]])
    # Verify source URDF and frame conventions against real simulator traces.
    source_fk = pose_matrix(trace['robot'][a:b]) @ tree.transform('right_wrist_yaw_link', source_positions, n)
    fk_position_error = np.linalg.norm(source_fk[:, :3, 3]-measured_wrist[:, :3, 3], axis=-1).max()
    fk_rotation_error = np.max(np.abs(source_fk[:, :3, :3]-measured_wrist[:, :3, :3]))
    if fk_position_error > .001 or fk_rotation_error > .002:
        raise ValueError(f'Source FK/recording mismatch: position={fk_position_error}, rotation={fk_rotation_error}')
    target_root_matrix = pose_matrix(np.asarray(target_root))
    nominal = dict(zip(BODY_JOINTS, nominal_body_pose()))
    target_torso = target_root_matrix @ tree.transform('torso_link', nominal, 1)[0]
    alignment = target_torso @ np.linalg.inv(source_torso[0])
    if not np.allclose(alignment[:3, :3] @ [0., 0., 1.], [0., 0., 1.], atol=1e-4):
        raise ValueError('Scene alignment rotates gravity; requires a different reference construction')
    data = retime_segment(trace, segment, source_dt=metadata['reference_dt'])
    count = len(data['time_s'])
    q = np.tile(nominal_body_pose(), (count, 1)).astype(float)
    qd = np.zeros_like(q)
    body_arm_ids = joint_indices(BODY_JOINTS, RIGHT_ARM)
    teacher_arm_ids = joint_indices(names, RIGHT_ARM)
    q[:, body_arm_ids] = data['joint_pos'][:, teacher_arm_ids]
    qd[:, body_arm_ids] = data['joint_vel'][:, teacher_arm_ids]
    target_positions = {name:q[:, i] for i, name in enumerate(BODY_JOINTS)}
    wrist = target_root_matrix @ tree.transform('right_wrist_yaw_link', target_positions, count)
    # Compare at original samples to avoid mixing joint-Hermite and pose-SLERP
    # interpolants (they need not agree between samples).
    original_target = dict(nominal)
    original_target.update({name:source_positions[name] for name in RIGHT_ARM})
    converted_wrist = target_root_matrix @ tree.transform('right_wrist_yaw_link', original_target, n)
    alignment_error = np.max(np.abs(converted_wrist - alignment @ measured_wrist))
    if alignment_error > .002:
        raise ValueError(f'Torso alignment does not preserve the hand trajectory: {alignment_error}')
    result = dict(time_s=data['time_s'], body_q=q, body_qd=qd,
        root_pose=np.tile(target_root, (count, 1)), wrist_pose=matrix_pose(wrist),
        right_hand_q=data['joint_pos'][:, joint_indices(names, hand_joints('right'))],
        previous_right_hand_targets=data['commanded_joint_targets'][:, joint_indices(names, hand_joints('right'))],
        previous_right_arm_targets=data['commanded_joint_targets'][:, teacher_arm_ids],
        source_sample=data['source_sample'])
    for key in ('object', 'goal', 'table'):
        result[key+'_pose'] = matrix_pose(alignment @ pose_matrix(data[key]))
    # Report violations; never clip a teacher trajectory and claim it is intact.
    violations = {}
    for i, name in enumerate(BODY_JOINTS):
        limit = tree.joints[name].find('limit')
        position = max(float(np.max(float(limit.get('lower'))-q[:, i])),
                       float(np.max(q[:, i]-float(limit.get('upper')))), 0.)
        speed = max(float(np.max(np.abs(qd[:, i])))-float(limit.get('velocity')), 0.)
        if position > 1e-5 or speed > 1e-5:
            violations[name] = dict(position_excess_rad=position, velocity_excess_rad_s=speed)
    report = dict(source_fk_position_error_m=float(fk_position_error),
        source_fk_rotation_matrix_error=float(fk_rotation_error),
        scene_alignment=alignment.tolist(), hand_alignment_matrix_error=float(alignment_error),
        body_joint_order=list(BODY_JOINTS), hand_joint_order=list(hand_joints('right')),
        limit_violations=violations, same_robot_kinematic_alignment=True,
        wholebody_ik_performed=False, physics_validated=False, distillation_completed=False,
        command_semantics='Previous applied position targets; not next-action supervision',
        outstanding=['dynamic balance', 'self/table collisions', 'wrist tracking', 'object load transfer'])
    return result, report


def future_body_reference(reference, time_s, spacing_s=.1):
    """Ten planned samples at the GRAIL reference spacing, NOT ten 50 Hz ticks.

    Endpoints hold pose with zero velocity. Input is an offline PLANNED motion,
    never future states read from the running student simulation.
    """
    source = reference['time_s']
    if not np.isfinite(time_s) or spacing_s <= 0 or not np.all(np.diff(source)>0):
        raise ValueError('Invalid reference query')
    query = time_s + np.arange(10)*spacing_s
    clipped = np.clip(query, source[0], source[-1])
    q = np.stack([np.interp(clipped, source, reference['body_q'][:, i]) for i in range(29)], -1)
    qd = np.stack([np.interp(clipped, source, reference['body_qd'][:, i]) for i in range(29)], -1)
    qd[(query < source[0]) | (query >= source[-1])] = 0.
    root = _retime_pose(source, reference['root_pose'], clipped)
    return q, qd, root
