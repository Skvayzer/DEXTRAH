import xml.etree.ElementTree as ET
import numpy as np
import pytest
from dextrah_lab.wholebody.contract import BODY_JOINTS, hand_joints, nominal_body_pose
from dextrah_lab.wholebody.kinematics import UrdfKinematics, pose_matrix, matrix_pose
from dextrah_lab.wholebody.reference import Segment
from dextrah_lab.wholebody.teacher_bridge import RIGHT_ARM, align_teacher_segment, future_body_reference


def fixture(tmp_path):
    root = ET.Element('robot', name='bridge_test')
    ET.SubElement(root, 'link', name='pelvis')
    parents = {'waist_yaw_joint': 'pelvis', 'waist_roll_joint':'waist_yaw_link',
               'waist_pitch_joint':'waist_roll_link'}
    child_names = {'waist_pitch_joint':'torso_link'}
    for i, name in enumerate(RIGHT_ARM):
        parents[name] = 'torso_link' if i==0 else RIGHT_ARM[i-1].replace('_joint', '_link')
    for name in (*BODY_JOINTS, *hand_joints('right')):
        child = child_names.get(name, name.replace('_joint', '_link'))
        ET.SubElement(root, 'link', name=child)
        joint = ET.SubElement(root, 'joint', name=name, type='revolute')
        ET.SubElement(joint, 'parent', link=parents.get(name, 'pelvis'))
        ET.SubElement(joint, 'child', link=child)
        ET.SubElement(joint, 'origin', xyz='.04 0 .03', rpy='.1 .2 .3')
        ET.SubElement(joint, 'axis', xyz='0 1 0')
        ET.SubElement(joint, 'limit', lower='-3', upper='3', velocity='10', effort='2')
    path = tmp_path/'robot.urdf'
    ET.ElementTree(root).write(path)
    return path


def source(tmp_path):
    path = fixture(tmp_path)
    tree = UrdfKinematics(path)
    n = 121
    time = np.arange(n)/60
    names = list(reversed((*RIGHT_ARM, *hand_joints('right'))))
    q = np.tile(.1*time[:, None], (1, len(names)))
    static = dict(zip(BODY_JOINTS, nominal_body_pose()))
    position = dict(static, **{name:q[:, i] for i, name in enumerate(names)})
    root = np.tile([.2, .4, .75, 2**-.5, 0, 0, -2**-.5], (n, 1))
    body = np.stack([pose_matrix(root) @ tree.transform(name, position, n)
                     for name in ('torso_link', 'right_wrist_yaw_link')], 1)
    body_pose = matrix_pose(body)
    obj = np.tile([.4, .1, .9, 1, 0, 0, 0], (n, 1))
    trace = dict(step=np.arange(n), resets=np.zeros(n), goal_hits=(time>=1).astype(int),
        joint_pos=q, joint_vel=np.full_like(q, .1), commanded_joint_targets=q+.02,
        robot=root, body_pos=body_pose[..., :3], body_quat=body_pose[..., 3:],
        object=obj, table=obj.copy(), goal=obj.copy())
    meta = dict(joint_names=names, body_names=['torso_link', 'right_wrist_yaw_link'],
                visual_static_joint_pos=static, reference_dt=1/60)
    return path, trace, meta


def test_same_robot_alignment_preserves_relative_object_and_wrist(tmp_path):
    path, trace, meta = source(tmp_path)
    data, audit = align_teacher_segment(trace, meta, Segment(0, 121, 1), path)
    assert data['body_q'].shape == (101, 29)
    assert audit['source_fk_position_error_m'] < 1e-12
    assert audit['hand_alignment_matrix_error'] < 1e-12
    assert not audit['physics_validated'] and not audit['distillation_completed']
    old_wrist = pose_matrix(np.r_[trace['body_pos'][0, 1], trace['body_quat'][0, 1]])
    old_relative = np.linalg.inv(old_wrist) @ pose_matrix(trace['object'][0])
    new_relative = np.linalg.inv(pose_matrix(data['wrist_pose'][0])) @ pose_matrix(data['object_pose'][0])
    np.testing.assert_allclose(old_relative, new_relative, atol=1e-12)
    assert audit['limit_violations'] == {}
    assert data['previous_right_hand_targets'][0, 0] == .02


def test_moving_torso_and_bad_source_frames_fail_closed(tmp_path):
    path, trace, meta = source(tmp_path)
    trace['body_pos'][20, 0, 0] += .01
    with pytest.raises(ValueError, match='torso moves'):
        align_teacher_segment(trace, meta, Segment(0, 121, 1), path)
    trace['body_pos'][20, 0, 0] -= .01
    trace['body_pos'][20, 1, 0] += .01
    with pytest.raises(ValueError, match='FK/recording mismatch'):
        align_teacher_segment(trace, meta, Segment(0, 121, 1), path)


def test_future_spacing_and_endpoint_hold(tmp_path):
    path, trace, meta = source(tmp_path)
    data, _ = align_teacher_segment(trace, meta, Segment(0, 121, 1), path)
    q, qd, root = future_body_reference(data, .5)
    arm = BODY_JOINTS.index(RIGHT_ARM[0])
    np.testing.assert_allclose(q[:, arm], .1*(.5+np.arange(10)*.1), atol=1e-12)
    assert root.shape == (10, 7)
    q, qd, _ = future_body_reference(data, 1.95)
    np.testing.assert_allclose(q[1:], np.tile(data['body_q'][-1], (9, 1)))
    assert not qd[1:].any()


def test_missing_independent_joint_is_not_silently_zeroed(tmp_path):
    tree = UrdfKinematics(fixture(tmp_path))
    with pytest.raises(ValueError, match='Missing independent'):
        tree.transform('right_wrist_yaw_link', {}, 1)


def test_fk_honors_nonzero_locked_angle_and_mimic(tmp_path):
    path = fixture(tmp_path)
    tree = ET.parse(path)
    joint = tree.find(".//joint[@name='right_wrist_yaw_joint']")
    ET.SubElement(joint, 'mimic', joint='right_wrist_pitch_joint', multiplier='1.2', offset='.15')
    tree.write(path)
    fk = UrdfKinematics(path)
    q = dict(zip(BODY_JOINTS, nominal_body_pose()))
    q.pop('right_wrist_yaw_joint')
    expected = dict(q, right_wrist_yaw_joint=.15)
    np.testing.assert_allclose(fk.transform('right_wrist_yaw_link', q, 1),
                               fk.transform('right_wrist_yaw_link', expected, 1))
