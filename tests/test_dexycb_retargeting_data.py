from pathlib import Path

import numpy as np
import yaml

from dextrah_lab.retargeting.dexycb import (
    load_sequence,
    object_motion_window,
    palm_frame,
    palm_relative_fingertips,
)


def _right_hand() -> np.ndarray:
    joints = np.zeros((21, 3), dtype=np.float64)
    joints[0] = [0.5, -0.2, 1.0]
    joints[5] = joints[0] + [0.0, 0.04, 0.05]
    joints[9] = joints[0] + [0.0, 0.00, 0.06]
    joints[17] = joints[0] + [0.0, -0.04, 0.05]
    for index, y in zip((4, 8, 12, 16, 20), (0.06, 0.035, 0.0, -0.03, -0.055)):
        joints[index] = joints[0] + [0.01, y, 0.12]
    return joints


def test_palm_frame_is_orthonormal_and_right_handed():
    joints = np.stack((_right_hand(), _right_hand() + [0.2, 0.1, -0.3]))
    origin, rotation = palm_frame(joints)

    np.testing.assert_allclose(origin, joints[:, 0])
    np.testing.assert_allclose(
        np.swapaxes(rotation, -1, -2) @ rotation,
        np.broadcast_to(np.eye(3), (2, 3, 3)),
        atol=1.0e-12,
    )
    np.testing.assert_allclose(np.linalg.det(rotation), np.ones(2))


def test_palm_relative_fingertips_ignore_camera_rigid_transform():
    joints = _right_hand()
    angle = 0.7
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    transformed = joints @ rotation.T + np.array([1.0, -2.0, 0.3])

    np.testing.assert_allclose(
        palm_relative_fingertips(joints),
        palm_relative_fingertips(transformed),
        atol=1.0e-12,
    )


def test_object_motion_window_keeps_pregrasp_and_postgrasp_margin():
    translation = np.zeros((30, 3))
    translation[12:, 0] = np.arange(18) * 0.03

    result = object_motion_window(
        translation,
        margin=3,
        lookahead=2,
        immediate_threshold=0.02,
        future_threshold=0.04,
    )

    assert result.start == 9
    assert result.stop == 30


def test_load_sequence_uses_one_camera_and_skips_invalid_frames(tmp_path: Path):
    capture = tmp_path / "20200709-subject-01" / "capture-a"
    camera_a = capture / "100"
    camera_b = capture / "200"
    camera_a.mkdir(parents=True)
    camera_b.mkdir()
    metadata = {
        "mano_sides": ["right"],
        "ycb_ids": [1, 5],
        "ycb_grasp_ind": 1,
    }
    with (capture / "meta.yml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(metadata, stream)

    valid = _right_hand()[None, ...]
    invalid = np.full((1, 21, 3), -1.0)
    pose_y = np.zeros((2, 3, 4))
    pose_y[1, :, 3] = [0.1, 0.2, 0.3]
    np.savez(camera_a / "labels_000001.npz", joint_3d=invalid, pose_y=pose_y)
    np.savez(camera_a / "labels_000002.npz", joint_3d=valid, pose_y=pose_y)
    np.savez(camera_b / "labels_000002.npz", joint_3d=valid + 10.0, pose_y=pose_y)

    sequence = load_sequence(capture, trim_to_object_motion=False)

    assert sequence.camera_serial == "100"
    assert sequence.ycb_grasp_id == 5
    np.testing.assert_array_equal(sequence.frame_indices, [2])
    np.testing.assert_allclose(sequence.joints_camera[0], valid[0])
    np.testing.assert_allclose(sequence.object_translation_camera[0], [0.1, 0.2, 0.3])
