from pathlib import Path

import numpy as np
import torch

from dextrah_lab.retargeting.pca import fit_pca_action_space
from dextrah_lab.retargeting.revo2_kinematics import Revo2Kinematics
from dextrah_lab.retargeting.validation import validate_trajectory_arrays


URDF = (
    Path(__file__).parents[1]
    / "dextrah_lab/assets/revo2_description/urdf/revo2_right_hand.urdf"
)


def _valid_data():
    hand = Revo2Kinematics(URDF)
    fractions = torch.linspace(0.1, 0.9, 20)[:, None]
    q = hand.lower + fractions * (hand.upper - hand.lower)
    tips = hand.fingertip_positions(q).numpy()
    artifact = fit_pca_action_space(
        q.numpy(),
        joint_names=hand.actuated_joint_names,
        joint_lower=hand.lower.numpy(),
        joint_upper=hand.upper.numpy(),
        components=5,
    )
    arrays = {
        "joint_positions": q.numpy(),
        "robot_fingertips": tips,
        "scaled_human_fingertips": tips.copy(),
        "gamma": np.linspace(1.0, 0.0, len(q)),
        "fingertip_error": np.zeros((len(q), 5)),
    }
    return hand, artifact, arrays


def test_validation_accepts_consistent_trajectory() -> None:
    hand, artifact, arrays = _valid_data()

    result = validate_trajectory_arrays(arrays, hand, artifact)

    assert result.passed
    assert result.metrics["max_fk_difference_m"] < 1.0e-12


def test_validation_detects_fk_and_limit_corruption() -> None:
    hand, artifact, arrays = _valid_data()
    arrays["robot_fingertips"] = arrays["robot_fingertips"].copy()
    arrays["robot_fingertips"][0, 0, 0] += 0.01
    arrays["joint_positions"] = arrays["joint_positions"].copy()
    arrays["joint_positions"][1, 0] = artifact.joint_upper[0] + 0.1

    result = validate_trajectory_arrays(arrays, hand, artifact)

    assert not result.passed
    assert any("joint-limit" in failure for failure in result.failures)
    assert any("FK difference" in failure for failure in result.failures)
