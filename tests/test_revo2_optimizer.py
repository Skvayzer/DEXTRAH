from pathlib import Path

import numpy as np
import pytest
import torch

from dextrah_lab.retargeting.revo2_kinematics import Revo2Kinematics
from dextrah_lab.retargeting.revo2_optimizer import (
    RetargetingConfig,
    Revo2Retargeter,
    estimate_fingertip_scale,
    gamma_values,
    nominal_revo2_configuration,
)


URDF = (
    Path(__file__).parents[1]
    / "dextrah_lab/assets/revo2_description/urdf/revo2_right_hand.urdf"
)


def test_gamma_schedules_cover_documented_and_literal_behavior():
    torch.testing.assert_close(
        gamma_values(3, schedule="endpoint"),
        torch.tensor([1.0, 0.5, 0.0], dtype=torch.float64),
    )
    torch.testing.assert_close(
        gamma_values(4, schedule="paper_literal"),
        torch.tensor([0.75, 0.5, 0.25, 0.0], dtype=torch.float64),
    )


def test_nominal_grasps_respect_limits_and_power_is_more_curled():
    hand = Revo2Kinematics(URDF)
    power = nominal_revo2_configuration(hand, "power")
    precision = nominal_revo2_configuration(hand, "precision")

    assert torch.all(power > hand.lower)
    assert torch.all(power < hand.upper)
    assert torch.all(precision > hand.lower)
    assert torch.all(precision < hand.upper)
    assert power[2:].sum() > precision[2:].sum()


def test_fingertip_scale_matches_rms_extent():
    robot = np.ones((5, 3)) * 2.0
    human = np.ones((3, 5, 3))

    scale = estimate_fingertip_scale(human, robot)

    assert scale == pytest.approx(2.0)


def test_adam_recovers_synthetic_revo2_fingertips_at_imitation_endpoint():
    hand = Revo2Kinematics(URDF)
    expected_q = hand.lower + 0.45 * (hand.upper - hand.lower)
    target = hand.fingertip_positions(expected_q).unsqueeze(0)
    retargeter = Revo2Retargeter(
        hand,
        RetargetingConfig(
            mode="power",
            scale=1.0,
            regularization_weight=0.0,
            learning_rate=5.0e-2,
            iterations=600,
            minimum_iterations=100,
            convergence_tolerance=1.0e-12,
        ),
    )

    result = retargeter.retarget(target, initial_position=expected_q * 0.7)

    assert result.joint_positions.shape == (1, 6)
    assert result.robot_fingertips.shape == (1, 5, 3)
    assert result.fingertip_error.max() < 2.0e-4
    assert np.all(result.joint_positions > hand.lower.numpy())
    assert np.all(result.joint_positions < hand.upper.numpy())


def test_optimizer_sequence_is_warm_started_and_finite():
    hand = Revo2Kinematics(URDF)
    targets = hand.fingertip_positions(
        torch.stack(
            (
                hand.lower + 0.2 * (hand.upper - hand.lower),
                hand.lower + 0.4 * (hand.upper - hand.lower),
                hand.lower + 0.6 * (hand.upper - hand.lower),
            )
        )
    )
    retargeter = Revo2Retargeter(
        hand,
        RetargetingConfig(iterations=20, minimum_iterations=5),
    )

    result = retargeter.retarget(targets)

    assert result.joint_positions.shape == (3, 6)
    assert np.isfinite(result.total_loss).all()
    np.testing.assert_allclose(result.gamma, [1.0, 0.5, 0.0])
