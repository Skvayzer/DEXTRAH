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
    assert torch.all(precision >= hand.lower)
    assert torch.all(precision <= hand.upper)
    assert power[2:].sum() > precision[2:].sum()
    torch.testing.assert_close(
        power,
        torch.tensor([1.0, 0.75, 1.0, 1.0, 1.0, 1.0], dtype=torch.float64),
    )
    torch.testing.assert_close(
        precision[:2], torch.tensor([1.0, 0.75], dtype=torch.float64)
    )
    torch.testing.assert_close(
        precision[2:], torch.zeros(4, dtype=torch.float64)
    )


def test_fingertip_scale_matches_uniform_extent_ratio():
    robot = np.ones((5, 3)) * 2.0
    human = np.ones((3, 5, 3))

    scale = estimate_fingertip_scale(human, robot)

    assert scale == pytest.approx(2.0)


def test_fingertip_scale_is_robust_to_one_morphology_outlier():
    human = np.zeros((2, 5, 3))
    human[..., 2] = 1.0
    robot = np.zeros((5, 3))
    robot[:, 2] = [8.0, 2.0, 2.0, 2.0, 2.0]

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


def test_batched_optimizer_sequence_is_finite():
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
    assert not np.allclose(result.joint_positions[0], hand.lower.numpy())


def test_batched_optimizer_is_invariant_to_trajectory_grouping():
    hand = Revo2Kinematics(URDF)
    q = torch.stack(
        [
            hand.lower + fraction * (hand.upper - hand.lower)
            for fraction in (0.2, 0.4, 0.6, 0.3, 0.5)
        ]
    )
    targets = hand.fingertip_positions(q)
    config = RetargetingConfig(
        iterations=30,
        minimum_iterations=30,
        convergence_tolerance=0.0,
        gradient_clip_norm=1.0e-4,
    )
    retargeter = Revo2Retargeter(hand, config)

    first = retargeter.retarget(targets[:3])
    second = retargeter.retarget(targets[3:])
    combined_gamma = np.concatenate((first.gamma, second.gamma))
    combined = retargeter.retarget(targets, gamma_override=combined_gamma)

    np.testing.assert_allclose(
        combined.joint_positions,
        np.concatenate((first.joint_positions, second.joint_positions)),
        rtol=1.0e-10,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(combined.gamma, [1.0, 0.5, 0.0, 1.0, 0.0])


def test_retargeting_result_frame_slice_preserves_diagnostics():
    hand = Revo2Kinematics(URDF)
    targets = hand.fingertip_positions(
        hand.lower + 0.4 * (hand.upper - hand.lower)
    ).repeat(4, 1, 1)
    result = Revo2Retargeter(
        hand, RetargetingConfig(iterations=2, minimum_iterations=2)
    ).retarget(targets)

    sliced = result.frame_slice(1, 3)

    assert sliced.joint_positions.shape == (2, 6)
    np.testing.assert_array_equal(sliced.gamma, result.gamma[1:3])
    np.testing.assert_array_equal(
        sliced.fingertip_error, result.fingertip_error[1:3]
    )
    with pytest.raises(ValueError, match="frame slice"):
        result.frame_slice(2, 2)


def test_sequential_optimizer_remains_available_for_warm_starting():
    hand = Revo2Kinematics(URDF)
    target = hand.fingertip_positions(
        hand.lower + 0.4 * (hand.upper - hand.lower)
    ).unsqueeze(0)
    result = Revo2Retargeter(
        hand,
        RetargetingConfig(
            optimizer_execution="sequential",
            iterations=30,
            minimum_iterations=5,
        ),
    ).retarget(target)

    assert result.joint_positions.shape == (1, 6)
    assert np.isfinite(result.total_loss).all()


def test_gamma_override_supports_pure_imitation_calibration():
    hand = Revo2Kinematics(URDF)
    target = hand.fingertip_positions(
        hand.lower + 0.3 * (hand.upper - hand.lower)
    ).repeat(4, 1, 1)
    result = Revo2Retargeter(
        hand, RetargetingConfig(iterations=10, minimum_iterations=2)
    ).retarget(target, gamma_override=np.ones(4))

    np.testing.assert_array_equal(result.gamma, np.ones(4))
    with pytest.raises(ValueError, match="one value per frame"):
        Revo2Retargeter(
            hand, RetargetingConfig(iterations=1, minimum_iterations=0)
        ).retarget(target, gamma_override=np.ones(3))
