from pathlib import Path

import torch

from dextrah_lab.retargeting.revo2_kinematics import (
    REVO2_RIGHT_ACTUATED_JOINTS,
    Revo2Kinematics,
    saturate_joint_position,
    unsaturate_joint_position,
)


URDF = (
    Path(__file__).parents[1]
    / "dextrah_lab/assets/revo2_description/urdf/revo2_right_hand.urdf"
)


def test_revo2_urdf_has_six_hardware_commands_and_five_tips():
    hand = Revo2Kinematics(URDF)

    assert hand.actuated_joint_names == REVO2_RIGHT_ACTUATED_JOINTS
    assert hand.lower.tolist() == [0.0] * 6
    torch.testing.assert_close(
        hand.upper,
        torch.tensor([1.57, 1.03, 1.41, 1.41, 1.41, 1.41], dtype=torch.float64),
    )
    assert hand.fingertip_positions(torch.zeros(6, dtype=torch.float64)).shape == (5, 3)


def test_open_hand_fingertips_match_urdf_origins():
    hand = Revo2Kinematics(URDF)
    tips = hand.fingertip_positions(torch.zeros(6, dtype=torch.float64))

    # These are sums/compositions of the URDF origins at zero joint angle,
    # retained as an independent regression check on frame and chain ordering.
    expected = torch.tensor(
        [
            [0.01079818, 0.11153359, 0.03940770],
            [0.01980852, 0.03381904, 0.14578261],
            [0.01941412, 0.01171940, 0.15966979],
            [0.01851315, -0.01163653, 0.15582067],
            [0.01761257, -0.03326791, 0.13805994],
        ],
        dtype=torch.float64,
    )
    torch.testing.assert_close(tips, expected, atol=1.0e-7, rtol=1.0e-7)


def test_revo2_fk_is_batched_and_differentiable_through_mimics():
    hand = Revo2Kinematics(URDF)
    q = torch.full((3, 6), 0.3, dtype=torch.float64, requires_grad=True)

    tips = hand.fingertip_positions(q)
    tips.square().sum().backward()

    assert tips.shape == (3, 5, 3)
    assert q.grad is not None
    assert torch.isfinite(q.grad).all()
    assert torch.count_nonzero(q.grad) == q.numel()


def test_tanh_joint_saturation_round_trips_interior_positions():
    hand = Revo2Kinematics(URDF)
    position = hand.lower + 0.37 * (hand.upper - hand.lower)

    unconstrained = unsaturate_joint_position(position, hand.lower, hand.upper)
    restored = saturate_joint_position(unconstrained, hand.lower, hand.upper)

    torch.testing.assert_close(restored, position)
    at_limits = saturate_joint_position(
        unsaturate_joint_position(hand.lower, hand.lower, hand.upper),
        hand.lower,
        hand.upper,
    )
    assert torch.all(at_limits > hand.lower)
    assert torch.all(at_limits < hand.upper)
