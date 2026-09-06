import torch

from dextrah_lab.g1_adept import (
    CollisionBatch,
    ReducedAdeptFabric,
    ReducedAdeptFabricConfig,
)


def _fabric(num_dof: int = 13, batch: int = 2):
    lower = -torch.ones(num_dof)
    upper = torch.ones(num_dof)
    fabric = ReducedAdeptFabric(lower, upper)
    fabric.reset(torch.zeros(batch, num_dof))
    return fabric


def test_target_attractor_moves_toward_target():
    fabric = _fabric()
    target = torch.full((2, 13), 0.5)

    state = fabric.step(target)

    assert torch.all(state.position > 0.0)
    assert torch.all(state.velocity > 0.0)
    assert torch.all(state.acceleration > 0.0)


def test_joint_limit_term_pushes_away_from_upper_limit():
    cfg = ReducedAdeptFabricConfig(
        target_gain_arm=0.0,
        target_gain_hand=0.0,
        target_damping_arm=0.0,
        target_damping_hand=0.0,
        cspace_damping=0.0,
    )
    lower = -torch.ones(13)
    upper = torch.ones(13)
    fabric = ReducedAdeptFabric(lower, upper, config=cfg)
    initial = torch.zeros(1, 13)
    initial[:, 0] = 0.999
    fabric.reset(initial)

    state = fabric.step(initial)

    assert state.acceleration[0, 0] < 0.0
    assert state.position[0, 0] < initial[0, 0]


def test_collision_clearance_pushes_along_positive_gradient():
    fabric = _fabric(batch=1)
    target = torch.zeros(1, 13)
    clearance = torch.tensor([[0.001]])
    jacobian = torch.zeros(1, 1, 13)
    jacobian[..., 2] = 1.0

    state = fabric.step(target, CollisionBatch(clearance, jacobian))

    assert state.acceleration[0, 2] > 0.0
    assert state.position[0, 2] > 0.0


def test_far_collision_constraint_is_inactive():
    fabric = _fabric(batch=1)
    target = torch.zeros(1, 13)
    clearance = torch.tensor([[1.0]])
    jacobian = torch.randn(1, 1, 13)

    state = fabric.step(target, CollisionBatch(clearance, jacobian))

    torch.testing.assert_close(state.position, torch.zeros_like(state.position))


def test_acceleration_and_jerk_are_bounded():
    fabric = _fabric(batch=1)
    state = fabric.step(torch.ones(1, 13))
    dt = fabric.config.timestep
    arm_jerk_step = fabric.config.max_arm_jerk * dt
    hand_jerk_step = fabric.config.max_hand_jerk * dt

    assert state.acceleration[0, :7].abs().max() <= arm_jerk_step + 1.0e-6
    assert state.acceleration[0, 7:].abs().max() <= hand_jerk_step + 1.0e-6
    assert state.acceleration[0, :7].abs().max() <= fabric.config.max_arm_acceleration
    assert state.acceleration[0, 7:].abs().max() <= fabric.config.max_hand_acceleration


def test_selected_reset_only_changes_requested_environments():
    fabric = _fabric(batch=3)
    fabric.step(torch.full((3, 13), 0.5))
    before = fabric.state.clone()
    env_ids = torch.tensor([1])
    fabric.reset(torch.full((1, 13), -0.25), env_ids)

    torch.testing.assert_close(fabric.state.position[0], before.position[0])
    torch.testing.assert_close(fabric.state.position[2], before.position[2])
    torch.testing.assert_close(fabric.state.position[1], torch.full((13,), -0.25))
    torch.testing.assert_close(fabric.state.velocity[1], torch.zeros(13))


def test_collision_shape_validation():
    fabric = _fabric(batch=2)
    collision = CollisionBatch(
        clearance=torch.zeros(2, 3),
        jacobian=torch.zeros(2, 3, 12),
    )

    try:
        fabric.step(torch.zeros(2, 13), collision)
    except ValueError as exc:
        assert "jacobian" in str(exc)
    else:
        raise AssertionError("invalid collision shape was accepted")
