import pytest
import torch

from dextrah_lab.tasks.dextrah_kuka_allegro.dextrah_kuka_allegro_utils import (
    compute_relative_cspace_target,
)


def test_relative_target_uses_previous_fabric_position():
    actions = torch.tensor([[1.0, -0.5, 0.25]])
    previous_fabric_position = torch.tensor([[0.2, 0.3, -0.4]])
    limits = torch.tensor([-2.0, -2.0, -2.0]), torch.tensor([2.0, 2.0, 2.0])

    target = compute_relative_cspace_target(
        actions, previous_fabric_position, *limits, max_joint_delta=0.1
    )

    torch.testing.assert_close(target, torch.tensor([[0.3, 0.25, -0.375]]))


def test_relative_target_clamps_actions_and_joint_limits():
    actions = torch.tensor([[2.0, -3.0, 1.0]])
    previous_fabric_position = torch.tensor([[0.95, -0.95, 0.0]])
    lower_limits = torch.tensor([-1.0, -1.0, -0.05])
    upper_limits = torch.tensor([1.0, 1.0, 0.05])

    target = compute_relative_cspace_target(
        actions,
        previous_fabric_position,
        lower_limits,
        upper_limits,
        max_joint_delta=0.1,
    )

    torch.testing.assert_close(target, torch.tensor([[1.0, -1.0, 0.05]]))


@pytest.mark.parametrize(
    ("actions", "previous", "lower", "upper", "max_delta"),
    [
        (torch.zeros(3), torch.zeros(1, 3), torch.zeros(3), torch.ones(3), 0.1),
        (torch.zeros(1, 3), torch.zeros(2, 3), torch.zeros(3), torch.ones(3), 0.1),
        (torch.zeros(1, 3), torch.zeros(1, 3), torch.zeros(2), torch.ones(3), 0.1),
        (torch.zeros(1, 3), torch.zeros(1, 3), torch.zeros(3), torch.ones(3), 0.0),
    ],
)
def test_relative_target_rejects_invalid_inputs(actions, previous, lower, upper, max_delta):
    with pytest.raises(ValueError):
        compute_relative_cspace_target(actions, previous, lower, upper, max_delta)
