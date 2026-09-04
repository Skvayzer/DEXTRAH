# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# 
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# Imports
import torch

def assert_equals(a, b) -> None:
    # Saves space typing out the full assert and text
    assert a == b, f"{a} != {b}"

def to_torch(x, dtype=torch.float, device='cuda:0', requires_grad=False):
    return torch.tensor(x, dtype=dtype, device=device, requires_grad=requires_grad)

@torch.jit.script
def scale(x, lower, upper):
    return 0.5 * (x + 1.0) * (upper - lower) + lower


def compute_absolute_action(
    raw_actions: torch.Tensor,
    lower_limits: torch.Tensor,
    upper_limits: torch.Tensor,
) -> torch.Tensor:
    N, D = raw_actions.shape
    assert_equals(lower_limits.shape, (D,))
    assert_equals(upper_limits.shape, (D,))

    # Apply actions to hand
    absolute_action = scale(
        x=raw_actions,
        lower=lower_limits,
        upper=upper_limits,
    )
    absolute_action = tensor_clamp(
        t=absolute_action,
        min_t=lower_limits,
        max_t=upper_limits,
    )

    return absolute_action


def compute_relative_cspace_target(
    raw_actions: torch.Tensor,
    previous_fabric_position: torch.Tensor,
    lower_limits: torch.Tensor,
    upper_limits: torch.Tensor,
    max_joint_delta: float = 0.1,
) -> torch.Tensor:
    """Convert bounded policy actions into ADEPT-style joint targets.

    ADEPT commands every arm and hand joint relative to the fabric state from
    the previous policy step.  Using the fabric state instead of the measured
    robot state keeps the target continuous when simulation state is noisy.
    """
    if raw_actions.ndim != 2:
        raise ValueError(f"raw_actions must have shape (batch, joints), got {raw_actions.shape}")

    batch_size, num_joints = raw_actions.shape
    if previous_fabric_position.shape != (batch_size, num_joints):
        raise ValueError(
            "previous_fabric_position must match raw_actions; "
            f"got {previous_fabric_position.shape} and {raw_actions.shape}"
        )
    if lower_limits.shape != (num_joints,) or upper_limits.shape != (num_joints,):
        raise ValueError(
            "joint limits must each have shape (joints,); "
            f"got {lower_limits.shape} and {upper_limits.shape}"
        )
    if max_joint_delta <= 0.0:
        raise ValueError(f"max_joint_delta must be positive, got {max_joint_delta}")

    bounded_actions = torch.clamp(raw_actions, -1.0, 1.0)
    target = previous_fabric_position + max_joint_delta * bounded_actions
    return tensor_clamp(target, lower_limits, upper_limits)

@torch.jit.script
def tensor_clamp(t, min_t, max_t):
    return torch.max(torch.min(t, max_t), min_t)
