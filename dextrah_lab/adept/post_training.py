"""ADEPT Appendix D post-training schedule and Appendix G losses."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping

import torch


class PostTrainingPhase(str, Enum):
    ACTOR_BC = "actor_bc"
    CRITIC_WARMUP = "critic_warmup"
    PPO = "ppo"
    COMPLETE = "complete"


@dataclass(frozen=True)
class PostTrainingConfig:
    bc_iterations: int = 40_000
    bc_actor_learning_rate: float = 1e-4
    bc_adr_level: int = 20
    mixed_policy_period: int = 2
    critic_warmup_epochs: int = 20
    fixed_actor_log_std: float = -2.0
    actor_learning_rate: float = 1e-5
    critic_learning_rate: float = 5e-5
    max_ppo_epochs: int = 200_000
    ppo_clip: float = 0.05
    observation_clip: float = 100.0
    goal_adr_level: int = 50
    goal_path_during_bc: bool = True
    goal_path_after_bc: bool = False
    external_object_wrenches: bool = False
    object_orientation_range: tuple[float, float] = (1.0, 1.0)
    object_goal_sharpness: float = 15.0
    peg_board_contact_penalty: float = 0.0
    receptacle_contact_penalty: float = 0.0

    def phase(self, bc_iteration: int, post_bc_epoch: int) -> PostTrainingPhase:
        if bc_iteration < self.bc_iterations:
            return PostTrainingPhase.ACTOR_BC
        if post_bc_epoch < self.critic_warmup_epochs:
            return PostTrainingPhase.CRITIC_WARMUP
        if post_bc_epoch < self.max_ppo_epochs:
            return PostTrainingPhase.PPO
        return PostTrainingPhase.COMPLETE

    def use_teacher_action(self, rollout_step: int) -> bool:
        """Deterministically alternate teacher/student actions during BC."""

        return rollout_step % self.mixed_policy_period == 0


def mahalanobis_distribution_loss(
    student_mean: torch.Tensor,
    student_std: torch.Tensor,
    teacher_mean: torch.Tensor,
    teacher_std: torch.Tensor,
    *,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Per-sample ADEPT behavior-cloning loss from Eq. (29)."""

    if not (
        student_mean.shape
        == student_std.shape
        == teacher_mean.shape
        == teacher_std.shape
    ):
        raise ValueError("all action-distribution tensors must have the same shape")
    precision_scale = teacher_std.square().clamp_min(epsilon)
    mean_term = ((student_mean - teacher_mean).square() / precision_scale).sum(-1).sqrt()
    std_term = ((student_std - teacher_std).square() / precision_scale).sum(-1).sqrt()
    return mean_term + std_term


def keypoint_pose_loss(
    predicted_keypoints: torch.Tensor,
    target_keypoints: torch.Tensor,
) -> torch.Tensor:
    """Per-sample eight-keypoint auxiliary loss from Eq. (30)."""

    if predicted_keypoints.shape != target_keypoints.shape:
        raise ValueError("predicted and target keypoints must have the same shape")
    if predicted_keypoints.shape[-2:] != (8, 3):
        raise ValueError("keypoints must have trailing shape (8, 3)")
    return (predicted_keypoints - target_keypoints).square().sum(dim=(-2, -1)).sqrt()


def soft_height_mask(
    object_height: torch.Tensor,
    *,
    threshold: float = 0.08,
    softness: float = 0.02,
    floor: float = 0.1,
) -> torch.Tensor:
    """Appendix H mask: retain a nonzero floor below table-clearance height."""

    if softness <= 0.0:
        raise ValueError("softness must be positive")
    if not 0.0 <= floor <= 1.0:
        raise ValueError("floor must lie in [0, 1]")
    weight = torch.sigmoid((object_height - threshold) / softness)
    return floor + (1.0 - floor) * weight


def distillation_loss(
    student_mean: torch.Tensor,
    student_std: torch.Tensor,
    teacher_mean: torch.Tensor,
    teacher_std: torch.Tensor,
    predicted_keypoints: torch.Tensor,
    target_keypoints: torch.Tensor,
    *,
    object_height: torch.Tensor | None = None,
    bc_weight: float = 1.0,
    auxiliary_weight: float = 20.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    bc = mahalanobis_distribution_loss(
        student_mean, student_std, teacher_mean, teacher_std
    )
    auxiliary = keypoint_pose_loss(predicted_keypoints, target_keypoints)
    mask = torch.ones_like(bc) if object_height is None else soft_height_mask(object_height)
    total = (mask * (bc_weight * bc + auxiliary_weight * auxiliary)).mean()
    return total, {
        "bc": bc.mean(),
        "auxiliary": auxiliary.mean(),
        "height_mask": mask.mean(),
    }


def set_trainable(module: torch.nn.Module, trainable: bool) -> None:
    module.train(trainable)
    for parameter in module.parameters():
        parameter.requires_grad_(trainable)


def transfer_input_columns(
    source_weight: torch.Tensor,
    target_weight: torch.Tensor,
    source_fields: Mapping[str, slice],
    target_fields: Mapping[str, slice],
) -> torch.Tensor:
    """Warm-start overlapping observation columns without assuming append-only obs.

    ADEPT's Stage-1 and Stage-2 observations overlap but neither contains the
    other. New downstream columns retain their target initialization.
    """

    if source_weight.ndim != 2 or target_weight.ndim != 2:
        raise ValueError("input weights must be matrices")
    if source_weight.shape[0] != target_weight.shape[0]:
        raise ValueError("source and target layers must have the same output width")
    result = target_weight.clone()
    for field in source_fields.keys() & target_fields.keys():
        source_slice = source_fields[field]
        target_slice = target_fields[field]
        source_width = source_slice.stop - source_slice.start
        target_width = target_slice.stop - target_slice.start
        if source_width != target_width:
            raise ValueError(f"field {field!r} changed width")
        result[:, target_slice] = source_weight[:, source_slice]
    return result


STAGE1_ACTOR_FIELDS = {
    "robot_dof_pos": slice(0, 23),
    "robot_dof_vel": slice(23, 46),
    "hand_pos": slice(46, 61),
    "hand_vel": slice(61, 76),
    "actions": slice(76, 99),
    "fabric": slice(99, 168),
    "fingertip_contacts": slice(168, 183),
    "object_pos": slice(183, 186),
    "object_rot": slice(186, 190),
    "object_goal": slice(190, 193),
    "object_goal_quat": slice(193, 197),
    "multi_object_idx": slice(197, 198),
    "object_scale": slice(198, 199),
    "pointcloud": slice(199, 391),
}

STAGE2_ACTOR_FIELDS = {
    "robot_dof_pos": slice(0, 23),
    "robot_dof_vel": slice(23, 46),
    "hand_pos": slice(46, 61),
    "hand_vel": slice(61, 76),
    "actions": slice(76, 99),
    "fabric": slice(99, 168),
    "fingertip_contacts": slice(168, 183),
    "object_goal": slice(183, 186),
    "object_goal_quat": slice(186, 190),
    "pointcloud": slice(190, 382),
    "receptacle_pose": slice(382, 389),
    "object_receptacle_contact": slice(389, 392),
}

