"""DextrAH-G Appendix-D human-motion retargeting adapted to Revo2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch

from .revo2_kinematics import (
    Revo2Kinematics,
    saturate_joint_position,
    unsaturate_joint_position,
)


GraspMode = Literal["power", "precision", "precision_tripod"]
GammaSchedule = Literal["endpoint", "paper_literal"]
OptimizerExecution = Literal["batched", "sequential"]


@dataclass(frozen=True)
class RetargetingConfig:
    """Explicit values for details not reported in DextrAH-G Appendix D."""

    mode: GraspMode = "power"
    scale: float = 1.0
    regularization_weight: float = 1.0e-3
    learning_rate: float = 3.0e-2
    iterations: int = 250
    minimum_iterations: int = 40
    convergence_tolerance: float = 1.0e-9
    gamma_schedule: GammaSchedule = "endpoint"
    gradient_clip_norm: float = 10.0
    optimizer_execution: OptimizerExecution = "batched"

    def validate(self) -> None:
        if self.mode not in {"power", "precision", "precision_tripod"}:
            raise ValueError(f"unsupported grasp mode {self.mode!r}")
        if self.scale <= 0.0:
            raise ValueError("scale must be positive")
        if self.regularization_weight < 0.0:
            raise ValueError("regularization_weight cannot be negative")
        if self.learning_rate <= 0.0 or self.iterations <= 0:
            raise ValueError("learning rate and iterations must be positive")
        if not 0 <= self.minimum_iterations <= self.iterations:
            raise ValueError("minimum_iterations must lie in [0, iterations]")
        if self.convergence_tolerance < 0.0 or self.gradient_clip_norm <= 0.0:
            raise ValueError("optimizer tolerances must be non-negative/positive")
        if self.gamma_schedule not in {"endpoint", "paper_literal"}:
            raise ValueError(f"unsupported gamma schedule {self.gamma_schedule!r}")
        if self.optimizer_execution not in {"batched", "sequential"}:
            raise ValueError(
                f"unsupported optimizer execution {self.optimizer_execution!r}"
            )


@dataclass(frozen=True)
class RetargetingResult:
    joint_positions: np.ndarray
    robot_fingertips: np.ndarray
    scaled_human_fingertips: np.ndarray
    gamma: np.ndarray
    total_loss: np.ndarray
    imitation_loss: np.ndarray
    closure_loss: np.ndarray
    regularization_loss: np.ndarray
    optimizer_iterations: np.ndarray
    converged: np.ndarray

    @property
    def fingertip_error(self) -> np.ndarray:
        return np.linalg.norm(
            self.robot_fingertips - self.scaled_human_fingertips, axis=-1
        )


def gamma_values(
    num_frames: int,
    *,
    schedule: GammaSchedule = "endpoint",
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Return Appendix-D human-to-closure blending weights.

    The prose says the first frame has all weight on imitation and the final
    frame has all weight on closure.  ``endpoint`` implements those endpoints.
    ``paper_literal`` preserves the printed ``1 - (i + 1) / n`` equation for
    reproducibility of the paper's small off-by-one ambiguity.
    """

    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    if schedule == "endpoint":
        if num_frames == 1:
            return torch.ones(1, dtype=dtype, device=device)
        return torch.linspace(1.0, 0.0, num_frames, dtype=dtype, device=device)
    if schedule == "paper_literal":
        index = torch.arange(num_frames, dtype=dtype, device=device)
        return 1.0 - (index + 1.0) / num_frames
    raise ValueError(f"unsupported gamma schedule {schedule!r}")


def nominal_revo2_configuration(
    hand: Revo2Kinematics, mode: GraspMode
) -> torch.Tensor:
    """Return a morphology-aware regularization posture inside URDF limits."""

    if mode == "power":
        fraction = torch.tensor(
            [0.82, 0.82, 0.82, 0.86, 0.90, 0.94], dtype=hand.lower.dtype
        )
    elif mode == "precision":
        fraction = torch.tensor(
            [0.82, 0.50, 0.48, 0.22, 0.08, 0.04], dtype=hand.lower.dtype
        )
    elif mode == "precision_tripod":
        fraction = torch.tensor(
            [0.82, 0.50, 0.48, 0.38, 0.08, 0.04], dtype=hand.lower.dtype
        )
    else:
        raise ValueError(f"unsupported grasp mode {mode!r}")
    return hand.lower + fraction.to(hand.lower.device) * (hand.upper - hand.lower)


def estimate_fingertip_scale(
    human_fingertips: np.ndarray | torch.Tensor,
    robot_reference_fingertips: np.ndarray | torch.Tensor,
) -> float:
    """Estimate one robust scale from matching wrist-to-tip lengths.

    Taking the median of per-finger ratios avoids letting the Revo2 thumb's
    large lateral offset dominate the morphology calibration.
    """

    human = torch.as_tensor(human_fingertips, dtype=torch.float64)
    robot = torch.as_tensor(robot_reference_fingertips, dtype=torch.float64)
    if human.shape[-2:] != (5, 3) or robot.shape[-2:] != (5, 3):
        raise ValueError("human and robot fingertips must end in shape (5, 3)")
    human_extent = torch.linalg.vector_norm(human, dim=-1)
    robot_extent = torch.linalg.vector_norm(robot, dim=-1)
    if torch.any(human_extent <= 1.0e-8) or torch.any(robot_extent <= 1.0e-8):
        raise ValueError("cannot estimate scale from degenerate fingertips")
    ratios = robot_extent / human_extent
    return float(torch.median(ratios).item())


class Revo2Retargeter:
    """Per-frame Adam retargeter with no PPO dependency."""

    def __init__(self, hand: Revo2Kinematics, config: RetargetingConfig) -> None:
        config.validate()
        self.hand = hand
        self.config = config

    def _closure_target_and_mask(
        self, q_regularization: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        regularized_tips = self.hand.fingertip_positions(q_regularization)
        if self.config.mode == "power":
            # A single palm-side point, as specified by Appendix D.  Deriving
            # it from the nominal grasp keeps the adaptation tied to the URDF.
            point = regularized_tips.mean(dim=-2)
            point = point.clone()
            point[2] *= 0.70
            mask = torch.ones(5, dtype=torch.bool, device=point.device)
        else:
            # The precision point lies at the center of the thumb/index pair.
            point = regularized_tips[:2].mean(dim=-2)
            if self.config.mode == "precision_tripod":
                mask = torch.tensor(
                    [True, True, True, False, False], device=point.device
                )
            else:
                mask = torch.ones(5, dtype=torch.bool, device=point.device)
        return point.expand(5, 3), mask

    def retarget(
        self,
        human_fingertips: np.ndarray | torch.Tensor,
        *,
        initial_position: np.ndarray | torch.Tensor | None = None,
    ) -> RetargetingResult:
        """Retarget one complete palm-relative human grasp trajectory."""

        human = torch.as_tensor(
            human_fingertips,
            dtype=self.hand.lower.dtype,
            device=self.hand.lower.device,
        )
        if human.ndim != 3 or human.shape[1:] != (5, 3):
            raise ValueError("human_fingertips must have shape (frames, 5, 3)")
        if not torch.isfinite(human).all():
            raise ValueError("human_fingertips contains non-finite values")

        human = human * self.config.scale
        gamma = gamma_values(
            len(human),
            schedule=self.config.gamma_schedule,
            dtype=human.dtype,
            device=human.device,
        )
        q_regularization = nominal_revo2_configuration(
            self.hand, self.config.mode
        ).to(human.device)
        closure_target, closure_mask = self._closure_target_and_mask(q_regularization)

        if initial_position is None:
            # Do not initialize exactly on a tanh-saturated joint limit: its
            # near-zero derivative can trap Adam at the open-hand boundary.
            lower = self.hand.lower.to(human.device)
            upper = self.hand.upper.to(human.device)
            previous = lower + 0.05 * (upper - lower)
        else:
            previous = torch.as_tensor(
                initial_position, dtype=human.dtype, device=human.device
            ).clone()
            if previous.shape != self.hand.lower.shape:
                raise ValueError("initial_position must contain six Revo2 commands")
            previous = torch.clamp(
                previous,
                self.hand.lower.to(human.device),
                self.hand.upper.to(human.device),
            )

        if self.config.optimizer_execution == "batched":
            return self._retarget_batched(
                human,
                gamma,
                q_regularization,
                closure_target,
                closure_mask,
                previous,
            )

        q_frames: list[torch.Tensor] = []
        tip_frames: list[torch.Tensor] = []
        total_losses: list[float] = []
        imitation_losses: list[float] = []
        closure_losses: list[float] = []
        regularization_losses: list[float] = []
        iteration_counts: list[int] = []
        converged_flags: list[bool] = []

        lower = self.hand.lower.to(human.device)
        upper = self.hand.upper.to(human.device)
        for frame_index in range(len(human)):
            unconstrained = torch.nn.Parameter(
                unsaturate_joint_position(previous, lower, upper).detach()
            )
            optimizer = torch.optim.Adam(
                [unconstrained], lr=self.config.learning_rate
            )
            old_loss: float | None = None
            converged = False
            completed_iterations = 0
            for iteration in range(self.config.iterations):
                optimizer.zero_grad(set_to_none=True)
                q = saturate_joint_position(unconstrained, lower, upper)
                robot_tips = self.hand.fingertip_positions(q)
                imitation = (robot_tips - human[frame_index]).square().sum()
                closure = (
                    robot_tips[closure_mask] - closure_target[closure_mask]
                ).square().sum()
                regularization = torch.linalg.vector_norm(q - q_regularization)
                weight = gamma[frame_index]
                total = (
                    weight * imitation
                    + (1.0 - weight) * closure
                    + self.config.regularization_weight * regularization
                )
                total.backward()
                torch.nn.utils.clip_grad_norm_(
                    [unconstrained], self.config.gradient_clip_norm
                )
                optimizer.step()
                completed_iterations = iteration + 1
                current_loss = float(total.detach().item())
                if (
                    old_loss is not None
                    and completed_iterations >= self.config.minimum_iterations
                    and abs(old_loss - current_loss) <= self.config.convergence_tolerance
                ):
                    converged = True
                    break
                old_loss = current_loss

            with torch.no_grad():
                previous = saturate_joint_position(unconstrained, lower, upper)
                tips = self.hand.fingertip_positions(previous)
                imitation = (tips - human[frame_index]).square().sum()
                closure = (
                    tips[closure_mask] - closure_target[closure_mask]
                ).square().sum()
                regularization = torch.linalg.vector_norm(
                    previous - q_regularization
                )
                weight = gamma[frame_index]
                total = (
                    weight * imitation
                    + (1.0 - weight) * closure
                    + self.config.regularization_weight * regularization
                )
            q_frames.append(previous.detach().cpu())
            tip_frames.append(tips.detach().cpu())
            total_losses.append(float(total.item()))
            imitation_losses.append(float(imitation.item()))
            closure_losses.append(float(closure.item()))
            regularization_losses.append(float(regularization.item()))
            iteration_counts.append(completed_iterations)
            converged_flags.append(converged)

        return RetargetingResult(
            joint_positions=torch.stack(q_frames).numpy(),
            robot_fingertips=torch.stack(tip_frames).numpy(),
            scaled_human_fingertips=human.detach().cpu().numpy(),
            gamma=gamma.detach().cpu().numpy(),
            total_loss=np.asarray(total_losses),
            imitation_loss=np.asarray(imitation_losses),
            closure_loss=np.asarray(closure_losses),
            regularization_loss=np.asarray(regularization_losses),
            optimizer_iterations=np.asarray(iteration_counts, dtype=np.int64),
            converged=np.asarray(converged_flags, dtype=bool),
        )

    def _retarget_batched(
        self,
        human: torch.Tensor,
        gamma: torch.Tensor,
        q_regularization: torch.Tensor,
        closure_target: torch.Tensor,
        closure_mask: torch.Tensor,
        initial_position: torch.Tensor,
    ) -> RetargetingResult:
        """Optimize every frame independently in one vectorized Adam solve."""

        lower = self.hand.lower.to(human.device)
        upper = self.hand.upper.to(human.device)
        initial = initial_position.expand(len(human), -1)
        unconstrained = torch.nn.Parameter(
            unsaturate_joint_position(initial, lower, upper).detach().clone()
        )
        optimizer = torch.optim.Adam([unconstrained], lr=self.config.learning_rate)
        old_loss: torch.Tensor | None = None
        converged = torch.zeros(len(human), dtype=torch.bool, device=human.device)
        completed_iterations = 0

        for iteration in range(self.config.iterations):
            optimizer.zero_grad(set_to_none=True)
            q = saturate_joint_position(unconstrained, lower, upper)
            robot_tips = self.hand.fingertip_positions(q)
            imitation = (robot_tips - human).square().sum(dim=(-2, -1))
            closure = (
                robot_tips[:, closure_mask] - closure_target[closure_mask]
            ).square().sum(dim=(-2, -1))
            regularization = torch.linalg.vector_norm(
                q - q_regularization, dim=-1
            )
            total = (
                gamma * imitation
                + (1.0 - gamma) * closure
                + self.config.regularization_weight * regularization
            )
            total.sum().backward()
            torch.nn.utils.clip_grad_norm_(
                [unconstrained], self.config.gradient_clip_norm
            )
            optimizer.step()
            completed_iterations = iteration + 1
            if (
                old_loss is not None
                and completed_iterations >= self.config.minimum_iterations
            ):
                converged = torch.abs(old_loss - total.detach()) <= (
                    self.config.convergence_tolerance
                )
                if bool(torch.all(converged)):
                    break
            old_loss = total.detach().clone()

        with torch.no_grad():
            q = saturate_joint_position(unconstrained, lower, upper)
            robot_tips = self.hand.fingertip_positions(q)
            imitation = (robot_tips - human).square().sum(dim=(-2, -1))
            closure = (
                robot_tips[:, closure_mask] - closure_target[closure_mask]
            ).square().sum(dim=(-2, -1))
            regularization = torch.linalg.vector_norm(
                q - q_regularization, dim=-1
            )
            total = (
                gamma * imitation
                + (1.0 - gamma) * closure
                + self.config.regularization_weight * regularization
            )

        return RetargetingResult(
            joint_positions=q.cpu().numpy(),
            robot_fingertips=robot_tips.cpu().numpy(),
            scaled_human_fingertips=human.cpu().numpy(),
            gamma=gamma.cpu().numpy(),
            total_loss=total.cpu().numpy(),
            imitation_loss=imitation.cpu().numpy(),
            closure_loss=closure.cpu().numpy(),
            regularization_loss=regularization.cpu().numpy(),
            optimizer_iterations=np.full(
                len(human), completed_iterations, dtype=np.int64
            ),
            converged=converged.cpu().numpy(),
        )
