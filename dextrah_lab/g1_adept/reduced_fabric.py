"""Block-sparse ADEPT-style fabric for the 13-D G1/Revo2 action space.

The public NVIDIA FABRICS implementation materializes dense task-space
collision metrics.  That is acceptable for ADEPT's smaller batches, but it is
unnecessarily expensive at the 24,576 environments used by SAPG.  This module
pulls scalar clearance metrics directly into configuration space, one
constraint at a time, so memory scales as ``O(batch * constraints * dof)``
instead of ``O(batch * (3 * spheres) ** 2)``.

It deliberately owns only controller math.  Isaac/PhysX supplies current link
poses and Jacobians through the runtime adapter, while PhysX remains
responsible for hand-object and object-table contact dynamics.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from dextrah_lab.adept.fabric_math import (
    joint_limit_metric_diagonal,
    normalized_joint_clearance,
)


@dataclass(frozen=True)
class ReducedAdeptFabricConfig:
    """Numerical controller parameters for right arm (7) + Revo2 (6)."""

    timestep: float = 1.0 / 60.0
    arm_dof: int = 7
    target_gain_arm: float = 80.0
    target_gain_hand: float = 120.0
    target_damping_arm: float = 18.0
    target_damping_hand: float = 12.0
    target_metric_arm: float = 1.0
    target_metric_hand: float = 0.35
    cspace_damping: float = 1.0

    joint_limit_metric_scalar: float = 0.02
    joint_limit_metric_exploder_offset: float = 0.02
    joint_limit_max_metric: float = 20.0
    joint_limit_gate_sharpness: float = 10.0
    joint_limit_gate_offset: float = 0.02
    joint_limit_acceleration: float = 20.0
    joint_limit_damping: float = 4.0

    collision_influence_distance: float = 0.08
    collision_minimum_distance: float = 0.002
    collision_metric_scalar: float = 0.02
    collision_metric_budget: float = 8.0
    collision_acceleration: float = 35.0
    collision_damping: float = 8.0

    max_arm_velocity: float = 2.0
    max_hand_velocity: float = 4.0
    max_arm_acceleration: float = 10.0
    max_hand_acceleration: float = 25.0
    max_arm_jerk: float = 400.0
    max_hand_jerk: float = 1000.0
    speed_energy_arm_weight: float = 0.75
    speed_energy_hand_weight: float = 0.25
    speed_energy_target: float = 1.0
    solve_regularization: float = 1.0e-4

    def validate(self, num_dof: int) -> None:
        if num_dof <= self.arm_dof:
            raise ValueError(
                f"num_dof must exceed arm_dof={self.arm_dof}, got {num_dof}"
            )
        positive = {
            "timestep": self.timestep,
            "collision_influence_distance": self.collision_influence_distance,
            "collision_minimum_distance": self.collision_minimum_distance,
            "collision_metric_budget": self.collision_metric_budget,
            "speed_energy_target": self.speed_energy_target,
            "solve_regularization": self.solve_regularization,
        }
        invalid = {name: value for name, value in positive.items() if value <= 0.0}
        if invalid:
            raise ValueError(f"fabric parameters must be positive: {invalid}")


@dataclass
class ReducedAdeptFabricState:
    position: torch.Tensor
    velocity: torch.Tensor
    acceleration: torch.Tensor

    def clone(self) -> "ReducedAdeptFabricState":
        return ReducedAdeptFabricState(
            self.position.clone(), self.velocity.clone(), self.acceleration.clone()
        )


@dataclass(frozen=True)
class CollisionBatch:
    """Signed clearances and their configuration-space Jacobians.

    ``clearance`` is positive when separated. ``jacobian`` is ``d(clearance)/dq``.
    Constraints farther than the configured influence distance have zero weight.
    """

    clearance: torch.Tensor
    jacobian: torch.Tensor
    enabled: torch.Tensor | None = None

    def validate(self, batch_size: int, num_dof: int) -> None:
        if self.clearance.ndim != 2:
            raise ValueError(
                f"clearance must have shape (batch, constraints), got {self.clearance.shape}"
            )
        expected = (*self.clearance.shape, num_dof)
        if self.jacobian.shape != expected:
            raise ValueError(
                f"jacobian must have shape {expected}, got {self.jacobian.shape}"
            )
        if self.clearance.shape[0] != batch_size:
            raise ValueError(
                f"collision batch has {self.clearance.shape[0]} environments, expected {batch_size}"
            )
        if self.enabled is not None and self.enabled.shape != self.clearance.shape:
            raise ValueError(
                f"enabled must have shape {self.clearance.shape}, got {self.enabled.shape}"
            )


class ReducedAdeptFabric:
    """Batched configuration-space attractor with ADEPT safety terms."""

    def __init__(
        self,
        lower_limits: torch.Tensor,
        upper_limits: torch.Tensor,
        *,
        config: ReducedAdeptFabricConfig | None = None,
    ) -> None:
        if lower_limits.ndim != 1 or upper_limits.shape != lower_limits.shape:
            raise ValueError("joint limits must be same-size one-dimensional tensors")
        if not torch.all(upper_limits > lower_limits):
            raise ValueError("every upper joint limit must exceed its lower limit")
        self.lower_limits = lower_limits
        self.upper_limits = upper_limits
        self.num_dof = int(lower_limits.numel())
        self.config = config or ReducedAdeptFabricConfig()
        self.config.validate(self.num_dof)
        self.state: ReducedAdeptFabricState | None = None

        dtype = lower_limits.dtype
        device = lower_limits.device
        arm = self.config.arm_dof
        hand = self.num_dof - arm
        self._target_gain = torch.tensor(
            [self.config.target_gain_arm] * arm
            + [self.config.target_gain_hand] * hand,
            dtype=dtype,
            device=device,
        )
        self._target_damping = torch.tensor(
            [self.config.target_damping_arm] * arm
            + [self.config.target_damping_hand] * hand,
            dtype=dtype,
            device=device,
        )
        self._target_metric = torch.tensor(
            [self.config.target_metric_arm] * arm
            + [self.config.target_metric_hand] * hand,
            dtype=dtype,
            device=device,
        )
        self._max_velocity = torch.tensor(
            [self.config.max_arm_velocity] * arm
            + [self.config.max_hand_velocity] * hand,
            dtype=dtype,
            device=device,
        )
        self._max_acceleration = torch.tensor(
            [self.config.max_arm_acceleration] * arm
            + [self.config.max_hand_acceleration] * hand,
            dtype=dtype,
            device=device,
        )
        self._max_jerk = torch.tensor(
            [self.config.max_arm_jerk] * arm
            + [self.config.max_hand_jerk] * hand,
            dtype=dtype,
            device=device,
        )
        self._speed_weights = torch.tensor(
            [self.config.speed_energy_arm_weight / arm] * arm
            + [self.config.speed_energy_hand_weight / hand] * hand,
            dtype=dtype,
            device=device,
        )

    @property
    def batch_size(self) -> int:
        if self.state is None:
            raise RuntimeError("fabric state has not been reset")
        return int(self.state.position.shape[0])

    def reset(self, position: torch.Tensor, env_ids: torch.Tensor | None = None) -> None:
        """Synchronize all or selected internal states with measured joints."""

        if position.ndim != 2 or position.shape[1] != self.num_dof:
            raise ValueError(
                f"position must have shape (batch, {self.num_dof}), got {position.shape}"
            )
        bounded = torch.clamp(position, self.lower_limits, self.upper_limits)
        if env_ids is None or self.state is None:
            self.state = ReducedAdeptFabricState(
                bounded.clone(), torch.zeros_like(bounded), torch.zeros_like(bounded)
            )
            return
        if bounded.shape[0] != env_ids.numel():
            raise ValueError("selected reset positions must match env_ids")
        self.state.position[env_ids] = bounded
        self.state.velocity[env_ids] = 0.0
        self.state.acceleration[env_ids] = 0.0

    def _joint_limit_terms(
        self, position: torch.Tensor, velocity: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.config
        upper_clearance, lower_clearance = normalized_joint_clearance(
            position, self.lower_limits, self.upper_limits
        )
        inv_range = (self.upper_limits - self.lower_limits).reciprocal()
        metric_total = torch.zeros_like(position)
        rhs_total = torch.zeros_like(position)

        for clearance, jacobian_diag in (
            (upper_clearance, -inv_range),
            (lower_clearance, inv_range),
        ):
            clearance_velocity = velocity * jacobian_diag
            metric, gate = joint_limit_metric_diagonal(
                clearance,
                clearance_velocity,
                metric_scalar=cfg.joint_limit_metric_scalar,
                metric_exploder_offset=cfg.joint_limit_metric_exploder_offset,
                max_metric=cfg.joint_limit_max_metric,
                gate_sharpness=cfg.joint_limit_gate_sharpness,
                gate_offset=cfg.joint_limit_gate_offset,
            )
            desired_clearance_acceleration = (
                cfg.joint_limit_acceleration * gate
                - cfg.joint_limit_damping * torch.minimum(
                    clearance_velocity, torch.zeros_like(clearance_velocity)
                )
            )
            metric_total += metric * jacobian_diag.square()
            rhs_total += (
                metric
                * jacobian_diag
                * desired_clearance_acceleration
            )
        return metric_total, rhs_total

    def _collision_terms(
        self, collision: CollisionBatch, velocity: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        collision.validate(velocity.shape[0], self.num_dof)
        cfg = self.config
        active = collision.clearance < cfg.collision_influence_distance
        if collision.enabled is not None:
            active &= collision.enabled.bool()

        distance = torch.clamp(
            collision.clearance, min=cfg.collision_minimum_distance
        )
        proximity = torch.clamp(
            1.0 - distance / cfg.collision_influence_distance, min=0.0
        )
        raw_weight = cfg.collision_metric_scalar * proximity.square() / distance.square()
        raw_weight *= active.to(raw_weight.dtype)
        raw_sum = raw_weight.sum(dim=1, keepdim=True)
        weight = raw_weight * torch.clamp(
            cfg.collision_metric_budget / torch.clamp(raw_sum, min=1.0e-12),
            max=1.0,
        )

        clearance_velocity = torch.einsum(
            "bcd,bd->bc", collision.jacobian, velocity
        )
        approaching = torch.minimum(
            clearance_velocity, torch.zeros_like(clearance_velocity)
        )
        desired_clearance_acceleration = (
            cfg.collision_acceleration * proximity
            - cfg.collision_damping * approaching
        )

        metric = torch.einsum(
            "bc,bci,bcj->bij", weight, collision.jacobian, collision.jacobian
        )
        rhs = torch.einsum(
            "bc,bcd->bd",
            weight * desired_clearance_acceleration,
            collision.jacobian,
        )
        return metric, rhs

    def step(
        self,
        target: torch.Tensor,
        collision: CollisionBatch | None = None,
    ) -> ReducedAdeptFabricState:
        """Advance one 60-Hz controller step and return its internal state."""

        if self.state is None:
            raise RuntimeError("call reset() before step()")
        if target.shape != self.state.position.shape:
            raise ValueError(
                f"target must have shape {self.state.position.shape}, got {target.shape}"
            )
        cfg = self.config
        q = self.state.position
        qd = self.state.velocity
        bounded_target = torch.clamp(target, self.lower_limits, self.upper_limits)

        desired_acceleration = (
            self._target_gain * (bounded_target - q)
            - self._target_damping * qd
        )
        diagonal_metric = self._target_metric.expand_as(q).clone()
        rhs = diagonal_metric * desired_acceleration

        limit_metric, limit_rhs = self._joint_limit_terms(q, qd)
        diagonal_metric += limit_metric
        rhs += limit_rhs - cfg.cspace_damping * qd

        root_metric = torch.diag_embed(diagonal_metric)
        if collision is not None and collision.clearance.shape[1] > 0:
            collision_metric, collision_rhs = self._collision_terms(collision, qd)
            root_metric += collision_metric
            rhs += collision_rhs

        identity = torch.eye(
            self.num_dof, device=q.device, dtype=q.dtype
        ).unsqueeze(0)
        qdd = torch.linalg.solve(
            root_metric + cfg.solve_regularization * identity,
            rhs.unsqueeze(-1),
        ).squeeze(-1)

        dt = cfg.timestep
        jerk_delta = self._max_jerk * dt
        qdd = torch.clamp(
            qdd,
            self.state.acceleration - jerk_delta,
            self.state.acceleration + jerk_delta,
        )
        qdd_scale = torch.amin(
            torch.clamp(
                self._max_acceleration / torch.clamp(qdd.abs(), min=1.0e-12),
                max=1.0,
            ),
            dim=1,
            keepdim=True,
        )
        qdd = qdd * qdd_scale

        qd_new = qd + dt * qdd
        qd_new = torch.clamp(qd_new, -self._max_velocity, self._max_velocity)
        speed_energy = torch.sum(self._speed_weights * qd_new.square(), dim=1)
        speed_scale = torch.sqrt(
            torch.clamp(
                cfg.speed_energy_target
                / torch.clamp(speed_energy, min=1.0e-12),
                max=1.0,
            )
        ).unsqueeze(-1)
        qd_new = qd_new * speed_scale

        q_new_unclamped = q + dt * qd_new
        q_new = torch.clamp(q_new_unclamped, self.lower_limits, self.upper_limits)
        hit_limit = q_new != q_new_unclamped
        qd_new = torch.where(hit_limit, torch.zeros_like(qd_new), qd_new)

        self.state = ReducedAdeptFabricState(q_new, qd_new, qdd)
        return self.state


__all__ = [
    "CollisionBatch",
    "ReducedAdeptFabric",
    "ReducedAdeptFabricConfig",
    "ReducedAdeptFabricState",
]
