# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""ADEPT-style full configuration-space fabric for KUKA-Allegro.

This adapter intentionally lives in DextrAH rather than modifying FABRICS. It
reuses FABRICS' collision, joint-limit, attractor, and integration machinery
while replacing DextrAH's palm-pose + hand-PCA policy interface with separate
arm and hand configuration-space attractors.
"""

import numpy as np
import torch

# urdfpy 0.0.22 still uses the alias removed in NumPy 1.24. Isaac Sim 5 pins
# NumPy 1.26, so retain compatibility without downgrading the simulator stack.
if not hasattr(np, "float"):
    np.float = float  # type: ignore[attr-defined]

from fabrics_sim.energy.euclidean_energy import EuclideanEnergy
from fabrics_sim.fabric_terms.attractor import Attractor
from fabrics_sim.fabric_terms.body_sphere_3d_repulsion import BaseFabricRepulsion
from fabrics_sim.fabric_terms.fabric_term import BaseFabricTerm
from fabrics_sim.fabrics.kuka_allegro_pose_fabric import KukaAllegroPoseFabric
from fabrics_sim.taskmaps.linear_taskmap import LinearMap
from fabrics_sim.taskmaps.maps_base import BaseMap
from fabrics_sim.taskmaps.robot_frame_origins_taskmap import RobotFrameOriginsTaskMap

from dextrah_lab.adept.fabric_math import (
    collision_acceleration,
    collision_metric_weights,
    joint_limit_metric_diagonal,
    normalize_collision_metric_per_sphere,
)


class NormalizedJointLimitMap(BaseMap):
    """Joint-limit clearance taskmap expressed in fraction-of-range units."""

    def __init__(
        self,
        lower_limits: list[float],
        upper_limits: list[float],
        batch_size: int,
        device: str,
        *,
        upper_side: bool,
    ):
        super().__init__(device)
        lower = torch.tensor(lower_limits, device=device)
        upper = torch.tensor(upper_limits, device=device)
        joint_range = upper - lower
        if torch.any(joint_range <= 0):
            raise ValueError("invalid URDF joint limits")
        boundary = upper if upper_side else lower
        sign = -1.0 if upper_side else 1.0
        self.boundary = boundary.unsqueeze(0).repeat(batch_size, 1)
        self.inverse_range = joint_range.reciprocal()
        jacobian = sign * torch.diag(self.inverse_range)
        self.jacobian = jacobian.unsqueeze(0).repeat(batch_size, 1, 1)
        self.upper_side = upper_side

    def forward_position(self, q, _features):
        displacement = self.boundary - q if self.upper_side else q - self.boundary
        return displacement * self.inverse_range, self.jacobian


class AdeptJointLimitRepulsion(BaseFabricTerm):
    """Normalized, smoothly gated joint-limit response from Appendix B."""

    def __init__(self, is_forcing_policy, params, device, graph_capturable):
        params = dict(params)
        # ADEPT v1 does not report these two gate values. Preserve the upstream
        # zero-velocity switch while making its smoothing finite and explicit.
        params.setdefault("velocity_gate_sharpness", 10.0)
        params.setdefault("velocity_gate_offset", params["breakaway_velocity"])
        super().__init__(
            is_forcing_policy, params, device, graph_capturable=graph_capturable
        )
        self.gate = None

    def metric_eval(self, x, xd, _features):
        diagonal, gate = joint_limit_metric_diagonal(
            x,
            xd,
            metric_scalar=self.params["metric_scalar"],
            metric_exploder_offset=self.params["metric_exploder_offset"],
            max_metric=self.params["max_metric"],
            gate_sharpness=self.params["velocity_gate_sharpness"],
            gate_offset=self.params["velocity_gate_offset"],
        )
        evaluated = torch.diag_embed(diagonal)
        if self.metric is None:
            self.metric = evaluated
            self.force = torch.zeros_like(x)
            self.gate = gate
        elif self.graph_capturable:
            self.metric.copy_(evaluated)
            self.gate.copy_(gate)
        else:
            self.metric = evaluated
            self.gate = gate

    def force_eval(self, _x, xd, _features):
        if self.is_forcing_policy:
            damping = (xd < 0).to(xd.dtype) * self.params["damping_gain"]
            xdd = self.gate * self.params["soft_relu_gain"] - damping * xd
        else:
            speed_squared = xd.square().sum(dim=1, keepdim=True)
            xdd = speed_squared * self.params["soft_relu_gain"] * self.gate
        evaluated = -torch.bmm(self.metric, xdd.unsqueeze(2)).squeeze(2)
        if self.graph_capturable:
            self.force.copy_(evaluated)
        else:
            self.force = evaluated


class AdeptBodySphereRepulsion(BaseFabricTerm):
    """Per-sphere-normalized, budgeted body collision response."""

    def __init__(
        self,
        is_forcing_policy,
        params,
        batch_size,
        sphere_radius,
        device,
        graph_capturable,
    ):
        params = dict(params)
        # ADEPT publishes the response equation but not either numeric budget.
        params.setdefault("forcing_metric_budget", 1.0)
        params.setdefault("geom_metric_budget", 1.0)
        super().__init__(
            is_forcing_policy, params, device, graph_capturable=graph_capturable
        )
        self.batch_size = batch_size
        self.sphere_radius = sphere_radius
        self.num_spheres = sphere_radius.shape[1]

    def metric_eval(self, x, _xd, features):
        if features is None:
            evaluated = torch.zeros(
                x.shape[0], x.shape[1], x.shape[1], device=self.device
            )
        else:
            # The stock Warp query supplies a block-diagonal directional
            # metric. ADEPT normalizes each 3x3 sphere block independently.
            import warp as wp

            raw_metric = wp.torch.to_torch(features.allocated_data["metric"])
            unit_metric, active = normalize_collision_metric_per_sphere(
                raw_metric, self.num_spheres
            )
            prefix = "forcing" if self.is_forcing_policy else "geom"
            weights = collision_metric_weights(
                features.signed_distance,
                self.sphere_radius,
                active,
                metric_scalar=self.params[f"{prefix}_metric_scalar"],
                metric_budget=self.params[f"{prefix}_metric_budget"],
                minimum_distance=self.params["rescaled_min_dist"],
            )
            row_weights = weights.repeat_interleave(3, dim=1).unsqueeze(-1)
            evaluated = unit_metric * row_weights

        if self.metric is None:
            self.metric = evaluated
            self.force = torch.zeros_like(x)
        elif self.graph_capturable:
            self.metric.copy_(evaluated)
        else:
            self.metric = evaluated

    def force_eval(self, x, xd, features):
        if features is None:
            xdd = torch.zeros_like(x)
        else:
            direction = features.accel_dir.reshape(
                self.batch_size, self.num_spheres, 3
            )
            velocity = xd.reshape(self.batch_size, self.num_spheres, 3)
            xdd = collision_acceleration(
                direction,
                velocity,
                gain=(
                    self.params["constant_accel"]
                    if self.is_forcing_policy
                    else self.params["constant_accel_geom"]
                ),
                damping=(
                    self.params["damping_gain"] if self.is_forcing_policy else 0.0
                ),
                geometric=not self.is_forcing_policy,
            ).reshape_as(x)
        evaluated = -torch.bmm(self.metric, xdd.unsqueeze(2)).squeeze(2)
        if self.graph_capturable:
            self.force.copy_(evaluated)
        else:
            self.force = evaluated


class ScaledEuclideanEnergy(EuclideanEnergy):
    """Euclidean kinetic energy with a fixed metric scale."""

    def __init__(self, batch_size: int, num_joints: int, device: str, scale: float):
        self.scale = scale
        super().__init__(batch_size, num_joints, device)
        self.metric.mul_(scale)

    def energy_eval(self, x: torch.Tensor, xd: torch.Tensor):
        energy = 0.5 * self.scale * torch.sum(xd * xd, dim=1).unsqueeze(1)
        return self.metric, self.force, energy


class AdeptKukaAllegroCspaceFabric(KukaAllegroPoseFabric):
    """Full 23-DoF fabric assembled from public ADEPT specifications.

    ADEPT does not publish the numerical arm/hand attractor gains. The values
    below preserve the corresponding public DextrAH palm and hand attractor
    defaults. They are explicit here so they can be calibrated independently
    when reference trajectories or official parameters become available.
    """

    NUM_ARM_JOINTS = 7
    NUM_HAND_JOINTS = 16

    ARM_FORCING_PARAMS = {
        "min_isotropic_mass": 1.0,
        "max_isotropic_mass": 1.0,
        "mass_sharpness": 10.0,
        "mass_switch_offset": 0.5,
        "conical_sharpness": 40.0,
        "conical_gain": 50.0,
        "damping": 50.0,
        "damping_sharpness": 10.0,
        "damping_radius": 0.2,
    }
    HAND_FORCING_PARAMS = {
        "min_isotropic_mass": 8.0,
        "max_isotropic_mass": 8.0,
        "mass_sharpness": 20.0,
        "mass_switch_offset": 0.5,
        "conical_sharpness": 40.0,
        "conical_gain": 50.0,
        "damping": 50.0,
        "damping_sharpness": 10.0,
        "damping_radius": 0.2,
    }

    def _joint_limits(self) -> tuple[list[float], list[float]]:
        lower, upper = [], []
        for joint in self.urdfpy_robot.joints:
            if joint.joint_type == "revolute":
                lower.append(joint.limit.lower)
                upper.append(joint.limit.upper)
        return lower, upper

    def add_joint_limit_repulsion(self) -> None:
        lower, upper = self._joint_limits()
        for upper_side, taskmap_name in (
            (True, "upper_joint_limit"),
            (False, "lower_joint_limit"),
        ):
            self.add_taskmap(
                taskmap_name,
                NormalizedJointLimitMap(
                    lower,
                    upper,
                    self.batch_size,
                    self.device,
                    upper_side=upper_side,
                ),
                graph_capturable=self.graph_capturable,
            )
            for is_forcing, fabric_name in (
                (True, "joint_limit_repulsion"),
                (False, "geom_joint_limit_repulsion"),
            ):
                self.add_fabric(
                    taskmap_name,
                    fabric_name,
                    AdeptJointLimitRepulsion(
                        is_forcing,
                        self.fabric_params["joint_limit_repulsion"],
                        self.device,
                        graph_capturable=self.graph_capturable,
                    ),
                )

    def add_body_repulsion(self) -> None:
        params = self.fabric_params["body_repulsion"]
        frames = params["collision_sphere_frames"]
        self.collision_sphere_radii = params["collision_sphere_radii"]
        if len(frames) != len(self.collision_sphere_radii):
            raise ValueError("collision sphere frames and radii must have equal length")

        pairs = list(params["collision_sphere_pairs"])
        if not pairs:
            for prefix1, prefix2 in params["collision_link_prefix_pairs"]:
                frames1 = [frame for frame in frames if prefix1 in frame]
                frames2 = [frame for frame in frames if prefix2 in frame]
                pairs.extend(
                    [frame1, frame2]
                    for frame1 in frames1
                    for frame2 in frames2
                )
        collision_matrix = torch.zeros(
            len(frames), len(frames), dtype=torch.int64, device=self.device
        )
        for frame1, frame2 in pairs:
            collision_matrix[frames.index(frame1), frames.index(frame2)] = 1

        self.add_taskmap(
            "body_points",
            RobotFrameOriginsTaskMap(
                self.urdf_path, frames, self.batch_size, self.device
            ),
            graph_capturable=self.graph_capturable,
        )
        sphere_radius = torch.tensor(
            self.collision_sphere_radii, device=self.device
        ).repeat(self.batch_size, 1)
        self.add_fabric(
            "body_points",
            "repulsion",
            AdeptBodySphereRepulsion(
                True,
                params,
                self.batch_size,
                sphere_radius,
                self.device,
                graph_capturable=self.graph_capturable,
            ),
        )
        self.add_fabric(
            "body_points",
            "geom_repulsion",
            AdeptBodySphereRepulsion(
                False,
                params,
                self.batch_size,
                sphere_radius,
                self.device,
                graph_capturable=self.graph_capturable,
            ),
        )
        self.base_fabric_repulsion = BaseFabricRepulsion(
            params,
            self.batch_size,
            sphere_radius,
            collision_matrix,
            self.device,
        )

    def _selector(self, start: int, size: int) -> torch.Tensor:
        selector = torch.zeros(size, self._num_joints, device=self.device)
        selector[:, start : start + size] = torch.eye(size, device=self.device)
        return selector

    def _add_body_part_attractors(
        self,
        taskmap_name: str,
        selector: torch.Tensor,
        forcing_params: dict,
    ) -> None:
        self.add_taskmap(
            taskmap_name,
            LinearMap(selector, self.device),
            graph_capturable=self.graph_capturable,
        )
        self.add_fabric(
            taskmap_name,
            "geometric_attractor",
            Attractor(
                False,
                dict(self.fabric_params["cspace_attractor"]),
                self.device,
                graph_capturable=self.graph_capturable,
            ),
        )
        self.add_fabric(
            taskmap_name,
            "forcing_attractor",
            Attractor(
                True,
                dict(forcing_params),
                self.device,
                graph_capturable=self.graph_capturable,
            ),
        )

    def construct_fabric(self) -> None:
        self.fabric_params["speed_control"].update(
            active=True,
            energy_target=1.0,
            damping=100.0,
        )

        self.add_joint_limit_repulsion()

        arm_selector = self._selector(0, self.NUM_ARM_JOINTS)
        hand_selector = self._selector(self.NUM_ARM_JOINTS, self.NUM_HAND_JOINTS)
        self._add_body_part_attractors(
            "arm_cspace", arm_selector, self.ARM_FORCING_PARAMS
        )
        self._add_body_part_attractors(
            "hand_cspace", hand_selector, self.HAND_FORCING_PARAMS
        )

        self.add_body_repulsion()

        # ADEPT assigns 75% of the speed budget to arm joint motion and 25%
        # to Cartesian palm translation. Finger motion is not energized.
        self.add_energy(
            "arm_cspace",
            "arm_energy",
            ScaledEuclideanEnergy(
                self.batch_size, self.NUM_ARM_JOINTS, self.device, scale=0.75
            ),
        )
        palm_taskmap_name = "palm_origin"
        self.add_taskmap(
            palm_taskmap_name,
            RobotFrameOriginsTaskMap(
                self.urdf_path, ["palm_link"], self.batch_size, self.device
            ),
            graph_capturable=self.graph_capturable,
        )
        self.add_energy(
            palm_taskmap_name,
            "palm_energy",
            ScaledEuclideanEnergy(self.batch_size, 3, self.device, scale=0.25),
        )

    def set_features(
        self,
        cspace_target: torch.Tensor,
        batched_cspace_position: torch.Tensor,
        batched_cspace_velocity: torch.Tensor,
        object_ids,
        object_indicator,
        cspace_damping_gain=None,
    ) -> None:
        expected_shape = (self.batch_size, self._num_joints)
        if cspace_target.shape != expected_shape:
            raise ValueError(
                f"cspace_target must have shape {expected_shape}, got {cspace_target.shape}"
            )

        arm_slice = slice(0, self.NUM_ARM_JOINTS)
        hand_slice = slice(self.NUM_ARM_JOINTS, self._num_joints)
        self.fabrics_features["arm_cspace"]["geometric_attractor"] = (
            self.default_config[:, arm_slice]
        )
        self.fabrics_features["arm_cspace"]["forcing_attractor"] = cspace_target[
            :, arm_slice
        ]
        self.fabrics_features["hand_cspace"]["geometric_attractor"] = (
            self.default_config[:, hand_slice]
        )
        self.fabrics_features["hand_cspace"]["forcing_attractor"] = cspace_target[
            :, hand_slice
        ]

        body_point_pos, jac = self.get_taskmap("body_points")(
            batched_cspace_position, None
        )
        body_point_vel = torch.bmm(
            jac, batched_cspace_velocity.unsqueeze(2)
        ).squeeze(2)
        self.base_fabric_repulsion.calculate_response(
            body_point_pos,
            body_point_vel,
            object_ids,
            object_indicator,
        )
        self.fabrics_features["body_points"]["repulsion"] = (
            self.base_fabric_repulsion
        )
        self.fabrics_features["body_points"]["geom_repulsion"] = (
            self.base_fabric_repulsion
        )

        if cspace_damping_gain is not None:
            self.fabric_params["cspace_damping"]["gain"] = cspace_damping_gain
