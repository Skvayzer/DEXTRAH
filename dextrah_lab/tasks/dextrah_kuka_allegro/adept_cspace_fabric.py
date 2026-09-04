# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""ADEPT-style full configuration-space fabric for KUKA-Allegro.

This adapter intentionally lives in DextrAH rather than modifying FABRICS. It
reuses FABRICS' collision, joint-limit, attractor, and integration machinery
while replacing DextrAH's palm-pose + hand-PCA policy interface with separate
arm and hand configuration-space attractors.
"""

import torch

from fabrics_sim.energy.euclidean_energy import EuclideanEnergy
from fabrics_sim.fabric_terms.attractor import Attractor
from fabrics_sim.fabrics.kuka_allegro_pose_fabric import KukaAllegroPoseFabric
from fabrics_sim.taskmaps.linear_taskmap import LinearMap
from fabrics_sim.taskmaps.robot_frame_origins_taskmap import RobotFrameOriginsTaskMap


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
