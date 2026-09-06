"""Isaac Lab tensor adapter for the reduced G1 collision fabric."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .collision_geometry import (
    G1_REVO2_CANONICAL_JOINT_NAMES,
    G1_REVO2_DYNAMIC_SPHERES,
    G1_REVO2_FIXED_SPHERES,
    build_g1_collision_batch,
    point_jacobian,
    reduce_jacobian_to_canonical,
)
from .reduced_fabric import CollisionBatch


def quaternion_apply_wxyz(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Rotate vectors by unit quaternions, with broadcast-compatible batches."""

    if quaternion.shape[-1] != 4 or vector.shape[-1] != 3:
        raise ValueError("quaternion/vector must end in four/three values")
    imaginary = quaternion[..., 1:]
    uv = torch.linalg.cross(imaginary, vector, dim=-1)
    uuv = torch.linalg.cross(imaginary, uv, dim=-1)
    return vector + 2.0 * (quaternion[..., :1] * uv + uuv)


@dataclass(frozen=True)
class IsaacCollisionAdapterConfig:
    table_surface_offset: float = 0.15
    include_self_collision: bool = True


class IsaacG1CollisionAdapter:
    """Convert PhysX link poses/Jacobians to 13-D signed clearances.

    The adapter never asks PhysX to model the manipulated object as an
    avoidance obstacle.  Hand-object and object-table interaction therefore
    remains contact-rich and is handled by the simulator.
    """

    def __init__(
        self,
        env,
        config: IsaacCollisionAdapterConfig | None = None,
    ) -> None:
        self.env = env
        self.config = config or IsaacCollisionAdapterConfig()
        if tuple(env._joint_names_canonical) != G1_REVO2_CANONICAL_JOINT_NAMES:
            raise RuntimeError(
                "G1 collision adapter joint order mismatch: "
                f"expected={G1_REVO2_CANONICAL_JOINT_NAMES}, "
                f"actual={tuple(env._joint_names_canonical)}"
            )

        body_names = tuple(env.robot.data.body_names)
        missing = sorted(
            {sphere.link_name for sphere in G1_REVO2_DYNAMIC_SPHERES}
            - set(body_names)
        )
        if missing:
            raise RuntimeError(f"collision sphere links missing from G1 articulation: {missing}")
        self.body_ids = torch.tensor(
            [body_names.index(sphere.link_name) for sphere in G1_REVO2_DYNAMIC_SPHERES],
            device=env.device,
            dtype=torch.long,
        )
        self.local_offsets = torch.tensor(
            [sphere.offset for sphere in G1_REVO2_DYNAMIC_SPHERES],
            device=env.device,
            dtype=torch.float32,
        )
        self.fixed_centers_local = torch.tensor(
            [sphere.center for sphere in G1_REVO2_FIXED_SPHERES],
            device=env.device,
            dtype=torch.float32,
        )
        self._body_count = len(body_names)

    def measured_canonical_position(self) -> torch.Tensor:
        return self.env.robot.data.joint_pos.index_select(
            1, self.env._canonical_joint_ids_lab
        )

    def _dynamic_sphere_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        env = self.env
        body_state = env.robot.data.body_state_w.index_select(1, self.body_ids)
        body_position = body_state[..., :3]
        body_quaternion = body_state[..., 3:7]
        local_offsets = self.local_offsets.to(body_position).unsqueeze(0).expand(
            env.num_envs, -1, -1
        )
        rotated_offsets = quaternion_apply_wxyz(body_quaternion, local_offsets)
        sphere_positions = body_position + rotated_offsets

        full_jacobians = env.robot.root_physx_view.get_jacobians()
        jacobian_body_count = full_jacobians.shape[1]
        if jacobian_body_count == self._body_count - 1:
            # Fixed-base PhysX omits the root body from articulation Jacobians.
            jacobian_body_ids = self.body_ids - 1
        elif jacobian_body_count == self._body_count:
            jacobian_body_ids = self.body_ids
        else:
            raise RuntimeError(
                "unexpected PhysX body Jacobian count: "
                f"got {jacobian_body_count}, expected {self._body_count - 1} "
                f"or {self._body_count}"
            )
        if torch.any(jacobian_body_ids < 0):
            raise RuntimeError("a dynamic sphere cannot be attached to the root body")
        frame_jacobians = full_jacobians.index_select(1, jacobian_body_ids)
        sphere_jacobians_full = point_jacobian(frame_jacobians, rotated_offsets)

        mimic_sources = 7 + env._mimic_source_hand_action_ids
        sphere_jacobians = reduce_jacobian_to_canonical(
            sphere_jacobians_full,
            env._canonical_joint_ids_lab,
            env._mimic_target_joint_ids,
            mimic_sources,
            env._mimic_multipliers.squeeze(0),
        )
        return sphere_positions, sphere_jacobians

    def _fixed_sphere_positions(self) -> torch.Tensor:
        env = self.env
        root_position = env.robot.data.root_pos_w
        root_quaternion = env.robot.data.root_quat_w
        local = self.fixed_centers_local.to(root_position).unsqueeze(0).expand(
            env.num_envs, -1, -1
        )
        quaternion = root_quaternion.unsqueeze(1).expand(-1, local.shape[1], -1)
        return root_position.unsqueeze(1) + quaternion_apply_wxyz(quaternion, local)

    def build(self) -> CollisionBatch:
        positions, jacobians = self._dynamic_sphere_state()
        fixed_positions = self._fixed_sphere_positions()
        table_height = (
            self.env.scene.env_origins[:, 2]
            + self.env._table_z_per_env
            + self.config.table_surface_offset
        )
        collision = build_g1_collision_batch(
            positions,
            jacobians,
            fixed_positions,
            table_height=table_height,
            include_self_collision=self.config.include_self_collision,
        )
        self.dynamic_positions = positions
        self.dynamic_jacobians = jacobians
        self.fixed_positions = fixed_positions
        self.clearance = collision.clearance
        return collision


__all__ = [
    "IsaacCollisionAdapterConfig",
    "IsaacG1CollisionAdapter",
    "quaternion_apply_wxyz",
]
