"""Runtime bridge from normalized policy actions to the frozen PCA task map."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch

from .pca import PCAArtifact


class FrozenPCAHandActionMap(torch.nn.Module):
    """The hand portion of DextrAH-G's geometric-fabric policy interface.

    The policy emits normalized actions.  These become absolute targets in the
    learned task coordinates.  A fabric evaluates the current coordinates as
    ``x = [0, A] q`` and attracts them to those targets while its joint limits,
    collision terms, damping, and integration remain active.

    This module also exposes affine reconstruction for diagnostics and simple
    position-control fallbacks.  Reconstruction is not a substitute for the
    geometric fabric in the ADEPT setup.
    """

    def __init__(
        self,
        artifact: PCAArtifact,
        *,
        expected_joint_names: Sequence[str] | None = None,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if expected_joint_names is not None and tuple(expected_joint_names) != (
            artifact.joint_names
        ):
            raise ValueError(
                "PCA artifact joint order does not match the runtime hand: "
                f"artifact={artifact.joint_names}, runtime={tuple(expected_joint_names)}"
            )
        self.joint_names = artifact.joint_names
        self.metadata = dict(artifact.metadata)
        self.register_buffer(
            "components",
            torch.as_tensor(artifact.components, dtype=dtype, device=device),
        )
        self.register_buffer(
            "mean", torch.as_tensor(artifact.mean, dtype=dtype, device=device)
        )
        self.register_buffer(
            "coordinate_min",
            torch.as_tensor(artifact.coordinate_min, dtype=dtype, device=device),
        )
        self.register_buffer(
            "coordinate_max",
            torch.as_tensor(artifact.coordinate_max, dtype=dtype, device=device),
        )
        self.register_buffer(
            "joint_lower",
            torch.as_tensor(artifact.joint_lower, dtype=dtype, device=device),
        )
        self.register_buffer(
            "joint_upper",
            torch.as_tensor(artifact.joint_upper, dtype=dtype, device=device),
        )
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        **kwargs,
    ) -> "FrozenPCAHandActionMap":
        return cls(PCAArtifact.load(path), **kwargs)

    @property
    def latent_dim(self) -> int:
        return self.components.shape[0]

    @property
    def joint_dim(self) -> int:
        return self.components.shape[1]

    def forward(self, hand_joint_positions: torch.Tensor) -> torch.Tensor:
        """Evaluate ``x = A q_hand`` with the matrix frozen as a buffer."""

        if hand_joint_positions.shape[-1] != self.joint_dim:
            raise ValueError(
                f"hand_joint_positions must end in {self.joint_dim} values"
            )
        return hand_joint_positions @ self.components.T

    def normalized_to_target(self, raw_actions: torch.Tensor) -> torch.Tensor:
        """Map policy actions from ``[-1, 1]`` to fitted coordinate bounds."""

        if raw_actions.shape[-1] != self.latent_dim:
            raise ValueError(f"raw_actions must end in {self.latent_dim} values")
        unit = 0.5 * (torch.clamp(raw_actions, -1.0, 1.0) + 1.0)
        return self.coordinate_min + unit * (
            self.coordinate_max - self.coordinate_min
        )

    def fabric_taskmap_matrix(self, arm_dof: int = 7) -> torch.Tensor:
        """Return the frozen G1 arm+Revo2 map ``[0_(5x7), A_(5x6)]``."""

        if arm_dof < 0:
            raise ValueError("arm_dof cannot be negative")
        return torch.cat(
            (
                torch.zeros(
                    self.latent_dim,
                    arm_dof,
                    dtype=self.components.dtype,
                    device=self.components.device,
                ),
                self.components,
            ),
            dim=1,
        )

    def reconstruct(self, task_coordinates: torch.Tensor) -> torch.Tensor:
        """Project task coordinates back into the PCA affine subspace."""

        if task_coordinates.shape[-1] != self.latent_dim:
            raise ValueError(
                f"task_coordinates must end in {self.latent_dim} values"
            )
        mean_coordinates = self.mean @ self.components.T
        return self.mean + (task_coordinates - mean_coordinates) @ self.components

    def reconstruct_clipped(self, task_coordinates: torch.Tensor) -> torch.Tensor:
        """Diagnostic/position-control reconstruction with hardware clipping."""

        return torch.clamp(
            self.reconstruct(task_coordinates), self.joint_lower, self.joint_upper
        )

    def extra_repr(self) -> str:
        return (
            f"latent_dim={self.latent_dim}, joint_dim={self.joint_dim}, "
            f"frozen={not self.components.requires_grad}"
        )


def load_fabric_pca_matrix(
    path: str | Path,
    *,
    arm_dof: int = 7,
    expected_joint_names: Sequence[str] | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Convenience entry point for FABRICS ``LinearMap`` construction."""

    action_map = FrozenPCAHandActionMap.from_file(
        path,
        expected_joint_names=expected_joint_names,
        device=device,
        dtype=dtype,
    )
    return action_map.fabric_taskmap_matrix(arm_dof)
