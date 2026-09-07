"""Diagnostic fingertip signals; independent of Isaac Sim and of the RL MDP.

Forces are PhysX *normal-contact resultant vectors*, not measured capacitive
forces, shear, or a calibrated pressure map. Opposing contacts may cancel in
the resultant. Finger order is explicit and never inferred from body order.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch


FINGERS = ("thumb", "index", "middle", "ring", "pinky")
# The G1 URDF's *_tip meshes are 0.5 mm markers, NOT tactile surfaces.
TIP_BODIES = tuple(f"right_{finger}_distal_link" for finger in FINGERS)
CONTACT_CHANNELS = ("object", "table", "probe")


def rotate_to_local(quaternion_wxyz: torch.Tensor, vector_w: torch.Tensor) -> torch.Tensor:
    """Inverse rotation, with matching leading dimensions (no translation)."""
    if quaternion_wxyz.shape[:-1] != vector_w.shape[:-1]:
        raise ValueError("quaternion and vector batch shapes must match")
    if quaternion_wxyz.shape[-1] != 4 or vector_w.shape[-1] != 3:
        raise ValueError("expected wxyz quaternions and xyz vectors")
    norm = quaternion_wxyz.norm(dim=-1, keepdim=True)
    if not torch.isfinite(norm).all() or torch.any(norm < 1.e-8):
        raise ValueError("invalid quaternion")
    quat = quaternion_wxyz / norm
    xyz = -quat[..., 1:]
    cross = 2 * torch.linalg.cross(xyz, vector_w)
    return vector_w + quat[..., :1] * cross + torch.linalg.cross(xyz, cross)


def adept_grasp_gate(force_n: torch.Tensor, threshold_n: float = 1.) -> torch.Tensor:
    """ADEPT's strict >1 N thumb AND any other finger condition.

    The caller decides whether these are all contacts or privileged,
    object-only contacts. The latter cannot be obtained from touch alone.
    """
    if force_n.shape[-1] != 5:
        raise ValueError("expected thumb/index/middle/ring/pinky")
    above = torch.isfinite(force_n) & (force_n > threshold_n)
    return above[..., 0] & above[..., 1:].any(dim=-1)


@dataclass(frozen=True)
class ContactFilterConfig:
    # Provisional diagnostic thresholds, NOT Revo2 hardware calibration.
    on_n: float = 0.15
    off_n: float = 0.08
    time_constant_s: float = 0.03

    def __post_init__(self):
        if not all(math.isfinite(x) for x in (self.on_n, self.off_n, self.time_constant_s)):
            raise ValueError("filter configuration must be finite")
        if not 0 <= self.off_n < self.on_n or self.time_constant_s < 0:
            raise ValueError("require 0 <= off < on and nonnegative time constant")


class FingertipContactFilter:
    """Batched, resettable scalar filtering with explicit invalid-sample flags.

    Call once per physics step. This does not pretend that a 120 Hz simulator
    is the real hand's 70 Hz tactile stream; hardware resampling/calibration
    belongs in a later, separately validated adapter.
    """

    def __init__(self, num_envs: int, device="cpu", config=None):
        self.config = config or ContactFilterConfig()
        self.raw_n = torch.zeros(num_envs, 5, device=device)
        self.filtered_n = torch.zeros_like(self.raw_n)
        self.derivative_n_s = torch.zeros_like(self.raw_n)
        self.duration_s = torch.zeros_like(self.raw_n)
        self.contact = torch.zeros_like(self.raw_n, dtype=torch.bool)
        self.valid = torch.zeros_like(self.contact)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        for value in (self.raw_n, self.filtered_n, self.derivative_n_s,
                      self.duration_s, self.contact, self.valid):
            value[ids] = 0

    def update(self, force_w: torch.Tensor, dt: float):
        if force_w.shape != (*self.raw_n.shape, 3):
            raise ValueError("expected force shape (num_envs, 5, 3)")
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        self.valid.copy_(torch.isfinite(force_w).all(dim=-1))
        # Invalid data are visibly invalid, never retained as a false grasp.
        raw = torch.linalg.vector_norm(torch.where(self.valid[..., None], force_w, 0.), dim=-1)
        previous = self.filtered_n.clone()
        tau = self.config.time_constant_s
        alpha = 1. if tau == 0 else -math.expm1(-dt / tau)
        self.raw_n.copy_(raw)
        self.filtered_n.lerp_(raw, alpha)
        self.filtered_n.masked_fill_(~self.valid, 0.)
        self.derivative_n_s.copy_((self.filtered_n - previous) / dt)
        self.derivative_n_s.masked_fill_(~self.valid, 0.)
        self.contact.copy_(self.valid & torch.where(
            self.contact, self.filtered_n > self.config.off_n,
            self.filtered_n >= self.config.on_n,
        ))
        self.duration_s.copy_(torch.where(self.contact, self.duration_s + dt, 0.))
        return self
