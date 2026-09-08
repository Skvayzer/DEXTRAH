"""Normal + shear readout, independent of Isaac Sim and policy registration.

PhysX normal contacts and friction anchors have INDEPENDENT indices/counts.
Their sum, projected onto a sensor plane, approximates its measured load.
This is not a calibrated model of the Revo2 pad's deformation or electronics.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .contact import aggregate_pad_contacts, rotate_to_local


@dataclass
class PadContactForces:
    normal_pairs_w: torch.Tensor
    normal_centroids_w: torch.Tensor
    normal_load_n: torch.Tensor
    reconstructed_normal_pairs_w: torch.Tensor
    friction_pairs_w: torch.Tensor
    reconstructed_friction_pairs_w: torch.Tensor

    @property
    def total_pairs_w(self):
        return self.normal_pairs_w + self.friction_pairs_w


def aggregate_pad_friction(raw_data, body_pos, body_quat, pad_origin,
                           pad_rotation, pad_bounds):
    """Sum get_friction_data(physics_dt) forces at in-pad friction anchors.

    Inputs are already newtons: do NOT divide by dt again. Buffer padding may
    contain arbitrary values; only indexed active anchors are read. Return
    masked and unmasked world-frame force sums, each (env, filter, xyz).
    """
    forces, points, counts, starts = raw_data
    if counts.ndim != 2 or starts.shape != counts.shape:
        raise ValueError("friction counts/starts must have shape (env, filter)")
    if forces.ndim != 2 or forces.shape[-1] != 3 or points.shape != forces.shape:
        raise ValueError("friction force/point buffers must have shape (capacity, 3)")
    counts, starts = counts.long(), starts.long()
    if torch.any(counts < 0):
        raise RuntimeError("negative friction count")
    n_env, n_channel = counts.shape
    flat_counts = counts.flatten()
    rows = torch.repeat_interleave(
        torch.arange(flat_counts.numel(), device=counts.device), flat_counts)
    masked = forces.new_zeros(n_env * n_channel, 3)
    reconstructed = torch.zeros_like(masked)
    if rows.numel():
        block = flat_counts.cumsum(0) - flat_counts
        indices = starts.flatten()[rows] + torch.arange(
            rows.numel(), device=counts.device) - block[rows]
        # Full capacity is ambiguous: PhysX may have truncated further anchors.
        if indices.min() < 0 or indices.max() >= len(points) or rows.numel() >= len(points):
            raise RuntimeError("friction-point buffer exhausted or invalid; increase capacity")
        world, vectors = points[indices], forces[indices]
        if not torch.isfinite(world).all() or not torch.isfinite(vectors).all():
            raise RuntimeError("non-finite active PhysX friction anchor")
        env_ids = rows // n_channel
        local = rotate_to_local(body_quat[env_ids], world - body_pos[env_ids])
        pad_points = (local - pad_origin) @ pad_rotation
        accepted = ((pad_points >= pad_bounds[0]) & (pad_points <= pad_bounds[1])).all(-1)
        reconstructed.index_add_(0, rows, vectors)
        masked.index_add_(0, rows, vectors * accepted[:, None])
    return masked.view(n_env, n_channel, 3), reconstructed.view(n_env, n_channel, 3)


def read_pad_contact_forces(view, dt, body_pos, body_quat, pad_origin,
                            pad_rotation, pad_bounds):
    """Read both PhysX buffers WITHOUT allowing shared counts to be overwritten.

    The installed tensor backend reuses counts/starts between the two getters.
    Finish normal aggregation before requesting friction; never fetch both raw
    tuples first or zip normal points with friction anchors. Missing friction
    support is an error, not a fabricated all-zero shear signal.
    """
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("physics dt must be finite and positive")
    args = (body_pos, body_quat, pad_origin, pad_rotation, pad_bounds)
    normal = aggregate_pad_contacts(view.get_contact_data(dt=dt), *args)
    friction = aggregate_pad_friction(view.get_friction_data(dt=dt), *args)
    return PadContactForces(*normal, *friction)


def project_pad_force(force_w, body_quat, pad_rotation, normal_axis):
    """Project TOTAL contact force onto the CAD pad normal and tangent plane.

    pad_rotation maps CAD pad coordinates to the distal-link frame. Its selected
    normal axis points OUT of the pad: compressive load is minus that component.
    Tangent axes are the remaining CAD axes, in ascending order. These axes are
    explicit geometric diagnostics, NOT a verified BrainCo 0-degree direction.
    """
    if normal_axis not in (0, 1, 2):
        raise ValueError("normal_axis must be 0, 1 or 2")
    if pad_rotation.shape != (3, 3):
        raise ValueError("pad_rotation must be a 3x3 rotation")
    eye = torch.eye(3, device=pad_rotation.device, dtype=pad_rotation.dtype)
    if (not torch.isfinite(pad_rotation).all()
            or not torch.allclose(pad_rotation.T @ pad_rotation, eye, atol=1.e-5, rtol=1.e-5)
            or not torch.isclose(torch.det(pad_rotation), pad_rotation.new_tensor(1.), atol=1.e-5)):
        raise ValueError("pad_rotation must be a proper orthonormal rotation")
    local = rotate_to_local(body_quat, force_w) @ pad_rotation
    if not torch.isfinite(local).all():
        raise ValueError("non-finite pad force")
    tangent_axes = [i for i in range(3) if i != normal_axis]
    tangential = local[..., tangent_axes]
    return {
        "normal_n": (-local[..., normal_axis]).clamp_min(0.),
        "signed_outward_n": local[..., normal_axis],
        "tangential_n": tangential,
        "tangential_magnitude_n": tangential.norm(dim=-1),
    }


def decode_revo2_force_packet(normal_raw, tangential_raw, direction_raw, status_raw):
    """Decode capacitive Revo2 *_1 fields, preserving validity separately.

    Cartesian shear uses the DEVICE angular convention: component 1 is 0 deg,
    component 2 is +90 deg (clockwise per protocol). A measured/verified frame
    mapping is required before equating these axes to CAD tangent coordinates.
    Sequence progress is not a substitute for valid status. No 25 N clipping:
    recorded values can exceed the documented linear range and remain visible.
    """
    tensors = (normal_raw, tangential_raw, direction_raw, status_raw)
    if any(t.shape != normal_raw.shape or t.device != normal_raw.device for t in tensors):
        raise ValueError("packet fields must share shape and device")
    if any(t.is_floating_point() or t.is_complex() or t.dtype == torch.bool for t in tensors):
        raise ValueError("expected encoded integer packet fields, not converted forces")
    normal_raw, tangential_raw, direction_raw, status_raw = [t.long() for t in tensors]
    if any(torch.any((t < 0) | (t > 65535)) for t in
           (normal_raw, tangential_raw, direction_raw, status_raw)):
        raise ValueError("packet fields must fit uint16")
    normal = normal_raw.float() * .01
    magnitude = tangential_raw.float() * .01
    flags = status_raw & 255
    force_valid = flags == 0
    direction_valid = force_valid & (direction_raw < 360) & (tangential_raw > 0)
    # Do not perform trigonometry on the invalid 65535 sentinel.
    angle = torch.where(direction_valid, direction_raw, 0).float() * (math.pi / 180.)
    tangent = magnitude[..., None] * torch.stack((angle.cos(), angle.sin()), -1)
    tangent = torch.where(direction_valid[..., None], tangent, 0.)
    return {
        "normal_n": normal, "tangential_magnitude_n": magnitude,
        "tangential_n": tangent, "force_valid": force_valid,
        "direction_valid": direction_valid, "status_flags": flags,
        "sensor_sequence": (status_raw >> 8) & 255,
        "above_linear_range": (normal > 25.) | (magnitude > 25.),
    }
