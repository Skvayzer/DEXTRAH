"""Differentiable forward kinematics for the BrainCo Revo2 hand.

The Revo2 exposes six actuators while its URDF contains five additional
revolute joints coupled through ``mimic`` tags.  Retargeting must optimize the
six hardware commands and evaluate fingertip positions after expanding those
couplings.  This module deliberately parses the URDF rather than duplicating
its transforms in Python.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import torch


REVO2_RIGHT_ACTUATED_JOINTS = (
    "right_thumb_metacarpal_joint",
    "right_thumb_proximal_joint",
    "right_index_proximal_joint",
    "right_middle_proximal_joint",
    "right_ring_proximal_joint",
    "right_pinky_proximal_joint",
)

REVO2_RIGHT_FINGERTIP_LINKS = (
    "right_thumb_tip_link",
    "right_index_tip_link",
    "right_middle_tip_link",
    "right_ring_tip_link",
    "right_pinky_tip_link",
)


@dataclass(frozen=True)
class _Mimic:
    joint: str
    multiplier: float
    offset: float


@dataclass(frozen=True)
class _Joint:
    name: str
    kind: str
    parent: str
    child: str
    xyz: tuple[float, float, float]
    rpy: tuple[float, float, float]
    axis: tuple[float, float, float]
    lower: float | None
    upper: float | None
    mimic: _Mimic | None


def _vector(element: ET.Element | None, attribute: str, default: str) -> tuple[float, ...]:
    value = default if element is None else element.get(attribute, default)
    return tuple(float(item) for item in value.split())


def _rpy_matrix(rpy: tuple[float, float, float], *, dtype: torch.dtype) -> torch.Tensor:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    # URDF fixed-axis RPY: Rz(yaw) @ Ry(pitch) @ Rx(roll).
    return torch.tensor(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=dtype,
    )


def _origin_matrix(joint: _Joint, *, dtype: torch.dtype) -> torch.Tensor:
    result = torch.eye(4, dtype=dtype)
    result[:3, :3] = _rpy_matrix(joint.rpy, dtype=dtype)
    result[:3, 3] = torch.tensor(joint.xyz, dtype=dtype)
    return result


def _axis_angle_matrix(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Return a batched homogeneous Rodrigues rotation matrix."""

    axis = axis / torch.linalg.vector_norm(axis)
    x, y, z = axis.unbind()
    zero = torch.zeros_like(angle)
    skew = torch.stack(
        (
            zero,
            -z.expand_as(angle),
            y.expand_as(angle),
            z.expand_as(angle),
            zero,
            -x.expand_as(angle),
            -y.expand_as(angle),
            x.expand_as(angle),
            zero,
        ),
        dim=-1,
    ).reshape(angle.shape + (3, 3))
    eye = torch.eye(3, dtype=angle.dtype, device=angle.device).expand(angle.shape + (3, 3))
    outer = axis[:, None] * axis[None, :]
    outer = outer.expand(angle.shape + (3, 3))
    cosine = torch.cos(angle)[..., None, None]
    sine = torch.sin(angle)[..., None, None]
    rotation = cosine * eye + (1.0 - cosine) * outer + sine * skew
    result = torch.zeros(angle.shape + (4, 4), dtype=angle.dtype, device=angle.device)
    result[..., :3, :3] = rotation
    result[..., 3, 3] = 1.0
    return result


def saturate_joint_position(
    unconstrained: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> torch.Tensor:
    """Map unconstrained variables differentiably into joint limits."""

    if lower.shape != upper.shape or unconstrained.shape[-1] != lower.numel():
        raise ValueError("joint limit shapes must match the last input dimension")
    if torch.any(upper <= lower):
        raise ValueError("every upper joint limit must exceed its lower limit")
    return lower + 0.5 * (torch.tanh(unconstrained) + 1.0) * (upper - lower)


def unsaturate_joint_position(
    position: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    *,
    epsilon: float = 1.0e-6,
) -> torch.Tensor:
    """Inverse of :func:`saturate_joint_position`, finite at either limit."""

    if not 0.0 < epsilon < 0.5:
        raise ValueError("epsilon must lie in (0, 0.5)")
    unit = (position - lower) / (upper - lower)
    unit = torch.clamp(unit, epsilon, 1.0 - epsilon)
    return torch.atanh(2.0 * unit - 1.0)


class Revo2Kinematics:
    """Five-fingertip FK parameterized by the six Revo2 motor positions."""

    def __init__(
        self,
        urdf_path: str | Path,
        *,
        base_link: str = "right_base_link",
        actuated_joint_names: tuple[str, ...] = REVO2_RIGHT_ACTUATED_JOINTS,
        fingertip_link_names: tuple[str, ...] = REVO2_RIGHT_FINGERTIP_LINKS,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        self.urdf_path = Path(urdf_path)
        self.base_link = base_link
        self.actuated_joint_names = tuple(actuated_joint_names)
        self.fingertip_link_names = tuple(fingertip_link_names)
        self.dtype = dtype
        self._actuated_index = {
            name: index for index, name in enumerate(self.actuated_joint_names)
        }
        if len(self._actuated_index) != len(self.actuated_joint_names):
            raise ValueError("actuated joint names must be unique")

        root = ET.parse(self.urdf_path).getroot()
        joints = tuple(self._parse_joint(element) for element in root.findall("joint"))
        self._joints = {joint.name: joint for joint in joints}
        child_to_joint = {joint.child: joint for joint in joints}
        if len(child_to_joint) != len(joints):
            raise ValueError("URDF contains a link with multiple parent joints")
        self._chains = tuple(
            self._chain_to_link(link, child_to_joint) for link in self.fingertip_link_names
        )

        missing = set(self.actuated_joint_names) - self._joints.keys()
        if missing:
            raise ValueError(f"actuated joints missing from URDF: {sorted(missing)}")
        for chain in self._chains:
            for joint in chain:
                if joint.kind not in {"fixed", "revolute", "continuous"}:
                    raise ValueError(f"unsupported Revo2 joint type {joint.kind!r}")
                if joint.kind != "fixed" and joint.name not in self._actuated_index:
                    if joint.mimic is None or joint.mimic.joint not in self._actuated_index:
                        raise ValueError(
                            f"joint {joint.name!r} is neither actuated nor coupled to an actuator"
                        )

        limits = [self._joints[name] for name in self.actuated_joint_names]
        if any(joint.lower is None or joint.upper is None for joint in limits):
            raise ValueError("all Revo2 actuators must have finite position limits")
        self.lower = torch.tensor([joint.lower for joint in limits], dtype=dtype)
        self.upper = torch.tensor([joint.upper for joint in limits], dtype=dtype)

    @staticmethod
    def _parse_joint(element: ET.Element) -> _Joint:
        origin = element.find("origin")
        axis = element.find("axis")
        limit = element.find("limit")
        mimic_element = element.find("mimic")
        mimic = None
        if mimic_element is not None:
            mimic = _Mimic(
                joint=mimic_element.attrib["joint"],
                multiplier=float(mimic_element.get("multiplier", "1")),
                offset=float(mimic_element.get("offset", "0")),
            )
        return _Joint(
            name=element.attrib["name"],
            kind=element.attrib["type"],
            parent=element.find("parent").attrib["link"],
            child=element.find("child").attrib["link"],
            xyz=_vector(origin, "xyz", "0 0 0"),
            rpy=_vector(origin, "rpy", "0 0 0"),
            axis=_vector(axis, "xyz", "1 0 0"),
            lower=None if limit is None or limit.get("lower") is None else float(limit.get("lower")),
            upper=None if limit is None or limit.get("upper") is None else float(limit.get("upper")),
            mimic=mimic,
        )

    def _chain_to_link(
        self, link: str, child_to_joint: dict[str, _Joint]
    ) -> tuple[_Joint, ...]:
        reverse_chain: list[_Joint] = []
        current = link
        while current != self.base_link:
            if current not in child_to_joint:
                raise ValueError(
                    f"link {link!r} is not connected to base {self.base_link!r}"
                )
            joint = child_to_joint[current]
            reverse_chain.append(joint)
            current = joint.parent
        return tuple(reversed(reverse_chain))

    @property
    def num_actuators(self) -> int:
        return len(self.actuated_joint_names)

    @property
    def num_fingertips(self) -> int:
        return len(self.fingertip_link_names)

    def _joint_angle(self, joint: _Joint, q: torch.Tensor) -> torch.Tensor:
        if joint.name in self._actuated_index:
            return q[..., self._actuated_index[joint.name]]
        assert joint.mimic is not None
        source = q[..., self._actuated_index[joint.mimic.joint]]
        return source * joint.mimic.multiplier + joint.mimic.offset

    def fingertip_positions(self, q: torch.Tensor) -> torch.Tensor:
        """Return palm-fixed fingertip positions with shape ``[..., 5, 3]``."""

        if q.ndim == 0 or q.shape[-1] != self.num_actuators:
            raise ValueError(f"q must end in {self.num_actuators} Revo2 actuator values")
        if not q.is_floating_point():
            raise ValueError("q must be floating point")

        batch_shape = q.shape[:-1]
        points: list[torch.Tensor] = []
        for chain in self._chains:
            transform = torch.eye(4, dtype=q.dtype, device=q.device).expand(
                batch_shape + (4, 4)
            )
            for joint in chain:
                origin = _origin_matrix(joint, dtype=q.dtype).to(q.device)
                transform = transform @ origin
                if joint.kind != "fixed":
                    axis = torch.tensor(joint.axis, dtype=q.dtype, device=q.device)
                    transform = transform @ _axis_angle_matrix(
                        axis, self._joint_angle(joint, q)
                    )
            points.append(transform[..., :3, 3])
        return torch.stack(points, dim=-2)

    def to(self, device: torch.device | str) -> "Revo2Kinematics":
        """Move cached limit tensors; FK constants follow each input tensor."""

        self.lower = self.lower.to(device)
        self.upper = self.upper.to(device)
        return self
