"""Per-env procedural objects for MuJoCo Warp, exactly from the generated URDFs.

MuJoCo Warp can vary geom size/pos/quat and body inertials per world, but not
geom *types*. Every object is a single body with four primitive slots:

    0 handle box, 1 handle capsule, 2 head box, 3 head capsule

Each env enables the slots its URDF uses (Isaac converts the URDF cylinders
to capsules of the same radius and cylinder length: `replace_cylinders_with_
capsules=True`) and shrinks the others to a 0.1 mm placeholder at the handle
centre, fully enclosed by the handle (never size 0: rbound 0 means "plane" to
the broadphase). Mass, COM and diagonal inertia are the URDF values that
Isaac uses (`link_density=0`).
"""
from dataclasses import dataclass
import math
import xml.etree.ElementTree as ET

import numpy as np

SLOTS = ('handle_box', 'handle_capsule', 'head_box', 'head_capsule')
SLOT_TYPES = ('box', 'capsule', 'box', 'capsule')
PLACEHOLDER = 1e-4


def quat_from_rpy(r, p, y):
    """URDF fixed-axis roll-pitch-yaw -> wxyz quaternion (R = Rz(y) Ry(p) Rx(r))."""
    cr, sr, cp, sp, cy, sy = (math.cos(r/2), math.sin(r/2), math.cos(p/2), math.sin(p/2),
                              math.cos(y/2), math.sin(y/2))
    return np.array([cr*cp*cy+sr*sp*sy, sr*cp*cy-cr*sp*sy, cr*sp*cy+sr*cp*sy, cr*cp*sy-sr*sp*cy])


@dataclass
class ObjectBank:
    size: np.ndarray      # (n_obj, 4, 3) MuJoCo geom size per slot
    pos: np.ndarray       # (n_obj, 4, 3)
    quat: np.ndarray      # (n_obj, 4, 4) wxyz
    used: np.ndarray      # (n_obj, 4) bool
    mass: np.ndarray      # (n_obj,)
    com: np.ndarray       # (n_obj, 3)
    inertia: np.ndarray   # (n_obj, 3) principal moments, axes aligned with the link frame

    @property
    def rbound(self):
        box = np.linalg.norm(self.size, axis=-1)
        capsule = self.size[..., 0]+self.size[..., 1]
        return np.where(np.array([t == 'box' for t in SLOT_TYPES]), box, capsule)

    @property
    def aabb(self):
        """MuJoCo geom_aabb: local-frame centre (zeros) and half extents."""
        half = self.size.copy()
        capsule = np.array([t == 'capsule' for t in SLOT_TYPES])
        half[..., capsule, 0] = self.size[..., capsule, 0]
        half[..., capsule, 1] = self.size[..., capsule, 0]
        half[..., capsule, 2] = self.size[..., capsule, 0]+self.size[..., capsule, 1]
        return np.concatenate((np.zeros_like(half), half), -1)


def parse_object(path):
    link = ET.parse(path).getroot().find('link')
    geoms = []
    for collision in link.findall('collision'):
        origin = collision.find('origin')
        xyz = np.array([float(v) for v in origin.get('xyz').split()])
        rpy = [float(v) for v in origin.get('rpy').split()]
        shape = collision.find('geometry')[0]
        if shape.tag == 'box':
            geoms.append(('box', np.array([float(v) for v in shape.get('size').split()])/2, xyz, quat_from_rpy(*rpy)))
        elif shape.tag == 'cylinder':
            r, length = float(shape.get('radius')), float(shape.get('length'))
            geoms.append(('capsule', np.array([r, length/2, 0.]), xyz, quat_from_rpy(*rpy)))
        else:
            raise ValueError(f'Unsupported object primitive {shape.tag} in {path}')
    if not 1 <= len(geoms) <= 2:
        raise ValueError(f'Expected handle (+head) in {path}, found {len(geoms)} collision geoms')
    inertial = link.find('inertial')
    if [float(v) for v in inertial.find('origin').get('rpy').split()] != [0., 0., 0.]:
        raise ValueError(f'Rotated inertial frame not supported: {path}')
    tensor = inertial.find('inertia')
    if any(float(tensor.get(k)) != 0. for k in ('ixy', 'ixz', 'iyz')):
        raise ValueError(f'Non-diagonal inertia: {path}')
    return (geoms, float(inertial.find('mass').get('value')),
            np.array([float(v) for v in inertial.find('origin').get('xyz').split()]),
            np.array([float(tensor.get(k)) for k in ('ixx', 'iyy', 'izz')]))


def build_object_bank(paths):
    n = len(paths)
    size = np.full((n, 4, 3), PLACEHOLDER)
    pos = np.zeros((n, 4, 3))
    quat = np.tile(np.array([1., 0., 0., 0.]), (n, 4, 1))
    used = np.zeros((n, 4), dtype=bool)
    mass, com, inertia = np.zeros(n), np.zeros((n, 3)), np.zeros((n, 3))
    for i, path in enumerate(paths):
        geoms, mass[i], com[i], inertia[i] = parse_object(path)
        for part, (kind, s, xyz, q) in enumerate(geoms):
            slot = 2*part+(kind == 'capsule')
            size[i, slot], pos[i, slot], quat[i, slot], used[i, slot] = s, xyz, q, True
    return ObjectBank(size, pos, quat, used, mass, com, inertia)
