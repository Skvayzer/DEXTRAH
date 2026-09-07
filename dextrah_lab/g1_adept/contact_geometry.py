"""CAD-derived, provisional Touch-pad windows on the existing G1 distal links.

These windows do not change collision geometry. The standalone Touch CAD is
aligned through its distal-link frame, not by substituting its different
joint transforms/mimic ratios into the training robot.
"""
from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import trimesh

from .contact import FINGERS, TIP_BODIES


@dataclass
class PadGeometry:
    mesh: trimesh.Trimesh  # in distal-link coordinates
    pad_to_distal: np.ndarray
    bounds: np.ndarray  # in Touch CAD frame, expanded by tolerance
    surface_point: np.ndarray  # on the ACTUAL G1 collision mesh
    outward: np.ndarray


def load_pad_geometry(g1_urdf: Path, touch_description: Path) -> list[PadGeometry]:
    touch = ET.parse(touch_description / "urdf/revo2_right_hand.urdf").getroot()
    g1 = ET.parse(g1_urdf).getroot()
    pads = []
    for finger, body in zip(FINGERS, TIP_BODIES):
        joint = touch.find(f"joint[@name='right_{finger}_touch_joint']")
        if joint is None or joint.find("parent").get("link") != body:
            raise ValueError(f"Touch CAD has incompatible parent frame for {finger}")
        origin = joint.find("origin")
        transform = trimesh.transformations.euler_matrix(*np.fromstring(origin.get("rpy"), sep=" "))
        transform[:3, 3] = np.fromstring(origin.get("xyz"), sep=" ")
        mesh = trimesh.load(touch_description / f"meshes/revo2_right_hand/right_{finger}_touch_link.STL", force="mesh")
        bounds = mesh.bounds + np.asarray([[-.002] * 3, [.002] * 3])
        mesh.apply_transform(transform)
        collision = g1.find(f"link[@name='{body}']/collision")
        filename = collision.find("geometry/mesh").get("filename")
        actual = trimesh.load(g1_urdf.parent / filename, force="mesh")
        # This model's collision origins are identity; fail rather than ignore a change.
        c_origin = collision.find("origin")
        if c_origin is not None:
            assert np.allclose(np.fromstring(c_origin.get("xyz", "0 0 0"), sep=" "), 0)
            assert np.allclose(np.fromstring(c_origin.get("rpy", "0 0 0"), sep=" "), 0)
        local = trimesh.transform_points(actual.vertices, np.linalg.inv(transform))
        inside = ((local >= bounds[0]) & (local <= bounds[1])).all(axis=-1)
        candidates = actual.vertices[inside]
        if len(candidates) < 10:
            raise ValueError(f"Touch/G1 geometry mismatch for {finger}: {len(candidates)} vertices in pad window")
        outward = transform[:3, 2 if finger == "thumb" else 0]
        projection = candidates @ outward
        surface = candidates[projection.argmax()].copy()
        pads.append(PadGeometry(mesh, transform, bounds, surface, outward.copy()))
    return pads
