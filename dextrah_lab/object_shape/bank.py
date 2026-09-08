"""Content-addressed BPS bank for the actual single-link procedural URDFs.

Only preprocessing runs on CPU. Training gathers a fixed per-environment row;
no nearest-neighbor searches, meshes or surface clouds enter the rollout.
"""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial import cKDTree
import trimesh

from .grail.compute_bps import fibonacci_sphere, normalize_points

SHAPE_DIM = 132  # 128 distances + centroid xyz (meters) + radius (meters)
BANK_VERSION = "grail-bps128-urdf-collision-surface16384-seed42-cylinder64-v1"


def load_primitive_urdf(path):
    root = ET.parse(path).getroot()
    links = root.findall("link")
    if len(links) != 1 or root.findall("joint"):
        raise ValueError("BPS training loader requires a single rigid-link procedural object")
    meshes = []
    for collision in links[0].findall("collision"):
        shapes = list(collision.find("geometry"))
        if len(shapes) != 1:
            raise ValueError("Expected exactly one primitive per collision")
        shape = shapes[0]
        if shape.tag == "box":
            dimensions = np.fromstring(shape.attrib["size"], sep=" ")
            if dimensions.shape != (3,) or np.any(dimensions <= 0):
                raise ValueError("Invalid box dimensions")
            mesh = trimesh.creation.box(dimensions)
        elif shape.tag == "cylinder":
            radius, height = float(shape.attrib["radius"]), float(shape.attrib["length"])
            if min(radius, height) <= 0:
                raise ValueError("Invalid cylinder dimensions")
            mesh = trimesh.creation.cylinder(radius=radius, height=height, sections=64)
        else:
            raise ValueError(f"Unsupported training geometry: {shape.tag}")
        origin = collision.find("origin")
        xyz = np.fromstring(origin.get("xyz", "0 0 0") if origin is not None else "0 0 0", sep=" ")
        rpy = np.fromstring(origin.get("rpy", "0 0 0") if origin is not None else "0 0 0", sep=" ")
        if xyz.shape != (3,) or rpy.shape != (3,) or not np.isfinite(np.r_[xyz, rpy]).all():
            raise ValueError("Invalid collision origin")
        matrix = trimesh.transformations.euler_matrix(*rpy)
        matrix[:3, 3] = xyz
        mesh.apply_transform(matrix)
        meshes.append(mesh)
    if not meshes:
        raise ValueError("URDF has no collision primitives")
    return trimesh.util.concatenate(meshes)


def mesh_features(mesh):
    """Same float32 GRAIL distance definition, accelerated with a CPU KD tree."""
    samples, _ = trimesh.sample.sample_surface(mesh, 16384, seed=42)
    points, centroid, radius = normalize_points(samples.astype(np.float32))
    if not np.isfinite(points).all() or radius <= 1e-6:
        raise ValueError("Invalid/degenerate object surface")
    basis = fibonacci_sphere(128).astype(np.float32)
    _, ids = cKDTree(points).query(basis, k=1)
    distances = np.linalg.norm(basis - points[ids], axis=1)
    return np.r_[distances, centroid, radius].astype(np.float32)


def build_bank(urdf_paths, cache_dir):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    bank, entries = [], []
    specification = dict(version=BANK_VERSION, numpy=np.__version__, trimesh=trimesh.__version__)
    for index, source in enumerate(urdf_paths):
        source = Path(source)
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        key = hashlib.sha256((json.dumps(specification, sort_keys=True) + digest).encode()).hexdigest()
        target = cache_dir / f"{key}.npz"
        if target.exists():
            with np.load(target, allow_pickle=False) as data:
                features, bounds = data["features"], data["bounds_m"]
        else:
            mesh = load_primitive_urdf(source)
            features, bounds = mesh_features(mesh), mesh.bounds
            with tempfile.NamedTemporaryFile(dir=cache_dir, suffix=".npz", delete=False) as tmp:
                np.savez_compressed(tmp, features=features, bounds_m=bounds)
                temp_name = tmp.name
            os.replace(temp_name, target)
        if features.shape != (SHAPE_DIM,) or not np.isfinite(features).all() or features[-1] <= 0:
            raise ValueError(f"Invalid cached BPS row {index}: {target}")
        bank.append(features)
        entries.append(dict(index=index, name=source.name, urdf_sha256=digest,
                            cache_key=key, bounds_m=bounds.tolist()))
        if (index+1) % 100 == 0:
            print(f"[BPS] Encoded/cached {index+1}/{len(urdf_paths)} objects", flush=True)
    bank = np.stack(bank)
    manifest = dict(specification=specification, entries=entries,
                    features_sha256=hashlib.sha256(bank.tobytes()).hexdigest(),
                    feature_layout="distances[128], centroid_m[3], radius_m[1]")
    return bank, manifest


def map_features(bank, indices):
    indices = np.asarray(indices)
    if indices.ndim != 1 or indices.dtype.kind not in "iu":
        raise ValueError("Expected integer environment-to-asset indices")
    if np.any(indices < 0) or np.any(indices >= len(bank)):
        raise ValueError("BPS asset index out of range")
    return bank[indices]
