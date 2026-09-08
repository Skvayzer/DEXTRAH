"""GRAIL-compatible BPS preprocessing for transformed triangle-mesh exports.

No physics, controller, PCA prior, or policy is imported or modified here.
The called GRAIL routines retain their research-only license in grail/LICENSE.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh

from .grail.compute_bps import compute_bps, fibonacci_sphere, normalize_points

GRAIL_COMMIT = "aa31d8242ac79b11545b9e3635f73014a227bdfc"


def load_object_mesh(path: Path) -> trimesh.Trimesh:
    """Flatten a GLB/mesh scene, applying every instance's full node transform.

    Coordinates must already be in meters and in the object's canonical frame.
    Deliberately no guessed unit conversion or principal-axis alignment.
    """
    scene = trimesh.load(str(path), force="scene", process=False)
    meshes = []
    for node in scene.graph.nodes_geometry:
        transform, name = scene.graph[node]
        geometry = scene.geometry[name]
        if not isinstance(geometry, trimesh.Trimesh):
            continue
        mesh = geometry.copy()
        mesh.apply_transform(transform)
        meshes.append(mesh)
    if not meshes:
        raise ValueError(f"No triangle meshes in {path}")
    mesh = trimesh.util.concatenate(meshes)
    if not np.isfinite(mesh.vertices).all() or mesh.area <= 0:
        raise ValueError("Expected a finite mesh with nonzero surface area")
    return mesh


@dataclass(frozen=True)
class BPSRepresentation:
    basis: np.ndarray
    surface_m: np.ndarray
    surface_normalized: np.ndarray
    centroid_m: np.ndarray
    scale_m: float
    distances: np.ndarray
    nearest_indices: np.ndarray

    @property
    def nearest_normalized(self):
        return self.surface_normalized[self.nearest_indices]

    @property
    def basis_m(self):
        return self.centroid_m + self.scale_m * self.basis

    @property
    def nearest_m(self):
        return self.surface_m[self.nearest_indices]


def encode_points(surface_m: np.ndarray, num_basis: int = 128) -> BPSRepresentation:
    """Use upstream centering, isotropic normalization, basis, and distances.

    The centroid is the *sample mean*, not the mass center or bounding-box
    center. Distances are to sampled points, not exact continuous triangles.
    """
    surface_m = np.asarray(surface_m, dtype=np.float32)
    if num_basis < 1:
        raise ValueError("num_basis must be positive")
    if (surface_m.ndim != 2 or surface_m.shape[1] != 3 or len(surface_m) < 2
            or not np.isfinite(surface_m).all()):
        raise ValueError("Expected finite surface points of shape (N>=2, 3)")
    normalized, centroid, scale = normalize_points(surface_m)
    if scale <= 1e-6:
        raise ValueError("Degenerate point cloud: normalization radius <= 1e-6 m")
    basis = fibonacci_sphere(num_basis).astype(np.float32)
    distances = compute_bps(normalized, basis)
    # Chunk the visualization correspondence calculation to bound memory use.
    nearest = np.concatenate([
        np.linalg.norm(basis[start:start + 16, None] - normalized[None], axis=-1).argmin(axis=1)
        for start in range(0, num_basis, 16)
    ])
    return BPSRepresentation(basis, surface_m, normalized, centroid,
                             float(scale), distances, nearest)


def encode_mesh(mesh: trimesh.Trimesh, num_basis: int = 128,
                surface_count: int = 16384, seed: int = 42) -> BPSRepresentation:
    """Always area-sample faces, including on low-vertex procedural primitives.

    This deliberately bypasses GRAIL process_usd's vertex-only shortcut. The
    local seeded sampler does not change global NumPy RNG state.
    """
    if surface_count < 2:
        raise ValueError("surface_count must be at least 2")
    surface, _ = trimesh.sample.sample_surface(mesh, surface_count, seed=seed)
    return encode_points(surface, num_basis)


def validate_representation(mesh: trimesh.Trimesh, bps: BPSRepresentation) -> dict:
    """Check correspondence/units and quantify the sampling approximation."""
    links = np.linalg.norm(bps.basis - bps.nearest_normalized, axis=1)
    links_m = np.linalg.norm(bps.basis_m - bps.nearest_m, axis=1)
    np.testing.assert_allclose(links, bps.distances, atol=1e-7, rtol=1e-6)
    np.testing.assert_allclose(links_m, bps.distances * bps.scale_m, atol=1e-7)
    np.testing.assert_allclose(np.linalg.norm(bps.basis, axis=1), 1., atol=1e-7)
    # Exact point-to-triangle distances, without the optional rtree dependency.
    # Chunk queries because larger user-supplied meshes may contain many faces.
    exact = np.concatenate([
        trimesh.proximity.closest_point_naive(mesh, points)[1]
        for points in np.array_split(bps.basis_m, len(bps.basis))
    ])
    error_m = links_m - exact
    if error_m.min() < -1e-6:
        raise ValueError("A sampled nearest point appears closer than the actual mesh")
    # Check all 128 rendered endpoints actually lie on the loaded mesh.
    endpoint_error = np.concatenate([
        trimesh.proximity.closest_point_naive(mesh, points)[1]
        for points in np.array_split(bps.nearest_m, len(bps.basis))
    ])
    if endpoint_error.max() > 1e-6:
        raise ValueError("Nearest-point marker is not on the mesh")
    return {
        "basis_count": len(bps.basis),
        "surface_sample_count": len(bps.surface_m),
        "mesh_vertices": len(mesh.vertices),
        "mesh_triangles": len(mesh.faces),
        "mesh_extents_m": mesh.extents.tolist(),
        "centroid_m": bps.centroid_m.tolist(),
        "normalization_radius_m": bps.scale_m,
        "descriptor_range": [float(bps.distances.min()), float(bps.distances.max())],
        "max_endpoint_off_mesh_m": float(endpoint_error.max()),
        "max_sampling_distance_error_m": float(error_m.max()),
        "mean_sampling_distance_error_m": float(error_m.mean()),
        "max_link_descriptor_error": float(np.abs(links - bps.distances).max()),
    }
