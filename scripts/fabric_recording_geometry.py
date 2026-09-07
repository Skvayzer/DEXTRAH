"""Independent NumPy checks for the recorded training sphere model."""
import numpy as np
import trimesh


def rotation(quaternion):
    return trimesh.transformations.quaternion_matrix(quaternion)[:3, :3]


def verify_sphere_centers(geometry, body_names, body_pos, body_quat, root_pose,
                          moving, fixed, tolerance=2.e-6):
    ids = {name: i for i, name in enumerate(body_names)}
    expected_moving = np.stack([
        body_pos[ids[s["link_name"]]] + rotation(body_quat[ids[s["link_name"]]]) @ s["offset"]
        for s in geometry["dynamic"]])
    expected_fixed = np.array([s["center"] for s in geometry["fixed"]]) @ rotation(root_pose[3:]).T + root_pose[:3]
    error = max(float(np.linalg.norm(expected_moving - moving, axis=1).max()),
                float(np.linalg.norm(expected_fixed - fixed, axis=1).max()))
    if error > tolerance:
        raise RuntimeError(f"Recorded fabric spheres do not match measured links/root: {error} m")
    return error


def sphere_clearances(geometry, moving, fixed, table_height):
    """Reconstruct the exact body/table/self ordering, without torch/PhysX."""
    specs = geometry["dynamic"]
    radii = np.array([s["radius"] for s in specs])
    fixed_radii = np.array([s["radius"] for s in geometry["fixed"]])
    moving_min = np.full(len(moving), np.inf)
    fixed_min = np.full(len(fixed), np.inf)
    parts = []
    body_ids = [i for i, s in enumerate(specs) if s["avoid_body"]]
    body = np.linalg.norm(moving[body_ids, None] - fixed[None], axis=-1)
    body -= radii[body_ids, None] + fixed_radii[None]
    parts.append(body.ravel())
    moving_min[body_ids] = body.min(axis=1)
    fixed_min = np.minimum(fixed_min, body.min(axis=0))
    table_ids = [i for i, s in enumerate(specs) if s["avoid_table"]]
    table = moving[table_ids, 2] - radii[table_ids] - table_height
    parts.append(table)
    moving_min[table_ids] = np.minimum(moving_min[table_ids], table)
    if geometry["include_self_collision"]:
        pairs = np.array(geometry["self_pairs"], dtype=int)
        a, b = pairs[:, 0], pairs[:, 1]
        gaps = np.linalg.norm(moving[a] - moving[b], axis=-1) - radii[a] - radii[b]
        parts.append(gaps)
        np.minimum.at(moving_min, a, gaps)
        np.minimum.at(moving_min, b, gaps)
    return np.concatenate(parts), moving_min, fixed_min
