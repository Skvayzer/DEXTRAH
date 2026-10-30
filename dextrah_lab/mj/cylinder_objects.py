"""Flat-ended cylinders as an opt-in object set of the mjlab manipulation env (cfg.assets.object_set = 'cylinders').

The procedural tools (play2perfect generate_handle_head_urdfs, pinned) simulate URDF cylinders as capsules; these
objects are true MuJoCo cylinders in a fifth object slot (SceneParams.cylinder_slot, objects.CYLINDER_SLOT). URDF: one
link `object_root`, one collision named "cylinder" with its axis along the link x axis (as the tools' handles), the
inertial at the centre (solid cylinder of the sampled mass). Own random generator: the global numpy state is left
alone (the P2P generator reseeds it). File names keep the family at stem.split('_')[1] == 'cylinder'.
"""
from pathlib import Path

import numpy as np

AXIS_RPY = '0 -1.5707963267948966 0'          # URDF cylinder axis (local z) -> link x


def cylinder_urdf(name, radius, length, mass):
    ixx = .5*mass*radius**2
    iyy = mass*(3*radius**2+length**2)/12
    geom = f'<origin xyz="0 0 0" rpy="{AXIS_RPY}"/><geometry><cylinder radius="{radius:.6f}" length="{length:.6f}"/></geometry>'
    return (f'<?xml version="1.0"?>\n<robot name="{name}">\n  <link name="object_root">\n'
            f'    <inertial><origin xyz="0 0 0" rpy="0 0 0"/><mass value="{mass:.6f}"/>'
            f'<inertia ixx="{ixx:.9g}" ixy="0" ixz="0" iyy="{iyy:.9g}" iyz="0" izz="{iyy:.9g}"/></inertial>\n'
            f'    <visual name="cylinder">{geom}</visual>\n'
            f'    <collision name="cylinder">{geom}</collision>\n  </link>\n</robot>\n')


def generate_cylinder_urdfs(out_dir, radii=(.025, .0275, .03), length=(.08, .20), mass=(.1, .5), per_radius=100,
                            seed=0, object_base_size=.04):
    """Returns (paths, scales (n, 3) = (L, 2r, 2r)/object_base_size, meta (n, 3) = (r, L, m)), radius-major order."""
    rng = np.random.default_rng(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths, scales, meta = [], [], []
    for r in radii:
        for _ in range(per_radius):
            L, m = round(float(rng.uniform(*length)), 6), round(float(rng.uniform(*mass)), 6)   # as written
            i = len(paths)
            name = f'{i:03d}_cylinder_r{1000*r:.2f}mm_L{1000*L:.1f}mm_m{1000*m:.0f}g'.replace('.', 'p')
            path = out/f'{name}.urdf'
            path.write_text(cylinder_urdf(name, r, L, m))
            paths.append(path)
            scales.append(np.array([L, 2*r, 2*r])/object_base_size)
            meta.append((r, L, m))
    return paths, np.array(scales), np.array(meta)
