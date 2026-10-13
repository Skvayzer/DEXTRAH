"""Ground truth for disputed contacts: do the object primitives overlap the finger convex hulls?"""
import sys, glob, numpy as np, mujoco
from scipy.spatial import ConvexHull
cap = sys.argv[1]
base = mujoco.MjModel.from_binary_path(f'{cap}/model.mjb')
obj = [b for b in range(base.nbody) if base.body(b).name.startswith('object')][0]
ogeoms = np.nonzero(base.geom_bodyid == obj)[0]
FIELDS = ['qpos', 'qvel', 'ctrl', 'act', 'mocap_pos', 'mocap_quat', 'xfrc_applied', 'qacc_warmstart']
gname = lambda m, g: (m.geom(int(g)).name or m.body(int(m.geom_bodyid[g])).name).replace('robot/', '')
rng = np.random.default_rng(0)

def prim_points(m, d, g, k=6000):
    s = m.geom_size[g]
    if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
        p = rng.uniform(-1, 1, (k, 3))*s
    else:   # capsule: segment along z, radius s[0], half length s[1]
        r = s[0]*np.cbrt(rng.uniform(0, 1, k)); dirs = rng.normal(size=(k, 3)); dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        p = dirs*r[:, None]; p[:, 2] += rng.uniform(-s[1], s[1], k)
    return d.geom_xpos[g]+p@d.geom_xmat[g].reshape(3, 3).T

for path in sorted(glob.glob(f'{cap}/eject_*.npz'))[:12]:
    rec = dict(np.load(path))
    m = mujoco.MjModel.from_binary_path(f'{cap}/model.mjb')
    for k in ['geom_size', 'geom_pos', 'geom_quat', 'geom_rbound']:
        getattr(m, k)[ogeoms] = rec['model_'+k][ogeoms]
    d = mujoco.MjData(m)
    for f in FIELDS:
        if getattr(d, f).size: getattr(d, f)[:] = rec[f].reshape(getattr(d, f).shape)
    mujoco.mj_kinematics(m, d)
    out = []
    for g in ogeoms:
        if m.geom_size[g][0] < 1e-3: continue   # placeholder slot
        pts = prim_points(m, d, g)
        for h in range(m.ngeom):
            if m.geom_type[h] != mujoco.mjtGeom.mjGEOM_MESH or 'right_' not in m.body(int(m.geom_bodyid[h])).name: continue
            mid = m.geom_dataid[h]
            v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid]+m.mesh_vertnum[mid]]
            vw = d.geom_xpos[h]+v@d.geom_xmat[h].reshape(3, 3).T
            if np.linalg.norm(vw.mean(0)-d.geom_xpos[g]) > 0.4: continue
            hull = ConvexHull(vw)
            sd = pts@hull.equations[:, :3].T+hull.equations[:, 3]      # <=0 inside every face
            inside = (sd <= 0).all(1)
            fromto = np.zeros(6)
            dist = mujoco.mj_geomDistance(m, d, int(g), int(h), 0.05, fromto)
            if inside.any() or dist < 0.002:
                depth = float((-sd[inside].max(1)).max()) if inside.any() else 0.
                out.append(f'{gname(m, g)}|{gname(m, h)}: sampled-inside {inside.mean()*100:.1f}% depth>={depth*1e3:.1f}mm, C geomDistance {dist*1e3:.1f}mm')
    print('===', path.split('/')[-1], out if out else 'no overlap and no hull within 2 mm')
