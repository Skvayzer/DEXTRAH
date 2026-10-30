#!/usr/bin/env python3
"""Upright cylinder (r 2.75 cm, L 14 cm, 0.3 kg) on the table box in MuJoCo Warp: true cylinder geom vs a 24-gon
prism mesh, 4 worlds, 2 s at training options. Usage: python cylinder_warp_contact_check.py [cpu|cuda:0].
2026-10-30 (Warp CPU): cylinder 1 contact per world, max tilt 0.65 deg, |v| 0.087 at 2 s, sinks 1.0 mm; prism 4
contacts, 0.12 deg, |v| 0.014, 0.6 mm."""
import numpy as np, mujoco, warp as wp, mujoco_warp as mjw, sys
def model(kind, r=.0275, L=.14, n=24):
    s = mujoco.MjSpec()
    o = s.option; o.timestep = 1/120; o.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    o.cone = mujoco.mjtCone.mjCONE_ELLIPTIC; o.impratio = 10; o.iterations = 20; o.ls_iterations = 20
    t = s.worldbody.add_geom(name='table', type=mujoco.mjtGeom.mjGEOM_BOX, size=[.24, .2, .15], pos=[0, 0, .65])
    b = s.worldbody.add_body(name='obj', pos=[0, 0, .80+L/2+.0005])
    b.add_freejoint()
    if kind == 'cylinder':
        g = b.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[r, L/2, 0])
    else:
        a = np.linspace(0, 2*np.pi, n, endpoint=False)
        v = np.concatenate([np.stack([r*np.cos(a), r*np.sin(a), np.full(n, z)], 1) for z in (-L/2, L/2)])
        s.add_mesh(name='prism', uservert=v.reshape(-1).tolist())
        g = b.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname='prism')
    g.mass = .3; g.condim = 4; g.friction = [.5, .005, 1e-4]
    t.condim = 4
    return s.compile()
dev = sys.argv[1] if len(sys.argv) > 1 else 'cpu'
for kind in ('cylinder', 'prism'):
    m = model(kind)
    with wp.ScopedDevice(dev):
        wm = mjw.put_model(m); wd = mjw.make_data(m, nworld=4, nconmax=16, njmax=64)
        q = wp.to_torch(wd.qpos)
        z0 = q[:, 2].clone()
        tilt, ncon = [], []
        for k in range(240):
            mjw.step(wm, wd)
            qq = q.cpu().numpy()
            R = np.array([np.reshape(mujoco.mju_quat2Mat(np.zeros(9), w[3:7].astype(float)) or 0, ()) for w in qq])
            zz = []
            for w in qq:
                M = np.zeros(9); mujoco.mju_quat2Mat(M, w[3:7].astype(float)); zz.append(np.degrees(np.arccos(np.clip(M[8], -1, 1))))
            tilt.append(max(zz))
            ncon.append(int(wp.to_torch(wd.nacon).cpu().numpy()[0]))
        v = wp.to_torch(wd.qvel).cpu().numpy()
    print(f'{kind}: contacts (all 4 worlds) at t=0.1/1/2 s: {ncon[11]}/{ncon[119]}/{ncon[-1]} | max tilt over 2 s {max(tilt):.3f} deg | final |v| {np.abs(v).max():.2e} | dz {float(q[0,2]-z0[0])*1000:.2f} mm')
