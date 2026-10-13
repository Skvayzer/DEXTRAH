"""Single-world mujoco_warp replay of captured launch steps: contact forces and variants."""
import sys, glob, numpy as np, mujoco, warp as wp, mujoco_warp as mjw
cap = sys.argv[1]
base = mujoco.MjModel.from_binary_path(f'{cap}/model.mjb')
obj = [b for b in range(base.nbody) if base.body(b).name.startswith('object')][0]
ogeoms = np.nonzero(base.geom_bodyid == obj)[0]
dof = int(base.jnt_dofadr[[j for j in range(base.njnt) if base.jnt_bodyid[j] == obj][0]])
name = lambda g: (base.geom(int(g)).name or base.body(int(base.geom_bodyid[g])).name).replace('robot/', '')
FIELDS = ['qpos', 'qvel', 'ctrl', 'act', 'mocap_pos', 'mocap_quat', 'xfrc_applied', 'qacc_warmstart']
robot_geoms = np.array([base.body(int(b)).name.startswith('robot') for b in base.geom_bodyid])

def build(rec, variant):
    m = mujoco.MjModel.from_binary_path(f'{cap}/model.mjb')
    for k in ['geom_size', 'geom_pos', 'geom_quat', 'geom_rbound']:
        getattr(m, k)[ogeoms] = rec['model_'+k][ogeoms]
    for k in ['body_mass', 'body_ipos', 'body_iquat', 'body_inertia']:
        getattr(m, k)[obj] = rec['model_'+k][obj]
    for g in ogeoms:
        s = m.geom_size[g]
        m.geom_aabb[g] = [0, 0, 0, s[0], s[1], s[2]] if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX else [0, 0, 0, s[0], s[0], s[0]+s[1]]
    if 'solref' in variant: m.geom_solref[robot_geoms] = variant['solref']
    if 'solimp' in variant: m.geom_solimp[robot_geoms] = variant['solimp']
    if 'impratio' in variant: m.opt.impratio = variant['impratio']
    if 'cone' in variant: m.opt.cone = variant['cone']
    if 'no_placeholders' in variant:   # unused slots: no collisions
        for g in ogeoms:
            if m.geom_size[g][0] < 1e-3: m.geom_contype[g] = m.geom_conaffinity[g] = 0
    hand_act = [a for a in range(m.nu) if m.actuator(a).name.startswith('robot/right_') and m.actuator_biasprm[a][2] != 0 and m.actuator_biasprm[a][1] == 0]
    if 'servo_off' in variant:
        for a in hand_act: m.actuator_gainprm[a][0] = 0; m.actuator_biasprm[a][2] = 0
    if 'armature' in variant:
        for a in hand_act: m.dof_armature[m.jnt_dofadr[m.actuator_trnid[a][0]]] = variant['armature']
    if 'integrator' in variant: m.opt.integrator = variant['integrator']
    d = mujoco.MjData(m)
    for f in FIELDS:
        if getattr(d, f).size: getattr(d, f)[:] = rec[f].reshape(getattr(d, f).shape)
    if 'freeze_hand' in variant:   # zero robot velocities (is the launch from finger motion?)
        d.qvel[6:dof] = 0
    return m, d

def step(m, d, report=False, substeps=1):
    if substeps > 1: m.opt.timestep /= substeps
    wm = mjw.put_model(m); wd = mjw.put_data(m, d, nconmax=256, njmax=2000)
    for _ in range(substeps): mjw.step(wm, wd)
    wp.synchronize()
    v = np.linalg.norm(wd.qvel.numpy()[0, dof:dof+3])
    if not report:
        return v, None
    # contact forces of the (pre-integration) solve
    nacon = int(wd.nacon.numpy()[0])
    ids = wp.array(np.arange(nacon, dtype=np.int32), dtype=int)
    force = wp.zeros(nacon, dtype=wp.spatial_vector)
    mjw.contact_force(wm, wd, ids, False, force)
    f = force.numpy(); geom = wd.contact.geom.numpy()[:nacon]; dist = wd.contact.dist.numpy()[:nacon]
    rows = [(name(g0)+'|'+name(g1), dd*1e3, f[i, 0]) for i, ((g0, g1), dd) in enumerate(zip(geom, dist)) if obj in (base.geom_bodyid[g0], base.geom_bodyid[g1])]
    rows.sort(key=lambda r: -abs(r[2]))
    return v, [f'{p} d={dd:.1f}mm fn={fn:.0f}N' for p, dd, fn in rows[:6]]

VARIANTS = {
    'as_env': {},
    'servo_off': dict(servo_off=1),
    'armature_1e-4': dict(armature=1e-4),
    'armature_1e-3': dict(armature=1e-3),
    'substeps4': dict(substeps=4),
    'substeps8': dict(substeps=8),
}
for path in sorted(glob.glob(f'{cap}/eject_*.npz')):
    rec = dict(np.load(path))
    mass = float(rec['model_body_mass'][obj])
    print(f'=== {path.split("/")[-1]} object mass {mass*1e3:.0f} g, env post v {np.linalg.norm(rec["post_qvel"][dof:dof+3]):.2f}')
    for k, var in VARIANTS.items():
        m, d = build(rec, var)
        if k == 'C_implicit':   # MuJoCo C (correct BVH not needed for detection here: compare against C implicitfast)
            m2, d2 = build(rec, {}); 
            try:
                mujoco.mj_step(m, d); vi = np.linalg.norm(d.qvel[dof:dof+3]); mujoco.mj_step(m2, d2); vf = np.linalg.norm(d2.qvel[dof:dof+3])
                print(f'  C implicit {vi:.2f} vs C implicitfast {vf:.2f} (C misses contacts: BVH)'); continue
            except Exception as e: print('C implicit failed', e); continue
        v, rows = step(m, d, report=(k == 'as_env'), substeps=var.get('substeps', 1))
        print(f'  {k:16s} v {v:6.2f} m/s' + (f'   top contacts: {rows}' if rows else ''))
