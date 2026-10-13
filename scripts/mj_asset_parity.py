#!/usr/bin/env python3
"""Stage 1 of the mjlab port: compare the MuJoCo model with what PhysX simulated.

CPU only. Reads the Isaac dump (`scripts/dump_isaac_reference.py`), writes the
reference JSON consumed by the mjlab env (joint/body order, limits, defaults),
and reports every mismatch: body masses, COMs, principal inertias, joint
limits, armature, PD gains, effort limits, object inertials for all 1200
objects, and collision-shape counts per link.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('dump', type=Path, help='outputs/isaac_reference_<job>/isaac_reference.npz')
    p.add_argument('--write-reference', action='store_true')
    args = p.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import mujoco
    from dextrah_lab.mj.compat import install
    install()
    from dextrah_lab.mj.robot import build_robot_spec, IMPLICIT_DRIVE_DT
    from dextrah_lab.mj.objects import build_object_bank
    from dextrah_lab.mj.compat import p2p
    ref = np.load(args.dump, allow_pickle=False)
    joints, bodies = list(ref['joint_names']), list(ref['body_names'])
    out = dict(joints=len(joints), bodies=len(bodies), issues=[])
    issue = lambda *a: out['issues'].append(' '.join(str(x) for x in a))
    if args.write_reference:
        dest = Path(__file__).resolve().parents[1]/'dextrah_lab/mj/isaac_reference.json'
        dest.write_text(json.dumps(dict(
            source=str(args.dump), joint_names=joints, body_names=bodies,
            joint_pos_limits=ref['joint_pos_limits'].tolist(), joint_vel_limits=ref['joint_vel_limits'].tolist(),
            default_joint_pos=ref['default_joint_pos'].tolist(), default_root_state=ref['default_root_state'].tolist(),
            joint_effort_limits=ref['joint_effort_limits'].tolist(), joint_stiffness=ref['joint_stiffness'].tolist(),
            joint_damping=ref['joint_damping'].tolist(), joint_armature=ref['joint_armature'].tolist()), indent=1))
        out['reference_written'] = str(dest)
    m = build_robot_spec().compile()
    jid = {m.joint(i).name: i for i in range(m.njnt)}
    bid = {m.body(i).name: i for i in range(m.nbody)}
    if set(joints) != set(jid)-{'floating_base_joint'}:
        issue('joint name sets differ', sorted(set(joints) ^ (set(jid)-{'floating_base_joint'})))
    if set(bodies) != set(bid)-{'world'}:
        issue('body name sets differ', sorted(set(bodies) ^ (set(bid)-{'world'})))
    # Bodies
    worst = dict(mass=0., com=0., inertia=0.)
    for k, name in enumerate(bodies):
        b = bid[name]
        dm = abs(m.body_mass[b]-ref['body_masses'][k])
        dc = np.abs(m.body_ipos[b]-ref['body_coms'][k][:3]).max()
        ie = np.sort(np.linalg.eigvalsh(ref['body_inertias'][k].reshape(3, 3)))
        im = np.sort(m.body_inertia[b])
        di = np.abs(ie-im).max()/max(ie.max(), 1e-9)
        worst['mass'], worst['com'], worst['inertia'] = max(worst['mass'], dm), max(worst['com'], dc), max(worst['inertia'], di)
        if dm > 1e-5 or dc > 1e-5 or di > 1e-3:
            issue(f'body {name}: mass {m.body_mass[b]:.6g} vs {ref["body_masses"][k]:.6g}, com diff {dc:.2e}, inertia rel diff {di:.2e}')
    out['body_worst'] = worst
    # Joints / drives
    act = {m.actuator(i).name: i for i in range(m.nu)}
    for k, name in enumerate(joints):
        j = jid[name]
        lo, hi = m.jnt_range[j]
        if abs(lo-ref['joint_pos_limits'][k, 0]) > 1e-5 or abs(hi-ref['joint_pos_limits'][k, 1]) > 1e-5:
            issue(f'joint {name}: range {lo:.5f},{hi:.5f} vs {ref["joint_pos_limits"][k]}')
        dof = m.jnt_dofadr[j]
        if abs(m.dof_armature[dof]-ref['joint_armature'][k]) > 1e-7:
            issue(f'joint {name}: armature {m.dof_armature[dof]} vs {ref["joint_armature"][k]}')
        a = act[name]
        # kv carries the implicit-stiffness term kp*dt (see robot.py); compare the PhysX kd.
        kp = m.actuator_gainprm[a, 0]
        kd = -m.actuator_biasprm[a, 2]-kp*IMPLICIT_DRIVE_DT
        if abs(kp-ref['joint_stiffness'][k]) > 1e-3*max(1, kp) or abs(kd-ref['joint_damping'][k]) > 1e-3*max(1, kd):
            issue(f'joint {name}: kp/kd {kp:.4f}/{kd:.4f} vs {ref["joint_stiffness"][k]:.4f}/{ref["joint_damping"][k]:.4f}')
        effort = m.actuator_forcerange[a, 1]
        if abs(effort-ref['joint_effort_limits'][k]) > 1e-4*max(1, effort):
            issue(f'joint {name}: effort {effort} vs Isaac {ref["joint_effort_limits"][k]}')
        if abs(ref['joint_friction'][k]) > 0:
            issue(f'joint {name}: Isaac joint friction {ref["joint_friction"][k]} (MuJoCo 0)')
    out['isaac_velocity_limits'] = {n: float(v) for n, v in zip(joints, ref['joint_vel_limits'])}
    out['isaac_dof_max_velocities_physx'] = {n: float(v) for n, v in zip(joints, ref['dof_max_velocities'])}
    # Collision shapes per link
    geoms_per_body = np.bincount(m.geom_bodyid, minlength=m.nbody)
    for name, count in zip(ref['shape_link_names'], ref['shapes_per_link']):
        if name in bid and geoms_per_body[bid[name]] != count:
            issue(f'link {name}: {geoms_per_body[bid[name]]} MuJoCo geoms vs {count} PhysX shapes')
    # Materials
    mats = ref['shape_materials']
    out['robot_shape_friction_values'] = sorted(set(np.round(mats[:, 0], 4).tolist()))
    out['object_friction_values'] = sorted(set(np.round(ref['object_materials'][:, :, 0].ravel(), 4).tolist()))
    out['table_friction_values'] = sorted(set(np.round(ref['table_materials'][:, 0], 4).tolist()))
    # Objects
    gen = p2p('generate_objects')
    import tempfile
    cfg_assets = dict(handle_head_types=('hammer', 'screwdriver', 'marker', 'spatula', 'eraser', 'brush'),
                      num_per_type=100, shuffle=True)
    with tempfile.TemporaryDirectory() as d:
        paths, _ = gen.generate_handle_head_urdfs(out_dir=d, **cfg_assets)
        bank = build_object_bank(paths)
    a = ref['object_asset_index']
    dm = np.abs(bank.mass[a]-ref['object_masses']).max()
    coms, inertias = ref['object_coms'], ref['object_inertias']
    if coms.ndim == 1:
        # Dump of job 1112 kept only element 0 of the (N, 7) COM pose and the
        # (N, 9) inertia: compare COM x and Ixx (frames are axis-aligned).
        dc = np.abs(bank.com[a, 0]-coms).max()
        di = (np.abs(bank.inertia[a, 0]-inertias)/inertias).max()
    else:
        dc = np.abs(bank.com[a]-coms[:, :3]).max()
        ie = np.sort(np.linalg.eigvalsh(inertias.reshape(-1, 3, 3)), -1)
        di = (np.abs(ie-np.sort(bank.inertia[a], -1)).max(-1)/ie.max(-1)).max()
    out['objects'] = dict(envs=len(a), max_mass_diff=float(dm), max_com_diff=float(dc), max_inertia_rel_diff=float(di))
    if dm > 1e-6 or dc > 1e-6 or di > 1e-4:
        issue(f'objects: mass {dm:.2e}, com {dc:.2e}, inertia {di:.2e}')
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
