#!/usr/bin/env python3
"""Replay a captured non-finite MuJoCo Warp world (dextrah_lab/mj/nan_capture.py).

1. Rebuild the world as a single-world mujoco_warp model: the compiled
   MjModel (model.mjb) with every captured Model array restored (per-world
   expanded fields from this world, shared fields as they were on the GPU).
2. Fidelity: step each buffered pre-step input and compare with the next
   buffered input (resets or the task's object spin clamp between steps
   show up as differences; qpos of the robot is untouched by both).
3. Failing step, stage by stage as mujoco_warp.forward/step run it:
   kinematics, inertia, collision, constraint assembly, transmission,
   velocity, actuation, smooth acceleration, constraint solver, integration.
   Reports the first stage that leaves non-finite values and the contacts /
   constraint rows involved.
4. The same state stepped by C MuJoCo (float64) for comparison.

Writes replay.json next to the dump.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def resolve(root, dotted):
    obj = root
    parts = dotted.split('.')
    for p in parts[:-1]:
        obj = getattr(obj, p)
    return obj, parts[-1]


def nonfinite_fields(d, wp, mjw_types):
    """Non-finite entries in a single-world Data (active contacts / rows only)."""
    import torch
    nacon = int(wp.to_torch(d.nacon)[0])
    nefc = int(wp.to_torch(d.nefc)[0])
    bad = {}
    for name, value in walk(d):
        if not isinstance(value, wp.array) or value.size == 0:
            continue
        try:
            t = wp.to_torch(value)
        except Exception:
            continue
        if not torch.is_floating_point(t):
            continue
        if name.startswith('contact.'):
            t = t[:nacon]
        elif name.startswith('efc.') and t.ndim >= 2 and t.shape[0] == 1:
            t = t[:, :min(nefc, d.njmax)]
        n = int((~torch.isfinite(t)).sum())
        if n:
            bad[name] = n
    return bad


def walk(obj, prefix=''):
    import dataclasses
    for f in dataclasses.fields(obj):
        value = getattr(obj, f.name, None)
        if dataclasses.is_dataclass(value):
            yield from walk(value, prefix+f.name+'.')
        else:
            yield prefix+f.name, value


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('dump', type=Path, help='nan_capture/nan_step<k> directory')
    p.add_argument('--world', type=int, help='default: first captured world')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--nconmax', type=int, default=512, help='single-world contact pool (no drops)')
    p.add_argument('--njmax', type=int, help='default: the run value')
    args = p.parse_args()
    import mujoco
    import torch
    import warp as wp
    import mujoco_warp as mjw
    from mujoco_warp._src import (smooth, collision_driver, constraint, forward, solver, types as mjw_types)
    wp.init()
    dump = args.dump
    summary = json.loads((dump/'summary.json').read_text())
    meta = json.loads((dump/'model_scalars.json').read_text())
    info = json.loads((dump.parent/'capture_info.json').read_text())
    inputs = np.load(dump/'inputs.npz')
    world = args.world if args.world is not None else meta['worlds'][0]
    k_model = meta['worlds'].index(world)
    k_in = list(inputs['worlds']).index(world)
    steps = list(inputs['steps'])
    out = dict(dump=str(dump), world=world, ring_steps=steps, capture=summary['per_world'].get(str(world)))
    njmax = args.njmax or info['njmax']

    mjm = mujoco.MjModel.from_binary_path(str(dump/'model.mjb'))
    saved = np.load(dump/'model.npz')
    with wp.ScopedDevice(args.device):
        m = mjw.put_model(mjm)
        restored, changed, mismatch = 0, [], []
        for name in saved.files:
            if name.endswith('#full'):
                continue
            try:
                obj, attr = resolve(m, name)
                cur = getattr(obj, attr)
            except AttributeError:
                mismatch.append([name, 'missing'])
                continue
            if not isinstance(cur, wp.array):
                continue
            val = saved[name]
            t = wp.to_torch(cur)
            if name in meta['expanded']:
                # Batched field expanded per world (single-world shape (1, ...)),
                # or a model dimension equal to nworld (whole array saved).
                if t.shape[0] == 1 and tuple(t.shape[1:]) == val.shape[1:]:
                    val = val[k_model][None]
                elif name+'#full' in saved.files:
                    val = saved[name+'#full']
                else:
                    mismatch.append([name, list(t.shape), list(val.shape), 'per-world candidate'])
                    continue
            if tuple(t.shape) != val.shape:
                if val.shape[0] == 1 and tuple(t.shape[1:]) == val.shape[1:]:
                    pass
                else:
                    mismatch.append([name, list(t.shape), list(val.shape)])
                    continue
            new = torch.as_tensor(val, device=t.device).to(t.dtype).reshape(t.shape)
            if not torch.equal(t, new):
                changed.append(name)
                t.copy_(new)
            restored += 1
        scal_diff = {}
        for name, value in meta['scalars'].items():
            try:
                obj, attr = resolve(m, name)
                cur = getattr(obj, attr)
            except AttributeError:
                continue
            cur_v = int(cur.value) if hasattr(cur, 'value') else cur
            if isinstance(cur_v, (bool, int, float, str)) and cur_v != value:
                scal_diff[name] = [cur_v, value]
                if name.startswith('opt.'):
                    setattr(obj, attr, type(cur)(value))
        out['model_restore'] = dict(arrays=restored, changed_from_mjb=changed, mismatched=mismatch,
                                    scalar_differences=scal_diff)

        def fresh_data():
            return mjw.put_data(mjm, mujoco.MjData(mjm), nworld=1, nconmax=args.nconmax, njmax=njmax)

        def load(d, slot):
            for f in inputs.files:
                if f in ('worlds', 'steps'):
                    continue
                arr = getattr(d, f, None)
                if arr is None or arr.size == 0:
                    continue
                t = wp.to_torch(arr)
                t.copy_(torch.as_tensor(inputs[f][slot, k_in], device=t.device).to(t.dtype).reshape(t.shape))

        def state(d, f):
            return wp.to_torch(getattr(d, f))[0].double().cpu().numpy()

        # 2. Fidelity over the buffered steps.
        fid = []
        for s in range(len(steps)-1):
            d = fresh_data()
            load(d, s)
            mjw.step(m, d)
            wp.synchronize()
            nxt = inputs['qpos'][s+1, k_in].astype(np.float64)
            q = state(d, 'qpos')
            fid.append(dict(step=int(steps[s]), next_step=int(steps[s+1]), finite=bool(np.isfinite(q).all()),
                            qpos_max_abs_diff=float(np.nanmax(np.abs(q-nxt))),
                            qpos_argmax=int(np.nanargmax(np.abs(q-nxt)))))
        out['fidelity'] = fid

        # 3. Failing step: full step, then stage by stage.
        last = len(steps)-1
        d = fresh_data()
        load(d, last)
        mjw.step(m, d)
        wp.synchronize()
        out['full_step'] = dict(nonfinite=nonfinite_fields(d, wp, mjw_types),
                                nacon=int(wp.to_torch(d.nacon)[0]), nefc=int(wp.to_torch(d.nefc)[0]),
                                solver_niter=int(wp.to_torch(d.solver_niter)[0]),
                                overflow=int(wp.to_torch(d.overflow)[0]))
        d = fresh_data()
        load(d, last)
        sleep = bool(m.opt.enableflags & mjw_types.EnableBit.SLEEP)
        stages = [
            ('kinematics', lambda: forward.fwd_kinematics(m, d)),
            ('crb', lambda: (smooth.crb(m, d), smooth.tendon_armature(m, d))),
            ('collision', lambda: collision_driver.collision(m, d)),
            ('make_constraint', lambda: constraint.make_constraint(m, d)),
            ('transmission', lambda: smooth.transmission(m, d)),
            ('velocity', lambda: forward.fwd_velocity(m, d)),
            ('actuation', lambda: forward.fwd_actuation(m, d)),
            ('acceleration', lambda: forward.fwd_acceleration(m, d, factorize=True)),
            ('solve', lambda: solver.solve(m, d)),
            ('integrate', lambda: forward.implicit(m, d) if m.opt.integrator in (
                mjw_types.IntegratorType.IMPLICITFAST, mjw_types.IntegratorType.IMPLICIT) else forward.euler(m, d)),
        ]
        staged, first = [], None
        seen = set(nonfinite_fields(d, wp, mjw_types))   # inputs already non-finite
        staged.append(dict(stage='inputs', nonfinite=sorted(seen)))
        for name, fn in stages:
            if sleep:
                staged.append(dict(stage=name, note='sleep enabled: staged replay not exact'))
            fn()
            wp.synchronize()
            bad = nonfinite_fields(d, wp, mjw_types)
            new = {k: v for k, v in bad.items() if k not in seen}
            seen |= set(bad)
            staged.append(dict(stage=name, new_nonfinite=new))
            if new and first is None:
                first = name
        out['stages'] = staged
        out['first_nonfinite_stage'] = first
        # Contacts and constraint rows of the failing step (after the stages).
        nacon = int(wp.to_torch(d.nacon)[0])
        c = {f: wp.to_torch(getattr(d.contact, f))[:nacon].cpu().numpy()
             for f in ('dist', 'pos', 'frame', 'geom', 'dim', 'friction', 'efc_address', 'includemargin')}
        gx = wp.to_torch(d.geom_xpos)[0].cpu().numpy()
        gm = wp.to_torch(d.geom_xmat)[0].cpu().numpy()
        gsize = wp.to_torch(m.geom_size)[0].cpu().numpy()
        contacts = []
        for i in range(nacon):
            g = [int(x) for x in c['geom'][i]]
            contacts.append(dict(
                i=i, geom=g, names=[mjm.geom(x).name or f'geom{x}' for x in g],
                types=[mujoco.mjtGeom(int(mjm.geom_type[x])).name for x in g],
                bodies=[mjm.body(int(mjm.geom_bodyid[x])).name for x in g],
                size=[gsize[x].tolist() for x in g],
                dist=float(c['dist'][i]), pos=c['pos'][i].tolist(), normal=c['frame'][i].reshape(3, 3)[0].tolist(),
                dim=int(c['dim'][i]), efc=[int(x) for x in np.ravel(c['efc_address'][i])[:int(c['dim'][i])]],
                finite=bool(np.isfinite(c['dist'][i]) and np.isfinite(c['pos'][i]).all() and np.isfinite(c['frame'][i]).all()),
                geom_xpos=[gx[x].tolist() for x in g], geom_xmat=[gm[x].reshape(-1).tolist() for x in g]))
        out['contacts'] = contacts
        nefc = min(int(wp.to_torch(d.nefc)[0]), njmax)
        rows = {}
        for f in ('type', 'id', 'pos', 'D', 'aref', 'force', 'vel', 'margin'):
            arr = getattr(d.efc, f, None)
            if arr is not None and arr.size:
                rows[f] = wp.to_torch(arr)[0, :nefc].double().cpu().numpy().tolist()
        out['efc_rows'] = rows
        for f in ('qacc_smooth', 'qacc', 'qfrc_constraint', 'qfrc_actuator', 'qfrc_bias', 'qvel', 'qpos'):
            out.setdefault('vectors', {})[f] = wp.to_torch(getattr(d, f))[0].double().cpu().numpy().tolist()

    # 4. C MuJoCo (float64) on the same state, per-world fields applied.
    try:
        for name in meta['expanded']:
            if hasattr(mjm, name) and name in saved.files:
                target = getattr(mjm, name)
                val = saved[name][k_model]
                if target.size == val.size:
                    target[:] = val.reshape(target.shape)
        mjd = mujoco.MjData(mjm)
        for f in inputs.files:
            if f in ('worlds', 'steps') or not hasattr(mjd, f):
                continue
            target = getattr(mjd, f)
            val = inputs[f][last, k_in]
            if np.size(target) == np.size(val):
                if np.ndim(target) == 0:
                    setattr(mjd, f, float(val))
                else:
                    target[:] = np.asarray(val).reshape(np.shape(target))
        # Stage by stage: mj_step/mj_forward reset the whole state on a bad
        # qacc (mjWARN_BADQACC), which would hide the result.
        c_stages, c_first = [], None
        def c_bad():
            names = ('qacc_smooth', 'qfrc_smooth', 'qacc', 'qfrc_constraint', 'efc_force', 'efc_J', 'efc_D',
                     'efc_aref', 'actuator_force', 'qfrc_actuator', 'qfrc_bias', 'xpos', 'xquat', 'cinert', 'qM')
            out_ = [n for n in names if np.size(getattr(mjd, n, ())) and not np.isfinite(getattr(mjd, n)).all()]
            if mjd.ncon and not (np.isfinite(mjd.contact.dist).all() and np.isfinite(mjd.contact.frame).all()):
                out_.append('contact')
            return out_
        c_seen = set(c_bad())
        for name, fn in (('fwdPosition', mujoco.mj_fwdPosition), ('fwdVelocity', mujoco.mj_fwdVelocity),
                         ('fwdActuation', mujoco.mj_fwdActuation), ('fwdAcceleration', mujoco.mj_fwdAcceleration),
                         ('fwdConstraint', mujoco.mj_fwdConstraint)):
            fn(mjm, mjd)
            bad = set(c_bad())
            c_stages.append(dict(stage=name, new_nonfinite=sorted(bad-c_seen)))
            if bad-c_seen and c_first is None:
                c_first = name
            c_seen |= bad
            if name == 'fwdPosition':
                c_contacts = [dict(geom=[int(mjd.contact.geom1[i]), int(mjd.contact.geom2[i])],
                                   dist=float(mjd.contact.dist[i])) for i in range(mjd.ncon)]
        out['c_mujoco'] = dict(first_nonfinite_stage=c_first, stages=c_stages, ncon=len(c_contacts),
                               contacts=c_contacts, nefc=int(mjd.nefc),
                               qacc_finite=bool(np.isfinite(mjd.qacc).all()),
                               qacc_max_abs=float(np.nanmax(np.abs(mjd.qacc))),
                               warnings={mujoco.mjtWarning(i).name: int(mjd.warning[i].number)
                                         for i in range(len(mjd.warning)) if mjd.warning[i].number})
    except Exception as e:
        out['c_mujoco'] = dict(error=repr(e))
    path = dump/f'replay_w{world}.json'
    path.write_text(json.dumps(out, indent=1, default=float))
    brief = {k: out[k] for k in ('world', 'first_nonfinite_stage', 'full_step', 'fidelity', 'model_restore')}
    brief['c_mujoco'] = {k: v for k, v in out['c_mujoco'].items() if k != 'contacts'}
    brief['nonfinite_contacts'] = [x for x in out['contacts'] if not x['finite']]
    print(json.dumps(brief, indent=1, default=float))


if __name__ == '__main__':
    main()
