#!/usr/bin/env python3
"""Trace mujoco_warp's Newton solver, kernel group by kernel group, on a captured NaN step.

The failing step of run 1213 (world 11650) goes NaN in about half of the
repeated single-world solves of identical inputs (scripts/mj_nan_batch_replay.py),
always after 4 iterations. This runs that step R times as mujoco_warp.step
does (forward up to the smooth acceleration, then solver.solve), but drives the
solver loop itself: init_context, then per iteration _linesearch,
_update_constraint, _update_gradient (Hessian + Cholesky + search) and
_solve_done, snapshotting the solver context after each. For every NaN run it
reports the first (iteration, phase, fields) that went non-finite, and, for the
Hessian assembled in that iteration, its float64 eigenvalues and the pivots of
a float32 upper Cholesky (as Warp's tile_cholesky computes it, no pivot
guard), mapped to joints. Finite runs give the same numbers for comparison.
Writes solver_trace.json (+ .npz of the first NaN and first finite run).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mj_nan_replay import load_inputs, restore_model  # noqa: E402

CTX_FIELDS = ('Jaref', 'grad', 'Mgrad', 'search', 'jv', 'quad', 'alpha', 'grad_dot', 'newton_decrement',
              'improvement', 'search_dot', 'h', 'hfactor', 'done')


def cholesky_pivots_f32(h):
    """Upper Cholesky in float32, the order of Warp's scalar_cholesky_impl; returns pivots before sqrt."""
    a = h.astype(np.float32)
    n = a.shape[0]
    u = np.zeros_like(a)
    piv = np.zeros(n, np.float32)
    for j in range(n):
        s = a[j, j]
        for k in range(j):
            s = np.float32(s - u[k, j]*u[k, j])
        piv[j] = s
        if not s > 0:
            return piv[:j+1], j
        r = np.float32(np.sqrt(s))
        u[j, j] = r
        for i in range(j+1, n):
            t = a[j, i]
            for k in range(j):
                t = np.float32(t - u[k, i]*u[k, j])
            u[j, i] = np.float32(t/r)
    return piv, None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('dump', type=Path)
    p.add_argument('--world', type=int)
    p.add_argument('--repeats', type=int, default=30)
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()
    import mujoco
    import torch
    import warp as wp
    import mujoco_warp as mjw
    from mujoco_warp._src import forward, solver, types as mjw_types
    wp.init()
    dump = args.dump
    meta = json.loads((dump/'model_scalars.json').read_text())
    info = json.loads((dump.parent/'capture_info.json').read_text())
    inputs = np.load(dump/'inputs.npz')
    world = args.world if args.world is not None else meta['worlds'][0]
    k_model, k_in = meta['worlds'].index(world), list(inputs['worlds']).index(world)
    slot = len(inputs['steps'])-1
    mjm = mujoco.MjModel.from_binary_path(str(dump/'model.mjb'))
    saved = np.load(dump/'model.npz')
    nv = mjm.nv
    dof_name = [f'{mjm.joint(int(mjm.dof_jntid[i])).name}[{i - int(mjm.jnt_dofadr[mjm.dof_jntid[i]])}]'
                for i in range(nv)]
    runs, keep = [], {}
    with wp.ScopedDevice(args.device):
        m, restore = restore_model(mjm, saved, meta, k_model)
        out = dict(dump=str(dump), world=world, nv=nv, nv_pad=m.nv_pad, is_sparse=bool(m.is_sparse),
                   iterations=int(m.opt.iterations), cone=int(m.opt.cone), solver=int(m.opt.solver),
                   model_restore=restore)
        for r in range(args.repeats):
            d = mjw.put_data(mjm, mujoco.MjData(mjm), nworld=1, nconmax=512, njmax=info['njmax'])
            load_inputs(d, inputs, slot, k_in)
            forward.fwd_position(m, d, factorize=False)
            forward.fwd_velocity(m, d)
            forward.fwd_actuation(m, d)
            forward.fwd_acceleration(m, d, factorize=True)
            ctx = solver._create_solver_context(m, d)
            warmstart = not (m.opt.disableflags & mjw_types.DisableBit.WARMSTART)
            wp.launch(solver._solve_init_dof(warmstart, m.is_sparse), dim=(d.nworld, m.nv),
                      inputs=[d.nefc, d.qacc_warmstart, d.qacc_smooth], outputs=[d.qacc, d.qfrc_constraint])
            nefc = int(wp.to_torch(d.nefc)[0])
            snaps = []

            def snap(it, phase):
                wp.synchronize()
                s = dict(it=it, phase=phase)
                for f in CTX_FIELDS:
                    a = getattr(ctx, f, None)
                    if a is not None and a.size:
                        t = wp.to_torch(a)[0]
                        s[f] = t.double().cpu().numpy() if torch.is_floating_point(t) else t.cpu().numpy()
                for f in ('force', 'state'):
                    s['efc.'+f] = wp.to_torch(getattr(d.efc, f))[0, :nefc].cpu().numpy()
                s['qacc'] = wp.to_torch(d.qacc)[0].double().cpu().numpy()
                s['nonfinite'] = sorted(k for k, v in s.items() if isinstance(v, np.ndarray)
                                        and np.issubdtype(v.dtype, np.floating) and not np.isfinite(v).all())
                snaps.append(s)
                return s

            solver.init_context(m, d, ctx, grad=True)
            snap(0, 'init_context')
            nsolving = wp.full(shape=(1,), value=1, dtype=int)
            for it in range(1, int(m.opt.iterations)+1):
                solver._linesearch(m, d, ctx)
                snap(it, 'linesearch')
                solver._update_constraint(m, d, ctx, track_changes=False, stable_fast=False)
                snap(it, 'update_constraint')
                solver._update_gradient(m, d, ctx)
                snap(it, 'update_gradient')
                wp.launch(solver._solve_done, dim=d.nworld,
                          inputs=[m.nv, m.opt.tolerance, m.opt.iterations, m.stat.meaninertia, ctx.grad_dot,
                                  ctx.newton_decrement, ctx.improvement, ctx.done],
                          outputs=[d.solver_niter, nsolving, ctx.done])
                if bool(wp.to_torch(ctx.done)[0]):
                    break
            first = next((s for s in snaps if s['nonfinite']), None)
            nan = first is not None
            rec = dict(run=r, nan=nan, iterations=snaps[-1]['it'],
                       first_nonfinite=None if not nan else dict(it=first['it'], phase=first['phase'],
                                                                fields=first['nonfinite']))
            # Hessian of every update_gradient (and init): spectrum + float32 pivots.
            hs = []
            for s in snaps:
                if s['phase'] not in ('init_context', 'update_gradient') or 'h' not in s:
                    continue
                h = s['h'][:nv, :nv]
                if not np.triu(h, 1).any():
                    hsym = np.tril(h) + np.tril(h, -1).T       # lower triangle stored
                elif not np.tril(h, -1).any():
                    hsym = np.triu(h) + np.triu(h, 1).T        # upper triangle stored
                else:
                    hsym = (h + h.T)/2
                entry = dict(it=s['it'], h_finite=bool(np.isfinite(hsym).all()))
                if entry['h_finite']:
                    ev = np.linalg.eigvalsh(hsym)
                    piv, fail = cholesky_pivots_f32(hsym)
                    entry.update(eig_min=float(ev[0]), eig_max=float(ev[-1]),
                                 cond=float(ev[-1]/ev[0]) if ev[0] > 0 else None,
                                 f32_pivot_min=float(piv.min()), f32_fail_dof=None if fail is None else dof_name[fail],
                                 diag_min=float(np.diag(hsym).min()),
                                 diag_min_dof=dof_name[int(np.argmin(np.diag(hsym)))])
                if 'hfactor' in s:
                    hf = s['hfactor'][:nv, :nv]
                    bad = ~np.isfinite(np.diag(hf))
                    entry['hfactor_first_nonfinite_dof'] = dof_name[int(np.argmax(bad))] if bad.any() else None
                entry['alpha'] = float(np.ravel(s.get('alpha', [np.nan]))[0])
                entry['states'] = np.bincount(s['efc.state'].astype(int), minlength=5).tolist()
                hs.append(entry)
            rec['hessians'] = hs
            rec['alphas'] = [float(np.ravel(s['alpha'])[0]) for s in snaps if s['phase'] == 'linesearch']
            runs.append(rec)
            if nan and 'nan' not in keep:
                keep['nan'] = snaps
            if not nan and 'ok' not in keep:
                keep['ok'] = snaps
            print(json.dumps({k: v for k, v in rec.items() if k != 'hessians'}), flush=True)
            print('   ', json.dumps(hs[-3:]), flush=True)
    out['runs'] = runs
    out['nan_runs'] = sum(x['nan'] for x in runs)
    (dump/'solver_trace.json').write_text(json.dumps(out, indent=1, default=str))
    for k, snaps in keep.items():
        flat = {}
        for i, s in enumerate(snaps):
            for f, v in s.items():
                if isinstance(v, np.ndarray):
                    flat[f'{i:03d}_{s["it"]}_{s["phase"]}_{f}'] = v
        np.savez_compressed(dump/f'solver_trace_{k}.npz', **flat)
    print('NAN RUNS', out['nan_runs'], 'of', len(runs))


if __name__ == '__main__':
    main()
