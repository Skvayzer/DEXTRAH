"""E1 gradient census: which action channels carry a usable gradient in which manipulation situation?

For states from scripted M-like table episodes (approach, close, lift, hold, lower, release; structured.scenes),
labelled by situation from contacts and object-palm relative motion, compare for each action channel and outcome:

  local  g0 : MuJoCo's own first-order derivative (central difference, delta = 1e-6, no noise) - what an exact
              differentiable simulator (or our structured model, whose forward equals MuJoCo's) returns;
  smooth gs : derivative of the noise-averaged outcome E[y(a + xi)] (per-step Gaussian noise on executed finger
              targets and palm velocity, sigma level s; central differences with common random numbers) - the
              signal the physics offers under realistic noise, the target of rev. 2.1's reference gradients.

Channels (offsets held over the horizon): 6 finger commands (policy interface: distal = coupling x proximal) in rad,
palm linear velocity (m/s) and angular velocity (rad/s, world frame). Outcomes after h = 1, 2, 4, 8 control steps:
object position and rotation in the palm frame (relative to the start), total hand-object normal force, slip
margin 1 - max_k |f_t,k| / (mu_k f_n,k) over hand-object contacts (-1 when no contact), object-table normal force,
object height above the table.

Metrics per (situation, channel group, outcome, h, sigma): fraction of states with a (numerically) zero local
derivative, median |g0| and |gs|, cosine(g0, gs) over the channel group, split-half reliability of gs
(cosine of two independent noise halves = rev. 2.1 gate G1), sign agreement.
"""
import argparse
import json
import os
import time
from multiprocessing import Pool
from pathlib import Path

import mujoco
import numpy as np

from dextrah_lab.neurosim.reference.held_states import mat
from dextrah_lab.neurosim.structured.scenes import table_episode, tool_testbed

H = (1, 2, 4, 8)
SIGMAS = dict(low=(.005, .01, .05), mid=(.02, .03, .2), high=(.05, .1, .5))     # finger rad, palm m/s, palm rad/s
COUPLING = [(0,), (1, 2), (3, 4), (5, 6), (7, 8), (9, 10)]
COUPLE_K = {2: 1., 4: 1.155, 6: 1.155, 8: 1.155, 10: 1.155}
# FINGER_MODE=intent (E2): fingers are driven by a grip intent u in [-1, 1] per command group, target = q + band*u
# (band = effort/kp, the servo's linear band): for a blocked joint the torque is kp*band*u = effort*u, i.e. torque
# control. The base intent is the logged command's direction, u0 = clip((target - q)/band, -0.5, 0.5); the finger
# channel offsets u (unitless). Finger noise sigma (fraction): low 0.1, mid 0.3, high 0.5.
FINGER_MODE = os.environ.get('FINGER_MODE', 'absolute')
BAND = np.array([.5, 1.1, 1.1, 2., 2., 2., 2., 2., 2., 2., 2.])/1200.
if FINGER_MODE == 'intent':
    SIGMAS = {k: (dict(low=.1, mid=.3, high=.5)[k], v[1], v[2]) for k, v in SIGMAS.items()}
OUTCOMES = ('rel_pos', 'rel_rot', 'grip_force', 'slip_margin', 'table_force', 'height', 'utilization', 'capacity')
# utilization: aggregate friction utilization sum|f_t| / sum mu f_n over hand-object contacts (0 without contact);
# capacity: sum mu f_n / (m g). Both continuous in the contact forces (the max-over-contacts slip margin is not).
CHANNELS = ['finger']*6+['palm_lin']*3+['palm_ang']*3


def cmd_to_targets(cmd_off):
    """6 command offsets -> 11 target offsets (distal joints follow by coupling)."""
    t = np.zeros(11)
    for c, js in enumerate(COUPLING):
        for j in js:
            t[j] = cmd_off[c]*COUPLE_K.get(j, 1.)
    return t


def contact_terms(tb):
    m, d = tb.m, tb.d
    grip, table, umax, n, ft_sum, cap = 0., 0., 0., 0, 0., 0.
    f = np.zeros(6)
    for c in tb.contacts():
        if c['pair'] == 'hand_object':
            i = [k for k in range(d.ncon) if d.contact[k].geom1 == c['geom1'] and d.contact[k].geom2 == c['geom2']
                 and np.allclose(d.contact[k].pos, c['pos'])][0]
            mujoco.mj_contactForce(m, d, i, f)
            fn = max(f[0], 1e-9)
            grip += f[0]
            umax = max(umax, np.hypot(f[1], f[2])/(d.contact[i].friction[0]*fn))
            ft_sum += np.hypot(f[1], f[2])
            cap += d.contact[i].friction[0]*max(f[0], 0.)
            n += 1
        elif c['pair'] == 'object_table':
            table += float(c['force_local'][0])
    return grip, (1-umax) if n else -1., table, ft_sum/max(cap, 1e-9) if n else 0., cap


def height(tb, top):
    from dextrah_lab.neurosim.structured.scenes import _geom_extent_down
    m, d = tb.m, tb.d
    return min(float(d.geom_xpos[g][2]-_geom_extent_down(m, d, g)) for g in tb.object_geoms)-top


def outcome(tb, palm, rel0, top):
    d = tb.d
    o = tb.object_qadr
    Rp = mat(palm[3:])
    p = Rp.T @ (d.qpos[o:o+3]-palm[:3])-rel0[0]
    Rr = rel0[1].T @ (Rp.T @ mat(d.qpos[o+3:o+7]))
    rv = np.zeros(3)
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(Rr).reshape(-1))
    mujoco.mju_quat2Vel(rv, q, 1.)
    grip, margin, table, util, cap = contact_terms(tb)
    mass = float(tb.m.body_mass[tb.object_body])
    return dict(rel_pos=p, rel_rot=rv, grip_force=np.array([grip]), slip_margin=np.array([margin]),
                table_force=np.array([table]), height=np.array([height(tb, top)]), utilization=np.array([util]),
                capacity=np.array([cap/(mass*9.81)]))


def perturbed_palm(palm_traj, k, v, w, dt):
    """Palm pose k of the logged trajectory shifted by constant world-frame velocity offsets (v, w) over k*dt."""
    p = palm_traj[k].copy()
    p[:3] += v*k*dt
    q = p[3:].copy()
    qw = np.zeros(4)
    mujoco.mju_axisAngle2Quat(qw, w/max(np.linalg.norm(w), 1e-300), np.linalg.norm(w)*k*dt)
    mujoco.mju_mulQuat(q, qw, q)
    p[3:] = q/np.linalg.norm(q)
    return p


def rollout(tb, state, palm_traj, targets, a_off, noise, top):
    """Outcomes at the horizons H. a_off (12,): 6 command offsets, palm v (3), palm w (3); noise (Hmax, 12) per
    step (same layout, added to the offsets at that step; palm noise integrates like a velocity)."""
    tb.set_state(state)
    dt = 1/60                                   # control step
    Rp0 = mat(palm_traj[0][3:])
    o = tb.object_qadr
    rel0 = (Rp0.T @ (tb.d.qpos[o:o+3]-palm_traj[0][:3]), Rp0.T @ mat(tb.d.qpos[o+3:o+7]))
    out = {}
    pv, pw = np.zeros(3), np.zeros(3)
    prev = palm_traj[0].copy()
    acc_pos, acc_q = np.zeros(3), np.array([1., 0., 0., 0.])
    for k in range(max(H)):
        a = a_off+noise[k]
        if FINGER_MODE == 'intent':
            q = tb.d.qpos[tb.qadr]
            u = np.clip((targets[k]-q)/BAND, -.5, .5)+np.array([a[c] for c, js in enumerate(COUPLING) for _ in js])
            tgt = q+BAND*np.clip(u, -1., 1.)
        else:
            tgt = targets[k]+cmd_to_targets(a[:6])
        nxt = perturbed_palm(palm_traj, k+1, a_off[6:9], a_off[9:12], dt)
        # noise: an extra velocity this step, accumulated
        acc_pos = acc_pos+noise[k, 6:9]*dt
        qn = np.zeros(4)
        wn = noise[k, 9:12]
        mujoco.mju_axisAngle2Quat(qn, wn/max(np.linalg.norm(wn), 1e-300), np.linalg.norm(wn)*dt)
        mujoco.mju_mulQuat(acc_q, qn, acc_q.copy())
        nxt[:3] += acc_pos
        q2 = np.zeros(4)
        mujoco.mju_mulQuat(q2, acc_q, nxt[3:])
        nxt[3:] = q2/np.linalg.norm(q2)
        tb.step(tgt, prev, nxt)
        prev = nxt
        if k+1 in H:
            out[k+1] = outcome(tb, nxt, rel0, top)
    return out


def flat(o):
    return np.concatenate([o[k] for k in OUTCOMES])


def jacobian(tb, state, palm_traj, targets, top, delta, sigma, draws, rng):
    """(len(H), n_out, 12) central-difference Jacobian; noise-averaged with common random numbers if sigma."""
    hmax = max(H)
    J = {h: [] for h in H}
    scale = np.r_[[sigma[0]]*6, [sigma[1]]*3, [sigma[2]]*3] if sigma else None
    noises = [rng.normal(size=(hmax, 12))*scale for _ in range(draws)] if sigma else [np.zeros((hmax, 12))]
    for ch in range(12):
        e = np.zeros(12)
        e[ch] = delta[ch]
        acc = {h: 0. for h in H}
        for nz in noises:
            yp = rollout(tb, state, palm_traj, targets, e, nz, top)
            ym = rollout(tb, state, palm_traj, targets, -e, nz, top)
            for h in H:
                acc[h] = acc[h]+(flat(yp[h])-flat(ym[h]))/(2*delta[ch])
        for h in H:
            J[h].append(acc[h]/len(noises))
    return np.stack([np.stack(J[h], -1) for h in H])


def label(tb, ep, k):
    tb.set_state(ep.states[k])
    cs = tb.contacts()
    hand = any(c['pair'] == 'hand_object' and c['dist'] < 0 for c in cs)
    table = any(c['pair'] == 'object_table' and c['dist'] < 0 for c in cs)
    d, o = tb.d, tb.object_dadr
    dt = 1/60                                   # control step
    pv = (ep.palm[k+1][:3]-ep.palm[k][:3])/dt
    qrel = np.zeros(4)
    mujoco.mju_mulQuat(qrel, np.array([ep.palm[k][3], -ep.palm[k][4], -ep.palm[k][5], -ep.palm[k][6]]), ep.palm[k+1][3:])
    wl = np.zeros(3)
    mujoco.mju_quat2Vel(wl, qrel, dt)
    pw = mat(ep.palm[k][3:]) @ wl
    po = d.qpos[tb.object_qadr:tb.object_qadr+3]
    vrel = d.qvel[o:o+3]-(pv+np.cross(pw, po-ep.palm[k][:3]))
    wrel = mat(d.qpos[tb.object_qadr+3:tb.object_qadr+7]) @ d.qvel[o+3:o+6]-pw
    ph = ep.phase[k]
    moving = np.linalg.norm(vrel) > .02 or np.linalg.norm(wrel) > .3
    if not hand:
        return 'approach' if table and ph in ('approach', 'close') else ('free' if not table else 'other')
    if table:
        return {'close': 'closing', 'lift': 'liftoff', 'lower': 'touchdown', 'release': 'release'}.get(ph, 'other')
    if ph == 'lower':
        return 'lowering'
    return 'slip' if moving else 'hold'


def episode_task(args):
    seed, = args
    rng = np.random.default_rng(seed)
    ep, tb = table_episode(rng)
    labs = [label(tb, ep, k) for k in range(len(ep.states)-max(H))]
    return seed, ep, labs


def census_task(args):
    seed, ep, k, lab, sig_names, draws = args
    rng = np.random.default_rng(10_000+seed*1000+k)
    tb = tool_testbed(ep.tool)
    top = .8
    palm_traj = ep.palm[k:k+max(H)+1]
    targets = ep.targets[k:k+max(H)]
    st = ep.states[k]
    t0 = time.time()
    deltas = [float(x) for x in os.environ.get('LOCAL_DELTAS', '1e-6').split(',')]
    res = dict(seed=seed, k=k, label=lab)
    for i, dl in enumerate(deltas):                 # noise-free central differences at several step sizes
        res['local' if i == 0 else f'local_{dl:g}'] = jacobian(tb, st, palm_traj, targets, top, np.full(12, dl), None, 1, rng)
    for s in sig_names:
        sig = SIGMAS[s]
        delta = np.r_[[.2*sig[0]]*6, [.2*sig[1]]*3, [.2*sig[2]]*3]
        a = jacobian(tb, st, palm_traj, targets, top, delta, sig, draws//2, rng)
        b = jacobian(tb, st, palm_traj, targets, top, delta, sig, draws//2, rng)
        res[s] = (a+b)/2
        res[s+'_halves'] = (a, b)
    res['sec'] = time.time()-t0
    return res


def summarize(results, sig_names):
    n_out = sum({'rel_pos': 3, 'rel_rot': 3}.get(k, 1) for k in OUTCOMES)
    slices, i = {}, 0
    for k in OUTCOMES:
        w = {'rel_pos': 3, 'rel_rot': 3}.get(k, 1)
        slices[k] = slice(i, i+w)
        i += w
    groups = dict(finger=slice(0, 6), palm_lin=slice(6, 9), palm_ang=slice(9, 12))
    labels = sorted({r['label'] for r in results})
    table = []
    for lab in labels:
        R = [r for r in results if r['label'] == lab]
        for oname, osl in slices.items():
            for gname, gsl in groups.items():
                for hi, h in enumerate(H):
                    g0 = np.stack([r['local'][hi][osl, gsl].reshape(-1) for r in R])
                    row = dict(situation=lab, outcome=oname, channel=gname, h=h, n=len(R),
                               zero_local=float(np.mean(np.abs(g0).max(1) < 1e-7)),
                               med_local=float(np.median(np.linalg.norm(g0, axis=1))))
                    for key in [k for k in R[0] if k.startswith('local_')]:
                        gd = np.stack([r[key][hi][osl, gsl].reshape(-1) for r in R])
                        gm = np.stack([r['mid'][hi][osl, gsl].reshape(-1) for r in R]) if 'mid' in R[0] else None
                        nd = np.linalg.norm(gd, axis=1)
                        row[key+'_zero'] = float(np.mean(np.abs(gd).max(1) < 1e-7))
                        row[key+'_med'] = float(np.median(nd))
                        if gm is not None:
                            nm = np.linalg.norm(gm, axis=1)
                            ok = (nd > 1e-12) & (nm > 1e-12)
                            c = np.where(ok, (gd*gm).sum(1)/np.maximum(nd*nm, 1e-300), np.nan)
                            row[key+'_cos_mid'] = float(np.nanmedian(c)) if np.isfinite(c).any() else None
                    for s in sig_names:
                        gs = np.stack([r[s][hi][osl, gsl].reshape(-1) for r in R])
                        a = np.stack([r[s+'_halves'][0][hi][osl, gsl].reshape(-1) for r in R])
                        b = np.stack([r[s+'_halves'][1][hi][osl, gsl].reshape(-1) for r in R])
                        nl, ns = np.linalg.norm(g0, axis=1), np.linalg.norm(gs, axis=1)
                        ok = (nl > 1e-12) & (ns > 1e-12)
                        cos = np.where(ok, (g0*gs).sum(1)/np.maximum(nl*ns, 1e-300), np.nan)
                        na, nb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
                        rel = np.where((na > 1e-12) & (nb > 1e-12), (a*b).sum(1)/np.maximum(na*nb, 1e-300), np.nan)
                        row.update({f'{s}_med_smooth': float(np.median(ns)),
                                    f'{s}_cos_local': float(np.nanmedian(cos)) if np.isfinite(cos).any() else None,
                                    f'{s}_ratio_local': float(np.median(nl/np.maximum(ns, 1e-300))),
                                    f'{s}_reliability': float(np.nanmedian(rel)) if np.isfinite(rel).any() else None})
                    table.append(row)
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes', type=int, default=120)
    ap.add_argument('--per_situation', type=int, default=40)
    ap.add_argument('--sigmas', default='low,mid,high')
    ap.add_argument('--draws', type=int, default=32)
    ap.add_argument('--workers', type=int, default=24)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=os.environ.get('CENSUS_OUT', 'outputs/neurosim/census'))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sig_names = a.sigmas.split(',')
    print('CONFIG', json.dumps(vars(a)), 'LOCAL_DELTAS', os.environ.get('LOCAL_DELTAS', '1e-6'), 'TIGHT', os.environ.get('TESTBED_TIGHT_SOLVER', ''), 'FINGER_MODE', FINGER_MODE, 'TESTBED_SUBSTEPS', os.environ.get('TESTBED_SUBSTEPS', 2), 'SIGMAS', json.dumps({s: SIGMAS[s] for s in sig_names}), flush=True)
    t0 = time.time()
    with Pool(a.workers) as pool:
        eps = pool.map(episode_task, [(a.seed*100_000+i,) for i in range(a.episodes)])
        cand = {}
        for seed, ep, labs in eps:
            for k, lab in enumerate(labs):
                cand.setdefault(lab, []).append((seed, ep, k, lab))
        counts = {k: len(v) for k, v in cand.items()}
        lifted = sum(1 for _, ep, _ in eps if (np.array(ep.phase) == 'hold').any()
                     and ep.obj[:-1][np.array(ep.phase) == 'hold', 2].min()-ep.obj[0, 2] > .03)
        print('EPISODES', a.episodes, 'lifted', lifted, 'step labels', json.dumps(counts), '%.0fs' % (time.time()-t0), flush=True)
        rng = np.random.default_rng(a.seed)
        jobs = []
        for lab, items in cand.items():
            if lab == 'other':
                continue
            sel = rng.choice(len(items), min(a.per_situation, len(items)), replace=False)
            jobs += [(*items[i], sig_names, a.draws) for i in sel]
        print('CENSUS STATES', len(jobs), flush=True)
        results = []
        for i, r in enumerate(pool.imap_unordered(census_task, jobs)):
            results.append(r)
            if (i+1) % 25 == 0:
                print('PROGRESS %d/%d %.0fs' % (i+1, len(jobs), time.time()-t0), flush=True)
    np.save(out/'census_raw.npy', np.array(results, dtype=object), allow_pickle=True)
    table = summarize(results, sig_names)
    (out/'census_table.json').write_text(json.dumps(table, indent=1))
    print('SUMMARY situation outcome channel h=4: zero_local | med_local | ' +
          ' | '.join(f'{s}: med_smooth cos ratio reliab' for s in sig_names))
    for row in table:
        if row['h'] != 4:
            continue
        print('ROW', row['situation'], row['outcome'], row['channel'], 'n=%d' % row['n'], 'zero %.2f' % row['zero_local'],
              'local %.3g' % row['med_local'],
              ' '.join('%s %.3g %s %.3g %s' % (s, row[f'{s}_med_smooth'],
                                                 'nan' if row[f'{s}_cos_local'] is None else '%.2f' % row[f'{s}_cos_local'],
                                                 row[f'{s}_ratio_local'],
                                                 'nan' if row[f'{s}_reliability'] is None else '%.2f' % row[f'{s}_reliability'])
                       for s in sig_names), flush=True)
    print('DONE %.0fs -> %s' % (time.time()-t0, out), flush=True)


if __name__ == '__main__':
    main()
