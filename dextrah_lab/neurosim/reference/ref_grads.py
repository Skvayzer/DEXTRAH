"""Phase 1 reference gradients (plan rev. 2.1 §4) at logged states of the tool episodes (tool_episodes.py).

At a state s_t of an episode, rollouts of h <= 8 control steps replay the episode's own inputs (the executed finger
commands and palm poses of steps t..t+h-1), with
  * an action perturbation +-delta along a unit direction u, held over the horizon: on the 6 finger commands
    (rad; distal targets by coupling), or on the palm twist (6: linear m/s scaled by palm_lin, angular rad/s scaled by
    palm_ang; applied as a constant extra twist, so the palm pose drifts by (k+1) dt delta u by step k);
  * noise (the sigma sweep): i.i.d. Gaussian noise on the 6 executed commands every step (sigma_f rad) and on the palm
    pose every step (sigma_p m, sigma_r deg);
  * common random numbers: +delta, -delta and the unperturbed rollout share the noise draws.
Estimates per direction: mean over K draws of (y(+) - y(-)) / (2 delta) (directional derivative), two independent
halves of the draws (for the agreement gate G1), and the curvature (y(+) + y(-) - 2 y(0)) / delta^2.
With delta = 0 and no noise the rollout reproduces the logged next states bit for bit (checked per state).

Outcomes y at each horizon (OUTCOMES): object position and rotation change (world, rotation vector), object twist,
normal force of each finger and the palm, grip force, slip ratio (max over hand-object contacts of
|f_tangential| / (mu f_normal)), tilt change (angle of the object's z axis from vertical), object-table normal force.
"""
from dataclasses import dataclass, field

import mujoco
import numpy as np

from .held_states import axis_angle, mat, pose, targets_from_commands
from .tool_episodes import DT, FINGERS, PHASES, set_state

OUTCOMES = (('obj_dpos', 3), ('obj_drot', 3), ('obj_vel', 6), ('finger_n', 6), ('grip', 1), ('slip', 1), ('tilt', 1),
            ('table_n', 1))
NY = sum(n for _, n in OUTCOMES)
STRATA = ('pre_contact', 'forming', 'holding', 'touchdown', 'release', 'incidental')


@dataclass
class GradSpec:
    horizons: tuple = (1, 2, 4, 8)
    delta_f: float = .01                 # rad, finger commands
    delta_p: float = 1.                  # palm twist direction scale (u_lin * palm_lin, u_ang * palm_ang)
    palm_lin: float = .01                # m/s per unit
    palm_ang: float = .05                # rad/s per unit
    noise: tuple = ((0., 0., 0.), (.005, .001, .5), (.02, .001, .5), (.05, .003, .5))   # (sigma_f rad, sigma_p m, deg)
    draws: int = 32
    directions: int = 2                  # random unit directions per channel (finger, palm)


def rotvec(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.reshape(-1))
    v = np.zeros(3)
    mujoco.mju_quat2Vel(v, q, 1.)
    return v


def tilt(q):
    return float(np.arccos(np.clip(mat(q)[2, 2], -1., 1.)))


class Probe:
    """Outcome measurement on a testbed."""

    def __init__(self, tb):
        self.tb = tb
        self.objg = set(tb.object_geoms.tolist())
        self.hand = set(tb.hand_geoms.tolist())
        self.f = np.zeros(6)

    def slip(self):
        m, d, f, worst = self.tb.m, self.tb.d, self.f, 0.
        for i in range(d.ncon):
            c = d.contact[i]
            g = {int(c.geom1), int(c.geom2)}
            if g & self.objg and g & self.hand:
                mujoco.mj_contactForce(m, d, i, f)
                if f[0] > 1e-6:
                    worst = max(worst, float(np.hypot(f[1], f[2])/(c.friction[0]*f[0])))
        return worst

    def measure(self, o0):
        tb, d = self.tb, self.tb.d
        o = d.qpos[tb.object_qadr:tb.object_qadr+7]
        fr = tb.finger_forces()
        fn = np.array([fr[k] for k in FINGERS])
        tn = sum(c['force_local'][0] for c in tb.contacts() if c['pair'] == 'object_table')
        return np.concatenate((o[:3]-o0[:3], rotvec(mat(o[3:]) @ mat(o0[3:]).T), d.qvel[tb.object_dadr:tb.object_dadr+6],
                               fn, [fn.sum(), self.slip(), tilt(o[3:])-tilt(o0[3:]), tn]))


def palm_offset(p, k, lin, ang):
    """Pose p shifted by a constant extra twist (world linear lin m/s, world angular ang rad/s) for k+1 steps."""
    t = (k+1)*DT
    n = np.linalg.norm(ang)
    R = (axis_angle(ang, n*t) if n > 0 else np.eye(3)) @ mat(p[3:])
    return pose(p[:3]+lin*t, R)


def rollout(tb, probe, state, cmds, palms, horizons, du_f=None, du_p=None, noise=None, lo=None, hi=None):
    """Replay from state with the logged inputs (cmds (H, 6), palms (H, 7)) plus a held perturbation and per-step
    noise (dict cmd (H, 6), pos (H, 3), rot (H, 3) rotation vectors). Returns (len(horizons), NY) and whether any
    which of the 6 commands were clipped at a joint limit in any step (bool (6,))."""
    set_state(tb, state)
    o0 = tb.d.qpos[tb.object_qadr:tb.object_qadr+7].copy()
    out, clipped, H = [], np.zeros(6, bool), max(horizons)
    for k in range(H):
        cmd = cmds[k].copy()
        if du_f is not None:
            cmd = cmd+du_f
        if noise is not None:
            cmd = cmd+noise['cmd'][k]
        c = np.clip(cmd, lo, hi)
        clipped |= c != cmd
        p = palms[k]
        if du_p is not None:
            p = palm_offset(p, k, du_p[:3], du_p[3:])
        if noise is not None and (noise['pos'][k].any() or noise['rot'][k].any()):
            r = noise['rot'][k]
            p = pose(p[:3]+noise['pos'][k], (axis_angle(r, np.linalg.norm(r)) if np.linalg.norm(r) > 0 else np.eye(3))
                     @ mat(p[3:]))
        prev = tb.d.qpos[tb.palm_qadr:tb.palm_qadr+7].copy()
        tb.step(targets_from_commands(c), prev, p)
        if k+1 in horizons:
            out.append(probe.measure(o0))
    return np.array(out), clipped


@dataclass
class StateGrads:
    stratum: str
    episode: int
    t: int
    replay_exact: bool
    gap: np.ndarray                      # 11 targets - q at t (band position)
    dirs_f: np.ndarray                   # (D, 6)
    dirs_p: np.ndarray                   # (D, 6)
    results: dict = field(default_factory=dict)   # noise index -> dict of arrays


def state_grads(tb, probe, state, cmds, palms, lo, hi, rng, spec=GradSpec(), next_states=None, rec=None):
    """All estimates at one state. cmds, palms: the episode's inputs for steps t..t+H-1."""
    H, hz = max(spec.horizons), spec.horizons
    D = spec.directions
    dirs_f = rng.normal(size=(D, 6)); dirs_f /= np.linalg.norm(dirs_f, axis=1, keepdims=True)
    dirs_p = rng.normal(size=(D, 6)); dirs_p /= np.linalg.norm(dirs_p, axis=1, keepdims=True)
    exact = None
    if next_states is not None and rec is not None:             # delta = 0, no noise: must replay the log exactly
        set_state(tb, state)
        exact = True
        for k in range(H):
            prev = tb.d.qpos[tb.palm_qadr:tb.palm_qadr+7].copy()
            tb.step(targets_from_commands(np.clip(cmds[k], lo, hi)), prev, palms[k])
            exact &= bool(np.array_equal(rec.get_state(), next_states[k]))
    results = {}
    for ni, (sf, sp, sr) in enumerate(spec.noise):
        K = 1 if sf == 0 and sp == 0 and sr == 0 else spec.draws
        noises = [None if K == 1 else dict(cmd=rng.normal(size=(H, 6))*sf, pos=rng.normal(size=(H, 3))*sp,
                                            rot=rng.normal(size=(H, 3))*np.radians(sr)) for _ in range(K)]
        g = np.zeros((2, 2*D, K, len(hz), NY))           # (+/-, channel-direction, draw, horizon, outcome)
        y0 = np.zeros((K, len(hz), NY))
        clip = np.zeros((D, K, 6), bool)
        for kd, nz in enumerate(noises):
            y0[kd], _ = rollout(tb, probe, state, cmds, palms, hz, noise=nz, lo=lo, hi=hi)
            for j in range(2*D):
                for si, sgn in enumerate((1., -1.)):
                    if j < D:
                        y, c = rollout(tb, probe, state, cmds, palms, hz, du_f=sgn*spec.delta_f*dirs_f[j], noise=nz,
                                       lo=lo, hi=hi)
                    else:
                        u = dirs_p[j-D]*spec.delta_p*sgn
                        y, c = rollout(tb, probe, state, cmds, palms, hz,
                                       du_p=np.concatenate((u[:3]*spec.palm_lin, u[3:]*spec.palm_ang)), noise=nz,
                                       lo=lo, hi=hi)
                    g[si, j, kd] = y
                    if j < D:
                        clip[j, kd] |= c
        delta = np.array([spec.delta_f]*D+[spec.delta_p]*D)[:, None, None, None]
        fd = (g[0]-g[1])/(2*delta)                       # (2D, K, horizons, NY) per-draw central differences
        curv = (g[0]+g[1]-2*y0[None])/delta**2
        half = K//2
        results[ni] = dict(K=K, grad=fd.mean(1), grad_se=fd.std(1)/np.sqrt(max(K, 1)) if K > 1 else np.zeros_like(fd[:, 0]),
                           grad_halves=np.stack((fd[:, :half].mean(1), fd[:, half:].mean(1))) if K > 1 else None,
                           curv=curv.mean(1), y0=y0.mean(0), y0_std=y0.std(0), clipped=clip.mean((0, 1)))
    return dirs_f, dirs_p, exact, results


def strata(sh, e):
    """Stratum label per step of episode e of a shard (or None), from the logged forces, contacts and phases."""
    a, b = sh['step_offsets'][e], sh['step_offsets'][e+1]
    ph = sh['phase'][a:b].astype(int)
    f = sh['forces'][a:b].sum(1)
    tn = sh['object_table_n'][a:b]
    n = b-a
    contact = f > .05
    lab = [None]*n
    for t in range(n):
        p = PHASES[ph[t]]
        if not contact[t]:
            if contact[t+1:t+9].any():
                lab[t] = 'pre_contact'
            continue
        if t >= 2 and not contact[t-2:t].all():
            lab[t] = 'forming'
        elif p in ('carry', 'regrip') and tn[t] <= 0 and f[t] > .5:
            lab[t] = 'holding'
        elif p in ('lower', 'press') and (tn[t] > 0 or (tn[t+1:t+5] > 0).any()):
            lab[t] = 'touchdown'
        elif p == 'release':
            lab[t] = 'release'
        elif p in ('sweep', 'retreat', 'settle'):
            lab[t] = 'incidental'
    return lab
