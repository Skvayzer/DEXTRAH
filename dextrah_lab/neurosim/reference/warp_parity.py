"""Phase 0.6 Warp parity on harvested tool holds (plan rev. 2.1): the same compiled testbed model (one per tool) and
the same acceptance test, in C MuJoCo (float64, tool_holds.accept) and in MuJoCo Warp (float32, batched: one world
per hold x shake direction). Both sides use identical shake directions (the same seeded generator per hold and
replica), so outcomes compare pairwise as well as in distribution.

Warp step = Testbed.step: palm pose and twist set before every physics substep (constant twist over the control
step), servo velocity command recomputed per substep from the current joint angles, palm put exactly on the
trajectory after the substeps. The end-of-step mj_forward of the C version is skipped (it changes nothing that the
next step reads). Outcome per world as tool_holds.accept: held (5 mm / 5 deg after the transient), dropped (no
finger or palm contact above 0.5 N at the end; forces from C MuJoCo's forward pass on Warp's final state), shifted.
"""
import numpy as np
import mujoco
import torch

from ..testbed import PHYSICS_DT, SUBSTEPS
from .held_states import axis_angle, mat, pose, random_unit, rotation_deg
from .tool_holds import AcceptSpec


def shake_dirs(seed, asset, hold, replica):
    """The two unit vectors (translation, rotation axis) tool_holds.accept draws from its generator."""
    rng = np.random.default_rng([seed, asset, hold, replica])
    return rng, random_unit(rng), random_unit(rng)


def palm_substeps(palm, nxt):
    """Testbed.step's palm trajectory: per-substep poses (SUBSTEPS, 7) and the constant twist (6: lin, body ang)."""
    T = SUBSTEPS*PHYSICS_DT
    lin = (nxt[:3]-palm[:3])/T
    dq = np.zeros(4)
    mujoco.mju_mulQuat(dq, np.array([palm[3], -palm[4], -palm[5], -palm[6]]), nxt[3:7])
    ang = np.zeros(3)
    mujoco.mju_quat2Vel(ang, dq, T)
    poses = np.empty((SUBSTEPS, 7))
    for k in range(SUBSTEPS):
        a = k/SUBSTEPS
        poses[k, :3] = palm[:3]+a*(nxt[:3]-palm[:3])
        q = palm[3:7].copy()
        mujoco.mju_quatIntegrate(q, ang, a*T)
        poses[k, 3:] = q
    return poses, np.concatenate((lin, ang))



def palm_schedule(palm, sd, sa, spec):
    """Per control step: (poses (SUBSTEPS, 7), twist (6), end pose (7)) for transient + settle + shake."""
    hold = palm_substeps(palm, palm)
    out = [(hold[0], hold[1], palm)]*(spec.transient_steps+spec.settle_steps)
    R = mat(palm[3:])
    prev = palm
    for k in range(spec.shake_steps):
        ph = np.sin(2*np.pi*(k+1)/spec.shake_steps)
        nxt = pose(palm[:3]+spec.shake_m*ph*sd, axis_angle(sa, np.radians(spec.shake_deg)*ph) @ R)
        p, v = palm_substeps(prev, nxt)
        out.append((p, v, nxt))
        prev = nxt
    return out


def _qmul(a, b):
    w1, x1, y1, z1 = a.unbind(-1)
    w2, x2, y2, z2 = b.unbind(-1)
    return torch.stack((w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2,
                        w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2), -1)


def _conj(q):
    return q*q.new_tensor([1., -1., -1., -1.])


def _rotate(q, v):
    return _qmul(_qmul(q, torch.cat((torch.zeros_like(v[..., :1]), v), -1)), _conj(q))[..., 1:]


def accept_warp(tb, holds, dirs, spec=AcceptSpec(), device='cuda:0', nconmax=96, njmax=600):
    """Batched acceptance test in MuJoCo Warp on tb's compiled model. holds: list of dicts (palm, hand_q,
    hand_target, object_qpos, table_top); dirs: list of (hold index, sd, sa), one world each. Returns one result
    dict per world (outcome, worst_mm, worst_deg, transient_mm, transient_deg) and the wall time of the steps."""
    import time
    import warp as wp
    import mujoco_warp as mjw
    m, d = tb.m, tb.d
    W = len(dirs)
    qpos0, mocap_pos, mocap_quat = [], [], []
    for h, _, _ in dirs:
        x = holds[h]
        tb.reset(x['palm'], x['hand_q'], x['object_qpos'], table_top=float(x['table_top']))
        qpos0.append(d.qpos.copy()); mocap_pos.append(d.mocap_pos.copy()); mocap_quat.append(d.mocap_quat.copy())
    sched = [palm_schedule(holds[h]['palm'], sd, sa, spec) for h, sd, sa in dirs]
    n_steps = len(sched[0])
    dev = torch.device(device)
    f32 = dict(dtype=torch.float32, device=dev)
    poses = torch.tensor(np.array([[s[k][0] for s in sched] for k in range(n_steps)]), **f32)    # (steps, W, SUB, 7)
    twist = torch.tensor(np.array([[s[k][1] for s in sched] for k in range(n_steps)]), **f32)    # (steps, W, 6)
    ends = torch.tensor(np.array([[s[k][2] for s in sched] for k in range(n_steps)]), **f32)     # (steps, W, 7)
    targets = torch.tensor(np.array([holds[h]['hand_target'] for h, _, _ in dirs]), **f32)
    gain, vmax = torch.tensor(tb.servo_gain, **f32), torch.tensor(tb.servo_vmax, **f32)
    qadr, act = torch.as_tensor(tb.qadr, device=dev), torch.as_tensor(tb.act, device=dev)
    pq, pd, oq = tb.palm_qadr, tb.palm_dadr, tb.object_qadr
    with wp.ScopedDevice(device), wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(dev))):
        wm = mjw.put_model(m)                          # torch writes and Warp kernels on one stream (no races)
        wd = mjw.make_data(m, nworld=W, nconmax=nconmax, njmax=njmax)
        Q, V, C = wp.to_torch(wd.qpos), wp.to_torch(wd.qvel), wp.to_torch(wd.ctrl)
        Q.copy_(torch.tensor(np.array(qpos0), **f32)); V.zero_()
        wp.to_torch(wd.mocap_pos).copy_(torch.tensor(np.array(mocap_pos), **f32))
        wp.to_torch(wd.mocap_quat).copy_(torch.tensor(np.array(mocap_quat), **f32))
        wp.to_torch(wd.qacc_warmstart).zero_()

        def rel():
            P, Pq = Q[:, pq:pq+3].double(), Q[:, pq+3:pq+7].double()
            O, Oq = Q[:, oq:oq+3].double(), Q[:, oq+3:oq+7].double()
            return _rotate(_conj(Pq), O-P), _qmul(_conj(Pq), Oq)

        def dev_of(r0):
            p, q = rel()
            ang = 2*torch.arccos(torch.clamp(torch.abs((q*r0[1]).sum(-1)/q.norm(dim=-1)/r0[1].norm(dim=-1)), max=1.))
            return (p-r0[0]).norm(dim=-1), torch.rad2deg(ang)

        torch.cuda.synchronize()
        t0 = time.time()
        r0 = rel()
        worst_m, worst_deg = torch.zeros(W, dtype=torch.float64, device=dev), torch.zeros(W, dtype=torch.float64, device=dev)
        transient = None
        for k in range(n_steps):
            for s in range(SUBSTEPS):
                Q[:, pq:pq+7] = poses[k, :, s]
                V[:, pd:pd+6] = twist[k]
                C[:, act] = torch.clamp(gain*(targets-Q[:, qadr]), -vmax, vmax)
                mjw.step(wm, wd)
            Q[:, pq:pq+7] = ends[k]
            V[:, pd:pd+6] = twist[k]
            dm, dd = dev_of(r0)
            worst_m, worst_deg = torch.maximum(worst_m, dm), torch.maximum(worst_deg, dd)
            if k+1 == spec.transient_steps:
                transient = (worst_m.clone(), worst_deg.clone())
                r0 = rel()
                worst_m.zero_(); worst_deg.zero_()
        wp.synchronize(); torch.cuda.synchronize()
        wall = time.time()-t0
        q_end, v_end, c_end = Q.double().cpu().numpy(), V.double().cpu().numpy(), C.double().cpu().numpy()
        warm = wp.to_torch(wd.qacc_warmstart).double().cpu().numpy()
        nan = ~np.isfinite(q_end).all(1)
    out = []
    for w in range(W):
        ok = bool(worst_m[w] <= spec.max_m and worst_deg[w] <= spec.max_deg) and not nan[w]
        if nan[w]:
            contact_end = 0
        else:
            d.qpos[:], d.qvel[:], d.ctrl[:], d.qacc_warmstart[:] = q_end[w], v_end[w], c_end[w], warm[w]
            d.mocap_pos[:], d.mocap_quat[:] = mocap_pos[w], mocap_quat[w]
            mujoco.mj_forward(m, d)
            contact_end = sum(v > .5 for v in tb.finger_forces().values())
        out.append(dict(outcome='held' if ok else ('dropped' if contact_end == 0 else 'shifted'), nan=bool(nan[w]),
                        worst_mm=1000*float(worst_m[w]), worst_deg=float(worst_deg[w]),
                        transient_mm=1000*float(transient[0][w]), transient_deg=float(transient[1][w])))
    return out, wall


def accept_c(tb, holds, dirs, spec=AcceptSpec()):
    """tool_holds.accept on the same worlds (its generator yields the same shake directions as shake_dirs)."""
    from .tool_holds import HoldPool, accept
    pool = HoldPool(keys=np.arange(len(holds)), asset=np.zeros(len(holds), int), family=['']*len(holds),
                    palm=np.array([x['palm'] for x in holds]), hand_q=np.array([x['hand_q'] for x in holds]),
                    hand_target=np.array([x['hand_target'] for x in holds]),
                    object_qpos=np.array([x['object_qpos'] for x in holds]),
                    table_top=np.array([x['table_top'] for x in holds]), arm_touch=np.zeros(len(holds), bool),
                    success=np.zeros(len(holds)))
    out = []
    for h, rng in dirs:
        r = accept(tb, pool, h, None, rng, spec)
        out.append({k: r[k] for k in ('outcome', 'worst_mm', 'worst_deg', 'transient_mm', 'transient_deg')})
    return out
