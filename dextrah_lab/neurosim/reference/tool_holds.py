"""Transfer M's stored holding states into the stage 3 testbed (plan rev. 2.1; held states from the policy, tools first).

The success-guided sampler of a manipulation run stores held states (families stable_grasp / near_goal): the whole
robot's qpos / qvel, every joint target, the object (asset id, pose at its COM) and the table height. Each one is
rebuilt in the testbed with the same tool (the run's handle-head object bank: 4 primitive slots at the COM, as
MjSgsTask.assign_objects), the palm at the recorded hand-base pose (forward kinematics of the whole robot), the 11
hand joint angles and the recorded TARGETS (so every servo keeps its saturation gap), and the recorded table.

Acceptance: with the palm prescribed and every velocity zeroed, settle 0.5 s, then one ±3 mm / ±3 deg palm shake
over 0.5 s; kept if the object stays within 5 mm and 5 deg of its recorded pose in the palm frame throughout. The
rejection rate (per object family) is the Phase 0.6 parity measurement on real contact states: MuJoCo Warp (float32,
compliant arm, condim of the source run) versus C MuJoCo (float64, prescribed palm, condim 4).
"""
from dataclasses import dataclass, field
import json
import tempfile
from pathlib import Path

import mujoco
import numpy as np
import torch

from ..testbed import HAND_JOINTS, HAND_ROOT, Testbed, TestbedSpec
from .held_states import COMMAND_JOINTS, axis_angle, mat, pose, random_unit, rotation_deg

HELD_FAMILIES = ('stable_grasp', 'near_goal')
SGS_FAMILIES = ('reaching', 'stable_grasp', 'near_goal')     # dextrah_lab.sgs.mj_task.FAMILIES order
BAND = {j: (.5 if 'metacarpal' in j else 1.1 if 'thumb' in j else 2.)/1200 for j in HAND_JOINTS}


def object_bank_for_run(run_dir):
    """(bank, families, table) of a run's procedural tools, rebuilt as the environment builds them (seed 42 pool)."""
    from dextrah_lab.mj.compat import p2p
    from dextrah_lab.mj.env import load_run_cfg
    from dextrah_lab.mj.objects import build_object_bank
    cfg = load_run_cfg(Path(run_dir))
    out = tempfile.mkdtemp(prefix='ns3_tools_')
    paths, _ = p2p('generate_objects').generate_handle_head_urdfs(
        handle_head_types=tuple(cfg.assets.handle_head_types), num_per_type=cfg.assets.num_assets_per_type,
        out_dir=out, shuffle=cfg.assets.shuffle_assets)
    bank = build_object_bank([str(p) for p in paths])
    families = [Path(p).stem.split('_')[1] for p in paths]
    table = dict(size=bank.size, pos=bank.pos-bank.com[:, None], quat=bank.quat, rbound=bank.rbound,
                 aabb=bank.aabb.reshape(len(paths), 4, 6), mass=bank.mass, inertia=bank.inertia, used=bank.used)
    return bank, families, table


def robot_model(condim=4):
    """The whole-body scene compiled on the CPU (qpos layout of the stored states)."""
    from mjlab.scene import Scene
    from dextrah_lab.mj.robot import ContactParams
    from dextrah_lab.mj.scene import SceneParams, build_scene_cfg
    return Scene(build_scene_cfg(1, SceneParams(contact=ContactParams(condim=condim)), robot_pose=None), 'cpu').compile()


def family_layout(settings):
    fams = list(settings['families'])
    num = settings['num_configs']
    sizes = [num//len(fams)+(i < num % len(fams)) for i in range(len(fams))]
    bounds = np.cumsum([0]+sizes)
    family = np.empty(num, dtype=np.int64)
    for i, f in enumerate(fams):
        family[bounds[i]:bounds[i+1]] = SGS_FAMILIES.index(f)
    return family


@dataclass
class AcceptSpec:
    transient_steps: int = 15                 # 0.25 s: the restored contacts' penetration resolves (reference pose after it)
    settle_steps: int = 15
    shake_steps: int = 30
    shake_m: float = .003
    shake_deg: float = 3.
    max_m: float = .005
    max_deg: float = 5.


@dataclass
class HoldPool:
    """A run's held states, decoded for the testbed."""
    keys: np.ndarray             # SGS configuration index
    asset: np.ndarray
    family: list
    palm: np.ndarray             # (n, 7) hand-base pose
    hand_q: np.ndarray           # (n, 11) HAND_JOINTS order
    hand_target: np.ndarray      # (n, 11)
    object_qpos: np.ndarray      # (n, 7) object body (COM frame) pose
    table_top: np.ndarray        # (n,)
    arm_touch: np.ndarray        # (n,) the object touched a non-hand robot body in the whole-body state
    success: np.ndarray          # (n,) M's sampler success rate for the configuration (nan if never tried)
    meta: dict = field(default_factory=dict)


def decode_pool(sgs_path, run_dir, limit=None, seed=0):
    state = torch.load(sgs_path, map_location='cpu', weights_only=False)
    c = {k: v.numpy() if torch.is_tensor(v) else v for k, v in state['cfg'].items()}
    family = family_layout(state['settings'])
    held = np.nonzero(state['filled'].numpy() & np.isin(family, [SGS_FAMILIES.index(f) for f in HELD_FAMILIES]))[0]
    if limit is not None and limit < len(held):
        held = np.sort(np.random.default_rng(seed).choice(held, limit, replace=False))
    m = robot_model()
    d = mujoco.MjData(m)
    # joint_pos / targets columns (the environment's joint order) -> qpos addresses, matched on the stored states
    q, jp = c['qpos'][held[:2000]], c['joint_pos'][held[:2000]]
    names = [m.joint(j).name for j in range(m.njnt)]
    qadr = {n: int(m.jnt_qposadr[j]) for j, n in enumerate(names)}
    col_name = []
    for col in range(jp.shape[1]):
        err = {n: np.abs(q[:, a]-jp[:, col]).max() for n, a in qadr.items() if n.startswith('robot/') and 'floating' not in n}
        best = min(err, key=err.get)
        if err[best] > 1e-4:
            raise RuntimeError(f'joint_pos column {col} matches no robot joint (best {best}, {err[best]:.2e})')
        col_name.append(best.replace('robot/', ''))
    hand_cols = [col_name.index(j) for j in HAND_JOINTS]
    obj_adr = qadr['object/object_free']
    palm_body = m.body('robot/'+HAND_ROOT).id
    _, fams, table = object_bank_for_run(run_dir)
    n = len(held)
    palm = np.zeros((n, 7))
    for i, k in enumerate(held):
        d.qpos[:] = c['qpos'][k]
        mujoco.mj_kinematics(m, d)
        palm[i] = np.concatenate((d.xpos[palm_body], d.xquat[palm_body]))
    # forearm / wrist support of the object in the whole-body state: not measured (writing a tool's geometry into
    # the compiled whole-body model leaves stale compile data in C MuJoCo; it needs a compile per tool)
    arm_touch = np.zeros(n, dtype=bool)
    sampler = state['sampler']
    counts = sampler['count'].numpy() if torch.is_tensor(sampler.get('count')) else np.asarray(sampler.get('count'))
    outcomes = sampler['outcomes'].numpy() if torch.is_tensor(sampler.get('outcomes')) else np.asarray(sampler.get('outcomes'))
    hist = outcomes.shape[1]
    used = np.minimum(counts, hist)
    success = np.where(used > 0, outcomes.sum(1)/np.maximum(used, 1), np.nan)
    return HoldPool(keys=held, asset=c['asset'][held].astype(np.int64), family=[fams[int(c['asset'][k])] for k in held],
                    palm=palm, hand_q=c['joint_pos'][held][:, hand_cols], hand_target=c['targets'][held][:, hand_cols],
                    object_qpos=c['qpos'][held][:, obj_adr:obj_adr+7], table_top=c['table_z'][held]+.15,
                    arm_touch=arm_touch, success=success[held],
                    meta=dict(sgs=str(sgs_path), run=str(run_dir), held=int(n), joint_columns=col_name, arm_touch='not measured'))


def _under(m, b, root):
    while b > 0:
        b = int(m.body_parentid[b])
        if b == root:
            return True
    return False


def tool_of(table, a):
    return {k: table[k][a] for k in ('size', 'pos', 'quat', 'used', 'mass', 'inertia')}


def accept(tb, pool, i, table, rng, spec=AcceptSpec()):
    """Rebuild hold i in testbed tb (compiled with hold i's tool) and run the acceptance test. Returns a result dict
    (the state after the settle, before the shake, in result['state'])."""
    palm = pool.palm[i]
    tb.reset(palm, pool.hand_q[i], pool.object_qpos[i], table_top=float(pool.table_top[i]))
    targets = pool.hand_target[i]
    R_p = mat(palm[3:])
    rel0 = (R_p.T @ (pool.object_qpos[i, :3]-palm[:3]), R_p.T @ mat(pool.object_qpos[i, 3:]))
    worst_m, worst_deg = 0., 0.

    def track():
        nonlocal worst_m, worst_deg
        pp, pq = tb.d.qpos[tb.palm_qadr:tb.palm_qadr+3], tb.d.qpos[tb.palm_qadr+3:tb.palm_qadr+7]
        op, oq = tb.d.qpos[tb.object_qadr:tb.object_qadr+3], tb.d.qpos[tb.object_qadr+3:tb.object_qadr+7]
        Rp = mat(pq)
        worst_m = max(worst_m, float(np.linalg.norm(Rp.T @ (op-pp)-rel0[0])))
        worst_deg = max(worst_deg, rotation_deg((Rp.T @ mat(oq)).T @ rel0[1]))
    initial_penetration = min([c['dist'] for c in tb.contacts() if c['pair'] == 'hand_object'], default=0.)
    for _ in range(spec.transient_steps):
        tb.step(targets, palm)
        track()
    transient_m, transient_deg = worst_m, worst_deg
    pp, pq = tb.d.qpos[tb.palm_qadr:tb.palm_qadr+3], tb.d.qpos[tb.palm_qadr+3:tb.palm_qadr+7]
    Rp = mat(pq)
    rel0 = (Rp.T @ (tb.d.qpos[tb.object_qadr:tb.object_qadr+3]-pp), Rp.T @ mat(tb.d.qpos[tb.object_qadr+3:tb.object_qadr+7]))
    worst_m, worst_deg = 0., 0.
    for _ in range(spec.settle_steps):
        tb.step(targets, palm)
        track()
    settled = tb.get_state()
    q_settled = tb.d.qpos[tb.qadr].copy()
    forces = tb.finger_forces()
    R = mat(palm[3:])
    sd, sa = random_unit(rng), random_unit(rng)
    prev = palm
    for k in range(spec.shake_steps):
        ph = np.sin(2*np.pi*(k+1)/spec.shake_steps)
        nxt = pose(palm[:3]+spec.shake_m*ph*sd, axis_angle(sa, np.radians(spec.shake_deg)*ph) @ R)
        tb.step(targets, prev, nxt)
        prev = nxt
        track()
    end = tb.finger_forces()
    ok = worst_m <= spec.max_m and worst_deg <= spec.max_deg
    contact_end = sum(v > .5 for v in end.values())
    outcome = 'held' if ok else ('dropped' if contact_end == 0 else 'shifted')
    return dict(accepted=ok, outcome=outcome, worst_mm=1000*worst_m, worst_deg=worst_deg, forces=forces, forces_end=end,
                transient_mm=1000*transient_m, transient_deg=transient_deg,
                initial_penetration_mm=-1000*initial_penetration, gaps_recorded=targets-pool.hand_q[i],
                gaps_settled=targets-q_settled, state=settled)


def run_acceptance(pool, run_dir, geometry='mesh', limit=None, seed=0, spec=AcceptSpec(), log_every=0):
    _, _, table = object_bank_for_run(run_dir)
    rng = np.random.default_rng(seed)
    n = len(pool.keys) if limit is None else min(limit, len(pool.keys))
    rows, cache = [], {}
    for i in range(n):
        a = int(pool.asset[i])
        if a not in cache:                                  # one compiled testbed per tool
            if len(cache) > 64:
                cache.pop(next(iter(cache)))
            cache[a] = Testbed(TestbedSpec(geometry=geometry, object_kind='tool', tool=tool_of(table, a)))
        r = accept(cache[a], pool, i, table, rng, spec)
        rows.append(r)
        if log_every and (i+1) % log_every == 0:
            acc = np.mean([x['accepted'] for x in rows])
            print(f'ACCEPT {i+1}/{n} accepted {100*acc:.1f}%', flush=True)
    return rows


def summarize(pool, rows):
    n = len(rows)
    acc = np.array([r['accepted'] for r in rows])
    fam = np.array(pool.family[:n])
    arm = pool.arm_touch[:n]
    out = dict(n=n, accepted=round(float(acc.mean()), 4),
               by_family={f: dict(n=int((fam == f).sum()), accepted=round(float(acc[fam == f].mean()), 3)) for f in sorted(set(fam))},
               arm_touch_share=round(float(arm.mean()), 3),
               accepted_arm_touch=round(float(acc[arm].mean()), 3) if arm.any() else None,
               accepted_no_arm_touch=round(float(acc[~arm].mean()), 3) if (~arm).any() else None,
               outcomes={o: round(float(np.mean([r['outcome'] == o for r in rows])), 3) for o in ('held', 'shifted', 'dropped')},
               worst_mm_quantiles=[round(float(x), 1) for x in np.quantile([r['worst_mm'] for r in rows], [.25, .5, .75])],
               worst_deg_quantiles=[round(float(x), 1) for x in np.quantile([r['worst_deg'] for r in rows], [.25, .5, .75])],
               transient_mm_median=round(float(np.median([r['transient_mm'] for r in rows])), 2),
               initial_penetration_mm_p90=round(float(np.quantile([r['initial_penetration_mm'] for r in rows], .9)), 2))
    g = np.array([r['gaps_settled'] for r in rows])[acc]
    if len(g):
        out['census_accepted'] = {j.replace('right_', '').replace('_joint', ''): dict(
            outside_band=round(float((np.abs(g[:, k]) > BAND[j]).mean()), 3),
            median_gap_mrad=round(1000*float(np.median(np.abs(g[:, k]))), 2),
            closing_side=round(float((g[:, k] > BAND[j]).mean()), 3)) for k, j in enumerate(HAND_JOINTS)}
    s = pool.success[:n]
    have = np.isfinite(s)
    if have.any():
        out['m_success_accepted_vs_rejected'] = [round(float(np.nanmean(s[acc & have])), 3) if (acc & have).any() else None,
                                                 round(float(np.nanmean(s[~acc & have])), 3) if (~acc & have).any() else None]
    return out
