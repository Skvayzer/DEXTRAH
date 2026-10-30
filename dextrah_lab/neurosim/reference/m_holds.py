"""M's own holding states (run 1454, best epoch 12662) rebuilt in the stage-3 testbed, on dep-1 without mjlab / Isaac.

Inputs (downloaded release m-1454-epoch12662): the success-guided sampler state (*.sgs.pt: 32768 configurations in
three contiguous family blocks reaching / stable_grasp / near_goal; per configuration the whole-scene qpos (robot 58:
floating base 7 + 51 joints in MuJoCo tree order, then the object free joint 7), joint positions / targets (51,
environment order), asset id, table height) and env_resolved.yaml (object bank settings).

Tool geometry: the run's procedural bank is regenerated with Play2Perfect's generate_handle_head_urdfs (a local
clone of Skvayzer/play2perfect-unforked-backup; env P2P_ROOT) with the run's settings, then parsed exactly as the
environment does (dextrah_lab.mj.objects.build_object_bank). validate_bank checks it against the sampler's own data:
for the reaching family the stored object pose rests on the table, so the lowest point of the rebuilt tool must sit
at the table top.

Hand state: the hand-base pose by forward kinematics of the whole robot (robot.build_robot_spec, floating base);
the 11 hand joints and targets by matching environment columns to qpos addresses (exact for the held families).
"""
import importlib.util
import os
import sys
import tempfile
from pathlib import Path

import mujoco
import numpy as np
import torch
import yaml

from ..testbed import HAND_JOINTS, HAND_ROOT
from .tool_holds import HELD_FAMILIES, SGS_FAMILIES, HoldPool, family_layout

P2P_ROOT = Path(os.environ.get('P2P_ROOT', Path.home()/'robotics/third_party/play2perfect'))
M_DIR = Path(os.environ.get('M_DIR', Path.home()/'m_1454'))


def _generator():
    base = P2P_ROOT/'isaacsimenvs/tasks/play/utils'
    pkg = 'p2p_play_utils'
    if pkg not in sys.modules:
        m = type(sys)(pkg)
        m.__path__ = [str(base)]
        sys.modules[pkg] = m
    spec = importlib.util.spec_from_file_location(pkg+'.generate_objects', base/'generate_objects.py')
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_env_cfg(path=None):
    class _L(yaml.SafeLoader):
        pass
    _L.add_constructor('tag:yaml.org,2002:python/tuple', lambda l, n: tuple(l.construct_sequence(n)))
    return yaml.load(Path(path or M_DIR/'params/env_resolved.yaml').read_text(), Loader=_L)


def object_bank(env_cfg=None):
    """(bank, families, table) of the run's procedural tools (same layout as tool_holds.object_bank_for_run)."""
    from dextrah_lab.mj.objects import build_object_bank
    a = (env_cfg or load_env_cfg())['assets']
    out = tempfile.mkdtemp(prefix='m1454_tools_')
    paths, _ = _generator().generate_handle_head_urdfs(handle_head_types=tuple(a['handle_head_types']),
                                                       num_per_type=a['num_assets_per_type'], out_dir=out,
                                                       shuffle=a['shuffle_assets'])
    bank = build_object_bank([str(p) for p in paths])
    families = [Path(p).stem.split('_')[1] for p in paths]
    table = dict(size=bank.size, pos=bank.pos-bank.com[:, None], quat=bank.quat, rbound=None,
                 mass=bank.mass, inertia=bank.inertia, used=bank.used)
    return bank, families, table


def load_sgs(path=None):
    return torch.load(path or M_DIR/'best/best_epoch12662.sgs.pt', map_location='cpu', weights_only=False)


def lowest_point(table, a, pose):
    """World z of the lowest point of tool a at object pose (7, COM frame)."""
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, pose[3:])
    R = R.reshape(3, 3)
    low = np.inf
    for k in range(4):
        if not table['used'][a][k]:
            continue
        gR = np.zeros(9)
        mujoco.mju_quat2Mat(gR, table['quat'][a][k])
        gR = R @ gR.reshape(3, 3)
        c = pose[:3]+R @ table['pos'][a][k]
        s = table['size'][a][k]
        if k in (0, 2):                                          # box
            low = min(low, c[2]-np.abs(gR[2]) @ s)
        else:                                                    # capsule along its z
            low = min(low, c[2]-abs(gR[2, 2])*s[1]-s[0])
    return low


def validate_bank(state, table, n=2000, seed=0):
    """Reaching-family spawns rest on the table: lowest tool point - table top (m), over n configurations."""
    fam = family_layout(state['settings'])
    c = state['cfg']
    idx = np.nonzero(state['filled'].numpy() & (fam == SGS_FAMILIES.index('reaching')))[0]
    idx = np.random.default_rng(seed).choice(idx, min(n, len(idx)), replace=False)
    gaps = np.array([lowest_point(table, int(c['asset'][i]), c['object_pose'][i].numpy().astype(np.float64))
                     - (float(c['table_z'][i])+.15) for i in idx])
    return gaps


def robot_model():
    from dextrah_lab.mj.robot import ContactParams, build_robot_spec
    return build_robot_spec(ContactParams(condim=4)).compile()


def decode(state, table, families, limit=None, seed=0):
    """HoldPool of the held families (stable_grasp, near_goal)."""
    c = {k: v.numpy() if torch.is_tensor(v) else v for k, v in state['cfg'].items()}
    fam = family_layout(state['settings'])
    held = np.nonzero(state['filled'].numpy() & np.isin(fam, [SGS_FAMILIES.index(f) for f in HELD_FAMILIES]))[0]
    if limit is not None and limit < len(held):
        held = np.sort(np.random.default_rng(seed).choice(held, limit, replace=False))
    m = robot_model()
    if m.nq != 58:
        raise RuntimeError(f'robot nq {m.nq} != 58')
    d = mujoco.MjData(m)
    names = [m.joint(j).name for j in range(m.njnt)]
    qadr = {n: int(m.jnt_qposadr[j]) for j, n in enumerate(names)}
    q, jp = c['qpos'][held[:2000]], c['joint_pos'][held[:2000]]
    col_name = []
    for col in range(jp.shape[1]):
        err = {n: np.abs(q[:, a]-jp[:, col]).max() for n, a in qadr.items() if 'floating' not in n}
        best = min(err, key=err.get)
        if err[best] > 1e-4:
            raise RuntimeError(f'joint_pos column {col} matches no robot joint ({best}, {err[best]:.2e})')
        col_name.append(best)
    hand_cols = [col_name.index(j) for j in HAND_JOINTS]
    base = m.body(HAND_ROOT).id
    palm = np.zeros((len(held), 7))
    for i, k in enumerate(held):
        d.qpos[:] = c['qpos'][k][:58]
        mujoco.mj_kinematics(m, d)
        palm[i] = np.concatenate((d.xpos[base], d.xquat[base]))
    sampler = state['sampler']
    counts, outcomes = sampler['count'].numpy(), sampler['outcomes'].numpy()
    used = np.minimum(counts, outcomes.shape[1])
    success = np.where(used > 0, outcomes.sum(1)/np.maximum(used, 1), np.nan)
    return HoldPool(keys=held, asset=c['asset'][held].astype(np.int64), family=[families[int(c['asset'][k])] for k in held],
                    palm=palm, hand_q=c['joint_pos'][held][:, hand_cols], hand_target=c['targets'][held][:, hand_cols],
                    object_qpos=c['qpos'][held][:, 58:65].astype(np.float64), table_top=c['table_z'][held]+.15,
                    arm_touch=np.zeros(len(held), bool), success=success[held],
                    meta=dict(held=int(len(held)), joint_columns=col_name,
                              sgs_family=[SGS_FAMILIES[fam[k]] for k in held]))


_CACHE = {}


def cached_pool():
    """(pool, table) of all held configurations, decoded once per process."""
    if 'pool' not in _CACHE:
        _, fams, table = object_bank()
        _CACHE['table'] = table
        _CACHE['pool'] = decode(load_sgs(), table, fams)
    return _CACHE['pool'], _CACHE['table']


def m_episode(rng, i=None, settle_steps=30, carry_steps=40, carry_speed=.2, carry_rate=1., lower_speed=.15,
              release_steps=30):
    """A census episode from M's hold i (random if None): the hold is rebuilt in the testbed (tool_holds.accept's
    reset), settles with M's targets for 0.5 s ('settle', not labelled), the palm then carries the tool with smooth
    random linear / angular motion ('hold'), lowers it to the table ('lower', until object-table contact + 10 steps),
    and the fingers open linearly ('release'). Returns (EpisodeLog, testbed) like scenes.table_episode."""
    from ..structured.scenes import EpisodeLog, tool_testbed
    from .held_states import axis_angle, mat, pose
    from .tool_holds import tool_of
    pool, table = cached_pool()
    i = int(rng.integers(len(pool.keys))) if i is None else i
    tool = tool_of(table, int(pool.asset[i]))
    tool['handle'] = pool.family[i]
    tb = tool_testbed(tool)
    top = float(pool.table_top[i])
    palm0 = pool.palm[i].copy()
    tb.reset(palm0, pool.hand_q[i], pool.object_qpos[i], table_top=top)
    targets = pool.hand_target[i].copy()
    d = tb.d
    log = dict(states=[], palm=[], targets=[], phase=[], obj=[], forces=[], table_force=[])
    cur = palm0.copy()

    def step(nxt, tgt, phase):
        nonlocal cur
        log['states'].append(tb.get_state())
        log['palm'].append(cur.copy())
        log['targets'].append(tgt.copy())
        log['phase'].append(phase)
        log['obj'].append(d.qpos[tb.object_qadr:tb.object_qadr+7].copy())
        tb.step(tgt, cur, nxt)
        log['forces'].append(tb.finger_forces())
        log['table_force'].append(sum(float(c['force_local'][0]) for c in tb.contacts() if c['pair'] == 'object_table'))
        cur = nxt
    for _ in range(settle_steps):
        step(cur.copy(), targets, 'settle')
    # carry: velocity follows a smoothed random walk (walking-like palm motion)
    v, w = np.zeros(3), np.zeros(3)
    dt = 1/60
    for _ in range(carry_steps):
        v = .9*v+.1*rng.normal(0, carry_speed, 3)
        w = .9*w+.1*rng.normal(0, carry_rate, 3)
        nxt = cur.copy()
        nxt[:3] += v*dt
        nxt[3:] = pose(np.zeros(3), axis_angle(w/max(np.linalg.norm(w), 1e-12), np.linalg.norm(w)*dt) @ mat(cur[3:]))[3:]
        step(nxt, targets, 'hold')
    for k in range(400):
        nxt = cur.copy()
        nxt[2] -= lower_speed*dt
        step(nxt, targets, 'lower')
        if log['table_force'][-1] > 0:
            for _ in range(10):
                step(cur.copy(), targets, 'lower')
            break
        if d.qpos[tb.object_qadr+2] < top-.05:          # dropped past the table edge
            break
    for k in range(release_steps):
        step(cur.copy(), targets*(1-(k+1)/release_steps), 'release')
    log['palm'].append(cur.copy())
    log['obj'].append(d.qpos[tb.object_qadr:tb.object_qadr+7].copy())
    return EpisodeLog(tool=tool, states=log['states'], palm=np.array(log['palm']), targets=np.array(log['targets']),
                      phase=log['phase'], obj=np.array(log['obj']), forces=log['forces'],
                      table_force=np.array(log['table_force'])), tb
