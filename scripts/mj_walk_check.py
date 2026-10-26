#!/usr/bin/env python3
"""Inference-only walking check in the mjlab whole-body env (frozen SONIC + released planner).

No optimizer and no training change. One env holds two groups of selected worlds:

walk         Bare SONIC: zero latent residual, mid-range fingers, released-planner reference
             (arms included) from constant body-frame commands: idle 0-2 s, command 2-10 s,
             idle afterwards. The empty hand leaves the object, so only robot falls and
             numerical failures end an episode.
carry        The checkpoint's SAPG leader (mean actions) grasps from a validated reaching start
             with the task's own goals. After a confirmed grasp (object lifted, >= 2 fingers on
             it, for 1 s) the object goal becomes the grasp pose fixed in the robot's yaw frame,
             SONIC follows the planner (legs/waist; the trained standing arm reference is kept,
             as in the 2026-09-15 Isaac test) and a known-map waypoint route walks to a second,
             virtual table: back off, sideways to the second table, forward to the same
             stand-off. Task success/timeout counting is off while carrying; object drops and
             robot falls still end the attempt. Further attempts follow while time remains.
carry_zero   As carry, but the latent residual is zero while walking (fingers stay the policy's):
             isolates whether the standing-trained residual opposes walking.

The planner bridge (dextrah_lab/wholebody/navigation_reference.py) and the waypoint driver
(navigation_transfer.WaypointVelocity) are the Isaac-era pieces, unchanged. SONIC orientation
tokens use the planner root orientation relative to the measured yaw, as there. Rendering:
wide view + tracking view; the second table is drawn translucent (it is not simulated).
"""
import argparse
import json
import os
from pathlib import Path
import traceback

import numpy as np

COMMANDS = (('back', (-.4, 0., 0.)), ('left', (0., .4, 0.)), ('right', (0., -.4, 0.)),
            ('turn_left', (0., 0., .2)), ('turn_right', (0., 0., -.2)))
TYPES = ('hammer', 'screwdriver', 'marker', 'spatula', 'eraser', 'brush')
PHASES = ('walk', 'grasp', 'carry', 'arrived')
GRASP_STEPS = 60            # 1 s at 60 Hz
FORCE_N = .1


class World:
    def __init__(self, env, role, name=None, command=None):
        self.env, self.role, self.name = env, role, name
        self.command_fixed = np.asarray(command if command is not None else (0., 0., 0.), float)
        self.command = np.zeros(3)
        self.phase = 'walk' if role == 'walk' else 'grasp'
        self.nav, self.route, self.offset = None, None, None
        self.grasp_count, self.attempts, self.attempt = 0, [], None
        self.falls = 0
        self.force_reset = False

    @property
    def navigating(self):
        return self.phase in ('walk', 'carry', 'arrived')


def body_frame(quat_wxyz, vec_w):
    w, x, y, z = quat_wxyz
    yaw = np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z))
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([c*vec_w[0]+s*vec_w[1], -s*vec_w[0]+c*vec_w[1]]), yaw


def run(args):
    import torch
    import mujoco_warp
    import dextrah_lab.mj.env as mjenv
    from isaaclab.utils.math import matrix_from_quat, quat_conjugate, quat_mul, yaw_quat
    from dextrah_lab.g1_adept.contact import FINGERS
    from dextrah_lab.sgs.mj_task import MjSgsTask, SgsSettings
    from dextrah_lab.wholebody.contract import BODY_JOINTS
    from dextrah_lab.wholebody.navigation_reference import NavigationReference, yaw_of
    from dextrah_lab.wholebody.navigation_transfer import WaypointVelocity
    from eval_wholebody_checkpoint import build_policy
    import record_wholebody_videos as rwv
    device = 'cuda:0'
    run_dir = args.checkpoint.resolve().parent.parent
    contract = json.loads((run_dir/'task_contract.json').read_text())
    if 'mujoco_warp_cone_fix' not in Path(mujoco_warp.__file__).parts:
        raise RuntimeError('Training used the vendored mujoco_warp cone Hessian fix; put it on PYTHONPATH')
    args.no_randomization = True
    env, cfg = rwv.build_env(args, run_dir, contract, device)
    # Object-only finger forces for the grasp check (the grasp-score module's contact channel;
    # scale 0 adds nothing to the reward, which this check does not use anyway).
    env.enable_grasp_reward(0.)
    model, payload = build_policy(env, args.checkpoint, contract, device)
    paths = env._object_urdf_paths
    asset = env._object_asset_index_per_env.cpu().numpy()
    kind = np.array([Path(paths[a]).stem.split('_')[1] for a in asset])

    # World selection before any rollout (no outcome filtering).
    worlds, used = {}, set()
    def take(t=None):
        for j in range(env.num_envs):
            if j not in used and (t is None or kind[j] == t):
                used.add(j)
                return j
        raise RuntimeError(f'No world left for {t}')
    for t in TYPES:
        for _ in range(args.carry_per_type):
            j = take(t); worlds[j] = World(env, 'carry')
        for _ in range(args.carry_zero_per_type):
            j = take(t); worlds[j] = World(env, 'carry_zero')
    for name, command in COMMANDS:
        for _ in range(args.walk_repeats):
            j = take(); worlds[j] = World(env, 'walk', name, command)
    selected = sorted(worlds)
    sel = torch.tensor(selected, device=device)

    settings = SgsSettings(**dict(contract['sgs']['settings'], eval_pool_size=0))
    state = torch.load(args.checkpoint.with_suffix('.sgs.pt'), map_location='cpu', weights_only=False)
    g = torch.Generator(device=device).manual_seed(args.seed)
    task = MjSgsTask(env, settings, generator=g, state=state)
    starts = rwv.ValidatedReaching(env, task, g, selected)

    planner_path = Path('/data1/users/konstantin.smirnov/G1-SONIC-planner/planner_sonic.onnx')
    first = NavigationReference(planner_path, speed_limit=max(.3, args.speed))
    for w in worlds.values():
        w.nav = NavigationReference(planner_path, session=first.session, speed_limit=max(.3, args.speed))
    arm = [i for i, n in enumerate(BODY_JOINTS) if any(p in n for p in ('shoulder', 'elbow', 'wrist'))]
    thumb = FINGERS.index('thumb')
    standing_q = env._reference_q.clone()
    table_a = np.array([0., 0., .65])
    lateral = np.array([-1., 0.])          # robot's right for the -90 deg start yaw (checked below)
    table_b = table_a.copy(); table_b[:2] += args.table_offset*lateral

    def masks():
        no_task = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
        no_success = torch.zeros_like(no_task)
        for j, w in worlds.items():
            no_task[j] = w.role == 'walk'
            no_success[j] = w.role != 'walk' and w.phase in ('carry', 'arrived')
        return no_task, no_success

    # Task terminations: walk worlds keep only robot fall / numerical failure (added later
    # by combine_terminations); carrying worlds drop success counting and timeouts.
    original_terminations = mjenv.termination_utils.compute_terminations
    def terminations(e):
        if e is not env:
            return original_terminations(e)
        no_task, no_success = masks()
        off = no_task | no_success
        force = torch.zeros_like(off)
        for j, w in worlds.items():
            force[j] = w.force_reset
        e._is_success = e._is_success & ~off
        terminated, truncated = original_terminations(e)
        r = e._termination_reasons
        r['timeout'] = (r['timeout'] & ~off) | force       # carry attempt over its time limit
        r['max_successes'] = r['max_successes'] & ~off
        r['fall'] = r['fall'] & ~no_task
        r['hand_far'] = r['hand_far'] & ~no_task
        return r['fall'] | r['hand_far'] | r['max_successes'], r['timeout']
    mjenv.termination_utils.compute_terminations = terminations

    # SONIC reference: planner frames for navigating worlds (as navigation_transfer_env).
    original_observation = env._wholebody_observation
    def wholebody_observation():
        ids = [j for j in selected if worlds[j].navigating]
        if ids:
            now = env._sim_step_counter*env.physics_dt
            rp = (env.robot.data.root_pos_w[ids]-env.scene.env_origins[ids]).cpu().numpy()
            rq = env.robot.data.root_quat_w[ids].cpu().numpy()
            bq = env.robot.data.joint_pos[ids][:, env._body_ids].cpu().numpy()
            Q, QD, R = [], [], []
            for k, j in enumerate(ids):
                w = worlds[j]
                root = np.r_[rp[k], rq[k]]
                if w.nav.frames is None:
                    w.nav.reset(root, bq[k], now)
                q, qd, ref = w.nav.reference(now, w.command, yaw_of(rq[k]))
                if w.role != 'walk':
                    q[:, arm] = standing_q[j][:, arm].cpu().numpy()
                    qd[:, arm] = 0.
                Q.append(q); QD.append(qd); R.append(ref[:, 3:])
            env._reference_q[ids] = torch.as_tensor(np.stack(Q), device=device)
            env._reference_qd[ids] = torch.as_tensor(np.stack(QD), device=device)
            ref_quat = torch.as_tensor(np.stack(R), device=device)
        extra = original_observation()
        if ids:
            n = len(ids)
            yq = quat_conjugate(yaw_quat(env.robot.data.root_quat_w[ids]))[:, None].expand(-1, 10, -1)
            delta = quat_mul(yq.reshape(-1, 4), ref_quat.reshape(-1, 4)).reshape(n, 10, 4)
            ori = matrix_from_quat(delta)[..., :2].reshape(n, 10, 6)
            env._reference_ori6 = env._reference_ori6.clone()
            env._reference_ori6[ids] = ori
            extra[ids, 930:] = env._sonic_reference.reference_tokens(env._reference_q[ids], env._reference_qd[ids], ori)
        return extra
    env._wholebody_observation = wholebody_observation

    class Hook:
        def before_reset(self, env_ids):
            starts.before_reset(env_ids)

        def after_reset(self, env_ids):
            starts.after_reset(env_ids)
            for j in env_ids.reshape(-1).tolist():
                w = worlds.get(j)
                if w is None:
                    continue
                env._reference_q[j] = standing_q[j]
                env._reference_qd[j] = 0.
                w.nav.frames = None
                w.grasp_count, w.route, w.offset = 0, None, None
                w.force_reset = False
                w.command[:] = 0.
                if w.role != 'walk':
                    w.phase = 'grasp'

        def after_step(self):
            starts.after_step()
    env._sgs = Hook()

    dt = env.step_dt
    steps = round(args.seconds/dt)
    stride = max(1, round(1/(args.fps*dt)))
    leader = torch.zeros(env.num_envs, 1, device=device)
    a = lambda x: x.detach().cpu().numpy().copy()
    frames = []
    walk_vel = {j: [] for j, w in worlds.items() if w.role == 'walk'}
    with torch.inference_mode():
        obs, _ = env.reset()
        rot0 = a(env.robot.data.root_quat_w[sel])
        # The route geometry assumes the training start yaw (-90 deg: facing world -Y).
        yaw0 = np.array([yaw_of(q) for q in rot0])
        if not np.allclose(yaw0, -np.pi/2, atol=.05):
            raise RuntimeError(f'Unexpected start yaw {yaw0}')
        states = tuple(s.to(device) for s in model.get_default_rnn_state())
        for step in range(steps):
            t = step*dt
            rp = a(env.robot.data.root_pos_w[sel]-env.scene.env_origins[sel])
            rq = a(env.robot.data.root_quat_w[sel])
            rv = a(env.robot.data.root_lin_vel_w[sel])
            wz = a(env.robot.data.root_ang_vel_w[sel][:, 2])
            # Commands and carry goals from the current state.
            goal_ids, goal_pose = [], []
            for k, j in enumerate(selected):
                w = worlds[j]
                if w.role == 'walk':
                    w.command[:] = w.command_fixed if 2. <= t < 10. else 0.
                    if 4. <= t < 10.:
                        vb, _ = body_frame(rq[k], rv[k])
                        walk_vel[j].append((vb[0], vb[1], wz[k]))
                    continue
                if w.phase in ('carry', 'arrived'):
                    if w.force_reset:
                        w.command[:] = 0.
                        continue
                    if w.phase == 'carry':
                        w.command[:] = w.route.update(np.r_[rp[k], rq[k]], rv[k], dt)
                        w.attempt.update(progress=int(w.route.index), remaining_m=round(w.route.distance, 3))
                        if w.route.done:
                            w.phase = 'arrived'
                            w.attempt.update(arrived_s=round(t-w.attempt['carry_start_s'], 2))
                        elif t-w.attempt['carry_start_s'] > args.carry_limit:
                            # End the attempt with a reset on the next step (no arrival in time).
                            w.attempt.update(outcome='no_arrival', end_s=round(t, 2))
                            w.attempt, w.force_reset = None, True
                            w.command[:] = 0.
                            continue
                    else:
                        w.command[:] = 0.
                    _, yaw = body_frame(rq[k], (0., 0.))
                    c, s = np.cos(yaw), np.sin(yaw)
                    off_xy, z, rel = w.offset
                    pos = np.r_[rp[k][:2]+np.array([c*off_xy[0]-s*off_xy[1], s*off_xy[0]+c*off_xy[1]]), z]
                    goal_ids.append(j); goal_pose.append(np.r_[pos, quat_np(yaw, rel)])
            if goal_ids:
                pose = torch.as_tensor(np.stack(goal_pose), device=device, dtype=torch.float32)
                pose[:, :3] += env.scene.env_origins[goal_ids]
                env.goal_viz.write_root_pose_to_sim(pose, torch.tensor(goal_ids, device=device))
            out = model(dict(obs=torch.cat((obs['policy'], leader), -1), is_train=False, prev_actions=None, rnn_states=states))
            states = out['rnn_states']
            action = out['mus'].clone()
            for j, w in worlds.items():
                if w.role == 'walk':
                    action[j] = 0.
                elif w.role == 'carry_zero' and w.phase in ('carry', 'arrived'):
                    action[j, :64] = 0.
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated | truncated
            for s_ in states:
                s_[:, done, :] = 0
            d = a(done[sel]).astype(bool)
            robot_fall = a(info['episode_final']['robot_fall'][sel]).astype(bool)
            reasons = {r: a(v[sel]) for r, v in env._termination_reasons.items()}
            lifted = a(env._lifted_object[sel]).astype(bool)
            force = a(env._last_contacts['finger_object_force_l1'][sel])
            touching = (force > FORCE_N).sum(-1)
            opos = a(env.object.data.root_pos_w[sel]-env.scene.env_origins[sel])
            oquat = a(env.object.data.root_quat_w[sel])
            for k, j in enumerate(selected):
                w = worlds[j]
                if d[k]:
                    why = [r for r, v in reasons.items() if v[k]]+(['robot_fall'] if robot_fall[k] else [])
                    w.falls += int(robot_fall[k])
                    if w.attempt is not None:
                        outcome = 'robot_fell' if robot_fall[k] else ('dropped' if 'fall' in why else '+'.join(why))
                        if w.attempt.get('arrived_s') is not None:
                            outcome = 'arrived_then_'+outcome
                        w.attempt.update(outcome=outcome, end_s=round(t+dt, 2))
                        w.attempt = None
                    continue
                if w.role != 'walk' and w.phase == 'grasp':
                    w.grasp_count = w.grasp_count+1 if (lifted[k] and touching[k] >= 2) else 0
                    if w.grasp_count >= GRASP_STEPS:
                        # Freeze the grasp pose in the robot's yaw frame; start the route.
                        _, yaw = body_frame(rq[k], (0., 0.))
                        c, s = np.cos(yaw), np.sin(yaw)
                        dxy = opos[k][:2]-rp[k][:2]
                        off = np.array([c*dxy[0]+s*dxy[1], -s*dxy[0]+c*dxy[1]])
                        w.offset = (off, float(opos[k][2]), rel_quat_np(yaw, oquat[k]))
                        x0, y0 = rp[k][0], rp[k][1]
                        stand = y0-table_a[1]
                        waypoints = [(x0, y0+args.backoff), (table_b[0], y0+args.backoff), (table_b[0], table_b[1]+stand)]
                        w.route = WaypointVelocity(waypoints, yaw, speed=args.speed)
                        w.phase = 'carry'
                        w.nav.frames = None
                        w.attempt = dict(grasp_s=round(t+dt, 2), carry_start_s=t+dt, waypoints=np.round(waypoints, 3).tolist())
                        w.attempts.append(w.attempt)
                if w.phase == 'arrived' and w.attempt is not None and 'holding_at_arrival' not in w.attempt:
                    w.attempt['holding_at_arrival'] = bool(lifted[k] and touching[k] >= 1)
            if step % stride == 0:
                gv = env.goal_viz.data
                frames.append(dict(time_s=t, qpos=a(env.sim.data.qpos[sel]),
                                   goal=a(torch.cat((gv.root_pos_w[sel], gv.root_quat_w[sel]), -1)),
                                   phase=np.array([PHASES.index(worlds[j].phase) for j in selected]),
                                   command=np.stack([worlds[j].command.copy() for j in selected]),
                                   vel_b=np.stack([np.r_[body_frame(rq[k], rv[k])[0], wz[k]] for k in range(len(selected))]),
                                   lifted=lifted.copy(), touching=touching.copy(),
                                   arrivals=np.array([sum(1 for x in worlds[j].attempts if x.get('arrived_s') is not None) for j in selected]),
                                   falls=np.array([worlds[j].falls for j in selected])))
    for k, j in enumerate(selected):
        w = worlds[j]
        if w.attempt is not None:
            holding = bool(frames[-1]['lifted'][k] and frames[-1]['touching'][k] >= 1)
            w.attempt['holding_at_end'] = holding
            w.attempt['outcome'] = ('arrived_holding_at_end' if holding else 'arrived_not_holding_at_end') \
                if w.phase == 'arrived' else 'running_at_end'
    stacked = {key: np.stack([f[key] for f in frames]) for key in frames[0]}
    np.savez_compressed(args.output/'trajectories.npz', **stacked)
    walk = {}
    for j, w in worlds.items():
        if w.role != 'walk':
            continue
        v = np.array(walk_vel[j]) if walk_vel[j] else np.zeros((1, 3))
        mean = v.mean(0)
        cmd = w.command_fixed
        lin = float(np.dot(mean[:2], cmd[:2])/(np.linalg.norm(mean[:2])*np.linalg.norm(cmd[:2])+1e-9)) if np.linalg.norm(cmd[:2]) else None
        walk[j] = dict(command=w.name, command_vec=cmd.tolist(), mean_vx=round(float(mean[0]), 3), mean_vy=round(float(mean[1]), 3),
                       mean_wz=round(float(mean[2]), 3), direction_cosine=None if lin is None else round(lin, 3), robot_falls=w.falls)
    carry = {j: dict(role=w.role, type=str(kind[j]), asset=int(asset[j]), attempts=w.attempts, robot_falls=w.falls)
             for j, w in worlds.items() if w.role != 'walk'}
    meta = dict(checkpoint=str(args.checkpoint), run=str(run_dir), epoch=int(payload['epoch']), fps=args.fps, step_dt=dt,
                seconds=args.seconds, speed=args.speed, backoff=args.backoff, table_offset=args.table_offset,
                table_a=table_a.tolist(), table_b=table_b.tolist(), selected=selected,
                envs=[dict(env=j, asset=int(asset[j]), type=str(kind[j]), role=worlds[j].role, name=worlds[j].name) for j in selected],
                walk=walk, carry=carry, planner=first.report(), domain_randomization='off', policy='SAPG leader, mean actions')
    (args.output/'recording.json').write_text(json.dumps(meta, indent=2, default=float))
    from dextrah_lab.mj.scene import object_indices
    body, slot_geoms = object_indices(env.sim.mj_model)
    bank = env.object_bank
    np.savez(args.output/'objects.npz', size=bank.size, pos=bank.pos, quat=bank.quat, used=bank.used, com=bank.com,
             slot_geoms=slot_geoms, body=body)
    return meta, cfg


def quat_np(yaw, rel):
    """yaw quaternion (wxyz) * rel."""
    q = np.array([np.cos(yaw/2), 0., 0., np.sin(yaw/2)])
    return quat_mul_np(q, rel)


def rel_quat_np(yaw, quat):
    q = np.array([np.cos(yaw/2), 0., 0., -np.sin(yaw/2)])
    return quat_mul_np(q, np.asarray(quat, float))


def quat_mul_np(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2,
                     w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2])


def render(args, mj_model, meta):
    import copy
    import imageio.v2 as imageio
    import mujoco
    from PIL import Image, ImageDraw, ImageFont
    from record_fixed_policy_videos import quat_mat
    from dextrah_lab.mj.scene import TABLE_SIZE
    tr = np.load(args.output/'trajectories.npz')
    ob = np.load(args.output/'objects.npz')
    slot_type = [mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_CAPSULE]*2
    font = ImageFont.load_default()
    W, H = args.width, args.height
    out_dir = args.output/'videos'
    out_dir.mkdir(exist_ok=True)
    table_b = np.array(meta['table_b'])
    for k, e in enumerate(meta['envs']):
        m = copy.copy(mj_model)
        a = e['asset']
        m.vis.global_.offwidth, m.vis.global_.offheight = max(W, 640), max(H, 480)
        m.vis.headlight.ambient[:] = (.45, .45, .45)
        m.vis.headlight.diffuse[:] = (.55, .55, .55)
        m.geom_rgba[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, 'table/table')] = (.62, .48, .33, 1.)
        for s, g in enumerate(ob['slot_geoms']):
            m.geom_size[g] = ob['size'][a, s]
            m.geom_pos[g] = ob['pos'][a, s]-ob['com'][a]
            m.geom_quat[g] = ob['quat'][a, s]
            m.geom_rgba[g] = (.95, .55, .1, 1.) if ob['used'][a, s] else (0, 0, 0, 0)
            m.geom_sameframe[g] = 0
        d = mujoco.MjData(m)
        renderer = mujoco.Renderer(m, H, W)
        opt = mujoco.MjvOption()
        opt.geomgroup[:] = 1
        wide = mujoco.MjvCamera()
        wide.lookat[:] = (table_b[0]/2, .5, .6)
        wide.azimuth, wide.elevation, wide.distance = 120., -25., 3.6
        track = mujoco.MjvCamera()
        track.azimuth, track.elevation, track.distance = 150., -15., 1.9
        carry = e['role'] != 'walk'
        title = f"{e['role']}{' '+e['name'] if e['name'] else ''}  {e['type']} asset {a}  env {e['env']}  |  epoch {meta['epoch']}"
        name = f"{k:02d}_{e['role']}_{e['name'] or e['type']}_env{e['env']}.mp4"
        writer = imageio.get_writer(out_dir/name, fps=meta['fps'], codec='libx264', quality=8, macro_block_size=1)
        for f in range(len(tr['time_s'])):
            d.qpos[:] = tr['qpos'][f, k]
            mujoco.mj_forward(m, d)
            track.lookat[:] = (d.qpos[0], d.qpos[1], .8)
            phase = PHASES[int(tr['phase'][f, k])]
            views = []
            for c in (wide, track):
                renderer.update_scene(d, camera=c, scene_option=opt)
                sc = renderer.scene
                if carry and sc.ngeom < sc.maxgeom:
                    mujoco.mjv_initGeom(sc.geoms[sc.ngeom], int(mujoco.mjtGeom.mjGEOM_BOX), np.array(TABLE_SIZE)/2, table_b,
                                        np.eye(3).flatten(), np.array([.62, .48, .33, .35], dtype=np.float32))
                    sc.ngeom += 1
                if carry and phase in ('grasp', 'carry', 'arrived'):
                    gp, gq = tr['goal'][f, k, :3], tr['goal'][f, k, 3:]
                    R = quat_mat(gq)
                    for s in range(4):
                        if not ob['used'][a, s] or sc.ngeom >= sc.maxgeom:
                            continue
                        mujoco.mjv_initGeom(sc.geoms[sc.ngeom], int(slot_type[s]), ob['size'][a, s], gp+R @ ob['pos'][a, s],
                                            (R @ quat_mat(ob['quat'][a, s])).flatten(), np.array([.1, .9, .2, .3], dtype=np.float32))
                        sc.ngeom += 1
                views.append(renderer.render())
            img = Image.fromarray(np.concatenate(views, 1))
            draw = ImageDraw.Draw(img)
            cmd, vel = tr['command'][f, k], tr['vel_b'][f, k]
            status = (f"t={tr['time_s'][f]:5.2f}s  phase {phase}  cmd vx {cmd[0]:+.2f} vy {cmd[1]:+.2f} wz {cmd[2]:+.2f}"
                      f"  measured vx {vel[0]:+.2f} vy {vel[1]:+.2f} wz {vel[2]:+.2f}  lifted {'yes' if tr['lifted'][f, k] else 'no'}"
                      f"  fingers on object {int(tr['touching'][f, k])}  arrivals {int(tr['arrivals'][f, k])}  robot falls {int(tr['falls'][f, k])}")
            draw.rectangle((0, 0, img.width, 30), fill=(0, 0, 0))
            draw.text((6, 2), title, fill=(255, 255, 255), font=font)
            draw.text((6, 16), status, fill=(255, 230, 120), font=font)
            note = ('translucent brown: second table (target, not simulated)   green: object goal (grasp pose held in the body frame while walking)'
                    if carry else 'bare SONIC + released planner, zero latent residual, empty hand')
            draw.text((6, img.height-14), note, fill=(255, 255, 255), font=font)
            writer.append_data(np.asarray(img))
        writer.close()
        renderer.close()
        print(f'VIDEO {out_dir/name}', flush=True)


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Run inside a Slurm GPU allocation')
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--num-envs', type=int, default=1200, help='>= assets: one env per asset')
    p.add_argument('--seconds', type=float, default=60.)
    p.add_argument('--speed', type=float, default=.4, help='navigation cruise speed (m/s)')
    p.add_argument('--backoff', type=float, default=.6, help='step back from the first table (m)')
    p.add_argument('--table-offset', type=float, default=1., help='second table centre, to the robot\'s right (m)')
    p.add_argument('--carry-limit', type=float, default=30., help='seconds per carry attempt before "no arrival"')
    p.add_argument('--carry-per-type', type=int, default=2)
    p.add_argument('--carry-zero-per-type', type=int, default=1)
    p.add_argument('--walk-repeats', type=int, default=2)
    p.add_argument('--fps', type=int, default=30)
    p.add_argument('--width', type=int, default=640)
    p.add_argument('--height', type=int, default=480)
    p.add_argument('--nconmax', type=int, default=64)
    p.add_argument('--njmax', type=int, default=300)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--no-render', action='store_true')
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    report = dict(completed=False)
    try:
        from dextrah_lab.mj.compat import install
        install()
        meta, cfg = run(args)
        report.update(walk=meta['walk'], carry=meta['carry'])
        print('WALK_CHECK_RESULTS '+json.dumps(dict(walk=meta['walk'], carry=meta['carry']), default=float), flush=True)
        if not args.no_render:
            import record_wholebody_videos as rwv
            render(args, rwv.compile_scene(cfg), meta)
        report.update(completed=True)
        print('WALK_CHECK_COMPLETE', flush=True)
    except BaseException as error:
        report.update(error=str(error), traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        (args.output/'result.json').write_text(json.dumps(report, indent=2, default=float))
        import sys
        sys.stdout.flush(); sys.stderr.flush()
        os._exit(0 if report['completed'] else 1)


if __name__ == '__main__':
    main()
