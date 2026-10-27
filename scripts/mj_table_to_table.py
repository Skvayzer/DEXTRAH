#!/usr/bin/env python3
"""Phase 2 evaluation: table-to-table transfer with a hard switch M -> C -> M (inference only).

M is the manipulation expert (frozen SAPG leader, mean actions), C the carry expert (carry run
checkpoint, mean actions). One attempt per world, all worlds in one env:

  grasp     M grasps from a validated reaching start on table A with the task's own goals.
            Confirmed grasp: object lifted with >= --grasp-min-fingers distal pads on it for 1 s.
  settle    switch to C (LSTM zero, as in its training): finger reference q_ref = M's finger
            command at the switch, arm reference = the arm joint angles at the switch, palm-frame
            hold position recorded, observation frame anchored at the robot pose; standing 1.5 s
            (C's 0.5 s warm-up + 1 s settle), table A still there.
  walk      released planner (legs/waist) following a waypoint route at --speed: back off, sideways
            to table B, forward to the same stand-off; C's context gets the body-frame command.
            One table body per world: once the robot has backed off (first waypoint reached) the
            table moves to table B's place, so the robot meets a real table at the destination.
  final     standing 2 s (C's final phase); arrival holding = palm-frame offset within 5 cm.
  place     switch to M (LSTM zero, as on its SGS held starts), standing reference, observation
            frame anchored at the episode start pose (table B is where table A was relative to the
            robot); goal = the object's resting pose on table A (the last still, untouched pose in
            the first 3 s: validated reaching starts drop the object from ~10 cm), moved to table B
            ('rest'), or the same raised to the lowest goal height M was trained on ('hover'), or at the
            same table-B spot the library resting pose nearest to the held orientation ('nearest',
            --place-library; least in-hand rotation, as the 'nearest' placement-goal training).
  release   only once table B supports the object or it is at the goal, and it is still: object-table
            normal force > 0.5 N or within 3 cm of the goal, object velocity averaged over 0.2 s
            < 5 cm/s (held objects chatter in the hand at ~10 cm/s instantaneous), for 0.25 s; then
            the fingers go to their pre-grasp commands with the arm command frozen, for 2 s.
            Support at the release (the strict rule) is reported per world.
  result    placed = resting on table B (contact, speed < 5 cm/s), no hand contact, robot upright.

C's end states at the hand-back (arrived holding) are saved in the SGS holding-state field layout
(handback_states.pt) as starts for a later placement fine-tune of M.

Carry v2 experts (contract version 2, docs/carry_v2_design.md) run with their own semantics, via the
helpers the training env uses: at the switch the carry posture for this grasp (nominal carry target,
no jitter); right-arm reference = minimum jerk from the grasp-time angles to that posture over C's
secure phase + the filtered arm channel (first 7 latent outputs); latent residual off; asymmetric
finger bound; v2 context. The settle phase lasts C's warm-up + secure phase. Hold rule (--hold-rule,
default 'support' for v2, 'offset' for v1): 'support' = drop when not supported by the hand (>= 1
pad / palm contact, within 15 cm of the palm, no table / floor contact) for C's unsupported_s or below
the floor height, holding at the hand-back = supported; 'offset' = v1's palm-frame offset rules.
--contact-condim builds the scene with that contact model (C v2 trains with condim 4).
"""
import argparse
import json
import math
import os
from pathlib import Path
import traceback

import numpy as np

TYPES = ('hammer', 'screwdriver', 'marker', 'spatula', 'eraser', 'brush')
PHASES = ('grasp', 'settle', 'walk', 'final', 'place', 'release', 'done')
GRASP, SETTLE, WALK, FINAL, PLACE, RELEASE, DONE = range(len(PHASES))
C_PHASES = (SETTLE, WALK, FINAL)
GRASP_STEPS = 60                    # 1 s at 60 Hz
FORCE_N = .1


def build_policy_dims(env, checkpoint, contract, device, obs_dim, hand_dim):
    """eval_wholebody_checkpoint.build_policy with explicit observation / hand-block sizes
    (the carry expert appends its context block after the hand block)."""
    import torch
    from rl_games.algos_torch import model_builder
    from dextrah_lab.g1_adept.touch_continuation import load_yaml
    from dextrah_lab.wholebody.sapg_network import load_touch_sources, register_sonic_models
    run = Path(checkpoint).resolve().parent.parent
    source, critic = load_touch_sources(device, Path(contract['source_checkpoint']), contract['source_sha256'])
    register_sonic_models(source, critic, env._sonic_reference, env._body_lower[0], env._body_upper[0], frozen=True,
                          hand_dim=hand_dim)
    params = load_yaml(run/'params/agent.yaml')['params']
    model = model_builder.ModelBuilder().load(params).build(dict(
        actions_num=70, input_shape=(obs_dim+32,), num_seqs=env.num_envs, value_size=1,
        normalize_value=True, normalize_input=True, type='extra_param',
        coef_ids=source.a2c_network.param_ids, coef_id_idx=obs_dim)).to(device)
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    payload = payload[0] if 0 in payload else payload
    model.load_state_dict(payload['model'], strict=True)
    model.a2c_network.adapter_input_clip = float(contract.get('adapter_input_clip', 0.))   # as trained
    return model.eval().requires_grad_(False), payload


def run(args):
    import torch
    import mujoco_warp
    import dextrah_lab.mj.env as mjenv
    from isaaclab.utils.math import matrix_from_quat, quat_apply_inverse, quat_conjugate, quat_mul, yaw_quat
    from dextrah_lab.mj.carry_env import context_dim, grasp_in_torso, install_moving_frame_observations, object_in_torso, obs_utils
    from dextrah_lab.mj.scene import TABLE_SIZE
    from dextrah_lab.sgs.mj_task import MjSgsTask, SgsSettings
    from dextrah_lab.wholebody.carry import MovingFrame, field_slice, yaw_of
    from dextrah_lab.wholebody.carry_compose import (RIGHT_ARM_JOINTS, CarrySettings, arm_delta_step, body_joint_masks,
                                                     carry_arm_reference, carry_context_v2, compose_body, finger_command,
                                                     hand_command, hand_support)
    from dextrah_lab.wholebody.contract import BODY_JOINTS
    from dextrah_lab.wholebody.hand_surface import HAND_SURFACE_DIM
    from dextrah_lab.wholebody.navigation_reference import NavigationReference, yaw_of as np_yaw
    from dextrah_lab.wholebody.navigation_transfer import WaypointVelocity
    from dextrah_lab.wholebody.task_contract import BODY_EXTRA_DIM, TASK_ACTOR_DIM
    from eval_wholebody_checkpoint import build_policy
    import record_wholebody_videos as rwv
    device = 'cuda:0'
    if 'mujoco_warp_cone_fix' not in Path(mujoco_warp.__file__).parts:
        raise RuntimeError('Training used the vendored mujoco_warp cone Hessian fix; put it on PYTHONPATH')
    m_run = args.m_checkpoint.resolve().parent.parent
    c_run = args.c_checkpoint.resolve().parent.parent
    m_contract = json.loads((m_run/'task_contract.json').read_text())
    c_contract = json.loads((c_run/'task_contract.json').read_text())
    carry = CarrySettings(**{k: tuple(v) if isinstance(v, list) else v
                             for k, v in c_contract['carry_mjlab']['settings'].items()})
    c_v2 = carry.version == 2
    if c_v2:
        if not carry.arm_channel:
            raise ValueError('A carry v2 expert on the route needs the arm channel')
    elif carry.stage != 2 or carry.residual_joints != 'arms' or carry.arm_reference != 'grasp':
        raise ValueError(f'Expected a stage-2 carry expert with arm ownership, got {carry.stage}/{carry.residual_joints}')
    if args.settle_s is None:
        args.settle_s = carry.warmup_s+carry.settle_s
    if args.hold_rule is None:
        args.hold_rule = 'support' if c_v2 else 'offset'
    print('ROUTE_C '+json.dumps(dict(version=carry.version, settle_s=args.settle_s, hold_rule=args.hold_rule,
                                     contact_condim=args.contact_condim)), flush=True)
    install_moving_frame_observations()
    args.no_randomization = True
    env, cfg = rwv.build_env(args, m_run, m_contract, device)
    env.enable_grasp_reward(0.)                      # object-only finger/palm forces for the checks
    env.max_episode_length = 10**9                   # attempts are timed here
    n = env.num_envs
    obs_dim, hand_dim = cfg.observation_space, env.hand_surface_dim
    m_model, m_payload = build_policy(env, args.m_checkpoint, m_contract, device)
    # Optional placement policy at table B (e.g. M fine-tuned with placement goals); M grasps at table A.
    p_model, p_payload = m_model, m_payload
    if args.place_checkpoint is not None:
        p_contract = json.loads((args.place_checkpoint.resolve().parent.parent/'task_contract.json').read_text())
        p_model, p_payload = build_policy(env, args.place_checkpoint, p_contract, device)
    c_ctx = context_dim(carry.version)
    c_model, c_payload = build_policy_dims(env, args.c_checkpoint, c_contract, device, obs_dim+c_ctx, hand_dim+c_ctx)
    sizes = obs_utils._obs_field_sizes(13, 5)
    goal_slice = field_slice(cfg.obs.obs_list, sizes, 'keypoints_rel_goal')
    goal_surface = slice(TASK_ACTOR_DIM+BODY_EXTRA_DIM+hand_dim-HAND_SURFACE_DIM, TASK_ACTOR_DIM+BODY_EXTRA_DIM+hand_dim)
    paths = env._object_urdf_paths
    asset = env._object_asset_index_per_env.cpu().numpy()
    kind = np.array([Path(paths[a]).stem.split('_')[1] for a in asset])
    worlds = np.arange(min(n, args.worlds or n))
    if args.place_goal == 'both':
        variant = np.array(['rest' if j % 2 == 0 else 'hover' for j in range(n)])
    else:
        variant = np.array([args.place_goal]*n)
    nearest_goals = None
    if args.place_goal == 'nearest':
        from dextrah_lab.mj.placement_goals import PlacementGoals, TABLE_TOP
        if args.place_library is None:
            raise ValueError('--place-goal nearest needs --place-library')
        nearest_goals = PlacementGoals(env, args.place_library, 1., orientation='nearest')
    video = []
    for t in TYPES:
        video += [int(j) for j in worlds if kind[j] == t][:args.video_per_type]
    video = sorted(video)

    settings = SgsSettings(**dict(m_contract['sgs']['settings'], eval_pool_size=0))
    state = torch.load(args.m_checkpoint.with_suffix('.sgs.pt'), map_location='cpu', weights_only=False)
    g = torch.Generator(device=device).manual_seed(args.seed)
    task = MjSgsTask(env, settings, generator=g, state=state)
    starts = rwv.ValidatedReaching(env, task, g, worlds.tolist())

    d = device
    z = lambda *shape, **kw: torch.zeros(n, *shape, device=d, **kw)
    phase = torch.full((n,), DONE, dtype=torch.long, device=d)
    phase[torch.as_tensor(worlds, device=d)] = GRASP
    phase_steps = z(dtype=torch.long)
    grasp_count, still_count, drop_count = z(dtype=torch.long), z(dtype=torch.long), z(dtype=torch.long)
    anchored = torch.zeros(n, dtype=torch.bool, device=d)
    home_pos, home_yaw = z(3), z()
    start_pos, start_yaw = z(3), z()
    q_ref, q_open = z(6), z(6)
    arm_ids = torch.tensor([i for i, name in enumerate(BODY_JOINTS) if any(p in name for p in ('shoulder', 'elbow', 'wrist'))], device=d)
    arm_ref = z(len(arm_ids))
    hold_ref = z(3)
    obj_pose0, goal_b = z(7), z(7)
    last_latent, frozen_fingers = z(64), z(6)
    velocity_ema = z(3)
    residual_mask = body_joint_masks(BODY_JOINTS, carry.residual_joints).to(d)
    standing_q = env._reference_q.clone()
    # carry v2 state
    right_arm = torch.tensor([BODY_JOINTS.index(j) for j in RIGHT_ARM_JOINTS], device=d)
    q_grasp, q_carry, arm_D, arm_now = z(7), z(7), z(7), z(7)
    c_steps, unsupported = z(dtype=torch.long), z(dtype=torch.long)
    carry_target = torch.tensor(carry.carry_target, device=d).expand(n, 3)
    torso_idx, palm_link = env.robot.body_names.index('torso_link'), env.robot.body_names.index('right_base_link')
    postures = None
    if c_v2:
        from dextrah_lab.wholebody.carry_posture import CarryPostures
        postures = CarryPostures(carry.postures, d, target=carry.carry_target, target_jitter=0.)
    lateral = torch.tensor([-1., 0., 0.], device=d)        # robot's right for the -90 deg start yaw (checked below)
    shift = args.table_offset*lateral                       # table A -> table B
    table_a, table_b = z(3), z(3)                           # per world, after the reset (table height varies)
    table_moved = torch.zeros(n, dtype=torch.bool, device=d)
    command = np.zeros((n, 3))
    nav, route = {}, {}
    planner_path = Path('/data1/users/konstantin.smirnov/G1-SONIC-planner/planner_sonic.onnx')
    first_nav = NavigationReference(planner_path, speed_limit=max(.3, args.speed))
    records = {int(j): dict(env=int(j), asset=int(asset[j]), type=str(kind[j]), place_goal=str(variant[j])) for j in worlds}
    handback = []

    # ---------------------------------------------------------------- hooks
    def moving_frame():
        data = env.robot.data
        root = data.root_pos_w-env.scene.env_origins
        yaw = yaw_of(data.root_quat_w)
        return MovingFrame(root_pos=root, root_yaw=yaw, home_pos=torch.where(anchored[:, None], home_pos, root),
                           home_yaw=torch.where(anchored, home_yaw, yaw))
    env.carry_moving_frame = moving_frame

    original_terminations = mjenv.termination_utils.compute_terminations
    def terminations(e):
        if e is not env:
            return original_terminations(e)
        off = phase != GRASP
        e._is_success = e._is_success & ~off
        terminated, truncated = original_terminations(e)
        r = e._termination_reasons
        r['timeout'] = torch.zeros_like(r['timeout'])
        r['max_successes'] = r['max_successes'] & ~off
        return r['fall'] | r['hand_far'] | r['max_successes'], r['timeout']
    mjenv.termination_utils.compute_terminations = terminations

    original_observation = env._wholebody_observation
    def wholebody_observation():
        ids = torch.nonzero(torch.isin(phase, torch.tensor(C_PHASES, device=d))).flatten().tolist()
        if ids:
            now = env._sim_step_counter*env.physics_dt
            rp = (env.robot.data.root_pos_w[ids]-env.scene.env_origins[ids]).cpu().numpy()
            rq = env.robot.data.root_quat_w[ids].cpu().numpy()
            bq = env.robot.data.joint_pos[ids][:, env._body_ids].cpu().numpy()
            Q, QD, R = [], [], []
            for k, j in enumerate(ids):
                if nav[j].frames is None:
                    nav[j].reset(np.r_[rp[k], rq[k]], bq[k], now)
                q, qd, ref = nav[j].reference(now, command[j], np_yaw(rq[k]))
                Q.append(q); QD.append(qd); R.append(ref[:, 3:])
            idx = torch.tensor(ids, device=d)
            q = torch.as_tensor(np.stack(Q), device=d)
            q[:, :, arm_ids] = arm_ref[idx][:, None]                 # C's arm reference: angles at the switch
            if c_v2:
                # C's episode time after this step (the training env counts the step before observing)
                tc = (c_steps[idx]+1).float()*env.step_dt
                arm_now[idx] = carry_arm_reference(q_grasp[idx], q_carry[idx], arm_D[idx], tc, carry.warmup_s, carry.settle_s)
                q[:, :, right_arm] = arm_now[idx][:, None]
            qd = torch.as_tensor(np.stack(QD), device=d)
            qd[:, :, arm_ids] = 0.
            env._reference_q[idx] = q
            env._reference_qd[idx] = qd
            ref_quat = torch.as_tensor(np.stack(R), device=d)
        extra = original_observation()
        if ids:
            k = len(ids)
            yq = quat_conjugate(yaw_quat(env.robot.data.root_quat_w[idx]))[:, None].expand(-1, 10, -1)
            delta = quat_mul(yq.reshape(-1, 4), ref_quat.reshape(-1, 4)).reshape(k, 10, 4)
            ori = matrix_from_quat(delta)[..., :2].reshape(k, 10, 6)
            env._reference_ori6 = env._reference_ori6.clone()
            env._reference_ori6[idx] = ori
            extra[idx, 930:] = env._sonic_reference.reference_tokens(env._reference_q[idx], env._reference_qd[idx], ori)
        return extra
    env._wholebody_observation = wholebody_observation

    c_mode = torch.zeros(n, dtype=torch.bool, device=d)
    def compose(body, residual):
        if not bool(c_mode.any()):
            return body
        bare = env._sonic_reference(env._body_extra[:, :930], env._reference_q, env._reference_qd,
                                    env._reference_ori6, torch.zeros_like(residual))
        return torch.where(c_mode[:, None], compose_body(bare, body, residual_mask), body)
    env._body_compose = compose

    class Hook:
        def before_reset(self, env_ids):
            starts.before_reset(env_ids)

        def after_reset(self, env_ids):
            starts.after_reset(env_ids)
            ids = env_ids.reshape(-1)
            env._reference_q[ids] = standing_q[ids]
            env._reference_qd[ids] = 0.

        def after_step(self):
            starts.after_step()
    env._sgs = Hook()

    # ---------------------------------------------------------------- helpers
    def palm():
        p = env.robot.data.body_state_w[:, env._palm_body_id, :]
        centre = obs_utils._apply_local_offset(p[:, :3], p[:, 3:7], env._palm_center_offset, (n,))
        return centre, p

    def hold_offset():
        centre, p = palm()
        return quat_apply_inverse(p[:, 3:7], env.object.data.root_pos_w-centre)

    def relative_speed():
        _, p = palm()
        palm_com = env.robot.data._d.body_com_pos_w[:, env.robot.body_perm][:, env._palm_body_id]
        obj_com = env.object.free_qpos[:, :3]
        rel = env.object.data.root_lin_vel_w-(p[:, 7:10]+torch.cross(p[:, 10:13], obj_com-palm_com, dim=-1))
        return rel.norm(dim=-1)

    def finger_command_now(ids):
        return hand_command(env._cur_targets[ids][:, env._hand_joint_ids], env._hand_lower[ids], env._hand_upper[ids])

    def c_observation(policy):
        obs = policy.clone()
        obs[:, goal_slice] = 0
        obs[:, goal_surface] = 0
        one_hot = torch.stack((phase == SETTLE, phase == WALK, phase == FINAL), -1).float()
        cmd = torch.as_tensor(command, device=d, dtype=torch.float32)
        torque = env.robot.data.applied_torque[:, env._hand_joint_ids].clamp(-5, 5)
        context = torch.cat((z(1), cmd, one_hot, z(6), torque), -1)
        if c_v2:
            torso = env.robot.data.body_state_w[:, torso_idx]
            flags, _, _ = support_state()
            context = torch.cat((context, carry_context_v2(object_in_torso(torso, env.object.data.root_pos_w), carry_target,
                                                           arm_now, arm_D, flags, z(2))), -1)
        return torch.cat((obs, context), -1)

    def support_state():
        c = getattr(env, '_last_contacts', None)
        if c is None:                                    # before the first step (no contact readings yet)
            return z(6), z(dtype=torch.long), torch.zeros(n, dtype=torch.bool, device=d)
        centre, _ = palm()
        return hand_support(c['finger_object_force_l1'], c['palm_object_force_l1'], c.get('object_table_force', z()),
                            c.get('object_ground_force', z()), (env.object.data.root_pos_w-centre).norm(dim=-1))

    def finish(j, outcome, t, **extra):
        records[j].update(outcome=outcome, end_s=round(t, 2), **extra)
        phase[j] = DONE

    # ---------------------------------------------------------------- rollout
    dt = env.step_dt
    steps = round(args.seconds/dt)
    stride = max(1, round(1/(args.fps*dt)))
    leader = z(1)
    frames = []
    vid = torch.tensor(video, device=d, dtype=torch.long)
    a = lambda x: x.detach().cpu().numpy().copy()
    with torch.inference_mode():
        obs, _ = env.reset()
        env.contacts.object_table = True                 # object-table normal force (placement support)
        env.contacts.object_ground = True                # object-floor normal force (support rule)
        table_a[:] = env.table.data.root_pos_w-env.scene.env_origins
        table_b[:] = table_a+shift
        yaw0 = yaw_of(env.robot.data.root_quat_w)
        if not bool((yaw0[torch.as_tensor(worlds, device=d)]+math.pi/2).abs().lt(.05).all()):
            raise RuntimeError(f'Unexpected start yaw {yaw0[:4].tolist()}')
        start_pos[:] = env.robot.data.root_pos_w-env.scene.env_origins
        start_yaw[:] = yaw0
        obj_pose0[:] = torch.cat((env.object.data.root_pos_w-env.scene.env_origins, env.object.data.root_quat_w), -1)
        q_open[:] = finger_command_now(torch.arange(n, device=d))
        hover = torch.as_tensor(variant == 'hover', device=d)
        rest_seen = torch.zeros(n, dtype=torch.bool, device=d)
        goal_ready = False
        m_states = tuple(s.to(d) for s in m_model.get_default_rnn_state())
        p_states = tuple(s.to(d) for s in p_model.get_default_rnn_state())
        c_states = tuple(s.to(d) for s in c_model.get_default_rnn_state())
        for step in range(steps):
            t = step*dt
            if not bool((phase != DONE).any()):
                break
            # -------- commands (C worlds) from the current state
            walk_ids = torch.nonzero(phase == WALK).flatten().tolist()
            if walk_ids:
                rp = a(env.robot.data.root_pos_w[walk_ids]-env.scene.env_origins[walk_ids])
                rq = a(env.robot.data.root_quat_w[walk_ids])
                rv = a(env.robot.data.root_lin_vel_w[walk_ids])
                for k, j in enumerate(walk_ids):
                    command[j] = route[j].update(np.r_[rp[k], rq[k]], rv[k], dt)
                    if route[j].index >= 1 and not bool(table_moved[j]):
                        pose = torch.zeros(1, 7, device=d)
                        pose[0, :3] = env.scene.env_origins[j]+table_b[j]
                        pose[0, 3] = 1.
                        env.table.write_root_pose_to_sim(pose, env_ids=torch.tensor([j], device=d))
                        table_moved[j] = True
                        records[j]['table_moved_s'] = round(t, 2)
                    if route[j].done:
                        phase[j], phase_steps[j] = FINAL, 0
                        command[j] = 0.
                        err = np.linalg.norm(rp[k][:2]-route[j].waypoints[-1])
                        records[j].update(arrived_s=round(t, 2), arrival_error_m=round(float(err), 3))
            for j in torch.nonzero(torch.isin(phase, torch.tensor((SETTLE, FINAL), device=d))).flatten().tolist():
                command[j] = 0.
            # -------- policies
            c_mode[:] = torch.isin(phase, torch.tensor(C_PHASES, device=d))
            m_out = m_model(dict(obs=torch.cat((obs['policy'], leader), -1), is_train=False, prev_actions=None, rnn_states=m_states))
            c_out = c_model(dict(obs=torch.cat((c_observation(obs['policy']), leader), -1), is_train=False,
                                 prev_actions=None, rnn_states=c_states))
            m_states, c_states = m_out['rnn_states'], c_out['rnn_states']
            m_mu, c_mu = m_out['mus'], c_out['mus']
            if p_model is not m_model:
                p_out = p_model(dict(obs=torch.cat((obs['policy'], leader), -1), is_train=False, prev_actions=None, rnn_states=p_states))
                p_states = p_out['rnn_states']
                m_mu = torch.where((phase == PLACE)[:, None], p_out['mus'], m_mu)
            if c_v2:
                c_action = torch.cat((torch.zeros_like(c_mu[:, :64]),
                                      finger_command(q_ref, c_mu[:, 64:70], carry.beta_fingers, carry.finger_close, carry.finger_open)), -1)
                arm_D[:] = torch.where(c_mode[:, None], arm_delta_step(arm_D, torch.tanh(c_mu[:, :7]), carry), arm_D)
            else:
                c_action = torch.cat((carry.beta_latent*torch.tanh(c_mu[:, :64]),
                                      (q_ref+carry.beta_fingers*torch.tanh(c_mu[:, 64:70])).clamp(-1, 1)), -1)
            action = torch.where(c_mode[:, None], c_action, m_mu)
            release = phase == RELEASE
            action = torch.where(release[:, None], torch.cat((last_latent, q_open), -1), action)
            keep = (phase == DONE)
            action = torch.where(keep[:, None], torch.cat((last_latent, frozen_fingers), -1), action)
            m_active = (phase == GRASP) | (phase == PLACE)
            last_latent = torch.where(m_active[:, None], m_mu[:, :64], last_latent)
            frozen_fingers = torch.where(keep[:, None], frozen_fingers, action[:, 64:70])
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated | truncated
            for s_ in m_states+c_states+(p_states if p_model is not m_model else ()):
                s_[:, done, :] = 0
            phase_steps += 1
            c_steps += c_mode.long()
            # -------- episode ends (robot fall, object on the floor, numerical failure)
            fell = info['episode_final']['robot_fall'].bool() if 'robot_fall' in info.get('episode_final', {}) else torch.zeros_like(done)
            reasons = {k: v for k, v in env._termination_reasons.items()}
            for j in torch.nonzero(done & (phase != DONE)).flatten().tolist():
                p = PHASES[int(phase[j])]
                finish(j, ('robot_fell_' if bool(fell[j]) else 'ended_')+p, t+dt,
                       termination=[k for k, v in reasons.items() if bool(v[j])])
            # -------- state for the checks
            lifted = env._lifted_object
            force = env._last_contacts['finger_object_force_l1']
            touching = (force > FORCE_N).sum(-1)
            hand_contact = force.sum(-1)+env._last_contacts.get('palm_object_force_l1', torch.zeros_like(force[:, 0]))
            support = env._last_contacts.get('object_table_force', z())
            offset = hold_offset()
            obj_speed = env.object.data.root_lin_vel_w.norm(dim=-1)
            rel_speed = relative_speed()
            velocity_ema += (dt/.2)*(env.object.data.root_lin_vel_w-velocity_ema)
            velocity_ema[done] = 0.
            obj_pos = env.object.data.root_pos_w-env.scene.env_origins
            if t < 3.:
                # Resting pose on table A: still, untouched, on the table (the start drops it from ~10 cm).
                resting_a = (obj_speed < .03) & (hand_contact < .05) & (support > .02) & (phase == GRASP)
                obj_pose0[resting_a] = torch.cat((obj_pos, env.object.data.root_quat_w), -1)[resting_a]
                rest_seen |= resting_a
                if step == round(1./dt):
                    for j in worlds.tolist():
                        records[j]['at_1s'] = dict(speed=round(float(obj_speed[j]), 3), hand_contact=round(float(hand_contact[j]), 3),
                                                   support_n=round(float(support[j]), 3), lifted=bool(lifted[j]),
                                                   height_above_table_m=round(float(obj_pos[j, 2]-table_a[j, 2]-TABLE_SIZE[2]/2), 3))
            elif not goal_ready:
                goal_b[:] = obj_pose0
                goal_b[:, :2] += shift[:2]
                goal_b[:, 2] = torch.where(hover, goal_b[:, 2].clamp(min=float(cfg.reset.target_volume_mins[2])), goal_b[:, 2])
                goal_ready = True
                for j in worlds.tolist():
                    records[j]['rest_pose_found'] = bool(rest_seen[j])
                    records[j]['goal_height_above_table_m'] = round(float(goal_b[j, 2]-table_b[j, 2]-TABLE_SIZE[2]/2), 3)
            # grasp -> settle (switch M -> C)
            ing = phase == GRASP
            grasp_count = torch.where(ing & lifted & (touching >= args.grasp_min_fingers), grasp_count+1, torch.zeros_like(grasp_count))
            for j in torch.nonzero(ing & (grasp_count >= GRASP_STEPS)).flatten().tolist():
                jj = torch.tensor([j], device=d)
                q_ref[j] = finger_command_now(jj)[0]
                arm_ref[j] = env.robot.data.joint_pos[j, env._body_ids][arm_ids]
                hold_ref[j] = offset[j]
                home_pos[j] = env.robot.data.root_pos_w[j]-env.scene.env_origins[j]
                home_yaw[j] = yaw_of(env.robot.data.root_quat_w[j:j+1])[0]
                anchored[j] = True
                for s_ in c_states:
                    s_[:, j, :] = 0
                c_steps[j], unsupported[j], arm_D[j] = 0, 0, 0.
                if c_v2:
                    body = env.robot.data.body_state_w[jj]
                    q_now = env.robot.data.joint_pos[jj][:, env._body_ids][:, right_arm]
                    t_po, palm_torso = grasp_in_torso(body[:, torso_idx], body[:, palm_link], env.object.data.root_pos_w[jj])
                    _, q_c, p_c = postures.select(t_po, palm_torso, q_now, carry_target[jj])
                    q_grasp[j], q_carry[j], arm_now[j] = q_now[0], q_c[0], q_now[0]
                    records[j]['carry_posture'] = dict(max_joint_change_rad=round(float((q_c-q_now).abs().max()), 3),
                                                       predicted_object_minus_target_m=round(float((p_c-carry_target[jj]).norm()), 3))
                nav[j] = NavigationReference(planner_path, session=first_nav.session, speed_limit=max(.3, args.speed))
                phase[j], phase_steps[j] = SETTLE, 0
                drop_count[j] = 0
                records[j]['grasp_s'] = round(t+dt, 2)
            for j in torch.nonzero(ing & (phase_steps*dt > args.grasp_limit)).flatten().tolist():
                finish(j, 'no_grasp', t+dt)
            # carrying: drops ('offset': v1's 12 cm beyond the hold for 6 steps; 'support': C v2's support rule;
            # both: or below the floor height)
            carrying = torch.isin(phase, torch.tensor(C_PHASES, device=d))
            far = carrying & ((offset-hold_ref).norm(dim=-1) > carry.drop_excess_m)
            drop_count = torch.where(far, drop_count+1, torch.zeros_like(drop_count))
            _, _, supported = support_state()
            counting = carrying & (c_steps.float()*dt > carry.warmup_s)
            unsupported = torch.where(counting & ~supported, unsupported+1, torch.zeros_like(unsupported))
            if args.hold_rule == 'support':
                lost = unsupported >= int(round(carry.unsupported_s/dt))
            else:
                lost = drop_count >= carry.drop_steps
            dropped = carrying & (lost | (obj_pos[:, 2] < carry.drop_floor_m))
            for j in torch.nonzero(dropped).flatten().tolist():
                finish(j, 'dropped_'+PHASES[int(phase[j])], t+dt)
            # settle -> walk
            for j in torch.nonzero((phase == SETTLE) & (phase_steps*dt >= args.settle_s)).flatten().tolist():
                x0, y0 = float(home_pos[j, 0]), float(home_pos[j, 1])
                waypoints = [(x0, y0+args.backoff), (float(table_b[j, 0]), y0+args.backoff), (float(table_b[j, 0]), y0)]
                route[j] = WaypointVelocity(waypoints, float(home_yaw[j]), speed=args.speed)
                phase[j], phase_steps[j] = WALK, 0
                records[j]['walk_start_s'] = round(t+dt, 2)
            for j in torch.nonzero((phase == WALK) & (phase_steps*dt > args.carry_limit)).flatten().tolist():
                finish(j, 'no_arrival', t+dt, remaining_m=round(route[j].distance, 3), waypoint=int(route[j].index),
                       position=[round(float(x), 3) for x in (env.robot.data.root_pos_w[j, :2]-env.scene.env_origins[j, :2])])
            # final -> place (switch C -> M) if still holding
            for j in torch.nonzero((phase == FINAL) & (phase_steps*dt >= args.final_s)).flatten().tolist():
                holding_offset = float((offset[j]-hold_ref[j]).norm()) < carry.retain_m
                records[j].update(holding_offset_at_handback=holding_offset, supported_at_handback=bool(supported[j]))
                holding = bool(supported[j]) if args.hold_rule == 'support' else holding_offset
                records[j]['holding_at_handback'] = holding
                if not holding:
                    finish(j, 'arrived_not_holding', t+dt)
                    continue
                handback.append(j)
                # Stance at table B vs the start stance at table A (robot frame of the start yaw).
                rel_b = env.robot.data.root_pos_w[j, :2]-env.scene.env_origins[j, :2]-table_b[j, :2]
                rel_a = start_pos[j, :2]-table_a[j, :2]
                dyaw = float(yaw_of(env.robot.data.root_quat_w[j:j+1])[0]-start_yaw[j])
                records[j]['stance_offset_m'] = [round(float(x), 3) for x in (rel_b-rel_a)]
                records[j]['stance_yaw_offset_deg'] = round(math.degrees(math.atan2(math.sin(dyaw), math.cos(dyaw))), 1)
                for s_ in m_states+(p_states if p_model is not m_model else ()):
                    s_[:, j, :] = 0
                home_pos[j], home_yaw[j] = start_pos[j], start_yaw[j]     # table B where table A was
                env._reference_q[j] = standing_q[j]
                env._reference_qd[j] = 0.
                if nearest_goals is not None:
                    a_j = env._object_asset_index_per_env[j:j+1]
                    if int(nearest_goals.count[a_j]) > 0:
                        k, q = nearest_goals.nearest(a_j, env.object.data.root_quat_w[j:j+1])
                        goal_b[j, 2] = table_b[j, 2]+TABLE_TOP+nearest_goals.height[k[0]]
                        goal_b[j, 3:] = q[0]
                        records[j]['nearest_goal'] = True
                pose = goal_b[j:j+1].clone()
                pose[:, :3] += env.scene.env_origins[j]
                env.goal_viz.write_root_pose_to_sim(pose, env_ids=torch.tensor([j], device=d))
                phase[j], phase_steps[j] = PLACE, 0
                still_count[j] = 0
                records[j]['handback_s'] = round(t+dt, 2)
            # place -> release (table support or at the goal, and still)
            placing = phase == PLACE
            goal_dist = (obj_pos-goal_b[:, :3]).norm(dim=-1)
            ready = placing & ((support > .5) | (goal_dist < .03)) & (velocity_ema.norm(dim=-1) < .05)
            still_count = torch.where(ready, still_count+1, torch.zeros_like(still_count))
            for j in torch.nonzero(placing & (still_count >= round(.25/dt))).flatten().tolist():
                records[j].update(release_s=round(t+dt, 2), support_at_release=bool(support[j] > .5),
                                  goal_distance_at_release_m=round(float(goal_dist[j]), 3),
                                  relative_speed_at_release_ms=round(float(rel_speed[j]), 3))
                phase[j], phase_steps[j] = RELEASE, 0
            for j in torch.nonzero(placing & (obj_pos[:, 2] < table_b[:, 2]-.1)).flatten().tolist():
                finish(j, 'dropped_place', t+dt)
            for j in torch.nonzero(placing & (phase_steps*dt > args.place_limit)).flatten().tolist():
                finish(j, 'no_release', t+dt, goal_distance_m=round(float(goal_dist[j]), 3), table_support=bool(support[j] > .5),
                       smoothed_speed_ms=round(float(velocity_ema[j].norm()), 3),
                       object_minus_goal_m=[round(float(x), 3) for x in (obj_pos[j]-goal_b[j, :3])])
            # release -> result
            for j in torch.nonzero((phase == RELEASE) & (phase_steps*dt >= args.release_s)).flatten().tolist():
                rel_b = (obj_pos[j]-table_b[j])
                on_b = bool((rel_b[:2].abs() <= torch.tensor(TABLE_SIZE[:2], device=d)/2).all())
                resting = bool(support[j] > .5) and float(obj_speed[j]) < .05
                free = float(hand_contact[j]) < .2
                q_err = quat_mul(quat_conjugate(goal_b[j:j+1, 3:]), env.object.data.root_quat_w[j:j+1])
                angle = float(2*torch.atan2(q_err[0, 1:].norm(), q_err[0, 0].abs()))
                placed = resting and on_b and free
                finish(j, 'placed' if placed else 'released_not_placed', t+dt, on_table_b=on_b, resting=resting,
                       hand_free=free, placement_error_xy_m=round(float((obj_pos[j, :2]-goal_b[j, :2]).norm()), 3),
                       placement_error_z_m=round(float(obj_pos[j, 2]-goal_b[j, 2]), 3),
                       placement_error_deg=round(math.degrees(angle), 1))
            # -------- hand-back states (C's end states, SGS holding-state layout)
            if handback:
                ids = torch.tensor(handback, device=d)
                origin = env.scene.env_origins[ids]
                setpoint = getattr(env.robot, '_setpoint', None)
                handback_store = dict(
                    asset=env._object_asset_index_per_env[ids].cpu(), qpos=env.sim.data.qpos[ids].cpu(),
                    qvel=env.sim.data.qvel[ids].cpu(), targets=env._cur_targets[ids].cpu(),
                    setpoint=(setpoint if setpoint is not None else env._cur_targets)[ids].cpu(),
                    table_z=env._table_z_per_env[ids].cpu(), object_init_z=env._object_init_z[ids].cpu(),
                    object_pose=torch.cat((env.object.data.root_pos_w[ids]-origin, env.object.data.root_quat_w[ids]), -1).cpu(),
                    joint_pos=env.robot.data.joint_pos[ids].cpu(), joint_vel=env.robot.data.joint_vel[ids].cpu(),
                    history=task._history_now(ids).cpu(), goal_pose=goal_b[ids].cpu(),
                    table_pose=(env.table.data.root_pos_w[ids]-origin).cpu(), env=ids.cpu())
                if task.na:
                    handback_store['act'] = env.sim.data.act[ids].cpu()
                path = args.output/'handback_states.pt'
                old = torch.load(path, weights_only=False) if path.exists() else None
                if old is not None:
                    handback_store = {k: torch.cat((old[k], v)) for k, v in handback_store.items()}
                torch.save(handback_store, path)
                handback.clear()
            if step % stride == 0 and len(video):
                frames.append(dict(time_s=t, qpos=a(env.sim.data.qpos[vid]),
                                   table=a(env.table.data.root_pos_w[vid]-env.scene.env_origins[vid]),
                                   goal=a(goal_b[vid]), phase=a(phase[vid]), command=command[video].copy(),
                                   support=a(support[vid]), touching=a(touching[vid]), lifted=a(lifted[vid])))
    for j in worlds:
        j = int(j)
        records[j].update(table_a=a(table_a[j]).round(3).tolist(), table_b=a(table_b[j]).round(3).tolist())
        if int(phase[j]) != DONE:
            records[j].update(outcome='running_at_end_'+PHASES[int(phase[j])])
    if frames:
        np.savez_compressed(args.output/'trajectories.npz', **{k: np.stack([f[k] for f in frames]) for k in frames[0]})
    from dextrah_lab.mj.scene import object_indices
    body, slot_geoms = object_indices(env.sim.mj_model)
    bank = env.object_bank
    np.savez(args.output/'objects.npz', size=bank.size, pos=bank.pos, quat=bank.quat, used=bank.used, com=bank.com,
             slot_geoms=slot_geoms, body=body)
    meta = dict(m_checkpoint=str(args.m_checkpoint), m_epoch=int(m_payload['epoch']), c_checkpoint=str(args.c_checkpoint),
                place_checkpoint=str(args.place_checkpoint) if args.place_checkpoint else None, place_epoch=int(p_payload['epoch']),
                c_epoch=int(c_payload['epoch']), carry_settings=c_contract['carry_mjlab']['settings'], video=video, step_dt=dt, fps=args.fps, args={k: str(v) for k, v in vars(args).items()},
                planner=first_nav.report(), policies='SAPG leaders, mean actions', domain_randomization='off')
    return meta, records, cfg


def summarize(records):
    rows = list(records.values())
    def stats(rs):
        nn = len(rs)
        if not nn:
            return dict(worlds=0)
        out = lambda o: sum(1 for r in rs if r.get('outcome', '').startswith(o))
        grasped = sum(1 for r in rs if 'grasp_s' in r)
        handed = sum(1 for r in rs if r.get('holding_at_handback'))
        placed = out('placed')
        errs = [r['placement_error_xy_m'] for r in rs if r.get('outcome') == 'placed']
        return dict(worlds=nn, grasped=grasped, arrived=sum(1 for r in rs if 'arrived_s' in r), arrived_holding=handed,
                    released=sum(1 for r in rs if 'release_s' in r),
                    released_with_table_support=sum(1 for r in rs if r.get('support_at_release')), placed=placed,
                    grasp_rate=round(grasped/nn, 3),
                    transfer_rate_given_grasp=round(handed/max(grasped, 1), 3),
                    placement_rate_given_handback=round(placed/max(handed, 1), 3),
                    end_to_end=round(placed/nn, 3),
                    robot_falls=sum(1 for r in rs if r.get('outcome', '').startswith('robot_fell')),
                    placed_error_xy_median_m=round(float(np.median(errs)), 3) if errs else None,
                    outcomes={o: sum(1 for r in rs if r.get('outcome') == o) for o in sorted({r.get('outcome') for r in rs})})
    s = dict(all=stats(rows))
    for v in ('rest', 'hover', 'nearest'):
        s['goal_'+v] = stats([r for r in rows if r['place_goal'] == v])
    for t in TYPES:
        s[t] = stats([r for r in rows if r['type'] == t])
    return s


def render(args, mj_model, meta, records):
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
    for k, j in enumerate(meta['video']):
        r = records[j]
        table_a, table_b = np.array(r['table_a']), np.array(r['table_b'])
        m = copy.copy(mj_model)
        a = r['asset']
        m.vis.global_.offwidth, m.vis.global_.offheight = max(W, 640), max(H, 480)
        m.vis.headlight.ambient[:] = (.45, .45, .45)
        m.vis.headlight.diffuse[:] = (.55, .55, .55)
        m.geom_rgba[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, 'table/table')] = (0, 0, 0, 0)   # drawn below at its pose
        for s, gid in enumerate(ob['slot_geoms']):
            m.geom_size[gid] = ob['size'][a, s]
            m.geom_pos[gid] = ob['pos'][a, s]-ob['com'][a]
            m.geom_quat[gid] = ob['quat'][a, s]
            m.geom_rgba[gid] = (.95, .55, .1, 1.) if ob['used'][a, s] else (0, 0, 0, 0)
            m.geom_sameframe[gid] = 0
        dd = mujoco.MjData(m)
        renderer = mujoco.Renderer(m, H, W)
        opt = mujoco.MjvOption()
        opt.geomgroup[:] = 1
        wide = mujoco.MjvCamera()
        wide.lookat[:] = ((table_a[0]+table_b[0])/2, .5, .6)
        wide.azimuth, wide.elevation, wide.distance = 120., -25., 3.6
        track = mujoco.MjvCamera()
        track.azimuth, track.elevation, track.distance = 150., -15., 1.9
        title = f"{r['type']} asset {a} env {j}  goal {r['place_goal']}  |  M epoch {meta['m_epoch']}  C epoch {meta['c_epoch']}  |  outcome {r.get('outcome')}"
        name = f"{k:02d}_{r['type']}_{r['place_goal']}_{r.get('outcome', 'none')}_env{j}.mp4"
        writer = imageio.get_writer(out_dir/name, fps=meta['fps'], codec='libx264', quality=8, macro_block_size=1)
        for f in range(len(tr['time_s'])):
            dd.qpos[:] = tr['qpos'][f, k]
            mujoco.mj_forward(m, dd)
            track.lookat[:] = (dd.qpos[0], dd.qpos[1], .8)
            ph = PHASES[int(tr['phase'][f, k])]
            table_now = tr['table'][f, k]
            moved = np.linalg.norm(table_now[:2]-table_b[:2]) < .01
            views = []
            for cam in (wide, track):
                renderer.update_scene(dd, camera=cam, scene_option=opt)
                sc = renderer.scene
                boxes = [(table_now, (.62, .48, .33, 1.))]
                boxes.append((table_a, (.62, .48, .33, .25)) if moved else (table_b, (.62, .48, .33, .25)))
                for pos, rgba in boxes:
                    if sc.ngeom < sc.maxgeom:
                        mujoco.mjv_initGeom(sc.geoms[sc.ngeom], int(mujoco.mjtGeom.mjGEOM_BOX), np.array(TABLE_SIZE)/2, pos,
                                            np.eye(3).flatten(), np.array(rgba, dtype=np.float32))
                        sc.ngeom += 1
                if ph in ('place', 'release'):
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
            cmd = tr['command'][f, k]
            policy = 'C (carry)' if ph in ('settle', 'walk', 'final') else ('M (manipulation)' if ph in ('grasp', 'place') else ph)
            status = (f"t={tr['time_s'][f]:5.2f}s  phase {ph}  policy {policy}  cmd vx {cmd[0]:+.2f} vy {cmd[1]:+.2f} wz {cmd[2]:+.2f}"
                      f"  lifted {'yes' if tr['lifted'][f, k] else 'no'}  fingers on object {int(tr['touching'][f, k])}"
                      f"  object-table force {tr['support'][f, k]:.1f} N")
            draw.rectangle((0, 0, img.width, 30), fill=(0, 0, 0))
            draw.text((6, 2), title, fill=(255, 255, 255), font=font)
            draw.text((6, 16), status, fill=(255, 230, 120), font=font)
            note = ('one simulated table per world: after the back-off it moves to table B (solid); translucent: the other position.'
                    '  green: placement goal')
            draw.text((6, img.height-14), note, fill=(255, 255, 255), font=font)
            writer.append_data(np.asarray(img))
        writer.close()
        renderer.close()
        print(f'VIDEO {out_dir/name}', flush=True)


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Run inside a Slurm GPU allocation')
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--m-checkpoint', type=Path, required=True, help='manipulation expert (with X.sgs.pt beside it)')
    p.add_argument('--c-checkpoint', type=Path, required=True, help='carry expert (stage 2)')
    p.add_argument('--place-checkpoint', type=Path, default=None, help='policy for the table-B placement (default: M)')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--num-envs', type=int, default=1200, help='>= assets: one env per asset')
    p.add_argument('--worlds', type=int, default=0, help='evaluated worlds (0: all)')
    p.add_argument('--seconds', type=float, default=150.)
    p.add_argument('--speed', type=float, default=.2, help='route cruise command (m/s); C trained to 0.2')
    p.add_argument('--backoff', type=float, default=.6)
    p.add_argument('--table-offset', type=float, default=1., help='table B centre, to the robot\'s right (m)')
    p.add_argument('--grasp-limit', type=float, default=20.)
    p.add_argument('--settle-s', type=float, default=None, help="default: C's warm-up + secure phase (v1 0.5 + 1.0 s)")
    p.add_argument('--hold-rule', choices=('offset', 'support'), default=None, help="default 'support' for carry v2, 'offset' for v1")
    p.add_argument('--contact-condim', type=int, default=None, help='scene contact model (C v2 trains with 4); raise --njmax')
    p.add_argument('--contact-torsional', type=float, default=None)
    p.add_argument('--carry-limit', type=float, default=70.)
    p.add_argument('--final-s', type=float, default=2.)
    p.add_argument('--place-limit', type=float, default=15.)
    p.add_argument('--release-s', type=float, default=2.)
    p.add_argument('--place-goal', choices=('rest', 'hover', 'both', 'nearest'), default='both')
    p.add_argument('--place-library', type=Path, default=None, help='resting-pose library (for --place-goal nearest)')
    p.add_argument('--grasp-min-fingers', type=int, default=1)
    p.add_argument('--video-per-type', type=int, default=2)
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
        meta, records, cfg = run(args)
        summary = summarize(records)
        report.update(meta=meta, summary=summary)
        (args.output/'worlds.json').write_text(json.dumps(records, indent=1, default=float))
        print('TABLE_TO_TABLE_RESULTS '+json.dumps(summary, default=float), flush=True)
        if not args.no_render:
            import record_wholebody_videos as rwv
            render(args, rwv.compile_scene(cfg), meta, records)
        report.update(completed=True)
        print('TABLE_TO_TABLE_COMPLETE', flush=True)
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
