#!/usr/bin/env python3
"""Log carry transitions for the hand-object world model (neural simulator), inference only.

Runs a carry expert C (mjlab carry env built from its run contract, as in training) and logs, every
control step and every env, everything a hand-object contact model needs (dextrah_lab/neurosim/data.py
lists the fields):

  state       object COM pose / velocity relative to the palm (palm frame), hand joint positions and
              velocities (actuated + passive);
  action      hand joint targets actually sent, the policy's raw outputs (arm channel, fingers);
  exogenous   palm velocity and gravity in the palm frame (the body / SONIC side), table pose
              relative to the palm and whether it is parked, the external force on the object (pushes);
  context     object asset, mass, COM offset (per episode), BPS shape features (per env);
  labels      hand-object contact forces (5 pads + palm), object-table / object-floor forces,
              supported by the hand, episode end with its reason (drop / fall / timeout / ...);
  body        root state and body joints, the walking command (for a later full-body model).

Actions: deterministic means (--action-noise 0), Gaussian noise on the raw outputs (SuperMimic collects
with N(0, 0.1^2)), or samples from the policy's own distribution (--action-noise policy).

Coverage of pre-drop states (review Q3, issue 4): a fixed share of worlds (--perturb-fraction) is perturbed:
Gaussian noise on the EXECUTED finger commands (the raw outputs are mapped back through the exact inverse of
the finger stabiliser, so tanh saturation does not shrink it), loosened-grip events (an offset toward open for
0.5-2 s), and extra horizontal object pushes through the env's own push mechanism. Logged per row
('loosen', 'extra_push') and per world ('perturbed' in context.pt).

Episode ends: values logged at a done step already belong to the reset episode; the end reason of the
finished episode is captured before the reset (the env's outcome hook), so transitions into a done step
carry only their end label.
"""
import argparse
import json
import math
import os
from pathlib import Path
import time
import traceback

END_REASONS = ('none', 'timeout', 'drop', 'fall', 'start_fail', 'other')
# body-side fields (for a later full-body model) stored in half precision
HALF = ('root_height', 'root_quat', 'root_linvel', 'root_angvel', 'body_q', 'body_qd', 'command', 'arm_now', 'arm_D',
        'raw_arm', 'carry_t')


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Run inside a Slurm GPU allocation')
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', type=Path, required=True, help='carry expert checkpoint')
    p.add_argument('--output', type=Path, required=True, help='new directory for the chunks')
    p.add_argument('--num-envs', type=int, default=2048)
    p.add_argument('--seconds', type=float, default=90.)
    p.add_argument('--chunk-steps', type=int, default=300)
    p.add_argument('--level', type=int, default=2)
    p.add_argument('--action-noise', default='0.1', help="'policy' (sample the policy) or a Gaussian std on the raw outputs")
    p.add_argument('--contact-condim', type=int, default=None, help='default: the run contract (C v2: 4)')
    p.add_argument('--perturb-fraction', type=float, default=0., help='share of worlds with coverage perturbations')
    p.add_argument('--target-noise', type=float, default=.3, help='perturbed worlds: std of the executed finger commands')
    p.add_argument('--loosen-rate', type=float, default=.2, help='perturbed worlds: loosening events per second')
    p.add_argument('--loosen-s', type=float, nargs=2, default=(.5, 2.))
    p.add_argument('--loosen-offset', type=float, nargs=2, default=(.1, .4), help='command units toward open')
    p.add_argument('--push-rate', type=float, default=.2, help='perturbed worlds: extra object pushes per second')
    p.add_argument('--push-n', type=float, default=3., help='extra push magnitude U(0, push_n) N, horizontal')
    p.add_argument('--push-s', type=float, nargs=2, default=(.1, .5))
    p.add_argument('--nconmax', type=int, default=64)
    p.add_argument('--njmax', type=int, default=400)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    report = dict(completed=False, args={k: str(v) for k, v in vars(args).items()})
    try:
        from dextrah_lab.mj.compat import install
        install()
        report.update(collect(args))
        report['completed'] = True
        print('NEUROSIM_COLLECT_COMPLETE '+json.dumps({k: report[k] for k in ('steps', 'transitions', 'ends')}), flush=True)
    except BaseException as error:
        report.update(error=str(error), traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        (args.output/'meta.json').write_text(json.dumps(report, indent=2))
        import sys
        sys.stdout.flush(); sys.stderr.flush()
        os._exit(0 if report['completed'] else 1)


def collect(args):
    import torch
    from isaaclab.utils.math import quat_apply_inverse, quat_conjugate, quat_mul
    from dextrah_lab.mj.carry_env import G1CarryMjEnv
    from dextrah_lab.mj.env import load_run_cfg
    from dextrah_lab.mj.factory import DEFAULT_MUJOCO, WORKSPACE
    from dextrah_lab.mj.robot import ContactParams
    from dextrah_lab.mj.scene import SceneParams
    from dextrah_lab.sgs.mj_task import SgsSettings
    from dextrah_lab.wholebody.carry_compose import CarrySettings, ClipSequencer
    from dextrah_lab.wholebody.sonic import FrozenSonic
    from mj_table_to_table import build_policy_dims
    device = 'cuda:0'
    torch.manual_seed(args.seed)
    run = args.checkpoint.resolve().parent.parent
    contract = json.loads((run/'task_contract.json').read_text())
    cm = contract['carry_mjlab']
    settings = CarrySettings(**{k: tuple(v) if isinstance(v, list) else v for k, v in cm['settings'].items()})
    task_run = Path(contract['task_config_source']).parent.parent
    task_contract = json.loads((task_run/'task_contract.json').read_text())
    cfg = load_run_cfg(task_run)
    cfg.seed = args.seed
    cfg.sonic_body.spawn_on_table = task_contract['body_termination']['spawn_on_table']
    holding = Path(cm['holding_state'])
    held_contract = json.loads((holding.parent.parent/'task_contract.json').read_text())
    sonic = FrozenSonic(WORKSPACE/'GRAIL', WORKSPACE/'G1-SONIC-models/checkpoint/SONIC/models/sonic_manipulation_base', device)
    trained = contract.get('contact_model', {})
    condim = args.contact_condim or trained.get('condim', 3)
    params = SceneParams(contact=ContactParams(condim=condim, torsional=trained.get('torsional', ContactParams().torsional)))
    env = G1CarryMjEnv(cfg, sonic=sonic, carry=settings, holding_state=holding,
                       held_settings=SgsSettings(**dict(held_contract['sgs']['settings'], eval_pool_size=0)),
                       clips_path=Path(cm['clips']), num_envs=args.num_envs, device=device, nconmax=args.nconmax,
                       njmax=args.njmax, mujoco_cfg=DEFAULT_MUJOCO, seed=args.seed, scene_params=params)
    env._curriculum.level = args.level
    env._seq.set_level(args.level)
    model, _ = build_policy_dims(env, args.checkpoint, contract, device, env.cfg.observation_space, env.hand_surface_dim)
    n, dt = env.num_envs, env.step_dt
    if not getattr(env, '_v2', False):
        raise ValueError('The collector logs carry v2 quantities (support, arm channel); use a carry v2 checkpoint')

    # End reasons of finished episodes, captured before the reset clears them.
    end_code = torch.zeros(n, dtype=torch.uint8, device=device)
    record = env._carry_record

    def recording(ids):
        code = torch.full((len(ids),), END_REASONS.index('other'), dtype=torch.uint8, device=device)
        t = env._carry_t[ids]
        timeout = t >= torch.round(env._seq.end[ids]/dt).long() if env._seq.variable else t >= int(round(settings.episode_s/dt))
        code[timeout] = END_REASONS.index('timeout')
        code[env._start_fail[ids]] = END_REASONS.index('start_fail')
        code[env._body_fallen[ids]] = END_REASONS.index('fall')
        code[env._dropped[ids]] = END_REASONS.index('drop')
        end_code[ids] = code
        return record(ids)
    env._carry_record = recording

    hand_ids = torch.unique(torch.cat((env._hand_joint_ids, env._passive_joint_ids))) if hasattr(env, '_passive_joint_ids') else env._hand_joint_ids
    act_pos = [int((hand_ids == i).nonzero()) for i in env._hand_joint_ids.tolist()]
    names = env.robot.joint_names
    obj_mj = int(env._held.body)
    gravity_w = torch.tensor([0., 0., -1.], device=device).expand(n, 3)

    def snapshot():
        centre, palm = env._palm()
        pq = palm[:, 3:7]
        obj = env.object.data
        com = env.object.free_qpos[:, :3]
        _, rel_lin, rel_ang = env._hold_kinematics()
        table = env.table.data.root_pos_w
        data = env.robot.data
        c = env._last_contacts
        zero = torch.zeros(n, device=device)
        return dict(
            obj_pos=quat_apply_inverse(pq, com-palm[:, :3]),
            obj_quat=quat_mul(quat_conjugate(pq), obj.root_quat_w),
            obj_linvel=rel_lin, obj_angvel=rel_ang,
            hand_q=data.joint_pos[:, hand_ids], hand_qd=data.joint_vel[:, hand_ids],
            hand_target=env._cur_targets[:, env._hand_joint_ids],
            palm_linvel=quat_apply_inverse(pq, palm[:, 7:10]), palm_angvel=quat_apply_inverse(pq, palm[:, 10:13]),
            gravity=quat_apply_inverse(pq, gravity_w),
            table_pos=quat_apply_inverse(pq, table-palm[:, :3]), table_parked=env._table_parked.float(),
            object_force=quat_apply_inverse(pq, env.sim.data.xfrc_applied[:, obj_mj, :3]),
            finger_force=c['finger_object_force_l1'], palm_force=c['palm_object_force_l1'],
            table_force=c.get('object_table_force', zero), ground_force=c.get('object_ground_force', zero),
            supported=env._supported.float(),
            mass=env._object_mass[:, 0], com_offset=env.object.com_offset.clone(),
            phase=env._seq.phase(env._episode_time()).float(), carry_t=env._carry_t.float(),
            root_height=data.root_pos_w[:, 2]-env.scene.env_origins[:, 2], root_quat=data.root_quat_w,
            root_linvel=quat_apply_inverse(data.root_quat_w, data.root_lin_vel_w),
            root_angvel=quat_apply_inverse(data.root_quat_w, data.root_ang_vel_w),
            body_q=data.joint_pos[:, env._body_ids], body_qd=data.joint_vel[:, env._body_ids],
            command=env._clip_command, arm_now=env._arm_now, arm_D=env._arm_D)

    noise = args.action_noise
    sigma = None if noise == 'policy' else float(noise)
    from dextrah_lab.wholebody.carry_compose import finger_command
    g = torch.Generator(device=device).manual_seed(args.seed+1)
    rnd = lambda *shape: torch.rand(*shape, device=device, generator=g)
    perturbed = rnd(n) < args.perturb_fraction
    loosen_left = torch.zeros(n, dtype=torch.long, device=device)
    loosen_amt = torch.zeros(n, device=device)
    close, open_ = settings.finger_close, settings.finger_open

    def raw_for_command(c, q_ref):
        # exact inverse of finger_command(q_ref, raw, beta, close, open) where it is not clamped
        d = c-q_ref
        t_ = torch.where(d >= 0, d/close, d/open_)
        return torch.atanh(t_.clamp(-.999, .999))
    obs, _ = env.reset()
    # contact reader exists after the first reset (the carry v2 env already enables both channels)
    env.contacts.object_table = True
    env.contacts.object_ground = True
    states = tuple(s.to(device) for s in model.get_default_rnn_state())
    leader = torch.zeros(n, 1, device=device)
    steps = round(args.seconds/dt)
    chunk, chunk_id, ends = {}, 0, {r: 0 for r in END_REASONS[1:]}
    # Row t: 'raw_*' / 'done' / 'end' describe the step into t; every other field is the state after it,
    # which for done envs is already the first state of the next episode (transition t-1 -> t invalid).
    t0 = time.time()
    with torch.inference_mode():
        for step in range(steps):
            out = model(dict(obs=torch.cat((obs['policy'], leader), -1), is_train=False, prev_actions=None, rnn_states=states))
            states = out['rnn_states']
            mus = out['mus']
            raw = mus
            if sigma is None:
                raw = out['actions'] if 'actions' in out else raw+out['sigmas']*torch.randn_like(raw)
            elif sigma > 0:
                raw = raw+sigma*torch.randn_like(raw)
            extra_push = torch.zeros(n, dtype=torch.bool, device=device)
            if bool(perturbed.any()):
                # loosening events (offset toward open), executed-command noise, extra object pushes
                start = perturbed & (loosen_left == 0) & (rnd(n) < args.loosen_rate*dt)
                lo, hi = args.loosen_s
                loosen_left = torch.where(start, torch.round((lo+(hi-lo)*rnd(n))/dt).long(), (loosen_left-1).clamp(min=0))
                lo, hi = args.loosen_offset
                loosen_amt = torch.where(start, lo+(hi-lo)*rnd(n), loosen_amt)
                q_ref = env._q_ref
                c = finger_command(q_ref, mus[:, 64:70], settings.beta_fingers, close, open_)
                c = c+args.target_noise*torch.randn(c.shape, device=device, generator=g)
                c = (c-(loosen_amt*(loosen_left > 0).float())[:, None]).clamp(-1., 1.)
                raw = raw.clone()
                raw[:, 64:70] = torch.where(perturbed[:, None], raw_for_command(c, q_ref), raw[:, 64:70])
                extra_push = perturbed & (env._object_steps == 0) & (rnd(n) < args.push_rate*dt)
                ids = torch.nonzero(extra_push).flatten()
                if ids.numel():
                    a = rnd(len(ids))*2*math.pi
                    direction = torch.stack((a.cos(), a.sin(), torch.zeros_like(a)), -1)
                    lo, hi = args.push_s
                    env._object_force[ids] = direction*(args.push_n*rnd(len(ids)))[:, None]
                    env._object_steps[ids] = torch.round((lo+(hi-lo)*rnd(len(ids)))/dt).long().clamp(min=1)
            end_code.zero_()
            obs, _, terminated, truncated, _ = env.step(raw)
            done = terminated | truncated
            for s_ in states:
                s_[:, done, :] = 0
            row = snapshot()
            row.update(raw_arm=raw[:, :7], raw_fingers=raw[:, 64:70], done=done.float(), end=end_code.float(),
                       loosen=(loosen_amt*(loosen_left > 0).float()), extra_push=extra_push.float())
            for k, v in row.items():
                chunk.setdefault(k, []).append(v.float().reshape(n, -1).clone())
            for r in END_REASONS[1:]:
                ends[r] += int((end_code == END_REASONS.index(r)).sum())
            if len(chunk['done']) == args.chunk_steps or step == steps-1:
                torch.save({k: torch.stack(v).to(torch.float16 if k in HALF else torch.float32).cpu() for k, v in chunk.items()},
                           args.output/f'chunk_{chunk_id:03d}.pt')
                chunk, chunk_id = {}, chunk_id+1
                print(f'NEUROSIM_CHUNK {chunk_id} step {step+1}/{steps} ends {ends} {time.time()-t0:.0f}s', flush=True)
    bps = env._bps_features.cpu()
    torch.save(dict(bps=bps, asset=env._object_asset_index_per_env.cpu(), perturbed=perturbed.cpu(),
                    perturbation=dict(fraction=args.perturb_fraction, target_noise=args.target_noise, loosen_rate=args.loosen_rate,
                                      loosen_s=list(args.loosen_s), loosen_offset=list(args.loosen_offset), push_rate=args.push_rate,
                                      push_n=args.push_n, push_s=list(args.push_s)),
                    hand_joint_names=[names[i] for i in hand_ids.tolist()],
                    actuated=act_pos, dt=dt, condim=condim, end_reasons=END_REASONS,
                    carry_settings=cm['settings'], checkpoint=str(args.checkpoint)), args.output/'context.pt')
    return dict(steps=steps, transitions=steps*n, ends=ends, chunks=chunk_id, condim=condim, envs=n, dt=dt,
                hand_joints=[names[i] for i in hand_ids.tolist()], seconds_wall=round(time.time()-t0, 1))


if __name__ == '__main__':
    main()
