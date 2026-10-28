#!/usr/bin/env python3
"""Stage-2 prerequisite: do the hand-object world model's action effects agree with the simulator's?

SuperMimic trains the policy with gradients of the imagined return with respect to actions. Before any such
gradient is used for the carry expert's fingers, check that the model moves outcomes the same way MuJoCo Warp
does when a finger command changes (finite differences in both, same perturbations):

  * worlds: --states holding states, each restored into 1 + 6 fingers x 2 signs x len(--deltas) worlds
    (no mass / COM / carry-target randomisation, no pushes, tables parked, per-step domain randomisation off --
    action / observation delays, object-state noise -- deterministic carry expert): identical except for
    the perturbation, so simulator differences are exact (MuJoCo Warp is deterministic);
  * perturbation: finger j's command +-delta (command units) on top of the expert's, for --duration-s,
    starting --t0-s after the restore (secure phase; the warm-up ends at 0.5 s);
  * outcomes over --horizon-s from t0: mean log(1 + pad + palm force) over the last 5 steps, the object's net
    palm-relative rotation and displacement (5-row averages at both ends), supported share, loss of the object;
  * the model rolls out from the logged state at t0 with each world's logged actions; exogenous inputs either
    each world's own ('teacher') or the baseline world's ('frozen', isolates the action);
  * per state and outcome: cosine similarity of the 12-24 simulator vs model differences, sign agreement where
    the simulator effect is clearly non-zero, and the simulator's own linearity (+delta vs -delta).
"""
import argparse
import json
import math
import os
from pathlib import Path
import time
import traceback


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Run inside a Slurm GPU allocation')
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', type=Path, required=True, help='carry expert (the policy the world model data came from)')
    p.add_argument('--world-model', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True, help='report .json')
    p.add_argument('--holding-state', type=Path, default=None, help='SGS state for the starts (default: the expert contract)')
    p.add_argument('--states', type=int, default=256)
    p.add_argument('--deltas', type=float, nargs='+', default=(.05, .1))
    p.add_argument('--t0-s', type=float, default=.6)
    p.add_argument('--duration-s', type=float, default=.25)
    p.add_argument('--horizon-s', type=float, default=.5)
    p.add_argument('--replicas', type=int, default=2, help='extra unperturbed copies per state (simulator noise floor)')
    p.add_argument('--bundle', type=int, default=1,
                   help='K copies of every (state, direction) with independent finger-command jitter; outcomes averaged '
                        '(expected effects, as Bundled Contact Gradients average perturbed branches)')
    p.add_argument('--jitter', type=float, default=0., help='std of the per-step finger-command jitter (command units)')
    p.add_argument('--randomization', choices=('off', 'training'), default='off',
                   help="the run config's per-step domain randomisation (action / observation delays, object-state "
                        "noise): 'off' (copies identical except for the perturbation; before 2026-10-28 it was on)")
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    report = dict(completed=False, args={k: str(v) for k, v in vars(args).items()})
    try:
        from dextrah_lab.mj.compat import install
        install()
        report.update(run(args))
        report['completed'] = True
    except BaseException as error:
        report.update(error=str(error), traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2))
        print('GRADIENT_CHECK_DONE '+json.dumps({k: report.get(k) for k in ('completed', 'summary')}), flush=True)
        import sys
        sys.stdout.flush(); sys.stderr.flush()
        os._exit(0 if report['completed'] else 1)


def run(args):
    import torch
    from dextrah_lab.neurosim.data import Layout, build_tensors, gather
    from dextrah_lab.neurosim.env_io import build_carry_env, hand_joint_ids, snapshot
    from dextrah_lab.neurosim.evaluate import project_so3
    from dextrah_lab.neurosim.geometry import rotation_angle
    from dextrah_lab.neurosim.model import HandObjectWorldModel
    from dextrah_lab.wholebody.carry_compose import finger_command
    t_start = time.time()
    dirs = [(0, 0., 0.)]*(1+args.replicas)+[(j+1, s, dl) for dl in args.deltas for j in range(6) for s in (1., -1.)]
    D = len(dirs)
    R = 1+args.replicas                                            # block layout per state: R baselines, then +- pairs
    K = max(1, args.bundle)
    n = args.states*D*K                                            # world = (state * D + direction) * K + copy
    # identical copies: no mass / COM / carry-target randomisation, no pushes, and every copy parks its table at the
    # end of the warm-up (table_keep_probability 0; before 2026-10-28 copies kept the table with probability 0.2)
    override = dict(mass_scale=(1., 1.), com_offset_m=0., disturbance=0., push_probability=0., carry_jitter=0.,
                    table_keep_probability=0.)
    env, policy, contract, settings = build_carry_env(args.checkpoint, n, settings_override=override,
                                                      holding_state=args.holding_state, seed=args.seed,
                                                      no_randomization=args.randomization == 'off')
    dev, dt = env.device, env.step_dt
    g = torch.Generator().manual_seed(args.seed)
    configs = env._sgs.configs
    chosen = torch.randperm(configs.numel(), generator=g)[:args.states].to(dev)
    assigned = chosen.repeat_interleave(D*K)               # world w -> holding config index (into configs)
    env._sgs.eval_mask[:] = False
    env._sgs.sampler.sample = lambda k, generator=None: assigned[:k]
    finger = torch.tensor([d[0] for d in dirs], device=dev).repeat_interleave(K).repeat(args.states)   # 0 = baseline
    sign = torch.tensor([d[1] for d in dirs], device=dev).repeat_interleave(K).repeat(args.states)
    delta = torch.tensor([d[2] for d in dirs], device=dev).repeat_interleave(K).repeat(args.states)
    jitter_g = torch.Generator(device=dev).manual_seed(args.seed+7)
    offset = torch.zeros(n, 6, device=dev)
    rows = (finger > 0).nonzero().flatten()
    offset[rows, finger[rows]-1] = sign[rows]*delta[rows]
    close, open_ = settings.finger_close, settings.finger_open
    hand_ids = hand_joint_ids(env)
    t0, P, H = round(args.t0_s/dt), round(args.duration_s/dt), round(args.horizon_s/dt)
    ended = torch.zeros(n, dtype=torch.bool, device=dev)
    lost = torch.zeros(n, dtype=torch.bool, device=dev)
    record = env._carry_record

    def recording(ids):
        lost[ids] |= env._dropped[ids]
        ended[ids] = True
        return record(ids)
    env._carry_record = recording
    obs, _ = env.reset()
    states = tuple(s.to(dev) for s in policy.get_default_rnn_state())
    leader = torch.zeros(n, 1, device=dev)
    log = {}
    with torch.inference_mode():
        for step in range(t0+H+1):
            out = policy(dict(obs=torch.cat((obs['policy'], leader), -1), is_train=False, prev_actions=None, rnn_states=states))
            states = out['rnn_states']
            raw = out['mus'].clone()
            perturb = t0 <= step < t0+P
            if perturb or args.jitter > 0:
                q_ref = env._q_ref
                c = finger_command(q_ref, raw[:, 64:70], settings.beta_fingers, close, open_)
                if perturb:
                    c = c+offset
                if args.jitter > 0:
                    c = c+args.jitter*torch.randn(c.shape, device=dev, generator=jitter_g)
                c = c.clamp(-1., 1.)
                d = c-q_ref
                t_ = torch.where(d >= 0, d/close, d/open_)
                raw[:, 64:70] = torch.atanh(t_.clamp(-.999, .999))
            obs, _, terminated, truncated, _ = env.step(raw)
            if step >= t0-2:
                row = snapshot(env, hand_ids)
                row['done'] = (terminated | truncated).float()
                row['end'] = torch.zeros(n, device=dev)
                for k, v in row.items():
                    log.setdefault(k, []).append(v.float().reshape(n, -1).clone())
    data = {k: torch.stack(v).cpu() for k, v in log.items()}
    # rows r = step - (t0 - 2): row 1 = the state before the first perturbed action (after step t0 - 1),
    # rows 2 .. H+1 = after the perturbed / later steps t0 .. t0+H-1; row 0 feeds the 'pre' inputs of row 1
    sim_s = time.time()-t_start
    # ---- world model
    ck = torch.load(args.world_model, map_location='cpu', weights_only=False)
    layout = Layout(**{k: v for k, v in ck['layout'].items() if k in Layout.__dataclass_fields__})
    wm = HandObjectWorldModel(layout, hidden=tuple(ck['hidden']), exo_mode=ck.get('exo_mode', 'post'))
    wm.load_state_dict(ck['state_dict'])
    wm.eval().to(dev)
    context = dict(bps=env._bps_features.float().cpu(), asset=env._object_asset_index_per_env.cpu(), dt=dt)
    t = build_tensors(data, context, exo_mode=ck.get('exo_mode', 'post'), lowpass_tau=ck.get('lowpass_tau') or None,
                      labels=ck.get('labels', 'lost'))
    t.pop('asset')
    t = {k: v.to(dev) for k, v in t.items()}
    s0 = torch.full((n,), 1, dtype=torch.long, device=dev)            # row 1: the last unperturbed state
    e = torch.arange(n, device=dev)
    idx_w = torch.arange(n, device=dev)
    base = (idx_w//(D*K))*(D*K)+idx_w % K                            # first baseline world of the same state and copy
    with torch.no_grad():
        w = gather(t, s0, e, H)
        w_frozen = dict(w)
        w_frozen['exo'] = w['exo'][base]                              # the baseline's exogenous inputs, own actions
        pred = {}
        for name, ww in (('teacher', w), ('frozen', w_frozen)):
            xs, Rs, fs, ls = wm.rollout(ww, H)
            pred[name] = dict(state=torch.stack(xs, 1), rot=torch.stack(Rs, 1), forces=torch.stack(fs, 1),
                              hazard=torch.stack([torch.sigmoid(l[:, 9]) for l in ls], 1),
                              support=torch.stack([torch.sigmoid(l[:, 8]) for l in ls], 1))

    def outcomes(state, rot, forces, support):
        """state (n, T, d) rows 0..H-1 after t0 (row -1 = start): force, net rotation (deg), displacement (mm), support."""
        force = forces[:, -5:, :6].sum(-1).mean(1)
        start_R = rot0_ref
        end_R = project_so3(rot[:, -5:].mean(1))
        net_rot = torch.rad2deg(rotation_angle(start_R, end_R))
        disp = 1e3*(state[:, -5:, :3].mean(1)-p0_ref).norm(dim=-1)
        return dict(force=force, rotation_deg=net_rot, displacement_mm=disp, support=support.mean(1))
    # start pose: rows 0..1 (unperturbed); outcomes over rows 2..H+1 (model: its H predictions)
    p0_ref = t['state'][0:2, :, :3].mean(0)
    rot0_ref = project_so3(t['rot'][0:2].mean(0))
    span = slice(2, H+2)
    sim = outcomes(t['state'][span].transpose(0, 1), t['rot'][span].transpose(0, 1),
                   t['forces'][span].transpose(0, 1), t['supported'][span].transpose(0, 1))
    sim['lost'] = lost.float()
    model = {name: outcomes(pr['state'], pr['rot'], pr['forces'], pr['support']) for name, pr in pred.items()}
    for name, pr in pred.items():
        model[name]['lost'] = 1-torch.prod(1-pr['hazard'], 1)
    # ---- compare finite differences
    from dextrah_lab.neurosim.evaluate import compare_finite_differences
    S = args.states
    keep = torch.cat((torch.zeros(1, dtype=torch.long), torch.arange(R, D))).to(dev)      # first baseline + the pairs
    # expected outcome of each (state, direction): mean over the K jittered copies
    avg = lambda v: v.reshape(S, D, K).mean(-1)
    sim = {k: avg(v).flatten() for k, v in sim.items()}
    model = {proto: {k: avg(v).flatten() for k, v in d.items()} for proto, d in model.items()}
    sel = lambda v: v.reshape(S, D)[:, keep].flatten()
    summary = {}
    for name in ('force', 'rotation_deg', 'displacement_mm', 'support', 'lost'):
        summary[name] = {proto: compare_finite_differences(sel(sim[name]), sel(model[proto][name]), S, len(keep)) for proto in ('teacher', 'frozen')}
        if R > 1:
            # simulator noise floor: identical unperturbed worlds (MuJoCo Warp atomics can reorder sums)
            b = sim[name].reshape(S, D)[:, :R]
            summary[name]['sim_noise_abs_median'] = round(float((b[:, 1:]-b[:, :1]).abs().median()), 5)
            summary[name]['sim_noise_identical_share'] = round(float((b[:, 1:] == b[:, :1]).float().mean()), 4)
        for dl in args.deltas:
            # per perturbation size: simulator antisymmetry and model agreement
            idx = [0]+[R+i for i, d in enumerate(dirs[R:]) if d[2] == dl]
            sub = torch.tensor(idx, device=dev)
            pick = lambda v: v.reshape(S, D)[:, sub].flatten()
            summary[name][f'delta_{dl:g}'] = {proto: compare_finite_differences(pick(sim[name]), pick(model[proto][name]), S, len(idx))
                                              for proto in ('teacher',)}
    return dict(summary=summary, worlds=n, states=S, directions=D-R, bundle=K, jitter=args.jitter, deltas=list(args.deltas), t0_s=args.t0_s,
                duration_s=args.duration_s, horizon_s=args.horizon_s, ended_in_window=int(ended.sum()), lost=int(lost.sum()),
                sim_seconds=round(sim_s, 1), wall_s=round(time.time()-t_start, 1))


if __name__ == '__main__':
    main()
