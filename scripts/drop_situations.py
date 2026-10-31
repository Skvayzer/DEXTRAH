#!/usr/bin/env python3
"""What happens in the second before M drops the tool? (mjlab, deterministic leader, fresh validated starts)

Same environment / policy / start pipeline as eval_wholebody_checkpoint.py. Every control step, per env, features
(palm frame): palm angular and linear speed, the tool's angular and linear speed relative to the palm, palm-tool
distance, mean right-hand finger joint speed, the tool's lift above its spawn height. A ring
buffer keeps the last W = 60 steps (1 s). When an episode ends with an object drop ('fall'), its window is saved with
the tool type, the goals reached and the episode step; as controls, windows of random running episodes (same step
range) are saved. Output: windows.npz; printed: feature medians, drops vs controls, early (no goal yet) vs late.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

W = 60
FEATS = ('palm_w', 'palm_v', 'tool_rel_w', 'tool_rel_v', 'dist', 'finger_speed', 'lift')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--num-envs', type=int, default=3000)
    p.add_argument('--max-steps', type=int, default=3000)
    p.add_argument('--control-rate', type=float, default=.002)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import torch
    from dextrah_lab.mj.compat import install
    install()
    from dextrah_lab.mj.factory import make_mj_env
    from dextrah_lab.sgs.mj_task import MjSgsTask, SgsSettings
    from isaaclab.utils.math import quat_apply_inverse, quat_mul, quat_conjugate
    from eval_wholebody_checkpoint import EvalStarts, build_policy
    device = 'cuda:0'
    torch.manual_seed(args.seed)
    run = args.checkpoint.resolve().parent.parent
    contract = json.loads((run/'task_contract.json').read_text())
    env = make_mj_env(args.num_envs, device, nconmax=64, njmax=300, run=run,
                      spawn_on_table=contract['body_termination']['spawn_on_table'])
    model, payload = build_policy(env, args.checkpoint, contract, device)
    settings = SgsSettings(**contract['sgs']['settings'])
    state = torch.load(args.checkpoint.with_suffix('.sgs.pt'), map_location='cpu', weights_only=False)
    g = torch.Generator(device=device).manual_seed(args.seed)
    task = MjSgsTask(env, settings, generator=g, state=state)
    asset_type = np.array([Path(x).stem.split('_')[1] for x in env._object_urdf_paths])
    names = env.robot.joint_names
    hand = torch.tensor([i for i, n in enumerate(names) if n.startswith('right_') and any(k in n for k in ('thumb', 'index', 'middle', 'ring', 'pinky'))],
                        device=device)
    print('HAND_JOINTS', [names[i] for i in hand.tolist()], 'epoch', payload['epoch'], flush=True)
    B = env.num_envs
    buf = torch.zeros(W, B, len(FEATS), device=device)
    qhist = torch.zeros(W, B, 4, device=device)
    qhist[..., 0] = 1
    count = torch.zeros(B, dtype=torch.long, device=device)
    saved = dict(drop=[], control=[])
    meta = dict(drop=[], control=[])

    class Hook(EvalStarts):
        def before_reset(self, env_ids):
            ids = env_ids.reshape(-1)
            done = ids[self.started[ids]]
            if self.recording and done.numel():
                fall = env._termination_reasons['fall'][done]
                d = done[fall & (count[done] > 5)]
                if d.numel():
                    saved['drop'].append(buf[:, d].permute(1, 0, 2).cpu().numpy())
                    meta['drop'] += [(int(e), str(asset_type[int(env._object_asset_index_per_env[e])]), int(env._successes[e]),
                                      int(count[e])) for e in d.tolist()]
                count[done] = 0
                buf[:, done] = 0
            super().before_reset(env_ids)

    hook = Hook(env, task, g)
    env._sgs = hook
    with torch.inference_mode():
        obs, _ = env.reset()
    hook.recording = True
    states = tuple(s.to(device) for s in model.get_default_rnn_state())
    c = torch.zeros(B, device=device)
    rng = np.random.default_rng(args.seed)
    with torch.inference_mode():
        for step in range(args.max_steps):
            out = model(dict(obs=torch.cat((obs['policy'], c[:, None]), -1), is_train=False, prev_actions=None, rnn_states=states))
            states = out['rnn_states']
            obs, _, terminated, truncated, _ = env.step(out['mus'])
            done = terminated | truncated
            for s in states:
                s[:, done, :] = 0
            hook.after_step()
            # features after the step (reset envs: start of a new episode; their buffers were cleared at reset)
            palm = env.robot.data.body_state_w[:, env._palm_body_id, :]
            obj = env.object.data.root_state_w
            pq, oq = palm[:, 3:7], obj[:, 3:7]
            pw, ow = palm[:, 10:13], obj[:, 10:13]
            r = obj[:, :3]-palm[:, :3]
            rel_v = quat_apply_inverse(pq, obj[:, 7:10]-palm[:, 7:10]-torch.cross(pw, r, dim=-1))
            rel_w = quat_apply_inverse(pq, ow-pw)
            fs = env.robot.data.joint_vel[:, hand].abs().mean(-1)
            lift = obj[:, 2]-env.scene.env_origins[:, 2]-env._object_init_z
            f = torch.stack((pw.norm(dim=-1), palm[:, 7:10].norm(dim=-1), rel_w.norm(dim=-1), rel_v.norm(dim=-1), r.norm(dim=-1), fs, lift), -1)
            buf = torch.roll(buf, -1, 0)
            buf[-1] = f
            count += 1
            # controls: random running episodes past W steps
            cand = torch.nonzero((count > W) & ~done).squeeze(-1)
            if cand.numel():
                pick = cand[torch.rand(cand.numel(), device=device) < args.control_rate]
                if pick.numel():
                    saved['control'].append(buf[:, pick].permute(1, 0, 2).cpu().numpy())
                    meta['control'] += [(int(e), str(asset_type[int(env._object_asset_index_per_env[e])]), int(env._successes[e]),
                                         int(count[e])) for e in pick.tolist()]
            if step % 500 == 0:
                print('STEP', step, 'drops', len(meta['drop']), 'controls', len(meta['control']), flush=True)
    X = {k: np.concatenate(v) if v else np.zeros((0, W, len(FEATS))) for k, v in saved.items()}
    np.savez_compressed(args.output/'windows.npz', drop=X['drop'], control=X['control'], feats=np.array(FEATS),
                        drop_meta=np.array(meta['drop'], dtype=object), control_meta=np.array(meta['control'], dtype=object))
    md, mc = meta['drop'], meta['control']
    print('WINDOWS drops', len(md), 'controls', len(mc), flush=True)
    for seg, sl in (('-1.0..-0.5 s', slice(0, 30)), ('-0.5..-0.1 s', slice(30, 54)), ('last 0.1 s', slice(54, 60))):
        for name, X_ in (('drop', X['drop']), ('control', X['control'])):
            if not len(X_):
                continue
            # windows of drops shorter than W are zero-padded at the start: use only full windows for early segments
            full = np.array([m[3] > W for m in (md if name == 'drop' else mc)])
            Xs = X_[full] if seg.startswith('-1.0') else X_
            print(f'SEG {seg:13s} {name:7s} n={len(Xs):5d} ' + ' '.join(f'{fe}={np.median(Xs[:, sl, i].mean(1)):.3f}' for i, fe in enumerate(FEATS)), flush=True)
    for phase, sel in (('early (0 goals)', lambda m: m[2] == 0), ('late (>=1 goal)', lambda m: m[2] > 0)):
        idx = np.array([sel(m) for m in md])
        if idx.any():
            print(f'DROPS {phase}: n={idx.sum()}, median episode step {np.median([m[3] for m, i in zip(md, idx) if i]):.0f}; last 0.5 s '
                  + ' '.join(f'{fe}={np.median(X["drop"][idx][:, 30:54, i].mean(1)):.3f}' for i, fe in enumerate(FEATS)), flush=True)
    for t in sorted(set(asset_type.tolist())):
        idx = np.array([m[1] == t for m in md])
        ci = np.array([m[1] == t for m in mc])
        if idx.sum() >= 10 and ci.sum() >= 10:
            print(f'TYPE {t:12s} drops {idx.sum():4d}: tool_rel_w {np.median(X["drop"][idx][:, 30:54, 2].mean(1)):.3f} vs control '
                  f'{np.median(X["control"][ci][:, 30:54, 2].mean(1)):.3f}; palm_w {np.median(X["drop"][idx][:, 30:54, 0].mean(1)):.3f} vs '
                  f'{np.median(X["control"][ci][:, 30:54, 0].mean(1)):.3f}; finger_speed {np.median(X["drop"][idx][:, 30:54, 5].mean(1)):.3f} vs '
                  f'{np.median(X["control"][ci][:, 30:54, 5].mean(1)):.3f}', flush=True)
    print('DONE', flush=True)
    import sys
    sys.stdout.flush()
    os._exit(0)


if __name__ == '__main__':
    main()
