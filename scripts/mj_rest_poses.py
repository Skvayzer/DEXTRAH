#!/usr/bin/env python3
"""Library of stable resting poses of every object on the table (inference only, no policy).

Each world starts from a validated reaching configuration of its object (the object is dropped
from ~10 cm, record_wholebody_videos.ValidatedReaching), the robot stands with bare SONIC (zero
latent residual, fingers at their reset command); after --settle-s the object pose is kept if it is
still (< 2 cm/s), supported by the table (object-table normal force > 0.02 N), untouched by the hand
and on the table top. Repeated --rounds times. Output (npz): asset, height of the object frame
above the table top, quaternion (wxyz), xy relative to the table centre.
"""
import argparse
import json
import os
from pathlib import Path


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Run inside a Slurm GPU allocation')
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True, help='run whose task / SGS state defines the starts')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--num-envs', type=int, default=1200)
    p.add_argument('--rounds', type=int, default=12)
    p.add_argument('--settle-s', type=float, default=1.5)
    p.add_argument('--nconmax', type=int, default=64)
    p.add_argument('--njmax', type=int, default=300)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    from dextrah_lab.mj.compat import install
    install()
    import numpy as np
    import torch
    from dextrah_lab.mj.scene import TABLE_SIZE
    from dextrah_lab.sgs.mj_task import MjSgsTask, SgsSettings
    from dextrah_lab.wholebody.carry_compose import hand_command
    import record_wholebody_videos as rwv
    device = 'cuda:0'
    run = args.checkpoint.resolve().parent.parent
    contract = json.loads((run/'task_contract.json').read_text())
    args.no_randomization = True
    env, cfg = rwv.build_env(args, run, contract, device)
    env.enable_grasp_reward(0.)                      # object-only finger / palm forces (untouched check)
    n = env.num_envs
    settings = SgsSettings(**dict(contract['sgs']['settings'], eval_pool_size=0))
    state = torch.load(args.checkpoint.with_suffix('.sgs.pt'), map_location='cpu', weights_only=False)
    g = torch.Generator(device=device).manual_seed(args.seed)
    task = MjSgsTask(env, settings, generator=g, state=state)
    starts = rwv.ValidatedReaching(env, task, g, list(range(n)))

    class Hook:
        def before_reset(self, env_ids):
            starts.before_reset(env_ids)

        def after_reset(self, env_ids):
            starts.after_reset(env_ids)

        def after_step(self):
            starts.after_step()
    env._sgs = Hook()
    env.max_episode_length = 10**9
    steps = round(args.settle_s/env.step_dt)
    out = dict(asset=[], height=[], quat=[], xy=[])
    kept_rounds = []
    with torch.inference_mode():
        for r in range(args.rounds):
            env.reset()
            env.contacts.object_table = True             # the contact reader exists after the first reset
            fingers = hand_command(env._cur_targets[:, env._hand_joint_ids], env._hand_lower, env._hand_upper)
            action = torch.cat((torch.zeros(n, 64, device=device), fingers), -1)
            alive = torch.ones(n, dtype=torch.bool, device=device)
            for _ in range(steps):
                _, _, terminated, truncated, _ = env.step(action)
                alive &= ~(terminated | truncated)
            c = env._last_contacts
            support = c.get('object_table_force', torch.zeros(n, device=device))
            touch = c['finger_object_force_l1'].sum(-1)+c['palm_object_force_l1']
            origin = env.scene.env_origins
            table = env.table.data.root_pos_w-origin
            obj = env.object.data.root_pos_w-origin
            rel = obj-table
            on_top = (rel[:, :2].abs() <= torch.tensor(TABLE_SIZE[:2], device=device)/2-.01).all(-1)
            still = env.object.data.root_lin_vel_w.norm(dim=-1) < .02
            keep = alive & still & (support > .02) & (touch < .05) & on_top
            kept_rounds.append(int(keep.sum()))
            print(f'ROUND {r} kept {int(keep.sum())}/{n}', flush=True)
            ids = torch.nonzero(keep).flatten()
            out['asset'].append(env._object_asset_index_per_env[ids].cpu())
            out['height'].append((rel[ids, 2]-TABLE_SIZE[2]/2).cpu())
            out['quat'].append(env.object.data.root_quat_w[ids].cpu())
            out['xy'].append(rel[ids, :2].cpu())
    data = {k: torch.cat(v).numpy() for k, v in out.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **data, table_top_height=TABLE_SIZE[2]/2)
    counts = np.bincount(data['asset'], minlength=int(env._object_asset_index_per_env.max())+1)
    print('REST_POSES '+json.dumps(dict(poses=int(len(data['asset'])), assets_with_poses=int((counts > 0).sum()),
          min_per_asset=int(counts.min()), median_per_asset=float(np.median(counts)), kept_rounds=kept_rounds,
          height_median=float(np.median(data['height'])))), flush=True)
    os._exit(0)


if __name__ == '__main__':
    main()
