#!/usr/bin/env python3
"""mjlab port: per-reason termination counts and key state signals (diagnostic)."""
import argparse
import json
import os
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--num-envs', type=int, default=120)
    p.add_argument('--seconds', type=float, default=5.)
    p.add_argument('--checkpoint', type=Path)
    args = p.parse_args()
    import torch
    sys.path.insert(0, str(Path(__file__).parent))
    from dextrah_lab.mj.factory import make_mj_env
    from mj_contact_capacity import build_policy
    env = make_mj_env(args.num_envs, 'cuda:0')
    n = env.num_envs
    obs, _ = env.reset()
    model = states = None
    if args.checkpoint:
        model, _ = build_policy(env, args.checkpoint, 'cuda:0')
        states = tuple(x.cuda() for x in model.get_default_rnn_state())
    passive = torch.zeros(n, 70, device='cuda:0'); passive[:, 64:] = -1
    counts, lengths = {}, []
    first = {}
    with torch.inference_mode():
        for step in range(round(args.seconds/env.step_dt)):
            if step == 0:
                o = env.object.data.root_pos_w[:4]
                first = dict(object_pos=o.tolist(), table_z=env._table_z_per_env[:4].tolist(),
                             palm=env.robot.data.body_pos_w[:4, env._palm_body_id].tolist(),
                             tips=env.robot.data.body_pos_w[0, env._fingertip_body_ids].tolist(),
                             fingertip_dist=env._curr_fingertip_distances[:4].tolist(),
                             goal=env.goal_viz.data.root_pos_w[:4].tolist(),
                             obs_absmax=float(obs['policy'].abs().max()), obs_mean_abs=float(obs['policy'].abs().mean()))
            if model is not None:
                out = model(dict(obs=torch.cat((obs['policy'], obs['policy'].new_zeros(n, 1)), -1), is_train=False,
                                 prev_actions=None, rnn_states=states))
                states, action = out['rnn_states'], out['mus']
            else:
                action = passive
            ep = env.episode_length_buf.clone()
            obs, r, te, tr, info = env.step(action)
            done = te | tr
            if states is not None:
                for s in states:
                    s[:, done, :] = 0
            final = info['episode_final']
            for k, v in final.items():
                if k.startswith('done_') or k in ('robot_fall', 'numerical_failure'):
                    counts[k] = counts.get(k, 0)+int((done & v.bool()).sum())
            counts['episodes'] = counts.get('episodes', 0)+int(done.sum())
            lengths += (ep[done]+1).tolist()
    print('EPISODE_STATS '+json.dumps(dict(counts=counts, mean_len_steps=sum(lengths)/max(len(lengths), 1),
                                            first_step=first)), flush=True)
    os._exit(0)


if __name__ == '__main__':
    main()
