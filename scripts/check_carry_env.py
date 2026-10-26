#!/usr/bin/env python3
"""Build the mjlab carry env (dextrah_lab/mj/carry_env.py) and step it with zero actions (hold at the
stored finger reference, bare SONIC walking): observation sizes, phases, table parking, outcomes.
Runs on CPU (MuJoCo Warp CPU device) or GPU; no policy, no training."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, default=Path('/data1/users/konstantin.smirnov/DEXTRAH-G1-WholeBody/outputs/0_mjlab_wb_imgen_dualclip_1454'))
    p.add_argument('--holding-state', type=Path, default=None)
    p.add_argument('--clips', type=Path, default=Path('/data1/users/konstantin.smirnov/DEXTRAH-G1-WholeBody/outputs/carry_clips/levels_v1.npz'))
    p.add_argument('--num-envs', type=int, default=12)
    p.add_argument('--steps', type=int, default=360)
    p.add_argument('--device', default='cpu')
    p.add_argument('--stage', type=int, default=2)
    p.add_argument('--level', type=int, default=2)
    p.add_argument('--mass-scale', type=float, nargs=2, default=(.7, 2.))
    p.add_argument('--com', type=float, default=.02)
    p.add_argument('--base-env', action='store_true', help='control: the plain manipulation env, same steps')
    args = p.parse_args()
    import torch
    from dextrah_lab.mj.compat import install
    install()
    from dextrah_lab.mj.carry_env import G1CarryMjEnv, CONTEXT_DIM
    from dextrah_lab.mj.env import load_run_cfg
    from dextrah_lab.mj.factory import DEFAULT_MUJOCO, WORKSPACE
    from dextrah_lab.sgs.mj_task import SgsSettings
    from dextrah_lab.wholebody.carry_compose import CarrySettings
    from dextrah_lab.wholebody.sonic import FrozenSonic
    holding = args.holding_state or args.run/'best/best_epoch12662.sgs.pt'
    contract = json.loads((args.run/'task_contract.json').read_text())
    cfg = load_run_cfg(args.run)
    cfg.sonic_body.spawn_on_table = contract['body_termination']['spawn_on_table']
    sonic = FrozenSonic(WORKSPACE/'GRAIL', WORKSPACE/'G1-SONIC-models/checkpoint/SONIC/models/sonic_manipulation_base', args.device)
    settings = CarrySettings(stage=args.stage, level=args.level, table_keep_probability=.25,
                             mass_scale=tuple(args.mass_scale), com_offset_m=args.com)
    if args.base_env:
        from dextrah_lab.mj.env import G1SonicTouchMjEnv
        env = G1SonicTouchMjEnv(cfg, sonic=sonic, num_envs=args.num_envs, device=args.device, nconmax=64, njmax=300,
                                mujoco_cfg=DEFAULT_MUJOCO)
        obs, _ = env.reset()
        a = torch.zeros(env.num_envs, 70, device=args.device)
        for step in range(args.steps):
            obs, reward, terminated, truncated, info = env.step(a)
            if step % 30 == 0:
                print(f'BASE_STEP {step} finite={bool(torch.isfinite(obs["policy"]).all())}', flush=True)
        print('CHECK_CARRY_ENV_OK base', flush=True)
        return
    env = G1CarryMjEnv(cfg, sonic=sonic, carry=settings, holding_state=holding,
                       held_settings=SgsSettings(**dict(contract['sgs']['settings'], eval_pool_size=0)),
                       clips_path=args.clips, num_envs=args.num_envs, device=args.device, nconmax=64, njmax=300,
                       mujoco_cfg=DEFAULT_MUJOCO)
    obs, _ = env.reset()
    assert obs['policy'].shape == (env.num_envs, cfg.observation_space), obs['policy'].shape
    assert obs['critic'].shape == (env.num_envs, cfg.state_space), obs['critic'].shape
    assert torch.isfinite(obs['policy']).all() and torch.isfinite(obs['critic']).all()
    ctx = obs['policy'][:, -CONTEXT_DIM:]
    print('OBS', tuple(obs['policy'].shape), tuple(obs['critic'].shape), 'context', ctx[0].tolist(), flush=True)
    print('GOAL_MASKED', float(obs['policy'][:, env._goal_policy].abs().max()), float(obs['policy'][:, env._goal_surface_policy].abs().max()))
    actions = torch.zeros(env.num_envs, 70, device=args.device)
    ends = []
    for step in range(args.steps):
        obs, reward, terminated, truncated, info = env.step(actions)
        assert torch.isfinite(obs['policy']).all() and torch.isfinite(reward).all()
        done = terminated | truncated
        if step % 30 == 0:
            t = env._episode_time()
            print(f'STEP {step} t={t[:4].tolist()} phase={env._seq.phase(t)[:6].tolist()} parked={env._table_parked[:6].int().tolist()} '
                  f'offset_cm={(100*env._offset_error[:6]).round().tolist()} clip={[env.clips.names[int(c)] for c in env._seq.clip[:3]]} '
                  f'reward={reward[:4].round(decimals=3).tolist()} done_now={int(done.sum())} done_total={sum(len(e) for e in ends)}', flush=True)
        if bool(done.any()):
            ends.append(done.nonzero().flatten().tolist())
    extras = {k: float(v) for k, v in env.extras.items() if k.startswith('carry/') and torch.is_tensor(v) and v.numel() == 1}
    print('CARRY_METRICS '+json.dumps(extras, indent=1), flush=True)
    print('CHECK_CARRY_ENV_OK', flush=True)


if __name__ == '__main__':
    main()
