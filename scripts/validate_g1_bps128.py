#!/usr/bin/env python3
"""Allocated-GPU integration check: every object, observation, terminal/reset path."""
import argparse
import json
import os
import sys
import traceback
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=1200)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch
from isaacsimenvs.tasks.play.play_env import PlayEnv
from dextrah_lab.tasks.g1_revo2_adept.g1_revo2_bps_env import G1Revo2BpsEnv, G1Revo2BpsEnvCfg


def main():
    cfg = G1Revo2BpsEnvCfg()
    cfg.seed = 42
    cfg.scene.num_envs = args.num_envs
    cfg.bps_artifact_dir = "outputs/bps128_validation"
    env = G1Revo2BpsEnv(cfg)
    try:
        assert len(env._bps_bank) == 1200
        assert env._bps_manifest["usd_reference_identity_checks"] == env.num_envs
        assert env._bps_features.shape == (env.num_envs, 132)
        assert env.single_observation_space["policy"].shape == (224,)
        assert env.single_observation_space["critic"].shape == (246,)
        assert env.observation_space.shape == (env.num_envs, 224)
        assert env.state_space.shape == (env.num_envs, 246)
        for hook in ("_pre_physics_step", "_apply_action", "_get_dones"):
            assert getattr(G1Revo2BpsEnv, hook) is getattr(PlayEnv, hook)
        for attr in ("adept_fabric", "collision_adapter", "_pca_action_map"):
            assert not hasattr(env, attr)
        env.reset()
        reset_count = terminal_count = 0
        with torch.inference_mode():
            for step in range(650):
                action = torch.zeros(env.num_envs, 13, device=env.device)
                action[:, 7:] = .5 * torch.sin(torch.tensor(step*.025, device=env.device))
                obs, reward, terminated, truncated, extras = env.step(action)
                for key, dimension in (("policy", 224), ("critic", 246)):
                    assert obs[key].shape == (env.num_envs, dimension)
                    assert torch.isfinite(obs[key]).all()
                    torch.testing.assert_close(obs[key][:, -132:], env._bps_features)
                assert torch.isfinite(reward).all()
                reset_count += int((terminated | truncated).sum())
                if "final_observation" in extras:
                    final = extras["final_observation"]["critic"]
                    assert final.shape == (env.num_envs, 246)
                    torch.testing.assert_close(final[:, -132:], env._bps_features)
                    terminal_count += 1
            assert reset_count > 0 and terminal_count > 0
            torch.testing.assert_close(env._bps_features, env._bps_bank[env._object_asset_index_per_env])
        print("BPS_RUNTIME_VALIDATED " + json.dumps(dict(steps=650, resets=reset_count,
            terminal_steps=terminal_count, objects=1200, envs=env.num_envs,
            actor_dim=224, critic_dim=246, actions=13, fabrics=False, pca=False)), flush=True)
    finally:
        env.close()


try:
    main()
except Exception:
    traceback.print_exc()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(1)
app.close()
