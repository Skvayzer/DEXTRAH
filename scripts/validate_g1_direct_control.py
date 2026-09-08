#!/usr/bin/env python3
"""Validate direct P2P action semantics and runtime tensors in an allocated GPU job."""
import argparse
import json
import os
import sys
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=96)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import torch
from isaacsimenvs.tasks.play.play_env import PlayEnv
from dextrah_lab.tasks.g1_revo2_adept.g1_revo2_adept_env_cfg import G1Revo2AdeptEnvCfg
from dextrah_lab.tasks.g1_revo2_adept.g1_revo2_direct_env import G1Revo2DirectEnv, G1Revo2DirectEnvCfg
import dextrah_lab.tasks.g1_revo2_adept.gym_setup  # noqa: F401


def main():
    cfg = G1Revo2DirectEnvCfg()
    original = G1Revo2AdeptEnvCfg()
    assert original.fabric.enabled and original.fabric.pca_enabled, "Shared config was mutated"
    assert not cfg.fabric.enabled and not cfg.fabric.pca_enabled
    assert not cfg.fabric.append_state_to_observations
    for hook in ("_pre_physics_step", "_apply_action", "_get_observations", "_get_dones"):
        assert getattr(G1Revo2DirectEnv, hook) is getattr(PlayEnv, hook), hook
    cfg.seed = 42
    cfg.scene.num_envs = args.num_envs
    cfg.assets.num_assets_per_type = 1
    # Isolate one-step target arithmetic, then restore normal randomization.
    cfg.domain_randomization.use_action_delay = False
    env = gym.make("G1-Revo2-SimToolReal-Repose-Direct", cfg=cfg).unwrapped
    try:
        env.reset()
        for attribute in ("adept_fabric", "collision_adapter", "_pca_action_map", "_cur_velocity_targets"):
            assert not hasattr(env, attribute), attribute
        obs = env._get_observations()
        assert obs["policy"].shape == (env.num_envs, 92)
        assert obs["critic"].shape == (env.num_envs, 114)
        actions = torch.zeros(env.num_envs, 13, device=env.device)
        actions[:, :7] = .4
        for hand_action in (-1., 0., 1.):
            previous = env._prev_targets.clone()
            actions[:, 7:] = hand_action
            env._pre_physics_step(actions)
            arm_raw = (previous[:, env._arm_joint_ids] + cfg.action.dof_speed_scale * env.step_dt * .4).clamp(env._arm_lower, env._arm_upper)
            arm_expected = cfg.action.arm_moving_average * arm_raw + (1 - cfg.action.arm_moving_average) * previous[:, env._arm_joint_ids]
            arm_expected = arm_expected.clamp(env._arm_lower, env._arm_upper)
            hand_raw = env._hand_lower + .5 * (hand_action + 1) * (env._hand_upper - env._hand_lower)
            hand_expected = cfg.action.hand_moving_average * hand_raw + (1 - cfg.action.hand_moving_average) * previous[:, env._hand_joint_ids]
            # Reset poses use physical joint bounds; original P2P action
            # targets additionally have the shared 0.02-rad inner margin.
            hand_expected = hand_expected.clamp(env._hand_lower, env._hand_upper)
            torch.testing.assert_close(env._cur_targets[:, env._arm_joint_ids], arm_expected)
            torch.testing.assert_close(env._cur_targets[:, env._hand_joint_ids], hand_expected)
            torch.testing.assert_close(env._cur_targets, env._prev_targets)
        print("PASS: inherited P2P control; full-range absolute hand targets; no fabric/PCA objects", flush=True)

        cfg.domain_randomization.use_action_delay = True
        env.reset()
        resets = 0
        with torch.inference_mode():
            for step in range(650):
                actions[:, :7] = .15 * torch.sin(torch.tensor(step * .015, device=env.device))
                actions[:, 7:] = .6 * torch.sin(torch.tensor(step * .025, device=env.device))
                obs, reward, terminated, truncated, extras = env.step(actions)
                assert all(torch.isfinite(value).all() for value in obs.values())
                assert torch.isfinite(reward).all()
                assert obs["policy"].shape[-1] == 92 and obs["critic"].shape[-1] == 114
                for key in ("ever_lifted", "any_success"):
                    assert extras["episode_final"][key].shape == (env.num_envs,)
                assert not any(key.startswith("fabric/") for key in extras)
                resets += int((terminated | truncated).sum())
            assert resets > 0, "No termination/reset path exercised"
            env._episode_ever_lifted[:] = True
            env._reset_idx(torch.tensor([0], device=env.device))
            assert not env._episode_ever_lifted[0] and env._episode_ever_lifted[1:].all()
        print("DIRECT_VALIDATION " + json.dumps({"steps": 650, "resets": resets,
              "actor_dim": 92, "critic_dim": 114, "actions": 13, "fabric": False, "pca": False}), flush=True)
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
