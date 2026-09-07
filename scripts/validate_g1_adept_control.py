#!/usr/bin/env python3
"""Exercise the actual G1 PhysX controller and SAPG observer before training.

Run inside an allocated GPU job with --headless. This is a scripted control
response check, not a learned grasp-success evaluation.
"""
import argparse
import json
import os
import sys
import traceback
from types import SimpleNamespace

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=64)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import torch
from isaacsimenvs.utils.rlgames_utils import EnvStatsAlgoObserver
from dextrah_lab.tasks.g1_revo2_adept.g1_revo2_adept_env_cfg import G1Revo2AdeptEnvCfg
import dextrah_lab.tasks.g1_revo2_adept.gym_setup  # noqa: F401


def check_observer():
    records = {}
    observer = EnvStatsAlgoObserver()
    observer.after_init(SimpleNamespace(
        writer=SimpleNamespace(add_scalar=lambda key, value, step: records.update({key: (float(value), step)})),
        games_to_track=10,
    ))
    infos = {
        "episode_cumulative": {"lifting_rew": torch.tensor([100., 100., 2., 4.])},
        "episode_final": {"any_success": torch.tensor([0., 0., 1., 0.])},
        "successes": torch.tensor([10., 20., 1., 3.]),
        "fabric/test": torch.tensor(0.05),
    }
    observer.process_infos(infos, torch.arange(4), ignore_env_boundary=2)
    observer.after_print_stats(1024, 1, 2.0)
    assert records["episode_cumulative/lifting_rew"] == (3., 1024)
    assert records["episode_final/any_success"] == (0.5, 1024)
    assert records["successes"] == (2., 1024)
    assert records["successes_per_block/block_0"] == (15., 1024)
    assert records["successes_per_block/block_1"] == (2., 1024)
    assert all(step == 1024 for _, step in records.values())
    print("PASS: leader-only episode stats, terminal successes, per-block stats, frame axis", flush=True)


def main():
    check_observer()
    cfg = G1Revo2AdeptEnvCfg()
    cfg.seed = 42
    cfg.scene.num_envs = args.num_envs
    cfg.assets.num_assets_per_type = 1
    cfg.domain_randomization.use_action_delay = False
    cfg.domain_randomization.use_obs_delay = False
    env = gym.make("Adept-G1-Revo2-SimToolReal-Repose", cfg=cfg).unwrapped
    try:
        env.reset()
        actions = torch.zeros(env.num_envs, 13, device=env.device)
        initial = env._prev_targets[:, env._canonical_joint_ids_lab].clone()
        env._pre_physics_step(actions)
        nominal = env._prev_targets[:, env._canonical_joint_ids_lab]
        torch.testing.assert_close(nominal[:, :7], initial[:, :7])
        midpoint = 0.5 * (env._hand_lower + env._hand_upper)
        expected = cfg.action.hand_moving_average * midpoint + (1 - cfg.action.hand_moving_average) * initial[:, 7:]
        torch.testing.assert_close(nominal[:, 7:], expected)

        # A positive arm command must accumulate from nominal targets even
        # though the downstream fabric is deliberately left behind.
        previous = nominal.clone()
        actions[:, :7] = 0.4
        env._pre_physics_step(actions)
        raw = torch.clamp(previous[:, :7] + cfg.action.dof_speed_scale * env.step_dt * 0.4, env._arm_lower, env._arm_upper)
        expected = cfg.action.arm_moving_average * raw + (1 - cfg.action.arm_moving_average) * previous[:, :7]
        torch.testing.assert_close(env._prev_targets[:, env._arm_joint_ids], expected)
        print("PASS: zero action closes toward midpoint; arm nominal accumulation matches P2P", flush=True)

        env.reset()
        snapshots = {}
        actions.zero_()
        # Advance physics directly so task drop/time-limit auto-resets cannot
        # disguise a blocked hand response. All fabric collision terms stay on.
        with torch.inference_mode():
            for label, command in (("open", -0.6), ("closed", 0.6)):
                actions[:, 7:] = command
                for _ in range(180):
                    env._pre_physics_step(actions)
                    for _ in range(cfg.decimation):
                        env._apply_action()
                        env.scene.write_data_to_sim()
                        env.sim.step(render=False)
                        env.scene.update(cfg.sim.dt)
                    env.common_step_counter += 1
                    q, qd = env._measured_canonical_state()
                    assert torch.isfinite(q).all() and torch.isfinite(qd).all()
                    assert torch.isfinite(env.adept_fabric.state.position).all()
                q, _ = env._measured_canonical_state()
                normalized = (q[:, 7:] - env._hand_lower) / (env._hand_upper - env._hand_lower)
                snapshots[label] = normalized.mean(dim=0)
            response = snapshots["closed"] - snapshots["open"]
            print("HAND_RESPONSE " + json.dumps({key: value.tolist() for key, value in {**snapshots, "travel_fraction": response}.items()}), flush=True)
            assert torch.all(response > 0.25), "a hand joint cannot traverse 25% of its range with fabric/PCA active"
            assert env._last_pca_weight == 0.05

            # Subset resets restore both nominal and fabric state, and must
            # leave every other environment's accumulator untouched.
            previous = env._prev_targets.clone()
            env._reset_idx(torch.tensor([0], device=env.device))
            torch.testing.assert_close(env._prev_targets[1:], previous[1:])
            q, _ = env._measured_canonical_state()
            torch.testing.assert_close(env._prev_targets[0, env._canonical_joint_ids_lab], q[0])
            # P2P reset uses hardware limits; fabric has the configured 0.02
            # rad inner margin. Its reset position is therefore margin-clamped.
            torch.testing.assert_close(
                env.adept_fabric.state.position[0],
                q[0].clamp(env.adept_fabric.lower_limits, env.adept_fabric.upper_limits),
            )
            obs = env._get_observations()
            assert obs["policy"].shape == (env.num_envs, 131)
            assert obs["critic"].shape == (env.num_envs, 153)
            assert all(torch.isfinite(value).all() for value in obs.values())
        print("PASS: physical hand response, finite states, subset resets, actor/critic dimensions", flush=True)
    finally:
        env.close()


try:
    main()
except Exception:
    # SimulationApp.close() can terminate the interpreter before Python prints
    # a pending exception. Preserve both the traceback and a nonzero exit code.
    traceback.print_exc()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(1)  # Kit's exit handler can otherwise replace the failure status.
app.close()
