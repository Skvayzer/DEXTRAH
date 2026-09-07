#!/usr/bin/env python3
"""Capture an original Play2Perfect SAPG rollout, without training or rendering.

Loads the run's saved Hydra configuration and strict actor weights. Keeps the
six trained coefficient settings, but evaluates the zero-entropy leader with
deterministic actions. Saves measured PhysX poses for a separate CPU renderer.
"""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--play2perfect-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=30.)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-envs", type=int, default=6)
    parser.add_argument("--env-id", type=int, default=0)
    parser.add_argument("--success-tolerance", type=float, default=.01)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite recording: {args.output}")
    if not 0 <= args.env_id < args.num_envs or args.seconds <= 0:
        raise ValueError("Invalid recording range")
    for path in (args.play2perfect_root, args.play2perfect_root / "rl_games"):
        sys.path.insert(0, str(path.resolve()))
    args.headless = True
    args.enable_cameras = False
    app = AppLauncher(args).app

    import numpy as np
    import torch
    import yaml
    import yourdfpy
    from isaaclab.envs.utils.spaces import (
        replace_env_cfg_spaces_with_strings, replace_strings_with_env_cfg_spaces,
    )
    from isaaclab.utils import replace_strings_with_slices
    from isaacsimenvs.tasks.play.play_env import PlayEnv
    from isaacsimenvs.tasks.play.play_env_cfg import PlayEnvCfg
    from isaacsimenvs.tasks.play.utils import scene_utils
    from isaacsimenvs.tasks.play.pose_viewer import object_urdf_for_env, table_urdf_for_env
    from isaacsimenvs.utils.rlgames_utils import register_rlgames_env
    from rl_games.torch_runner import Runner

    torch.set_num_threads(2)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    saved = yaml.safe_load(args.config.read_text())
    cfg = replace_env_cfg_spaces_with_strings(PlayEnvCfg())
    cfg.from_dict(replace_strings_with_slices(copy.deepcopy(saved["env"])))
    cfg = replace_strings_with_env_cfg_spaces(cfg)
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    cfg.sim.device = args.device
    # Only capacity is reduced: timesteps, solver, collisions, DR and actions
    # stay at the saved training settings. No cameras/visuals are needed.
    for name, capacity in {
        "gpu_found_lost_pairs_capacity": 2**18,
        "gpu_found_lost_aggregate_pairs_capacity": 2**19,
        "gpu_total_aggregate_pairs_capacity": 2**18,
        "gpu_max_rigid_contact_count": 2**18,
        "gpu_max_rigid_patch_count": 2**16,
        "gpu_collision_stack_size": 2**24,
    }.items():
        setattr(cfg.sim.physx, name, capacity)
    # The checkpoint has no environment/curriculum state. Pin the evaluation
    # criterion explicitly instead of silently restarting the easy curriculum.
    cfg.termination.eval_success_tolerance = args.success_tolerance
    cfg.termination.success_tolerance = args.success_tolerance
    if cfg.assets.robot_profile != "g1_brainco_right":
        raise ValueError("This recorder requires the original G1+Revo2 profile")
    dt = cfg.sim.dt * cfg.decimation
    stride = round(1 / (args.fps * dt))
    if stride < 1 or not math.isclose(stride * dt * args.fps, 1.):
        raise ValueError("Video FPS must divide the policy rate exactly")
    count = round(args.seconds / dt)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    checkpoint = checkpoint[0] if 0 in checkpoint else checkpoint
    agent = copy.deepcopy(saved["agent"])
    ac = agent["params"]["config"]
    blocks = ac["num_actors"] // ac["expl_coef_block_size"]
    if blocks != 6 or ac["expl_type"] != "mixed_expl_learn_param":
        raise ValueError("Expected the six-block original SAPG checkpoint")
    ac.update(num_actors=blocks, expl_coef_block_size=1, device=args.device,
              device_name=args.device, multi_gpu=False)
    ac["player"].update(deterministic=True, print_stats=False, evaluation=False)
    agent["params"]["seed"] = args.seed
    args.output.mkdir(parents=True)
    metadata = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": hashlib.file_digest(args.checkpoint.open("rb"), "sha256").hexdigest(),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_training_frames": int(checkpoint["frame"]),
        "saved_config": str(args.config.resolve()),
        "play2perfect_commit": subprocess.check_output(
            ["git", "-C", str(args.play2perfect_root), "rev-parse", "HEAD"], text=True).strip(),
        "seed": args.seed, "num_envs": args.num_envs, "recorded_env_id": args.env_id,
        "fps": args.fps, "policy_dt": dt, "seconds": count * dt,
        "policy": "SAPG zero-entropy leader; deterministic mean actions",
        "coefficient_id": 0., "trained_coefficient_ids": [50, 40, 30, 20, 10, 0],
        "success_tolerance_m": args.success_tolerance,
        "curriculum_state_available": checkpoint.get("env_state") is not None,
        "domain_randomization": "retained from saved training configuration",
        "selection": "fixed seed/environment, continuous unselected rollout; not a success-rate benchmark",
        "robot_urdf": str((args.play2perfect_root / cfg.assets.robot_urdf).resolve()),
        "presentation": "Measured active-body poses; unmodeled whole body is static visual context only",
    }
    (args.output / "source_config.yaml").write_text(args.config.read_text())
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    # MultiUsdFileCfg first materializes EVERY temporary template, even when
    # only six round-robin envs exist. Its first N assignments are exactly the
    # first N paths; omit only unreachable prototypes, not generated assets.
    original_builder = scene_utils.build_rigid_object_cfg
    def evaluation_builder(prim_path, usd_paths, **kwargs):
        selected = usd_paths[:args.num_envs] if prim_path.endswith("/Object") else usd_paths
        return original_builder(prim_path, selected, **kwargs)
    scene_utils.build_rigid_object_cfg = evaluation_builder
    try:
        env = PlayEnv(cfg)
    finally:
        scene_utils.build_rigid_object_cfg = original_builder
    try:
        assert env.observation_space.shape[-1] == 92, env.observation_space
        assert env.action_space.shape[-1] == 13, env.action_space
        wrapped = register_rlgames_env(env, rl_device=args.device,
            clip_obs=float(agent["params"]["env"]["clip_observations"]),
            clip_actions=float(agent["params"]["env"]["clip_actions"]))
        runner = Runner()
        runner.load(agent)
        player = runner.create_player()
        player.set_weights(checkpoint)  # strict state_dict loading, including normalizers
        player.has_batch_dimension = True
        player.intr_reward_coef_embd.fill_(0.)
        player.reset()
        obs = player.env_reset(wrapped)
        player.get_batch_size(obs, 1)
        print("CHECKPOINT_LOADED_STRICT " + json.dumps(metadata), flush=True)
        e = args.env_id
        # Export the exact generated primitive geometry before env.close removes
        # temporary assets. No re-generation or substitute tool is used to render.
        for name, resolver in (("object", object_urdf_for_env), ("table", table_urdf_for_env)):
            text, path = resolver(env, e)
            model = yourdfpy.URDF.load(path)
            model.scene.export(args.output / f"{name}.glb")
            (args.output / f"{name}.urdf").write_text(text)
            metadata[f"{name}_source"] = str(path)
        metadata.update(joint_names=env.robot.joint_names, body_names=env.robot.body_names,
                        actor_observations=92, actions=13,
                        generated_object_count=len(env._object_urdf_paths),
                        object_asset_index=int(env._object_asset_index_per_env[e]),
                        skipped_only_unassigned_temporary_object_prototypes=True)
        frames, events, step_metrics = [], [], []
        total_reward, goal_hits, resets = 0., 0, 0
        peak_lift = 0.
        begin = time.monotonic()

        def array(tensor):
            return tensor.detach().cpu().numpy().copy()

        def snapshot(step):
            origin = env.scene.env_origins[e]
            def pose(asset):
                return array(torch.cat((asset.data.root_pos_w[e] - origin, asset.data.root_quat_w[e])))
            return dict(step=step, joint_pos=array(env.robot.data.joint_pos[e]),
                body_pos=array(env.robot.data.body_pos_w[e] - origin),
                body_quat=array(env.robot.data.body_quat_w[e]), robot=pose(env.robot),
                object=pose(env.object), table=pose(env.table), goal=pose(env.goal_viz),
                reward=total_reward, goal_hits=goal_hits, resets=resets,
                lifted=bool(env._lifted_object[e]),
                goal_error=float(env._keypoints_max_dist[e]))

        with torch.inference_mode():
            for step in range(count):
                if step % stride == 0:
                    frames.append(snapshot(step))
                action = player.get_action(obs, is_deterministic=True)
                if not torch.isfinite(action).all():
                    raise RuntimeError("Non-finite checkpoint action")
                obs, reward, done, info = player.env_step(wrapped, action)
                if not torch.isfinite(obs).all() or not torch.isfinite(env.robot.data.joint_pos).all():
                    raise RuntimeError("Non-finite rollout state")
                total_reward += float(reward[e])
                hit = bool(env._is_success[e])
                goal_hits += int(hit)
                lift = float(.05 + env.object.data.root_pos_w[e, 2]
                    - env.scene.env_origins[e, 2] - env._object_init_z[e])
                peak_lift = max(peak_lift, lift)
                step_metrics.append(dict(step=step + 1, reward=float(reward[e]),
                    goal_hit=hit, done=bool(done[e]), lifted=bool(env._lifted_object[e]),
                    lift_height_m=lift, goal_error_m=float(env._keypoints_max_dist[e])))
                ended = done.reshape(-1).nonzero(as_tuple=False).flatten().to(player.device)
                if player.is_rnn and ended.numel():
                    for state in player.states:
                        state[:, ended, :] = 0.
                if bool(done[e]):
                    resets += 1
                    terminal = info.get("episode_final", {})
                    events.append({"step": step + 1, "episode_final": {
                        k: float(v[e]) for k, v in terminal.items()
                        if isinstance(v, torch.Tensor) and v.ndim == 1 and v.numel() == env.num_envs}})
                if (step + 1) % 120 == 0:
                    print(f"ROLLOUT {step+1}/{count} sim_s={(step+1)*dt:.1f} "
                          f"reward={total_reward:.2f} goals={goal_hits} resets={resets} "
                          f"wall_s={time.monotonic()-begin:.1f}", flush=True)
        np.savez_compressed(args.output / "trajectory.npz",
            **{key: np.stack([f[key] for f in frames]) for key in frames[0]})
        metadata.update(frames=len(frames), total_reward=total_reward, goal_hits=goal_hits,
                        resets=resets, peak_lift_height_m=peak_lift,
                        wall_seconds=time.monotonic()-begin, completed=True)
        (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2))
        (args.output / "metrics.json").write_text(json.dumps({"steps": step_metrics, "episodes": events}))
        print("RECORDING_CAPTURE_PASS " + json.dumps(metadata), flush=True)
    finally:
        env.close()
    app.close()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
