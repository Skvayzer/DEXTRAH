#!/usr/bin/env python3
"""Capture an original Play2Perfect SAPG rollout, without training or rendering.

Loads the run's saved Hydra configuration and strict actor weights. Keeps the
six trained coefficient settings, but evaluates the zero-entropy leader with
deterministic actions. Saves measured PhysX poses for a separate renderer.
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
    parser.add_argument("--fabrics", action="store_true",
                        help="Capture the fabric-enabled task; --config is its params directory")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=30.)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-envs", type=int, default=6)
    parser.add_argument("--env-id", type=int, default=0)
    parser.add_argument("--select-successful-env", action="store_true",
                        help="Record all envs; select one continuous rollout by goals, then sustained lift")
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
    from recording_selection import summarize_candidate, select_candidate
    from dataclasses import asdict
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
    if args.fabrics:
        from dextrah_lab.tasks.g1_revo2_adept.g1_revo2_adept_env import G1Revo2AdeptEnv
        from dextrah_lab.tasks.g1_revo2_adept.g1_revo2_adept_env_cfg import G1Revo2AdeptEnvCfg
        from dextrah_lab.g1_adept.collision_geometry import (
            G1_REVO2_DYNAMIC_SPHERES, G1_REVO2_FIXED_SPHERES, G1_REVO2_SELF_COLLISION_PAIRS,
        )
        class TupleLoader(yaml.SafeLoader):
            pass
        TupleLoader.add_constructor("tag:yaml.org,2002:python/tuple",
                                    lambda loader, node: tuple(loader.construct_sequence(node)))
        saved = {name: yaml.load((args.config / f"{name}.yaml").read_text(), Loader=TupleLoader)
                 for name in ("env", "agent")}
        cfg, env_class, expected_obs = G1Revo2AdeptEnvCfg(), G1Revo2AdeptEnv, 131
    else:
        saved = yaml.safe_load(args.config.read_text())
        cfg, env_class, expected_obs = replace_env_cfg_spaces_with_strings(PlayEnvCfg()), PlayEnv, 92
    # Isaac Lab's typed loader rejects an int over a None-typed seed default.
    cfg.seed = saved["env"].get("seed")
    cfg.from_dict(replace_strings_with_slices(copy.deepcopy(saved["env"])))
    if not args.fabrics:
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
        "fabrics_enabled": args.fabrics,
    }
    if args.fabrics:
        metadata["fabric_geometry"] = dict(
            dynamic=[asdict(s) for s in G1_REVO2_DYNAMIC_SPHERES],
            fixed=[asdict(s) for s in G1_REVO2_FIXED_SPHERES],
            self_pairs=G1_REVO2_SELF_COLLISION_PAIRS,
            include_self_collision=cfg.fabric.include_self_collision,
            influence_distance=cfg.fabric.collision_influence_distance,
            minimum_distance=cfg.fabric.collision_minimum_distance,
            table_surface_offset=cfg.fabric.table_surface_offset,
        )
        metadata["visual_static_joint_pos"] = scene_utils.G1_BODY_DEFAULT_JOINT_POS
        metadata["fabric_config"] = asdict(cfg.fabric) if hasattr(cfg.fabric, "__dataclass_fields__") else cfg.fabric.to_dict()
        for name in ("env", "agent"):
            (args.output / f"source_{name}.yaml").write_text((args.config / f"{name}.yaml").read_text())
    else:
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
        env = env_class(cfg)
    finally:
        scene_utils.build_rigid_object_cfg = original_builder
    try:
        assert env.observation_space.shape[-1] == expected_obs, env.observation_space
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
        metadata.update(joint_names=env.robot.joint_names, body_names=env.robot.body_names,
                        actor_observations=expected_obs, actions=13,
                        generated_object_count=len(env._object_urdf_paths),
                        skipped_only_unassigned_temporary_object_prototypes=True)
        frames = []
        events = [[] for _ in range(args.num_envs)]
        step_metrics = [[] for _ in range(args.num_envs)]
        total_reward = np.zeros(args.num_envs)
        goal_hits = np.zeros(args.num_envs, dtype=np.int64)
        resets = np.zeros(args.num_envs, dtype=np.int64)
        previous_episode_goals = np.zeros(args.num_envs)
        begin = time.monotonic()

        def array(tensor):
            return tensor.detach().cpu().numpy().copy()

        def snapshot(step):
            origin = env.scene.env_origins
            def pose(asset):
                return array(torch.cat((asset.data.root_pos_w - origin, asset.data.root_quat_w), dim=-1))
            result = dict(step=np.full(args.num_envs, step), joint_pos=array(env.robot.data.joint_pos),
                body_pos=array(env.robot.data.body_pos_w - origin[:, None, :]),
                body_quat=array(env.robot.data.body_quat_w), robot=pose(env.robot),
                object=pose(env.object), table=pose(env.table), goal=pose(env.goal_viz),
                reward=total_reward.copy(), goal_hits=goal_hits.copy(), resets=resets.copy(),
                lifted=array(env._lifted_object), goal_error=array(env._keypoints_max_dist),
                lift_height=array(.05 + env.object.data.root_pos_w[:, 2] - origin[:, 2] - env._object_init_z))
            if args.fabrics:
                # Sample after physics, using the training adapter itself.
                # These are measured proxies, not lagging pre-step positions.
                collision = env.collision_adapter.build()
                result.update(
                    fabric_dynamic=array(env.collision_adapter.dynamic_positions - origin[:, None, :]),
                    fabric_fixed=array(env.collision_adapter.fixed_positions - origin[:, None, :]),
                    fabric_clearance=array(collision.clearance),
                    fabric_table_height=array(env._table_z_per_env) + cfg.fabric.table_surface_offset,
                )
            return result

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
                rewards, dones = array(reward).reshape(-1), array(done).reshape(-1).astype(bool)
                total_reward += rewards
                # episode_final is captured before auto-reset, so terminal goal
                # hits are not lost when the environment clears its buffers.
                terminal = info.get("episode_final", {})
                episode_goals = array(terminal["successes"])
                hits = np.maximum(0, episode_goals - previous_episode_goals).astype(np.int64)
                goal_hits += hits
                previous_episode_goals = np.where(dones, 0, episode_goals)
                lift = array(.05 + env.object.data.root_pos_w[:, 2]
                    - env.scene.env_origins[:, 2] - env._object_init_z)
                final_extras = info.get("final_model_state_extras", {})
                if "lift_height" in final_extras:
                    lift[dones] = array(final_extras["lift_height"])[dones]
                lifted, errors = array(env._lifted_object), array(env._keypoints_max_dist)
                for e in range(args.num_envs):
                    step_metrics[e].append(dict(step=step + 1, reward=float(rewards[e]),
                        goal_hit=bool(hits[e]), done=bool(dones[e]), lifted=bool(lifted[e]),
                        lift_height_m=float(lift[e]), goal_error_m=float(errors[e])))
                ended = done.reshape(-1).nonzero(as_tuple=False).flatten().to(player.device)
                if player.is_rnn and ended.numel():
                    for state in player.states:
                        state[:, ended, :] = 0.
                resets += dones
                for e in np.flatnonzero(dones):
                    events[e].append({"step": step + 1, "episode_final": {
                        k: float(v[e]) for k, v in terminal.items()
                        if isinstance(v, torch.Tensor) and v.ndim == 1 and v.numel() == env.num_envs}})
                if (step + 1) % 120 == 0:
                    print(f"ROLLOUT {step+1}/{count} sim_s={(step+1)*dt:.1f} "
                          f"goals={goal_hits.tolist()} lifts_cm={np.round(lift*100, 1).tolist()} "
                          f"resets={resets.tolist()} "
                          f"wall_s={time.monotonic()-begin:.1f}", flush=True)
        summaries = [summarize_candidate(metrics, dt, cfg.reward.lifting_bonus_threshold)
                     for metrics in step_metrics]
        e = select_candidate(summaries) if args.select_successful_env else args.env_id
        stacked = {key: np.stack([f[key] for f in frames]) for key in frames[0]}
        np.savez_compressed(args.output / "trajectory.npz", **{key: value[:, e] for key, value in stacked.items()})
        if args.select_successful_env:
            np.savez_compressed(args.output / "candidates.npz", **stacked)
            metadata["selection"] = "Selected one continuous environment by goal count, sustained lift, then reward; not a success-rate benchmark"
        # Export the exact selected geometry before closing the environment.
        for name, resolver in (("object", object_urdf_for_env), ("table", table_urdf_for_env)):
            text, path = resolver(env, e)
            model = yourdfpy.URDF.load(path)
            model.scene.export(args.output / f"{name}.glb")
            (args.output / f"{name}.urdf").write_text(text)
            metadata[f"{name}_source"] = str(path)
        metadata.update(frames=len(frames), **summaries[e],
                        recorded_env_id=e, object_asset_index=int(env._object_asset_index_per_env[e]),
                        selection_candidates=summaries,
                        lift_metric_note="Reward lift metric = root height minus initial root height + 0.05 m; latched lifted flag is not sustained lift",
                        lifting_bonus_threshold_m=cfg.reward.lifting_bonus_threshold,
                        wall_seconds=time.monotonic()-begin, completed=True)
        (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2))
        (args.output / "metrics.json").write_text(json.dumps({"steps": step_metrics[e], "episodes": events[e]}))
        (args.output / "candidate_metrics.json").write_text(json.dumps({"steps": step_metrics, "episodes": events}))
        print("RECORDING_CAPTURE_PASS " + json.dumps(metadata), flush=True)
    finally:
        env.close()
    app.close()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
