"""GPU integration checks for the ADEPT Stage-1 environment."""

from __future__ import annotations

import argparse
import json
import os
import traceback

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Adept-Kuka-Allegro-Repose")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--rollout_steps", type=int, default=64)
parser.add_argument("--contact_steps", type=int, default=64)
parser.add_argument("--contact_gap", type=float, default=0.02)
parser.add_argument("--contact_velocity", type=float, default=-1.0)
parser.add_argument("--use_cuda_graph", action="store_true")
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab_tasks.utils import parse_env_cfg

import dextrah_lab.tasks.dextrah_kuka_allegro.gym_setup  # noqa: F401, E402
from dextrah_lab.tasks.dextrah_kuka_allegro.adept_mdp import ADEPT_PRIMITIVES


def _assert_finite(name: str, tensor: torch.Tensor) -> None:
    if not bool(torch.isfinite(tensor).all()):
        bad = int((~torch.isfinite(tensor)).sum().item())
        raise RuntimeError(f"{name} contains {bad} non-finite values")


def _step(env, actions):
    observations, rewards, terminated, truncated, extras = env.step(actions)
    return observations, rewards, terminated | truncated, extras


def main() -> None:
    cfg = parse_env_cfg(
        args.task,
        device=args.device,
        num_envs=args.num_envs,
        use_fabric=not args.disable_fabric,
    )
    cfg.use_cuda_graph = args.use_cuda_graph
    env = gym.make(args.task, cfg=cfg)
    base = env.unwrapped
    env.reset()
    print("ADEPT_VALIDATION_STAGE=environment_ready", flush=True)

    # Fire the two spherical primitives downward through the index/thumb tip
    # origins.  Starting outside the fingers and letting PhysX establish the
    # contact is both representative and much better conditioned than
    # teleporting an object into an overlapping collision configuration.
    spec_index = torch.round(
        base.object_id_scalar[:, 0] * (len(ADEPT_PRIMITIVES) - 1)
    ).long()
    sphere_specs = [
        (index, spec.dimensions[0])
        for index, spec in enumerate(ADEPT_PRIMITIVES)
        if spec.shape == "sphere"
    ]
    sphere_spec_indices = torch.tensor(
        [index for index, _radius in sphere_specs], device=base.device
    )
    sphere_env_mask = torch.any(
        spec_index[:, None] == sphere_spec_indices[None, :], dim=1
    )
    probe_env_ids = torch.nonzero(sphere_env_mask, as_tuple=False).flatten()[:2]
    if len(probe_env_ids) != 2:
        raise RuntimeError("contact validation requires both ADEPT sphere environments")
    index_env_id, thumb_env_id = probe_env_ids.cpu().tolist()

    local_position = torch.stack(
        (
            base.hand_pos[index_env_id, 1, :],
            base.hand_pos[thumb_env_id, -1, :],
        )
    )
    if not bool((local_position.abs() < 2.0).all()):
        raise RuntimeError("fingertip positions are not environment-local coordinates")
    sphere_radius = torch.tensor(
        [radius for _index, radius in sphere_specs], device=base.device
    )
    radius_by_env = sphere_radius[
        (spec_index[probe_env_ids, None] == sphere_spec_indices[None, :])
        .long()
        .argmax(dim=1)
    ] * base.object_scale[probe_env_ids, 0]
    local_position[:, 2] += radius_by_env + args.contact_gap
    world_position = local_position + base.scene.env_origins[probe_env_ids]
    orientation = torch.zeros(2, 4, device=base.device)
    orientation[:, 0] = 1.0
    base.object.write_root_pose_to_sim(
        torch.cat((world_position, orientation), dim=-1), env_ids=probe_env_ids
    )
    velocity = torch.zeros(2, 6, device=base.device)
    velocity[:, 2] = args.contact_velocity
    base.object.write_root_velocity_to_sim(velocity, env_ids=probe_env_ids)
    print("ADEPT_VALIDATION_STAGE=contact_probe_configured", flush=True)

    peak_force = torch.zeros(args.num_envs, 4, device=base.device)
    minimum_center_distance = torch.full((2,), float("inf"), device=base.device)
    for step in range(args.contact_steps):
        # Sample every physics step so a short collision impulse cannot fall
        # between the environment's four-substep control observations.
        base.sim.step(render=False)
        base.scene.update(cfg.sim.dt)
        base._compute_intermediate_values()
        print(f"ADEPT_VALIDATION_CONTACT_STEP={step + 1}", flush=True)
        force = torch.linalg.vector_norm(
            base.fingertip_contact_forces[:, 1:, :], dim=-1
        )
        peak_force = torch.maximum(peak_force, force)
        object_position = (
            base.object.data.root_pos_w[probe_env_ids]
            - base.scene.env_origins[probe_env_ids]
        )
        fingertip_position = torch.stack(
            (
                base.hand_pos[index_env_id, 1, :],
                base.hand_pos[thumb_env_id, -1, :],
            )
        )
        minimum_center_distance = torch.minimum(
            minimum_center_distance,
            torch.linalg.vector_norm(object_position - fingertip_position, dim=-1),
        )

    print("ADEPT_VALIDATION_STAGE=contact_probe_accumulated", flush=True)
    index_force = float(peak_force[index_env_id, 0].item())
    thumb_force = float(peak_force[thumb_env_id, -1].item())
    print(
        "ADEPT_VALIDATION_STAGE=contact_probe_measured "
        f"index_N={index_force:.6f} thumb_N={thumb_force:.6f} "
        f"index_min_center_m={minimum_center_distance[0].item():.6f} "
        f"thumb_min_center_m={minimum_center_distance[1].item():.6f} "
        f"index_radius_m={radius_by_env[0].item():.6f} "
        f"thumb_radius_m={radius_by_env[1].item():.6f}",
        flush=True,
    )
    if index_force <= cfg.contact_force_threshold:
        raise RuntimeError(
            f"index contact sensor did not cross {cfg.contact_force_threshold} N"
        )
    if thumb_force <= cfg.contact_force_threshold:
        raise RuntimeError(
            f"thumb contact sensor did not cross {cfg.contact_force_threshold} N"
        )
    print(
        "ADEPT_VALIDATION_STAGE=contact_ready "
        f"index_N={index_force:.6f} thumb_N={thumb_force:.6f}",
        flush=True,
    )

    env.reset()
    max_abs_qd = 0.0
    max_abs_qdd = 0.0
    minimum_joint_margin = float("inf")
    for _ in range(args.rollout_steps):
        actions = 0.1 * (
            2.0
            * torch.rand(args.num_envs, cfg.num_actions, device=base.device)
            - 1.0
        )
        observations, rewards, _dones, _extras = _step(env, actions)
        _assert_finite("policy observations", observations["policy"])
        _assert_finite("critic observations", observations["critic"])
        _assert_finite("rewards", rewards)
        _assert_finite("fabric position", base.fabric_q)
        _assert_finite("fabric velocity", base.fabric_qd)
        _assert_finite("fabric acceleration", base.fabric_qdd)

        lower_margin = base.fabric_q - base.robot_dof_lower_limits
        upper_margin = base.robot_dof_upper_limits - base.fabric_q
        minimum_joint_margin = min(
            minimum_joint_margin,
            float(torch.minimum(lower_margin, upper_margin).min().item()),
        )
        max_abs_qd = max(max_abs_qd, float(base.fabric_qd.abs().max().item()))
        max_abs_qdd = max(max_abs_qdd, float(base.fabric_qdd.abs().max().item()))

    if minimum_joint_margin < -1.0e-3:
        raise RuntimeError(
            f"fabric exceeded a URDF joint limit by {-minimum_joint_margin:.6f} rad"
        )
    print("ADEPT_VALIDATION_STAGE=rollout_ready", flush=True)

    report = {
        "contact": {
            "index_peak_newtons": index_force,
            "thumb_peak_newtons": thumb_force,
        },
        "rollout": {
            "steps": args.rollout_steps,
            "minimum_joint_margin_rad": minimum_joint_margin,
            "max_abs_velocity_rad_s": max_abs_qd,
            "max_abs_acceleration_rad_s2": max_abs_qdd,
        },
        "use_cuda_graph": args.use_cuda_graph,
    }
    print("ADEPT_VALIDATION=" + json.dumps(report, sort_keys=True), flush=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        # Isaac Sim 5 can block in SimulationApp.close() after an exception,
        # hiding the actual validation failure until Slurm kills the job.
        traceback.print_exc()
        os._exit(1)
    finally:
        simulation_app.close()
