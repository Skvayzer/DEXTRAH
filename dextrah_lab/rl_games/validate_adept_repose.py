"""GPU integration checks for the ADEPT Stage-1 environment."""

from __future__ import annotations

import argparse
import json

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Adept-Kuka-Allegro-Repose")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--rollout_steps", type=int, default=64)
parser.add_argument("--contact_steps", type=int, default=8)
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

    # Place alternating objects directly over the index and thumb sensor
    # origins. This is a diagnostic-only penetration that must produce an
    # unambiguous contact response after PhysX resolves it.
    base._compute_intermediate_values()
    local_position = base.hand_pos[:, 1, :].clone()
    local_position[1::2] = base.hand_pos[1::2, -1, :]
    world_position = local_position + base.scene.env_origins
    orientation = torch.zeros(args.num_envs, 4, device=base.device)
    orientation[:, 0] = 1.0
    base.object.write_root_pose_to_sim(
        torch.cat((world_position, orientation), dim=-1)
    )
    base.object.write_root_velocity_to_sim(
        torch.zeros(args.num_envs, 6, device=base.device)
    )

    zero_actions = torch.zeros(
        args.num_envs, cfg.num_actions, device=base.device
    )
    peak_force = torch.zeros(args.num_envs, 4, device=base.device)
    for _ in range(args.contact_steps):
        _step(env, zero_actions)
        force = torch.linalg.vector_norm(
            base.fingertip_contact_forces[:, 1:, :], dim=-1
        )
        peak_force = torch.maximum(peak_force, force)

    index_force = float(peak_force[0::2, 0].max().item())
    thumb_force = float(peak_force[1::2, -1].max().item())
    if index_force <= cfg.contact_force_threshold:
        raise RuntimeError(
            f"index contact sensor did not cross {cfg.contact_force_threshold} N"
        )
    if thumb_force <= cfg.contact_force_threshold:
        raise RuntimeError(
            f"thumb contact sensor did not cross {cfg.contact_force_threshold} N"
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
    print("ADEPT_VALIDATION=" + json.dumps(report, sort_keys=True))
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
