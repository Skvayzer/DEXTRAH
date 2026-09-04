"""Distill an ADEPT FMB teacher into the Appendix-H KUKA RGB student."""

import argparse
import sys

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Adept-Kuka-Allegro-FMB-Star-Vision")
parser.add_argument("--teacher_checkpoint", required=True)
parser.add_argument(
    "--teacher_stage", choices=("pretraining", "downstream"), default="downstream"
)
parser.add_argument("--stage1_student", default=None)
parser.add_argument("--output", required=True)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--iterations", type=int, default=100_000)
parser.add_argument("--save_interval", type=int, default=1_000)
parser.add_argument("--seed", type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from rl_games.algos_torch.model_builder import ModelBuilder

import isaaclab_tasks  # noqa: F401
import dextrah_lab.tasks.dextrah_kuka_allegro.gym_setup  # noqa: F401
from dextrah_lab.adept.actor_bc import checkpoint_weights
from dextrah_lab.adept.student import make_kuka_vision_student
from dextrah_lab.adept.vision_dagger import (
    VisionDAggerConfig,
    VisionDAggerTrainer,
    load_vision_backbone,
)
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg


def _build_model(agent_cfg, input_size, action_count, batch_size, device):
    params = agent_cfg["params"]
    config = params["config"]
    network = ModelBuilder().load(params)
    return network.build(
        {
            "actions_num": action_count,
            "input_shape": (input_size,),
            "batch_size": batch_size,
            "num_seqs": batch_size,
            "value_size": 1,
            "normalize_value": config["normalize_value"],
            "normalize_input": config["normalize_input"],
        }
    ).to(device)


def main():
    torch.manual_seed(args.seed)
    device = args.device or "cuda:0"
    env_cfg = parse_env_cfg(args.task, device=device, num_envs=args.num_envs)
    env_cfg.seed = args.seed
    env = gym.make(args.task, cfg=env_cfg)
    task = env.unwrapped

    teacher_task = (
        "Adept-Kuka-Allegro-Repose"
        if args.teacher_stage == "pretraining"
        else args.task
    )
    teacher_cfg = load_cfg_from_registry(teacher_task, "rl_games_cfg_entry_point")
    teacher_input_size = 391 if args.teacher_stage == "pretraining" else 392
    teacher = _build_model(
        teacher_cfg, teacher_input_size, task.num_actions, task.num_envs, device
    )
    teacher_payload = torch.load(
        args.teacher_checkpoint, map_location=device, weights_only=False
    )
    teacher.load_state_dict(checkpoint_weights(teacher_payload), strict=True)
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    student = make_kuka_vision_student().to(device)
    if args.stage1_student is not None:
        stage1_payload = torch.load(
            args.stage1_student, map_location=device, weights_only=False
        )
        load_vision_backbone(student, stage1_payload)

    trainer = VisionDAggerTrainer(
        env,
        teacher,
        student,
        cfg=VisionDAggerConfig(iterations=args.iterations),
        device=device,
        stage=(
            "vision_pretrain"
            if args.teacher_stage == "pretraining"
            else "downstream"
        ),
    )

    def report(iteration, terms):
        if iteration == 1 or iteration % 100 == 0:
            values = " ".join(f"{name}={value:.5f}" for name, value in terms.items())
            print(f"DAgger iteration {iteration}/{args.iterations}: {values}", flush=True)
        if iteration % args.save_interval == 0:
            trainer.save(args.output)

    try:
        trainer.run(callback=report)
        trainer.save(args.output)
    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
