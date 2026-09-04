"""Run ADEPT Algorithm 1 Stage-2 actor warm-start in Isaac Lab."""

import argparse
import sys

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Adept-Kuka-Allegro-FMB-Star")
parser.add_argument("--teacher_checkpoint", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--iterations", type=int, default=40_000)
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
from dextrah_lab.adept.actor_bc import ActorBCTrainer, checkpoint_weights
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg


def _build_model(agent_cfg, input_size, action_count, batch_size, device):
    params = agent_cfg["params"]
    config = params["config"]
    network = ModelBuilder().load(params)
    model = network.build(
        {
            "actions_num": action_count,
            "input_shape": (input_size,),
            "batch_size": batch_size,
            "num_seqs": batch_size,
            "value_size": 1,
            "normalize_value": config["normalize_value"],
            "normalize_input": config["normalize_input"],
        }
    )
    return model.to(device)


def main():
    torch.manual_seed(args.seed)
    device = args.device or "cuda:0"
    env_cfg = parse_env_cfg(args.task, device=device, num_envs=args.num_envs)
    env_cfg.seed = args.seed
    env = gym.make(args.task, cfg=env_cfg)
    task = env.unwrapped

    teacher_cfg = load_cfg_from_registry(
        "Adept-Kuka-Allegro-Repose", "rl_games_cfg_entry_point"
    )
    student_cfg = load_cfg_from_registry(args.task, "rl_games_cfg_entry_point")
    teacher = _build_model(
        teacher_cfg, task.num_teacher_observations - 1, task.num_actions, task.num_envs, device
    )
    # FMB sets num_teacher_observations=392 for Stage 2; the reposing teacher
    # view is explicitly 391-D.
    student = _build_model(
        student_cfg, task.num_teacher_observations, task.num_actions, task.num_envs, device
    )
    payload = torch.load(args.teacher_checkpoint, map_location=device, weights_only=False)
    teacher.load_state_dict(checkpoint_weights(payload), strict=True)
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    trainer = ActorBCTrainer(env, teacher, student, device=device)

    def report(iteration, loss):
        if iteration == 1 or iteration % 100 == 0:
            print(f"BC iteration {iteration}/{args.iterations}: loss={loss:.6f}", flush=True)
        if iteration % args.save_interval == 0:
            trainer.save(args.output)

    try:
        trainer.run(args.iterations, callback=report)
        trainer.save(args.output)
    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
