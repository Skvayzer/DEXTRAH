"""Training-only logging and complete checkpoints; no evaluation restarts."""
import json
from pathlib import Path
import time
import torch
from rl_games.common.algo_observer import AlgoObserver


class SonicTransferObserver(AlgoObserver):
    def __init__(self, env, output):
        self.env, self.output = env, Path(output)
        self.started = time.monotonic()
        self.stop_requested = None

    def after_init(self, algo):
        self.algo = algo
        network = algo.model.a2c_network
        self.initial_decoder = network.decoder[0].weight.detach().clone()
        self.initial_fingers = network.source.mu.weight[7:13].detach().clone()
        if algo.optimizer.state or algo.central_value_net.optimizer.state:
            raise RuntimeError('Architecture migration must start with fresh optimizers')
        if not all(p.requires_grad for p in network.decoder.parameters()):
            raise RuntimeError('SONIC decoder was accidentally kept frozen')

    def after_print_stats(self, frame, epoch_num, total_time):
        algo, env = self.algo, self.env
        network = algo.model.a2c_network
        free, total = torch.cuda.mem_get_info()
        elapsed = max(env.touch_model.steps*env.cfg.sim.dt, 1e-9)
        data = dict(frame=int(algo.frame), epoch=int(epoch_num), wall_seconds=time.monotonic()-self.started,
            device_used_gib=(total-free)/2**30, device_free_gib=free/2**30,
            torch_peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
            decoder_weight_change_l2=float((network.decoder[0].weight-self.initial_decoder).detach().norm()),
            finger_weight_change_l2=float((network.source.mu.weight[7:13]-self.initial_fingers).detach().norm()),
            critic_body_weight_l2=float(algo.central_value_net.model.a2c_network.actor_mlp[0].weight[:, 271:1265].detach().norm()),
            tactile_acquisition_hz=env.touch_model.acquisition_count/elapsed,
            tactile_publication_hz=env.touch_model.publication_count/elapsed,
            tactile_normal_mean_n=float(env.touch_raw[..., 0].mean()),
            tactile_shear_mean_n=float(env.touch_raw[..., 1:].norm(dim=-1).mean()),
            tactile_valid_fraction=float(env.touch_model.valid.float().mean()),
            body_falls_total=int(env._body_falls_total),
            numerical_failures_total=int(env._body_numerical_failures_total),
            numerical_failures_per_million_transitions=1e6*float(env._body_numerical_failures_total)/max(env._body_checked_transitions, 1),
            maximum_joint_speed_seen_rad_s=float(env._body_max_joint_speed),
            tolerance=env._current_success_tolerance,
            pelvis_height_mean_m=float((env.robot.data.root_pos_w[:, 2]-env.scene.env_origins[:, 2]).mean()))
        for key, value in data.items():
            algo.writer.add_scalar('sonic_transfer/'+key, value, frame)
        (self.output/'progress.json').write_text(json.dumps(data, indent=2))
        import wandb
        if wandb.run:
            wandb.run.summary.update(dict(optimizer_updates_started=True, experiment_status='training',
                                         current_frame=int(algo.frame), current_epoch=int(epoch_num)))
        # Keep source SAPG's best-return selection. This is an additional rolling
        # complete state, including BOTH optimizers; never reset the simulator.
        if epoch_num % 64 == 0 or not (self.output/'nn/latest.pth').exists() or self.stop_requested:
            algo.save(str(self.output/'nn/latest'))
        if self.stop_requested:
            algo.max_frames = int(algo.frame)
