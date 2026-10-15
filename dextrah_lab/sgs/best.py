"""Best checkpoint by the SGS evaluation envs instead of training reward (opt-in).

Under SGS the training reward describes the reset mix (easy in-hand starts
included) and falls when SGS moves to harder configurations, so rl_games'
reward-based "best" stops tracking task quality. This observer saves the best
checkpoint (same file and best/model.pth link as rl_games) when a rolling
evaluation statistic of the fresh-rho evaluation envs (MjSgsTask.eval_stats,
full window only) reaches a new maximum.
"""
import json
import os
from pathlib import Path
from types import MethodType

from rl_games.algos_torch import torch_ext

METRICS = {'sgs_eval_goals': 'goals_reached_count', 'sgs_eval_success': 'any_goal_success_rate'}


class SgsEvalBest:
    def __init__(self, env, metric, delay_epochs=0):
        if metric not in METRICS:
            raise ValueError(f'Unknown SGS best metric {metric}; choose from {sorted(METRICS)}')
        self.env, self.metric, self.key, self.delay = env, metric, METRICS[metric], int(delay_epochs)
        self.algo, self.best, self.start_epoch = None, float('-inf'), None

    def after_init(self, algo):
        self.algo = algo

    def disable_reward_best(self):
        """Keep rl_games' reward-based best from ever saving (call after resume)."""
        algo = self.algo
        algo.last_mean_rewards = float('inf')
        original = algo.update_epoch

        def update_epoch(this):
            epoch = original()
            this.last_mean_rewards = float('inf')
            return epoch

        algo.update_epoch = MethodType(update_epoch, algo)
        self.start_epoch = int(algo.epoch_num)
        return dict(metric=self.metric, statistic=f'sgs/eval_{self.key}', delay_epochs=self.delay,
                    starts_after_epoch=self.start_epoch+self.delay, reward_best_disabled=True)

    def after_print_stats(self, frame, epoch_num, total_time):
        if self.start_epoch is None or epoch_num < self.start_epoch+self.delay:
            return
        stats = self.env._sgs.eval_stats
        window = stats[self.key]
        if window.size < len(window.values):          # rolling window not yet refilled
            return
        value = float(window.mean())
        if value <= self.best:
            return
        self.best = value
        algo = self.algo
        path = os.path.join(algo.nn_dir, algo.config['name'])
        algo.save(path)
        best_dir = os.path.join(algo.experiment_dir, 'best')
        torch_ext.safe_filesystem_op(os.makedirs, best_dir, exist_ok=True)
        torch_ext.safe_symlink(os.path.relpath(path+'.pth', start=best_dir), os.path.join(best_dir, 'model.pth'))
        record = dict(metric=self.metric, value=value, epoch=int(epoch_num), frame=int(frame),
                      eval={k: float(r.mean()) for k, r in stats.items()}, episodes_in_window=window.size)
        Path(best_dir, 'best_eval.json').write_text(json.dumps(record, indent=2))
        algo.writer.add_scalar(f'sgs/best_eval_{self.key}', value, frame)
        print(f'SGS_BEST_EVAL {self.key}={value:.4f} '+' '.join(f'{k}={v:.4f}' for k, v in record['eval'].items())
              +f' epoch={epoch_num} frame={frame}', flush=True)
