import json
import os
import tempfile
import unittest
from types import SimpleNamespace

import torch

from dextrah_lab.sgs.best import SgsEvalBest
from dextrah_lab.sgs.mj_task import _Rolling


class Writer:
    def __init__(self):
        self.scalars = []

    def add_scalar(self, tag, value, step):
        self.scalars.append((tag, value, step))


class Algo:
    def __init__(self, root):
        self.nn_dir, self.experiment_dir = os.path.join(root, 'nn'), root
        os.makedirs(self.nn_dir)
        self.config, self.epoch_num, self.last_mean_rewards = dict(name='policy'), 100, -1e9
        self.writer, self.saved = Writer(), []

    def save(self, path, override_state=None):
        self.saved.append(path)
        open(path+'.pth', 'w').close()

    def update_epoch(self):
        self.epoch_num += 1
        self.last_mean_rewards = -1e9       # e.g. a delayed-best reset must not re-enable reward best
        return self.epoch_num


class SgsEvalBestTests(unittest.TestCase):
    def test_saves_on_new_eval_maximum_only_with_full_window(self):
        with tempfile.TemporaryDirectory() as root:
            stats = {k: _Rolling(4, 'cpu') for k in ('any_goal_success_rate', 'goals_reached_count', 'ever_lifted_rate')}
            env = SimpleNamespace(_sgs=SimpleNamespace(eval_stats=stats))
            algo = Algo(root)
            best = SgsEvalBest(env, 'sgs_eval_goals', delay_epochs=2)
            best.after_init(algo)
            report = best.disable_reward_best()
            self.assertEqual(report['starts_after_epoch'], 102)
            self.assertEqual(algo.update_epoch(), 101)
            self.assertEqual(algo.last_mean_rewards, float('inf'))     # reward best stays off
            stats['goals_reached_count'].add(torch.tensor([1., 1., 1., 1.]))
            best.after_print_stats(10, 101, 0.)                         # still in the delay
            self.assertEqual(algo.saved, [])
            stats['goals_reached_count'] = _Rolling(4, 'cpu')
            stats['goals_reached_count'].add(torch.tensor([1., 2.]))
            best.after_print_stats(20, 102, 0.)                         # window not full
            self.assertEqual(algo.saved, [])
            stats['goals_reached_count'].add(torch.tensor([3., 2.]))
            best.after_print_stats(30, 103, 0.)                         # mean 2.0: first best
            self.assertEqual(algo.saved, [os.path.join(root, 'nn', 'policy')])
            self.assertEqual(os.path.realpath(os.path.join(root, 'best', 'model.pth')), os.path.realpath(os.path.join(root, 'nn', 'policy.pth')))
            stats['goals_reached_count'].add(torch.tensor([0.]))        # mean 1.75: not better
            best.after_print_stats(40, 104, 0.)
            self.assertEqual(len(algo.saved), 1)
            stats['goals_reached_count'].add(torch.tensor([9.]))        # new best
            best.after_print_stats(50, 105, 0.)
            self.assertEqual(len(algo.saved), 2)
            record = json.loads(open(os.path.join(root, 'best', 'best_eval.json')).read())
            self.assertEqual((record['epoch'], record['frame']), (105, 50))
            self.assertAlmostEqual(record['value'], 3.5)
            self.assertIn(('sgs/best_eval_goals_reached_count', 3.5, 50), algo.writer.scalars)

    def test_rejects_unknown_metric(self):
        with self.assertRaises(ValueError):
            SgsEvalBest(None, 'reward')


if __name__ == '__main__':
    unittest.main()
