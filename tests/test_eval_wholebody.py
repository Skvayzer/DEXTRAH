"""scripts/eval_wholebody_checkpoint.py EvalStarts on the CPU whole-body stand-in."""
import importlib.util
from pathlib import Path
import unittest

import torch

import test_sgs_mj_task as fixed
from test_sgs_wholebody import N_ENV, make_env

spec = importlib.util.spec_from_file_location(
    'eval_wholebody_checkpoint', Path(__file__).resolve().parents[1]/'scripts/eval_wholebody_checkpoint.py')
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)


class EvalStartsTests(unittest.TestCase):
    def test_records_finished_episodes_and_writes_pool_starts(self):
        env, task, _ = fixed.make_validating(lambda call, n: torch.zeros(n, dtype=torch.bool), env=make_env())
        for k in ('hand_far', 'max_successes', 'timeout'):
            env._termination_reasons[k] = torch.zeros(N_ENV, dtype=torch.bool)
        hook = ev.EvalStarts(env, task, torch.Generator().manual_seed(0))
        ids = torch.arange(N_ENV)
        env.sim.data.qpos[:] = 5.
        hook.before_reset(ids)                      # initial reset: nothing to record
        hook.after_reset(ids)
        hook.recording = True
        k = hook.pick
        self.assertTrue(torch.equal(env._object_asset_index_per_env, task.eval_pool['asset'][k]))
        self.assertTrue(torch.equal(env.goal_viz.data.root_pos_w, task.eval_pool['goal_pose'][k, :3]))
        self.assertTrue(torch.equal(env.sim.data.qpos[:, :7], env.robot.data.default_root_state[:, :7]))
        self.assertEqual(hook.records, [])
        for _ in range(7):
            hook.after_step()
        env._successes[:3] = torch.tensor([0, 2, 5])
        env._termination_reasons['fall'][1] = True
        env._body_fallen[2] = True
        done = torch.tensor([0, 1, 2])
        assets = env._object_asset_index_per_env[done].clone()
        hook.before_reset(done)
        rows = torch.cat(hook.records)
        self.assertEqual(rows.shape, (3, len(ev.FIELDS)))
        self.assertEqual(rows[:, 0].tolist(), [0., 1., 2.])
        self.assertEqual(rows[:, 1].tolist(), assets.float().tolist())
        self.assertEqual(rows[:, 2].tolist(), [0., 2., 5.])
        self.assertEqual(rows[:, 4].tolist(), [0., 1., 0.])        # object drop
        self.assertEqual(rows[:, 8].tolist(), [0., 0., 1.])        # robot fall
        self.assertEqual(rows[:, 9].tolist(), [7., 7., 7.])        # steps
        self.assertEqual(hook.length[:3].tolist(), [0, 0, 0])


if __name__ == '__main__':
    unittest.main()


class ValidatedReachingTests(unittest.TestCase):
    def test_reaching_starts_of_own_object_with_standing_base(self):
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
        import record_wholebody_videos as rec
        env, task, _ = fixed.make_validating(lambda call, n: torch.zeros(n, dtype=torch.bool), env=make_env(), eval_pool_size=0)
        hook = rec.ValidatedReaching(env, task, torch.Generator().manual_seed(0), selected=[0, 5])
        env.sim.data.qpos[:] = 5.
        ids = torch.arange(N_ENV)
        before = env._object_asset_index_per_env.clone()
        for _ in range(3):
            hook.after_reset(ids)
            for i in (0, 5):
                c = hook.log[i][-1]['config']
                a, b = task.blocks['reaching']
                self.assertTrue(a <= c < b)
                self.assertEqual(int(task.cfg['asset'][c]), int(env._object_asset_index_per_env[i]))
                self.assertTrue(torch.allclose(env.object.data.root_pos_w[i], task.cfg['object_pose'][c, :3]))
        self.assertTrue(torch.equal(env._object_asset_index_per_env, before))       # objects never swapped
        self.assertTrue(torch.equal(env.sim.data.qpos[:, :7], env.robot.data.default_root_state[:, :7]))
        self.assertEqual(len(hook.log[0]), 3)
