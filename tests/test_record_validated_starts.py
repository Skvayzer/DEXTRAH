"""record_fixed_policy_videos --starts validated: resets use the checkpoint's
validated SGS reaching configurations of each env's own object."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import torch

from test_sgs_mj_task import N_ENV, make_validating

spec = importlib.util.spec_from_file_location(
    'record_fixed_policy_videos', Path(__file__).resolve().parents[1]/'scripts/record_fixed_policy_videos.py')
recorder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recorder)


class ValidatedStartsTests(unittest.TestCase):
    def test_resets_write_validated_configs_of_the_own_object(self):
        env, task, _ = make_validating(lambda call, n: torch.zeros(n, dtype=torch.bool))
        a, b = task.blocks['reaching']
        assets = task.cfg['asset'][a:b]
        bad = a+int(next(j for j in range(b-a) if int((assets == assets[j]).sum()) > 1))
        task.validated[bad] = False                                 # an unresolved slot must never be used
        task.sampler.outcomes[a:b, :2] = True
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'x.sgs.pt'
            torch.save(task.state_dict(), path)
            starts = recorder.ValidatedStarts(env, path, selected=[0, 5])
        self.assertEqual(starts.report['validated_reaching_configs'], b-a-1)
        ids = torch.arange(N_ENV)
        for _ in range(5):
            starts.after_reset(ids)
            for i in (0, 5):
                c = starts.log[i][-1]['config']
                self.assertTrue(a <= c < b and c != bad)
                self.assertEqual(int(task.cfg['asset'][c]), int(env._object_asset_index_per_env[i]))
                self.assertTrue(torch.allclose(env.object.data.root_pos_w[i], task.cfg['object_pose'][c, :3]))
                self.assertTrue(torch.allclose(env.goal_viz.data.root_pos_w[i], task.cfg['goal_pose'][c, :3]))
                self.assertTrue(torch.allclose(env._cur_targets[i], task.cfg['targets'][c]))
                self.assertAlmostEqual(starts.log[i][-1]['train_success_rate'], .5)
        self.assertEqual(len(starts.log[0]), 5)


if __name__ == '__main__':
    unittest.main()
