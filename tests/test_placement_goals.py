"""Opt-in placement goals (dextrah_lab/mj/placement_goals.py), CPU with a fake env."""
import math
import tempfile
import types
import unittest

import numpy as np
import torch

from dextrah_lab.mj.placement_goals import TABLE_TOP, PlacementGoals


class FakeGoal:
    def __init__(self, n):
        self.pose = torch.zeros(n, 7)

    def write_root_pose_to_sim(self, pose, env_ids):
        self.pose[env_ids] = pose


def fake_env(n=8):
    return types.SimpleNamespace(device='cpu', num_envs=n, extras={},
                                 _object_asset_index_per_env=torch.arange(n) % 3,
                                 _table_z_per_env=torch.full((n,), .65),
                                 scene=types.SimpleNamespace(env_origins=torch.zeros(n, 3)), goal_viz=FakeGoal(n))


class PlacementGoalTests(unittest.TestCase):
    def test_replaces_goals_with_library_poses(self):
        with tempfile.NamedTemporaryFile(suffix='.npz') as f:
            q = np.array([[1., 0, 0, 0], [math.cos(.5), math.sin(.5), 0, 0]])
            np.savez(f.name, asset=np.array([0, 1]), height=np.array([.02, .05]), quat=q)   # asset 2: no poses
            env = fake_env()
            pg = PlacementGoals(env, f.name, 1.)
            pg.apply(torch.arange(8))
            a = env._object_asset_index_per_env
            self.assertTrue(bool(pg.active[a < 2].all()) and not bool(pg.active[a == 2].any()))
            z = env.goal_viz.pose[:, 2]
            self.assertTrue(torch.allclose(z[a == 0], torch.full_like(z[a == 0], .65+TABLE_TOP+.02)))
            self.assertTrue(torch.allclose(z[a == 1], torch.full_like(z[a == 1], .65+TABLE_TOP+.05)))
            xy = env.goal_viz.pose[pg.active, :2]
            self.assertTrue(bool(((xy >= pg.xy_mins) & (xy <= pg.xy_maxs)).all()))
            quat = env.goal_viz.pose[pg.active, 3:]
            self.assertTrue(torch.allclose(quat.norm(dim=-1), torch.ones(len(quat)), atol=1e-5))
            # Yaw-only change of the library orientation: the tilt (angle of body z to world z) is kept.
            def up_z(qq):
                w, x, y, z_ = qq.unbind(-1)
                return 1-2*(x*x+y*y)
            self.assertTrue(torch.allclose(up_z(quat[a[pg.active] == 1]), up_z(torch.tensor(q[1:], dtype=torch.float32)).expand(int((a[pg.active] == 1).sum())), atol=1e-5))
            pg.record(pg.active.clone())
            self.assertEqual(pg.successes, int(pg.active.sum()))
            pg.apply(torch.tensor([0]))
            self.assertTrue(bool(pg.active[0]))          # p = 1: again a placement goal

    def test_probability_zero_share(self):
        with tempfile.NamedTemporaryFile(suffix='.npz') as f:
            np.savez(f.name, asset=np.arange(3), height=np.full(3, .03), quat=np.tile([1., 0, 0, 0], (3, 1)))
            env = fake_env(4000)
            env._object_asset_index_per_env = torch.zeros(4000, dtype=torch.long)
            pg = PlacementGoals(env, f.name, .3)
            pg.apply(torch.arange(4000))
            self.assertAlmostEqual(float(pg.active.float().mean()), .3, delta=.03)


if __name__ == '__main__':
    unittest.main()
