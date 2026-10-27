"""Carry posture prior (dextrah_lab/wholebody/carry_posture.py), CPU."""
import math
import unittest
from pathlib import Path

import numpy as np
import torch

from dextrah_lab.wholebody.carry_posture import CarryPostures, min_jerk, quat_apply

LIB = Path(__file__).resolve().parents[1]/'outputs/carry_postures/right_arm_v1.npz'


class CarryPostureTests(unittest.TestCase):
    def test_quat_apply_matches_mujoco(self):
        import mujoco
        q = np.array([math.cos(.4), .3*math.sin(.4), -.5*math.sin(.4), .81*math.sin(.4)])
        q /= np.linalg.norm(q)
        v = np.array([.1, -.2, .3])
        ref = np.zeros(3)
        mujoco.mju_rotVecQuat(ref, v, q)
        got = quat_apply(torch.tensor(q, dtype=torch.float64), torch.tensor(v)).numpy()
        self.assertTrue(np.allclose(got, ref, atol=1e-10))

    def test_min_jerk(self):
        s = torch.tensor([-.5, 0., .5, 1., 2.])
        self.assertTrue(torch.allclose(min_jerk(s), torch.tensor([0., 0., .5, 1., 1.])))

    @unittest.skipUnless(LIB.exists(), 'posture library not built')
    def test_select_reaches_target(self):
        cp = CarryPostures(LIB, 'cpu', target_jitter=0.)
        k = 4
        t_po = torch.tensor([[.04, 0., -.03]]*k)                     # object just below / ahead of the palm
        quat = cp.palm_quat[0].expand(k, -1)                          # grasp orientation = zero-pose palm
        q_grasp = torch.zeros(k, 7)
        target = cp.sample_targets(k)
        idx, q, p = cp.select(t_po, quat, q_grasp, target)
        self.assertEqual(q.shape, (k, 7))
        self.assertTrue(bool(((p-target).norm(dim=-1) < .05).all()), (p-target).norm(dim=-1))


if __name__ == '__main__':
    unittest.main()
