"""Grasp score after InterMimicGen App. C.2 (dextrah_lab/wholebody/grasp_reward.py)."""
import unittest

import torch

from dextrah_lab.wholebody.grasp_reward import THUMB, WEIGHTS, grasp_score

R = torch.full((5,), .007)                                   # pad effective radius 7 mm


def offsets(thumb_dir, finger_dir, gap=0.):
    """b_f for a thumb and the four fingers along unit directions, pad gap `gap`."""
    b = torch.zeros(1, 5, 3)
    for i in range(5):
        d = torch.tensor(thumb_dir if i == THUMB else finger_dir, dtype=torch.float32)
        b[0, i] = d/d.norm()*(R[i]+gap)
    return b


class GraspScoreTests(unittest.TestCase):
    def test_opposed_pinch_scores_full(self):
        G, p = grasp_score(offsets([0, 0, 1], [0, 0, -1]), R, torch.ones(1, 5), torch.ones(1))
        self.assertAlmostEqual(float(p['C_opp']), 1., places=5)
        self.assertAlmostEqual(float(G), sum(WEIGHTS.values()), places=5)

    def test_same_side_contact_gets_no_opposition_credit(self):
        G, p = grasp_score(offsets([0, 0, 1], [0, 0, 1]), R, torch.ones(1, 5), torch.zeros(1))
        self.assertAlmostEqual(float(p['C_opp']), 0., places=5)
        self.assertAlmostEqual(float(p['E']), 0., places=5)
        self.assertAlmostEqual(float(G), WEIGHTS['A']+WEIGHTS['C_any'], places=5)

    def test_perpendicular_is_half_opposed(self):
        _, p = grasp_score(offsets([0, 0, 1], [1, 0, 0]), R, torch.ones(1, 5), torch.zeros(1))
        self.assertAlmostEqual(float(p['C_opp']), .5, places=5)

    def test_no_force_no_contact_terms(self):
        G, p = grasp_score(offsets([0, 0, 1], [0, 0, -1]), R, torch.zeros(1, 5), torch.zeros(1))
        self.assertEqual(float(p['C_opp']), 0.)
        self.assertEqual(float(p['C_any']), 0.)
        self.assertAlmostEqual(float(G), WEIGHTS['A']+WEIGHTS['E'], places=5)   # proximity shaping only

    def test_gap_reduces_contact_and_proximity(self):
        _, near = grasp_score(offsets([0, 0, 1], [0, 0, -1], gap=.01), R, torch.ones(1, 5), torch.zeros(1))
        _, far = grasp_score(offsets([0, 0, 1], [0, 0, -1], gap=.03), R, torch.ones(1, 5), torch.zeros(1))
        self.assertAlmostEqual(float(near['C_opp']), (1-.01/.025)**2, places=4)  # C_h * C_f
        self.assertEqual(float(far['C_opp']), 0.)                                  # beyond 2.5 cm
        self.assertLess(float(far['A']), float(near['A']))

    def test_force_threshold(self):
        _, p = grasp_score(offsets([0, 0, 1], [0, 0, -1]), R, torch.full((1, 5), .05), torch.full((1,), .2))
        self.assertEqual(float(p['C_any']), 0.)
        self.assertEqual(float(p['C_palm']), 1.)


if __name__ == '__main__':
    unittest.main()
