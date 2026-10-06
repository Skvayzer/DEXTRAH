import math
import unittest
import torch
from dextrah_lab.wholebody.carry import (MovingFrame, WalkingClips, CarryRewardCfg, carry_reward,
    field_slice, grasp_gate, quat_from_yaw, quat_mul, transform_clean_observation, yaw_of)


def frame(root_xy=(0., 0.), root_yaw=0., home_yaw=-math.pi/2):
    n = 1
    return MovingFrame(root_pos=torch.tensor([[root_xy[0], root_xy[1], .76]]),
                       root_yaw=torch.full((n,), root_yaw),
                       home_pos=torch.tensor([[0., 0., .76]]), home_yaw=torch.full((n,), home_yaw))


class MovingFrameTests(unittest.TestCase):
    def test_identity_at_home_pose(self):
        f = frame(root_yaw=-math.pi/2)
        x = torch.tensor([[.3, -.2, 1.]])
        torch.testing.assert_close(f.position(x), x)
        q = torch.tensor([[.1, .2, .3, .9]]); q = q/q.norm()
        torch.testing.assert_close(f.quat_xyzw(q), q)

    def test_walking_keeps_body_relative_position(self):
        # Robot walked 1 m along world -X and turned to face world +X (yaw 0).
        home = frame(root_yaw=-math.pi/2)
        moved = frame(root_xy=(-1., 0.), root_yaw=0.)
        # A palm 0.3 m in front of the pelvis in each case.
        palm_home = torch.tensor([[0., -.3, 1.]])
        palm_moved = torch.tensor([[-.7, 0., 1.]])
        torch.testing.assert_close(moved.position(palm_moved), home.position(palm_home))
        torch.testing.assert_close(moved.vector(torch.tensor([[1., 0., 0.]])),
                                   torch.tensor([[0., -1., 0.]]), atol=1e-6, rtol=0)

    def test_clean_observation_transform_is_consistent(self):
        f = frame(root_xy=(.5, .5), root_yaw=.3)
        obs = dict(palm_pos=torch.randn(1, 3), palm_rot=torch.tensor([[0., 0., 0., 1.]]),
                   palm_vel=torch.randn(1, 6), object_rot=torch.tensor([[0., 0., 0., 1.]]),
                   object_vel=torch.randn(1, 6), fingertip_pos_rel_palm=torch.randn(1, 15),
                   keypoints_rel_palm=torch.randn(1, 12), keypoints_rel_goal=torch.randn(1, 12))
        ctx = dict(palm_pos=obs['palm_pos'], obj_pos=torch.randn(1, 3), goal_pos=torch.randn(1, 3),
                   obj_rot=torch.tensor([[1., 0., 0., 0.]]), goal_rot=torch.tensor([[1., 0., 0., 0.]]),
                   obj_linvel=torch.randn(1, 3), obj_angvel=torch.randn(1, 3))
        out, octx = transform_clean_observation(obs, ctx, f)
        torch.testing.assert_close(out['palm_pos'], octx['palm_pos'])
        # Norms of relative vectors are preserved.
        torch.testing.assert_close(out['keypoints_rel_palm'].reshape(4, 3).norm(dim=-1),
                                   obs['keypoints_rel_palm'].reshape(4, 3).norm(dim=-1))
        self.assertAlmostEqual(float(yaw_of(torch.cat((out['palm_rot'][:, 3:], out['palm_rot'][:, :3]), -1))),
                               float(f.angle), places=5)

    def test_field_slice(self):
        sizes = dict(a=2, b=3, c=4)
        self.assertEqual(field_slice(['a', 'b', 'c'], sizes, 'c'), slice(5, 9))


def clips():
    t = torch.arange(101)/50
    q = torch.zeros(2, 101, 29)
    q[0, :, 0] = t                       # clip 0: joint 0 ramps at 1 rad/s
    yaw = .2*t                           # clip 0 turns at 0.2 rad/s from +0.5 rad
    root = torch.stack((quat_from_yaw(yaw+.5), quat_from_yaw(torch.zeros_like(t))))
    vel = torch.zeros(2, 101, 3); vel[0, :, 0] = .4
    cmd = torch.zeros(2, 101, 3); cmd[0, 25:75, 0] = .4
    return WalkingClips(q, root, vel, cmd, [101, 51], ['walk', 'idle'])


class WalkingClipTests(unittest.TestCase):
    def test_future_samples_velocity_and_hold(self):
        c = clips()
        q, qd, root, v, cmd = c.reference(torch.tensor([0, 1]), torch.tensor([.5, 5.]), torch.tensor([0., 1.]))
        self.assertEqual(q.shape, (2, 10, 29))
        torch.testing.assert_close(q[0, :, 0], torch.clamp(.5+torch.arange(10)*.1, max=2.))
        torch.testing.assert_close(qd[0, :5, 0], torch.ones(5), atol=1e-4, rtol=0)
        # Past its end the idle clip holds the last frame with zero velocity.
        torch.testing.assert_close(qd[1], torch.zeros(10, 29))
        torch.testing.assert_close(v[0], torch.tensor([.4, 0., 0.]))
        torch.testing.assert_close(cmd[0], torch.tensor([.4, 0., 0.]))

    def test_heading_anchor(self):
        c = clips()
        _, _, root, _, _ = c.reference(torch.tensor([0]), torch.tensor([0.]), torch.tensor([-1.5]))
        # Clip start yaw is re-anchored to the measured heading.
        self.assertAlmostEqual(float(yaw_of(root[0, 0])), -1.5, places=5)
        self.assertAlmostEqual(float(yaw_of(root[0, 5])), -1.5+.2*.5, places=4)


class GateAndRewardTests(unittest.TestCase):
    def test_grasp_gate_needs_lift_contact_and_low_slip(self):
        normal = torch.tensor([[0., 1., 0., 0., 0.], [0.]*5, [1., 0., 0., 0., 0.]])
        gate = grasp_gate(torch.full((3,), .1), normal, torch.tensor([0., 0., .5]))
        self.assertEqual(gate.tolist(), [True, False, False])
        self.assertFalse(bool(grasp_gate(torch.tensor([.01]), normal[:1], torch.zeros(1))))

    def test_reward_terms(self):
        cfg = CarryRewardCfg()
        n = 2
        total, terms = carry_reward(cfg, distance=torch.tensor([.05, .5]), home_distance=torch.full((n,), .05),
            normal_n=torch.tensor([[1., 1., 0., 0., 0.], [0.]*5]), rel_linear=torch.zeros(n, 3),
            rel_angular=torch.zeros(n, 3), body_velocity=torch.tensor([[.4, 0., 0.], [0., 0., 0.]]),
            planned_velocity=torch.tensor([[.4, 0., 0.], [.4, 0., 0.]]), latent_delta=torch.zeros(n, 64),
            finger_delta=torch.zeros(n, 6), fell=torch.tensor([False, True]), dropped=torch.tensor([False, True]))
        torch.testing.assert_close(terms['hold'], torch.tensor([1., 0.]))
        self.assertAlmostEqual(float(terms['track'][0]), .75, places=5)
        self.assertLess(float(terms['track'][1]), .75)
        self.assertEqual(float(terms['fall'][1]), -10.)
        self.assertGreater(float(total[0]), float(total[1]))


if __name__ == '__main__':
    unittest.main()
