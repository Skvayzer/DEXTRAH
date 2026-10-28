"""Table observation and goals-follow-the-table (dextrah_lab/wholebody/table_obs.py), CPU only, fake env."""
import math
import types
import unittest

import torch

from dextrah_lab.wholebody.hand_surface import HAND_SURFACE_POINTS
from dextrah_lab.wholebody.table_obs import (BODY_TABLE_LINKS, CRITIC_DIM, TABLE_OBS_DIM, TableObservation,
                                             install_goals_follow_table, nearest_edge, rotate_z, soft_min, yaw_of_quat)

SIZE = (.475, .4, .3)                  # scene.TABLE_SIZE: half extents .2375 .2 .15


def quat_yaw(yaw):
    return torch.tensor([math.cos(yaw/2), 0., 0., math.sin(yaw/2)])


def fake_env(n=2, table_centre=(0., .5, .65), table_yaw=0.):
    names = ['pelvis', 'torso_link', *BODY_TABLE_LINKS[:4], *BODY_TABLE_LINKS[5:]]
    B = len(names)
    state = torch.zeros(n, B, 13)
    state[..., 3] = 1.
    torso = names.index('torso_link')
    state[:, torso, :3] = torch.tensor([0., 0., 1.0])           # torso 0.5 m in front of the table centre (-y)
    knee = names.index('right_knee_link')
    state[:, knee, :3] = torch.tensor([0., .25, .5])            # 5 cm from the table's near side (y = .3), below the top
    robot = types.SimpleNamespace(body_names=names, data=types.SimpleNamespace(
        body_state_w=state, root_pos_w=torch.tensor([[0., 0., .75]]).repeat(n, 1), root_quat_w=torch.tensor([[1., 0, 0, 0]]).repeat(n, 1)))
    table = types.SimpleNamespace(data=types.SimpleNamespace(root_pos_w=torch.tensor([table_centre]).repeat(n, 1),
                                                             root_quat_w=quat_yaw(table_yaw).repeat(n, 1)))
    env = types.SimpleNamespace(robot=robot, table=table, scene=types.SimpleNamespace(env_origins=torch.zeros(n, 3)),
                                device='cpu', num_envs=n, _keypoint_offsets=torch.tensor([[[0., 0., -.02], [0., 0., .02]]]).repeat(n, 1, 1),
                                body_table_names=['right_wrist_yaw_link', 'right_knee_link'],
                                _body_table_force_step=torch.tensor([[3., 0.], [0., 7.]]).repeat(n, 1)[:n],
                                _last_contacts={'object_table_force': torch.tensor([1., 0.]).repeat(n)[:n]})
    return env


class FakeHand:
    def __init__(self, n, palm_z=.9):
        self.pts = torch.zeros(n, HAND_SURFACE_POINTS+4, 3)
        self.pts[:, :, 1] = .45
        self.pts[:, :, 2] = palm_z
        self.pts[:, 1, 2] = .81                                  # one fingertip 1 cm above the 0.80 m top

    def points(self):
        return self.pts


class TableObsTests(unittest.TestCase):
    def test_helpers(self):
        half = torch.tensor([.2375, .2])
        v = nearest_edge(torch.tensor([[.2, 0.], [0., -.5]]), half)
        self.assertTrue(torch.allclose(v[0], torch.tensor([.0375, 0.]), atol=1e-6))     # inside: nearest side (+x)
        self.assertTrue(torch.allclose(v[1], torch.tensor([0., .3]), atol=1e-6))        # outside: clamped point
        self.assertAlmostEqual(float(soft_min(torch.tensor([[.1, .5, .3]]))), .1, places=4)
        self.assertAlmostEqual(float(yaw_of_quat(quat_yaw(.3))), .3, places=5)
        self.assertTrue(torch.allclose(rotate_z(torch.tensor([1., 0.]), torch.tensor(math.pi/2)), torch.tensor([0., 1.]), atol=1e-6))

    def test_block(self):
        n = 2
        env = fake_env(n)
        obs = TableObservation(env, FakeHand(n), noise=False, keypoints_fn=lambda p, q, o: p[:, None]+o, table_size=SIZE)
        self.assertEqual(obs.dim, TABLE_OBS_DIM)
        obj = (torch.tensor([[0., .45, .85]]).repeat(n, 1), torch.tensor([[1., 0, 0, 0]]).repeat(n, 1))
        actor, critic = obs.observation(obj, obj)
        self.assertEqual(tuple(actor.shape), (n, TABLE_OBS_DIM))
        top = .65+.15
        # plane: top relative to pelvis and palm (decimetres), normal up
        self.assertAlmostEqual(float(critic[0, 0]), (top-.75)*10, places=4)
        self.assertAlmostEqual(float(critic[0, 1]), (top-.9)*10, places=4)
        self.assertTrue(torch.allclose(critic[0, 2:5], torch.tensor([0., 0., 1.])))
        # edges: torso at y = 0, table near side at y = .3 -> 3 dm straight ahead (+y in the torso frame)
        self.assertTrue(torch.allclose(critic[0, 5:7], torch.tensor([0., 3.]), atol=1e-5))
        self.assertAlmostEqual(float(critic[0, 7]), 3., places=4)
        # hand heights: fingertip 1 cm above the top -> 0.1 dm; palm 10 cm above -> 1 dm
        hand = critic[0, 12:12+HAND_SURFACE_POINTS]
        self.assertAlmostEqual(float(hand[1]), .1, places=4)
        self.assertAlmostEqual(float(hand[0]), 1., places=4)
        # body: right knee 5 cm in front of the near side -> vector (0, .5 dm, 0)
        body = critic[0, 12+HAND_SURFACE_POINTS:12+HAND_SURFACE_POINTS+3*len(BODY_TABLE_LINKS)].reshape(-1, 3)
        knee = BODY_TABLE_LINKS.index('right_knee_link')
        self.assertTrue(torch.allclose(body[knee], torch.tensor([0., .5, 0.]), atol=1e-5))
        # object support point: soft minimum of the keypoints (0.83 / 0.87 m; 1 cm temperature) -> ~0.3 dm above the top
        self.assertAlmostEqual(float(critic[0, -CRITIC_DIM-1]), .3-.1*math.log1p(math.exp(-4.)), places=4)
        # critic forces vs zeros in the actor
        self.assertTrue(torch.allclose(critic[:, -3:], torch.log1p(torch.tensor([[1., 3., 0.], [0., 0., 7.]]))))
        self.assertTrue(torch.equal(actor[:, -3:], torch.zeros(n, 3)))
        self.assertTrue(torch.allclose(actor[:, :-3], critic[:, :-3]))            # no noise: same features

    def test_noise_and_yaw(self):
        n = 64
        env = fake_env(n, table_yaw=math.radians(30))
        obs = TableObservation(env, FakeHand(n), noise=True, step_noise=0., keypoints_fn=lambda p, q, o: p[:, None]+o, table_size=SIZE)
        obs.reset(torch.arange(n))
        obj = (torch.tensor([[0., .45, .85]]).repeat(n, 1), torch.tensor([[1., 0, 0, 0]]).repeat(n, 1))
        actor, critic = obs.observation(obj, obj)
        dh = (actor[:, 0]-critic[:, 0])/10                       # height bias in metres
        self.assertLessEqual(float(dh.abs().max()), .015+1e-6)
        self.assertGreater(float(dh.std()), .003)
        rel = torch.atan2(critic[0, 11], critic[0, 10])
        self.assertAlmostEqual(float(rel), math.radians(30), places=4)          # table yaw relative to the torso


class GoalsFollowTableTests(unittest.TestCase):
    def make(self):
        written = {}

        class Viz:
            def __init__(self):
                self.data = types.SimpleNamespace(root_pos_w=torch.zeros(3, 3))

            def write_root_pose_to_sim(self, pose, env_ids=None):
                self.data.root_pos_w[env_ids] = pose[:, :3]

        def reset_goal_pose(env, env_ids, mode):
            prev = env.goal_viz.data.root_pos_w[env_ids].clone()
            new = torch.full((len(env_ids), 7), 1.) if mode == 'absolute' else torch.cat((prev+.01, torch.ones(len(env_ids), 4)), -1)
            written['prev'] = prev
            env.goal_viz.write_root_pose_to_sim(new, env_ids=env_ids)
        ru = types.SimpleNamespace(_reset_goal_pose=reset_goal_pose)
        env = types.SimpleNamespace(goal_viz=Viz(), _table_z_per_env=torch.tensor([.65, .55, .75]))
        install_goals_follow_table(ru, .65)
        return ru, env, written

    def test_absolute_delta_and_intercepted_writer(self):
        ru, env, written = self.make()
        ids = torch.arange(3)
        ru._reset_goal_pose(env, ids, 'absolute')
        self.assertTrue(torch.allclose(env.goal_viz.data.root_pos_w[:, 2], torch.tensor([1., .9, 1.1])))
        ru._reset_goal_pose(env, ids, 'delta')
        self.assertTrue(torch.allclose(written['prev'][:, 2], torch.ones(3)))       # delta starts in nominal coordinates
        self.assertTrue(torch.allclose(env.goal_viz.data.root_pos_w[:, 2], torch.tensor([1.01, .91, 1.11])))
        # SGS-style interception: the recorded goal is shifted, the live goals untouched
        rec = {}
        live = env.goal_viz.data.root_pos_w.clone()
        env.goal_viz.write_root_pose_to_sim = lambda pose, env_ids=None: rec.update(goal=pose.clone())
        ru._reset_goal_pose(env, torch.tensor([1]), 'absolute')
        self.assertAlmostEqual(float(rec['goal'][0, 2]), .9, places=6)
        self.assertTrue(torch.equal(env.goal_viz.data.root_pos_w, live))
        install_goals_follow_table(ru, .65)                                          # idempotent
        self.assertTrue(getattr(ru._reset_goal_pose, 'follows_table', False))


if __name__ == '__main__':
    unittest.main()
