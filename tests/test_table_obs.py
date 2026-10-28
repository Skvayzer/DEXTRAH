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



class SgsTableHeightTests(unittest.TestCase):
    def test_lowest_keypoint(self):
        from dextrah_lab.sgs.mj_task import lowest_keypoint_z
        offsets = torch.tensor([[[0., 0., -.03], [0., 0., .03], [.05, 0., 0.]]]).repeat(4, 1, 1)
        pose = torch.tensor([[0., 0., 1., 1., 0., 0., 0.]]).repeat(4, 1)
        self.assertTrue(torch.allclose(lowest_keypoint_z(pose, offsets), torch.full((4,), .97)))
        flip = pose.clone()
        flip[:, 3:7] = torch.tensor([math.cos(math.pi/4), 0., math.sin(math.pi/4), 0.])   # 90 deg about y: x -> -z
        self.assertTrue(torch.allclose(lowest_keypoint_z(flip, offsets), torch.full((4,), .95), atol=1e-6))

    def fake_task(self, n_cfg=8):
        """Configurations 0-3 resting, 4-7 held (object lowest keypoint 0.92 m; goals 1.00 m); held 6-7 have a
        robot geom over the table down to 0.85 m (palm-up grasp)."""
        from dextrah_lab.sgs.mj_task import FAMILIES, HEIGHT_BINS, MjSgsTask
        pose = torch.tensor([[0., 0., .95, 1., 0., 0., 0.]]).repeat(n_cfg, 1)
        goal = torch.tensor([[0., 0., 1.0, 1., 0., 0., 0.]]).repeat(n_cfg, 1)
        cfg = dict(table_z=torch.full((n_cfg,), .65), object_pose=pose, goal_pose=goal, asset=torch.zeros(n_cfg, dtype=torch.long))
        fit_tops = torch.arange(21)*.01+.70
        table_fit = torch.ones(n_cfg, 21, dtype=torch.bool)
        table_fit[6:, 13:] = False                                  # robot clear of tables up to a 0.82 m top
        family = torch.tensor([0]*4+[1]*4)
        g = torch.Generator().manual_seed(0)

        class Sampler:
            available = torch.ones(n_cfg, dtype=torch.bool)

            def sample(self, k, generator=None):
                return torch.randint(0, n_cfg, (k,), generator=g)
        me = types.SimpleNamespace(cfg=cfg, family=family, tab=dict(keypoints=torch.tensor([[[0., 0., -.03], [0., 0., .03]]])),
                                   goal_band=(.10, .28), table_lo=.55, table_hi=.75, device='cpu', generator=g,
                                   fit_stats=dict(held_drawn=0., held_rejected=0., fallback=0.), sampler=Sampler(),
                                   blocks={'reaching': (0, 4), 'stable_grasp': (4, 8)}, fit_tops=fit_tops, table_fit=table_fit)
        me._fits = lambda k, z: MjSgsTask._fits(me, k, z)
        return me, MjSgsTask, FAMILIES

    def test_fits(self):
        me, T, _ = self.fake_task()
        k = torch.tensor([0, 4, 4, 4, 4, 6, 6])
        z = torch.tensor([.75, .55, .65, .735, .75, .665, .69])   # table tops 0.90, 0.70, 0.80, 0.885, 0.90, 0.815, 0.84
        # resting always fits; held: top <= min(object 0.92, robot) - 0.03 and goal (1.00) 0.10-0.28 above the top
        # (0.70 top: goal 0.30 above, out of the band; config 6: its robot clearance mask ends at a 0.82 top)
        self.assertEqual(T._fits(me, k, z).tolist(), [True, False, True, True, False, True, False])

    def test_heights_drawn_first_independent_of_start_type(self):
        me, T, _ = self.fake_task()
        chosen_all, heights_all = [], []
        for _ in range(400):
            chosen = me.sampler.sample(64)
            c, h = T._draw_heights(me, chosen.clone(), )
            self.assertTrue(bool(T._fits(me, c, h).all()))
            chosen_all.append(c)
            heights_all.append(h)
        c, h = torch.cat(chosen_all), torch.cat(heights_all)
        top = h+.15
        held = me.family[c] == 1
        # heights uniform over 0.70-0.90 for all starts (not moved); held starts only where they fit (top 0.72-0.87)
        self.assertGreater(float(top.min()), .699)
        self.assertLess(float(top.max()), .901)
        self.assertAlmostEqual(float(top.mean()), .80, delta=.005)
        self.assertTrue(bool(((top[held] >= .72-1e-6) & (top[held] <= .89+1e-6)).all()))
        palm_up = c >= 6
        self.assertTrue(bool((top[palm_up] <= .82+1e-6).all()))
        self.assertGreater(me.fit_stats['held_rejected'], 0.)

    def test_table_fit_mask(self):
        """A hand capsule above the table (lowest point 0.93 m) blocks table tops within 3 cm of it; a leg 2 cm beside
        the box face (side gap, horizontal normal) blocks nothing."""
        import mujoco
        import numpy as np
        from dextrah_lab.sgs.mj_task import table_fit_mask
        xml = """<mujoco><worldbody>
          <body name='base' mocap='true' pos='0 0 .6'><body name='table' pos='0 0 .05'><geom type='box' size='.2 .2 .15'/></body></body>
          <body name='hand' pos='0 0 .95'><freejoint/><geom type='capsule' size='.02 .05' quat='.7071 0 .7071 0'/></body>
          <body name='leg' pos='.27 0 .5'><freejoint/><geom type='box' size='.05 .05 .2'/></body>
        </worldbody></mujoco>"""
        m = mujoco.MjModel.from_xml_string(xml)
        tops = np.array([.70, .80, .89, .91, .95])
        robot = [m.body('hand').id, m.body('leg').id]
        q, stored = m.qpos0[None].copy(), np.array([.80])
        mask, on_table = table_fit_mask(m, q, tops, stored, robot, m.body('table').id, .03)
        self.assertEqual(mask[0].tolist(), [True, True, True, False, False])
        self.assertFalse(bool(on_table[0]))
        only_leg, _ = table_fit_mask(m, q, tops, stored, [m.body('leg').id], m.body('table').id, .03)
        self.assertTrue(bool(only_leg.all()))
        # the hand 1 cm above its stored table (top 0.92): lower tables fit, any raise does not
        mask, _ = table_fit_mask(m, q, np.array([.80, .92, .925]), np.array([.92]), robot, m.body('table').id, .03)
        self.assertEqual(mask[0].tolist(), [True, True, False])

    def test_table_fit_mask_stored_contact(self):
        """A hip leaning 1 mm into the table's side face when stored: the same contact at other heights is not new;
        a table lowered onto a hand under its bottom is."""
        import mujoco
        import numpy as np
        from dextrah_lab.sgs.mj_task import table_fit_mask
        xml = """<mujoco><worldbody>
          <body name='base' mocap='true' pos='0 0 .65'><body name='table'><geom type='box' size='.2 .2 .15'/></body></body>
          <body name='hip' pos='.249 0 .6'><freejoint/><geom type='box' size='.05 .05 .05'/></body>
          <body name='hand' pos='0 0 .43'><freejoint/><geom type='sphere' size='.02'/></body>
        </worldbody></mujoco>"""
        m = mujoco.MjModel.from_xml_string(xml)
        q = m.qpos0[None].copy()
        tops = np.array([.70, .74, .80, .85])
        mask, on_table = table_fit_mask(m, q, tops, np.array([.80]), [m.body('hip').id], m.body('table').id, .03)
        self.assertTrue(bool(on_table[0]))
        self.assertTrue(bool(mask.all()))
        # hand (top of the sphere at 0.45) under the box bottom (0.50 at a 0.80 top): bottom 0.40-0.44 cuts it
        mask, on_table = table_fit_mask(m, q, tops, np.array([.80]), [m.body('hand').id], m.body('table').id, .03)
        self.assertFalse(bool(on_table[0]))
        self.assertEqual(mask[0].tolist(), [False, False, True, True])

    def test_worst_bin(self):
        from dextrah_lab.sgs.mj_task import _Rolling, _WorstBin
        bins = [_Rolling(4, 'cpu') for _ in range(2)]
        w = _WorstBin(bins)
        bins[0].add(torch.tensor([1., 1., 1., 0.]))
        self.assertEqual(w.size, 0)
        bins[1].add(torch.tensor([1., 0., 0., 0.]))
        self.assertEqual(w.size, 4)
        self.assertAlmostEqual(float(w.mean()), .25+1e-3*.5, places=6)

    def test_height_bins(self):
        from dextrah_lab.sgs.mj_task import HEIGHT_BINS, MjSgsTask, _Rolling
        me = types.SimpleNamespace(height_edges=torch.linspace(.55, .75, HEIGHT_BINS+1))
        stats = [_Rolling(16, 'cpu') for _ in range(HEIGHT_BINS)]
        MjSgsTask._add_by_height(me, stats, torch.tensor([.56, .57, .74, .66]), torch.tensor([1., 0., 1., 1.]))
        self.assertEqual([r.size for r in stats], [2, 0, 1, 1])
        self.assertAlmostEqual(float(stats[0].mean()), .5)


class OwnCounterTests(unittest.TestCase):
    def test_added_channels_learn_existing_kept(self):
        from rl_games.algos_torch.running_mean_std import RunningMeanStd
        from dextrah_lab.wholebody.table_obs import install_own_counter
        rms = RunningMeanStd((6,))
        with torch.no_grad():
            rms.running_mean[:] = torch.tensor([1., 2., 3., 4., 0., 0.], dtype=torch.float64)
            rms.running_var[:] = torch.tensor([1., 1., 1., 1., 1., 1.], dtype=torch.float64)
            rms.count.fill_(1e9)
        install_own_counter(rms, 4)
        rms.train()
        g = torch.Generator().manual_seed(0)
        for _ in range(20):
            x = torch.randn(512, 6, generator=g)
            x[:, :4] += torch.tensor([1., 2., 3., 4.])                     # existing channels: as trained
            x[:, 4:] = x[:, 4:]*2+5                                          # added channels: mean 5, std 2
            out = rms(x)
        self.assertTrue(torch.allclose(rms.running_mean[:4].float(), torch.tensor([1., 2., 3., 4.]), atol=1e-3))
        self.assertTrue(torch.allclose(rms.running_mean[4:].float(), torch.tensor([5., 5.]), atol=.1))
        self.assertTrue(torch.allclose(rms.running_var[4:].float(), torch.tensor([4., 4.]), atol=.3))
        self.assertLess(float(out[:, 4:].mean().abs()), .2)                  # normalised with the own estimate
        self.assertNotIn('own_count', rms.state_dict())                      # checkpoint layout unchanged
        rms.eval()
        y = rms(x)
        self.assertTrue(torch.allclose(y, out, atol=1e-3))

if __name__ == '__main__':
    unittest.main()
