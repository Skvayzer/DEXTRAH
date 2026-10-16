"""MjSgsTask bookkeeping on a CPU stand-in for the mjlab fixed-body env.

The stand-in runs the real Play2Perfect reset functions (via dextrah_lab.mj.compat),
so the reaching capture is checked against them; physics is not simulated.
"""
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch

from dextrah_lab.mj.compat import install

install()
from dextrah_lab.mj.objects import ObjectBank  # noqa: E402
from dextrah_lab.sgs import mj_task  # noqa: E402
from dextrah_lab.sgs.mj_task import FAMILIES, MjSgsTask, SgsSettings  # noqa: E402

N_ENV, N_ASSET, J, NQ, NV = 24, 6, 5, 12, 11


class Rigid:
    def __init__(self, n, com=None, mass=None):
        self.data = SimpleNamespace(root_pos_w=torch.zeros(n, 3), root_quat_w=torch.tensor([[1., 0, 0, 0]]).repeat(n, 1),
                                    root_lin_vel_w=torch.zeros(n, 3), root_ang_vel_w=torch.zeros(n, 3), default_mass=mass)
        self.com_offset, self.default_mass = com, mass
        self.writes = []

    def write_root_pose_to_sim(self, pose, env_ids=None):
        self.writes.append(('pose', env_ids.clone(), pose.clone()))
        self.data.root_pos_w[env_ids], self.data.root_quat_w[env_ids] = pose[:, :3], pose[:, 3:]

    def write_root_velocity_to_sim(self, vel, env_ids=None):
        self.writes.append(('vel', env_ids.clone(), vel.clone()))


class Robot:
    def __init__(self, n):
        lim = torch.stack((-torch.ones(n, J), torch.ones(n, J)), -1)
        self.data = SimpleNamespace(default_joint_pos=torch.linspace(-.5, .5, J).repeat(n, 1), joint_pos_limits=lim,
                                    joint_pos=torch.zeros(n, J), joint_vel=torch.zeros(n, J))
        self.setpoint = torch.zeros(n, J)

    def write_joint_state_to_sim(self, pos, vel, joint_ids=None, env_ids=None):
        self.data.joint_pos[env_ids], self.data.joint_vel[env_ids] = pos, vel

    def reset_setpoint(self, env_ids, values):
        self.setpoint[env_ids] = values


def make_env():
    g = torch.Generator().manual_seed(0)
    bank = ObjectBank(size=np.random.rand(N_ASSET, 4, 3), pos=np.random.rand(N_ASSET, 4, 3), quat=np.random.rand(N_ASSET, 4, 4),
                      used=np.ones((N_ASSET, 4), bool), mass=np.random.rand(N_ASSET)+.1, com=np.random.rand(N_ASSET, 3)*.01,
                      inertia=np.random.rand(N_ASSET, 3))
    asset = torch.arange(N_ENV) % N_ASSET
    t = lambda x: torch.as_tensor(x, dtype=torch.float32)
    model = SimpleNamespace(geom_size=t(bank.size[asset]), geom_pos=t(bank.pos)[asset]-t(bank.com)[asset][:, None],
                            geom_quat=t(bank.quat[asset]), geom_rbound=t(bank.rbound[asset]),
                            geom_aabb=t(bank.aabb).view(N_ASSET, 4, 2, 3)[asset].clone(),
                            body_mass=torch.zeros(N_ENV, 3), body_inertia=torch.zeros(N_ENV, 3, 3))
    model.body_mass[:, 1] = t(bank.mass[asset])
    model.body_inertia[:, 1] = t(bank.inertia[asset])
    shared = SimpleNamespace(shape=(1, 3))
    sim = SimpleNamespace(model=model, wp_model=SimpleNamespace(body_subtreemass=shared, body_invweight0=shared, dof_invweight0=shared),
                          data=SimpleNamespace(qpos=torch.randn(N_ENV, NQ, generator=g), qvel=torch.randn(N_ENV, NV, generator=g)),
                          mj_model=SimpleNamespace(na=0))
    reset = SimpleNamespace(fixed_start_pose=None, fixed_goal_pose=None, fixed_trajectory_file='', table_reset_xy_range_m=(0., 0.),
                            table_reset_yaw_range_deg=0., reset_dof_pos_random_interval_arm=.1, reset_dof_pos_random_interval_fingers=.1,
                            reset_dof_vel_random_interval=.5, table_reset_z=.65, table_reset_z_range=.01,
                            object_spawn_xy_mins=(-.12, .06), object_spawn_xy_maxs=(.12, .16), table_object_z_offset=.25,
                            reset_position_noise_x=.1, reset_position_noise_y=.1, reset_position_noise_z=.02,
                            target_volume_mins=(-.16, .02, .9), target_volume_maxs=(.16, .18, 1.08), target_volume_region_scale=1.,
                            goal_sampling_type='delta', delta_goal_distance=.1, delta_rotation_degrees=90.)
    cfg = SimpleNamespace(reset=reset, reward=SimpleNamespace(object_base_size=.04, keypoint_scale=1.5))
    scale = torch.rand(N_ASSET, 3)+.5
    corners = torch.tensor(((1, 1, 1), (1, 1, -1), (-1, -1, 1), (-1, -1, -1)), dtype=torch.float32)
    bps = np.random.rand(N_ASSET, 7).astype(np.float32)
    mass = t(bank.mass[asset])[:, None].clone()
    env = SimpleNamespace(
        _fixed_body=True, cfg=cfg, device='cpu', num_envs=N_ENV, sim=sim, object_bank=bank,
        _object_urdf_paths=[f'a{i}' for i in range(N_ASSET)], _bps_bank=bps, _bps_features=torch.as_tensor(bps[asset.numpy()]),
        _object_asset_index_per_env=asset.clone(), _object_scale_per_env=scale[asset].clone(),
        _keypoint_offsets=corners[None]*(scale[asset]*.04*1.5*.5)[:, None],
        object=Rigid(N_ENV, t(bank.com[asset]).clone(), mass), table=Rigid(N_ENV), robot=Robot(N_ENV),
        goal_viz=Rigid(N_ENV), scene=SimpleNamespace(env_origins=torch.zeros(N_ENV, 3)),
        _prev_targets=torch.zeros(N_ENV, J), _cur_targets=torch.zeros(N_ENV, J), _table_z_per_env=torch.full((N_ENV,), .65),
        _object_init_z=torch.zeros(N_ENV), _arm_joint_ids=torch.tensor([0, 1, 2]), _hand_target_joint_ids=torch.tensor([3, 4]),
        _passive_joint_ids=torch.zeros(0, dtype=torch.long), _mimic_target_joint_ids=torch.zeros(0, dtype=torch.long),
        _hold_joint_ids=torch.zeros(0, dtype=torch.long), _successes=torch.zeros(N_ENV, dtype=torch.long),
        _episode_ever_lifted=torch.zeros(N_ENV, dtype=torch.bool), _lifted_object=torch.zeros(N_ENV, dtype=torch.bool),
        _termination_reasons=dict(fall=torch.zeros(N_ENV, dtype=torch.bool)), reset_buf=torch.zeros(N_ENV, dtype=torch.bool),
        touch_force_w=torch.zeros(N_ENV, 5, 3), extras={})
    env._object_mass = env.object.default_mass[:, 0:1]
    return env


def make_task(env=None, **kw):
    env = env or make_env()
    settings = SgsSettings(**dict(dict(num_configs=30, history=4, eval_every=4, harvest_per_step=3, stats_window=50), **kw))
    with mock.patch.object(mj_task, 'object_indices', lambda m: (1, np.arange(4))):
        return env, MjSgsTask(env, settings)


def make_validating(reject, env=None, state=None, **kw):
    """Task with validate_spawns; the settle check is replaced by `reject(call, n)`
    returning a bool mask of rejected candidates (physics is not simulated here)."""
    env = env or make_env()
    calls = []
    def fake_check(self, cand, assets):
        calls.append(len(assets))
        return ~reject(len(calls), len(assets))
    settings = SgsSettings(**dict(dict(num_configs=30, history=4, eval_every=4, harvest_per_step=3, stats_window=50,
                                       validate_spawns=True, eval_pool_size=8), **kw))
    with mock.patch.object(mj_task, 'object_indices', lambda m: (1, np.arange(4))), \
            mock.patch.object(MjSgsTask, '_spawn_check', fake_check):
        return env, MjSgsTask(env, settings, state=state), calls


class MjSgsTaskTests(unittest.TestCase):
    def test_validation_redraws_rejected_starts(self):
        torch.manual_seed(7)
        # First round rejects every other candidate; later rounds accept all.
        env, task, calls = make_validating(lambda call, n: (torch.arange(n) % 2 == 0) if call == 1 else torch.zeros(n, dtype=torch.bool))
        self.assertEqual(calls[:2], [10, 5])                 # 10 reaching slots, 5 redrawn
        self.assertTrue(bool(task.validated[:10].all()) and not bool(task.validated[10:].any()))
        self.assertEqual(sorted(torch.bincount(task.cfg['asset'][:10], minlength=N_ASSET).tolist()), [1, 1, 2, 2, 2, 2])
        self.assertIsNotNone(task.eval_pool)
        self.assertEqual(len(task.eval_pool['asset']), 8)
        # Evaluation envs start from the validated pool.
        ids = torch.arange(N_ENV)
        task.before_reset(ids)
        task.after_reset(ids)
        e = ids[ids % 4 == 3]
        k = task.eval_current[e]
        self.assertTrue(torch.equal(env.goal_viz.data.root_pos_w[e], task.eval_pool['goal_pose'][k, :3]))
        self.assertTrue(torch.equal(env._object_asset_index_per_env[e], task.eval_pool['asset'][k]))
        self.assertTrue(torch.equal(env.robot.data.joint_pos[e], task.eval_pool['joint_pos'][k]))

    def test_validation_gives_up_gracefully(self):
        # Slot 0 never passes; it keeps an unvalidated rho draw and is logged.
        env, task, calls = make_validating(lambda call, n: torch.arange(n) == 0, spawn_max_rounds=3)
        self.assertEqual(calls[:3], [10, 1, 1])
        self.assertFalse(bool(task.validated[0]))
        self.assertTrue(bool(task.validated[1:10].all()))
        self.assertEqual(len(task.spawn_stats['unresolved']), 2)       # reaching slot 0 + eval pool slot 0
        self.assertTrue(bool(task.filled[:10].all()))

    def test_resume_revalidates_unchecked_reaching(self):
        torch.manual_seed(8)
        env, old = make_task()                               # saved without validation
        ids = torch.arange(N_ENV)
        old.before_reset(ids); old.after_reset(ids)
        env._successes[:] = 1
        old.before_reset(ids)
        state = old.state_dict()
        before = state['cfg']['object_pose'].clone()
        # The check rejects reaching slots 0 and 1 once.
        env2, new, calls = make_validating(lambda call, n: (torch.arange(n) < 2) if call == 1 else torch.zeros(n, dtype=torch.bool), state=state)
        self.assertEqual(calls[:2], [10, 2])
        self.assertTrue(bool(new.validated[:10].all()))
        self.assertFalse(torch.equal(new.cfg['object_pose'][:2], before[:2].to(new.device)))
        self.assertTrue(torch.equal(new.cfg['object_pose'][2:10], before[2:10].to(new.device)))
        self.assertEqual(new.sampler.count[:2].tolist(), [0, 0])
        self.assertFalse(bool(new.sampler.outcomes[:2].any()))
        self.assertTrue(torch.equal(new.sampler.outcomes[2:], old.sampler.outcomes[2:]))

    def test_blocks_and_reaching_capture(self):
        torch.manual_seed(3)
        env, task = make_task()
        self.assertEqual(task.blocks, {'reaching': (0, 10), 'stable_grasp': (10, 20), 'near_goal': (20, 30)})
        c = task.cfg
        # Reaching only until held states exist.
        self.assertEqual(task.sampler.available.tolist(), [True]*10+[False]*20)
        r = slice(0, 10)
        # Captured from the real reset functions: object at table + .25 (+-.02), goal in the volume.
        self.assertTrue(torch.all((c['object_pose'][r, 2]-c['table_z'][r]-.25).abs() <= .02+1e-6))
        self.assertTrue(torch.all((c['goal_pose'][r, :3] >= torch.tensor((-.16, .02, .9))-1e-6) &
                                  (c['goal_pose'][r, :3] <= torch.tensor((.16, .18, 1.08))+1e-6)))
        self.assertTrue(torch.allclose(c['object_pose'][r, 3:].norm(dim=-1), torch.ones(10)))
        self.assertTrue(torch.equal(c['targets'][r], c['joint_pos'][r]))
        self.assertTrue(torch.allclose(c['object_init_z'][r], c['object_pose'][r, 2]))
        self.assertEqual(sorted(torch.bincount(c['asset'][r], minlength=N_ASSET).tolist()), [1, 1, 2, 2, 2, 2])
        # Capture left the env untouched.
        self.assertTrue(torch.equal(env._table_z_per_env, torch.full((N_ENV,), .65)))
        self.assertEqual(env.table.writes, [])
        for obj in (env.robot, env.table, env.object, env.goal_viz):   # capture patches removed
            self.assertFalse({'write_joint_state_to_sim', 'write_root_pose_to_sim', 'write_root_velocity_to_sim'} & set(vars(obj)))

    def test_reset_cycle_records_outcomes_and_swaps_objects(self):
        torch.manual_seed(4)
        env, task = make_task()
        ids = torch.arange(N_ENV)
        task.before_reset(ids)                           # initial reset: nothing recorded
        self.assertEqual(int(task.sampler.count.sum()), 0)
        task.after_reset(ids)
        train = ids[ids % 4 != 3]
        k = task.current[train]
        self.assertTrue(bool((k >= 0).all()) and bool((k < 10).all()))
        self.assertTrue(bool((task.current[ids % 4 == 3] == -1).all()))
        # Objects, robot, goal of training envs follow their configuration.
        self.assertTrue(torch.equal(env._object_asset_index_per_env[train], task.cfg['asset'][k]))
        task._verify_tables()
        self.assertTrue(torch.equal(env.robot.data.joint_pos[train], task.cfg['joint_pos'][k]))
        self.assertTrue(torch.equal(env.goal_viz.data.root_pos_w[train], task.cfg['goal_pose'][k, :3]))
        self.assertTrue(torch.equal(env._cur_targets[train], task.cfg['targets'][k]))
        self.assertTrue(torch.equal(env.robot.setpoint[train], task.cfg['setpoint'][k]))
        # Episode end: two successes, one failure; evaluation envs go to eval stats only.
        env._successes[train[:2]] = torch.tensor([1, 3])
        env._successes[3] = 2                            # an eval env
        done = torch.cat((train[:3], torch.tensor([3])))
        expected = task.current[train[:3]].clone()
        task.before_reset(done)
        self.assertEqual(int(task.sampler.count.sum()), 3)
        for cfg_id, ok in zip(expected.tolist(), [True, True, False]):
            row = task.sampler.outcomes[cfg_id]
            self.assertIn(ok, row.tolist())
        self.assertAlmostEqual(float(task.eval_stats['any_goal_success_rate'].mean()), 1.)
        self.assertAlmostEqual(float(task.eval_stats['goals_reached_count'].mean()), 2.)
        self.assertAlmostEqual(float(task.train_stats['reaching'].mean()), 2/3)

    def test_held_state_restore_and_harvest_cap(self):
        torch.manual_seed(5)
        env, task = make_task(harvest_asset_cap=1.)      # cap = ceil(10/6) = 2 per asset and family
        ids = torch.arange(N_ENV)
        task.before_reset(ids)
        task.after_reset(ids)
        # Every training env holds its object inside the goal volume.
        env._lifted_object[:] = True
        env.touch_force_w[:, 0, 0] = 1.
        env.touch_force_w[:, 2, 0] = 1.
        env.object.data.root_pos_w[:] = torch.tensor([0., .1, 1.])
        env._table_z_per_env[:] = .65
        for _ in range(10):
            task.after_step()
        for f in ('stable_grasp', 'near_goal'):
            self.assertEqual(task.fill_count[f], 10)
            self.assertLessEqual(int(task.asset_harvest[f].max()), 2)
        self.assertTrue(task.harvest_done)
        self.assertTrue(bool(task.sampler.available.all()))
        a, b = task.blocks['near_goal']
        goal, obj = task.cfg['goal_pose'][a:b], task.cfg['object_pose'][a:b]
        self.assertTrue(bool(((goal[:, :3]-obj[:, :3]).abs() <= .11+1e-6).all()))   # within 1.1 goal steps
        # Restore one held configuration into env 0 exactly.
        k = torch.tensor([a+3])
        task.current[0] = k
        env.sim.data.qpos[0] = 0.
        task.assign_objects(torch.tensor([0]), task.cfg['asset'][k])
        task.after_reset(torch.tensor([0]))
        self.assertTrue(torch.equal(env.sim.data.qpos[0], task.cfg['qpos'][a+3]))
        self.assertTrue(torch.equal(env.sim.data.qvel[0], task.cfg['qvel'][a+3]))
        self.assertTrue(bool(env._lifted_object[0]) and bool(env._episode_ever_lifted[0]))
        self.assertEqual(int(env._object_asset_index_per_env[0]), int(task.cfg['asset'][a+3]))
        task._verify_tables()
        # Metrics publish without errors.
        self.assertIn('sgs/near_goal/sampling_share', env.extras)
        self.assertAlmostEqual(sum(float(env.extras[f'sgs/{f}/sampling_share']) for f in FAMILIES), 1., places=5)

    def test_state_roundtrip(self):
        torch.manual_seed(6)
        env, task = make_task()
        ids = torch.arange(N_ENV)
        task.before_reset(ids)
        task.after_reset(ids)
        env._successes[:] = 1
        task.before_reset(ids)
        state = task.state_dict()
        _, other = make_task(kappa=5.)                   # kernel retuned: allowed
        other.load_state_dict(state)
        self.assertTrue(torch.equal(other.sampler.outcomes, task.sampler.outcomes))
        for key in task.cfg:
            self.assertTrue(torch.equal(other.cfg[key], task.cfg[key]), key)
        _, wrong = make_task(num_configs=33)
        with self.assertRaises(ValueError):
            wrong.load_state_dict(state)


if __name__ == '__main__':
    unittest.main()
