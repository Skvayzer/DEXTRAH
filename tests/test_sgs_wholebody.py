"""MjSgsTask on a CPU stand-in for the frozen-SONIC whole-body env.

Extends the fixed-body stand-in (test_sgs_mj_task) with a floating base (first
7 qpos / 6 qvel), 29 body joints, SONIC's timed history and the whole-body
reset buffers. Physics is not simulated: the checks cover what a configuration
writes, and that a held state restores SONIC's history frame for frame.
"""
import unittest
from unittest import mock

import torch

import test_sgs_mj_task as fixed
from dextrah_lab.wholebody.timed_history import TimedSonicHistory

N_ENV = fixed.N_ENV
ROOT = (0., .42, .75, .7071068, 0., 0., -.7071068)


class SpawnStub:
    """Table spawn: position from the asset id, so the capture's asset alignment is checked."""

    def __init__(self):
        self.calls = []

    def place(self, quat, table_z, spawn_min, spawn_max, env_ids, assets=None):
        self.calls.append(assets.clone())
        z = table_z+.16+assets.float()*1e-3
        return torch.stack((assets.float()*.01, torch.full_like(z, .1), z), -1)


def make_env(table_spawn=False):
    with mock.patch.multiple(fixed, J=31, NQ=45, NV=44):
        env = fixed.make_env()
    n = N_ENV
    env._fixed_body, env._wholebody_ready = False, True
    env._body_ids = torch.arange(29)
    env._body_nominal = torch.linspace(-.1, .1, 29)
    env._body_scales = torch.full((29,), .5)
    env._last_body_action = torch.zeros(n, 29)
    env._body_history = TimedSonicHistory(n, 1/60, 'cpu')
    env._body_extra_step, env._sim_step_counter = None, 0
    env._body_fallen = torch.zeros(n, dtype=torch.bool)
    env._body_numerical_failure = torch.zeros(n, dtype=torch.bool)
    env.robot.data.default_root_state = torch.tensor(ROOT+(0.,)*6).repeat(n, 1)

    def write_root_state_to_sim(root, env_ids=None):
        env.sim.data.qpos[env_ids, :7] = root[:, :7]
        env.sim.data.qvel[env_ids, :6] = root[:, 7:]
    env.robot.write_root_state_to_sim = write_root_state_to_sim

    def body_terms():
        q, v = env.sim.data.qpos, env.sim.data.qvel
        return v[:, 3:6].clone(), q[:, 7:36]-env._body_nominal, v[:, 6:35].clone(), env._last_body_action.clone(), q[:, 3:6].clone()
    env._body_terms = body_terms
    if table_spawn:
        env._table_spawn = SpawnStub()
    return env


def observe(env):
    """env._wholebody_observation's history update."""
    h = env._body_history
    if env._body_extra_step != env._sim_step_counter:
        h.push(*env._body_terms())
    else:
        h.initialize_fresh(*env._body_terms())
    env._body_extra_step = env._sim_step_counter
    return h.value()


def advance(env, g):
    """One policy step of motion; the executed body action follows the body targets (apply_wholebody_action)."""
    env._sim_step_counter += 2
    env.sim.data.qpos += torch.randn(env.sim.data.qpos.shape, generator=g)*.05
    env.sim.data.qvel += torch.randn(env.sim.data.qvel.shape, generator=g)*.05
    env._cur_targets[:, :29] = torch.randn(N_ENV, 29, generator=g)*.3
    env._last_body_action[:] = (env._cur_targets[:, :29]-env._body_nominal)/env._body_scales


class WholeBodySgsTests(unittest.TestCase):
    def test_reaching_capture_with_table_spawn_and_standing_reset(self):
        torch.manual_seed(11)
        env, task = fixed.make_task(make_env(table_spawn=True))
        self.assertTrue(task.wholebody)
        c, r = task.cfg, slice(*task.blocks['reaching'])
        # Table spawn applied to each draw's own asset, init height from the placed pose.
        self.assertTrue(torch.allclose(c['object_pose'][r, 0], c['asset'][r].float()*.01))
        self.assertTrue(torch.allclose(c['object_pose'][r, 2], c['table_z'][r]+.16+c['asset'][r].float()*1e-3))
        self.assertTrue(torch.allclose(c['object_init_z'][r], c['object_pose'][r, 2]))
        self.assertEqual(tuple(c['history'].shape), (30, 12, 93))
        ids = torch.arange(N_ENV)
        env.sim.data.qpos[:] = 5.                       # a fallen robot from the previous episode
        env.sim.data.qvel[:] = 1.
        task.before_reset(ids)
        task.after_reset(ids)
        train = ids[ids % 4 != 3]
        k = task.current[train]
        self.assertTrue(torch.equal(env.sim.data.qpos[train, :7], env.robot.data.default_root_state[train, :7]))
        self.assertFalse(bool(env.sim.data.qvel[train, :6].any()))
        self.assertTrue(bool(env._body_history.fresh[train].all()))
        self.assertTrue(torch.allclose(env._last_body_action[train], (c['targets'][k][:, :29]-env._body_nominal)/env._body_scales))
        self.assertTrue(torch.equal(env.robot.data.joint_pos[train], c['joint_pos'][k]))
        # Robot falls are tracked per family and for the evaluation envs.
        env._body_fallen[train[:2]] = True
        env._body_fallen[3] = True
        task.before_reset(torch.cat((train[:3], torch.tensor([3]))))
        self.assertAlmostEqual(float(task.robot_fall_stats['reaching'].mean()), 2/3)
        self.assertAlmostEqual(float(task.eval_stats['robot_fall_rate'].mean()), 1.)
        task._publish()
        self.assertIn('sgs/reaching/train_robot_fall_rate', env.extras)
        self.assertIn('sgs/eval_robot_fall_rate', env.extras)

    def test_validated_eval_pool_starts_standing(self):
        torch.manual_seed(12)
        env, task, _ = fixed.make_validating(lambda call, n: torch.zeros(n, dtype=torch.bool), env=make_env())
        ids = torch.arange(N_ENV)
        env.sim.data.qpos[:] = 5.
        task.before_reset(ids)
        task.after_reset(ids)
        e = ids[ids % 4 == 3]
        k = task.eval_current[e]
        self.assertTrue(torch.equal(env.sim.data.qpos[e, :7], env.robot.data.default_root_state[e, :7]))
        self.assertTrue(torch.equal(env.goal_viz.data.root_pos_w[e], task.eval_pool['goal_pose'][k, :3]))
        self.assertTrue(bool(env._body_history.fresh[e].all()))

    def test_held_state_restores_sonic_history_exactly(self):
        torch.manual_seed(13)
        g = torch.Generator().manual_seed(0)
        env, task = fixed.make_task(make_env())
        ids = torch.arange(N_ENV)
        task.before_reset(ids)
        task.after_reset(ids)
        observe(env)
        for _ in range(15):
            advance(env, g)
            observe(env)
        a, _ = task.blocks['stable_grasp']
        store = lambda env_id: task._store_held('stable_grasp', torch.tensor([env_id]), env._object_asset_index_per_env[[env_id]])

        def restore(env_id, slot):
            task.current[env_id] = slot
            task.assign_objects(torch.tensor([env_id]), task.cfg['asset'][[slot]])
            task.after_reset(torch.tensor([env_id]))

        # (1) Stored before this step's observation pushed the frame.
        advance(env, g)
        store(0)
        truth1 = observe(env)[0].clone()                # what env 0 sees when it continues
        state1 = env.sim.data.qpos[0].clone()
        # (2) Stored after a terminal observation already pushed this step's frame.
        advance(env, g)
        observe(env)
        store(0)
        truth2 = observe(env)[0].clone()                # the step's observation: no second push
        # Restore (1) at a step whose observation still pushes, and at one that already pushed.
        advance(env, g)
        restore(1, a)
        self.assertTrue(torch.equal(env.sim.data.qpos[1], state1))
        self.assertTrue(torch.allclose(observe(env)[1], truth1))
        advance(env, g)
        observe(env)
        restore(2, a)
        self.assertTrue(torch.allclose(observe(env)[2], truth1))
        # Restore (2) likewise.
        advance(env, g)
        restore(5, a+1)
        self.assertTrue(torch.allclose(observe(env)[5], truth2))
        self.assertFalse(bool(env._body_history.fresh[[1, 2, 5]].any()))
        self.assertTrue(bool(env._lifted_object[[1, 2, 5]].all()))

    def test_held_mask_excludes_fallen_robots(self):
        env, task = fixed.make_task(make_env())
        ids = torch.arange(N_ENV)
        task.before_reset(ids)
        task.after_reset(ids)
        observe(env)
        env._lifted_object[:] = True
        env.touch_force_w[:, 0, 0] = 1.
        env.touch_force_w[:, 2, 0] = 1.
        env.object.data.root_pos_w[:] = torch.tensor([0., .1, 1.])
        env._body_fallen[0] = True
        env._body_numerical_failure[1] = True
        held = task.held_mask()
        self.assertFalse(bool(held[0]) or bool(held[1]))
        self.assertTrue(bool(held[2]))
        self.assertFalse(bool(held[3]))                 # evaluation env

    def test_state_roundtrip_and_scene_check(self):
        torch.manual_seed(14)
        env, task = fixed.make_task(make_env())
        task.cfg['history'].normal_()
        state = task.state_dict()
        self.assertTrue(state['wholebody'])
        _, other = fixed.make_task(make_env())
        other.load_state_dict(state)
        self.assertTrue(torch.equal(other.cfg['history'], task.cfg['history']))
        _, fixed_task = fixed.make_task()
        with self.assertRaises(ValueError):
            fixed_task.load_state_dict(state)


if __name__ == '__main__':
    unittest.main()
