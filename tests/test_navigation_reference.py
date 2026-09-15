import unittest
from types import SimpleNamespace
import numpy as np
from dextrah_lab.wholebody.navigation_reference import (
    NavigationReference, ISAAC_FROM_MUJOCO, sample_motion, velocity_inputs, slerp)
from dextrah_lab.wholebody.navigation_transfer import WaypointVelocity, NavigationTransferDriver


class FakePlanner:
    def get_inputs(self):
        values = velocity_inputs([0, 0, 0], 0, np.tile(np.r_[0, 0, .75, 1, np.zeros(32)], (4, 1)))
        return [SimpleNamespace(name=k, shape=list(v.shape),
            type='tensor(float)' if v.dtype == np.float32 else 'tensor(int64)') for k, v in values.items()]

    def get_outputs(self):
        return [SimpleNamespace(name='mujoco_qpos'), SimpleNamespace(name='num_pred_frames')]

    def run(self, _, feeds):
        output = np.repeat(feeds['context_mujoco_qpos'][:, :1], 64, axis=1)
        output[0, :, 0] += np.arange(64)/30*.12 if feeds['mode'][0] else 0
        return output, np.array([64], np.int32)


class NavigationTests(unittest.TestCase):
    def test_stop_selects_idle_not_default_walk_speed(self):
        context = np.tile(np.r_[0, 0, .75, 1, np.zeros(32)], (4, 1))
        inputs = velocity_inputs([0, 0, 0], 0, context)
        self.assertEqual(inputs['mode'][0], 0)
        self.assertEqual(inputs['target_vel'][0], -1.)

    def test_rightward_body_command_uses_world_negative_x(self):
        context = np.tile(np.r_[0, 0, .75, 1, np.zeros(32)], (4, 1))
        inputs = velocity_inputs([0, -.12, 0], -np.pi/2, context)
        np.testing.assert_allclose(inputs['movement_direction'], [[-1, 0, 0]], atol=1e-7)
        np.testing.assert_allclose(inputs['facing_direction'], [[0, -1, 0]], atol=1e-7)
        self.assertEqual(inputs['mode'][0], 1)

    def test_joint_order_roundtrip(self):
        isaac = np.arange(29.)
        mujoco = np.zeros(29)
        mujoco[ISAAC_FROM_MUJOCO] = isaac
        np.testing.assert_array_equal(mujoco[ISAAC_FROM_MUJOCO], isaac)
        self.assertEqual(mujoco[22], 12)  # right shoulder pitch

    def test_slerp_sign_and_endpoint_hold(self):
        frames = np.tile(np.r_[0., 0., .75, 1., np.zeros(32)], (5, 1))
        frames[-1, 3:7] *= -1
        result = sample_motion(frames, np.array([-1., .01, 2.]))
        np.testing.assert_allclose(np.linalg.norm(result[:, 3:7], axis=1), 1)
        np.testing.assert_allclose(slerp([1, 0, 0, 0], [-1, 0, 0, 0], .5), [1, 0, 0, 0])

    def test_reference_shapes_and_initial_joint_identity(self):
        planner = NavigationReference('fake', session=FakePlanner())
        planner.reset([0, 0, .75, 1, 0, 0, 0], np.arange(29.), 0.)
        q, qd, root = planner.reference(0., [0, 0, 0], 0.)
        np.testing.assert_allclose(q, np.tile(np.arange(29.), (10, 1)))
        np.testing.assert_allclose(qd, 0.)
        self.assertEqual(root.shape, (10, 7))
        q, qd, root = planner.reference(.2, [.12, 0, 0], 0.)
        self.assertEqual(planner.inferences, 2)
        self.assertTrue(np.isfinite(root).all())

    def test_speed_limit_rejected(self):
        with self.assertRaises(ValueError):
            velocity_inputs([1, 0, 0], 0, np.zeros((4, 36)))

    def test_waypoint_command_does_not_move_robot_state(self):
        nav = WaypointVelocity([[-.6, .42]], -np.pi/2)
        root = np.array([0, .42, .75, np.sqrt(.5), 0, 0, -np.sqrt(.5)])
        original = root.copy()
        for _ in range(100):
            command = nav.update(root, np.zeros(6), 1/60)
        np.testing.assert_allclose(root, original)
        self.assertAlmostEqual(command[1], -.12)
        np.testing.assert_allclose(nav.update(root, np.zeros(6), 1/60, hold=True), 0.)

    def test_arrival_requires_stopped_body(self):
        nav = WaypointVelocity([[0, 0]], 0.)
        root = [0, 0, .75, 1, 0, 0, 0]
        for _ in range(90):
            nav.update(root, [.2, 0, 0, 0, 0, 0], 1/60)
        self.assertFalse(nav.done)
        for _ in range(70):
            nav.update(root, np.zeros(6), 1/60)
        self.assertTrue(nav.done)

    def test_carry_target_moves_with_base_not_remote_receiver(self):
        driver = NavigationTransferDriver([[-.02, -.01, -.01], [.02, .01, .01]])
        driver.reset([0, .1, 1, 1, 0, 0, 0], [-.625, 0, .65, 1, 0, 0, 0], [0, .42, .75])
        driver.root_pose = np.array([0, .42, .75, 1, 0, 0, 0.])
        for _ in range(40):
            driver.update([0, .1, 1, 1, 0, 0, 0], np.zeros(6), driver.root_pose[:3], .8, 0, 1, 1/60)
        first = driver.goal.copy()
        driver.root_pose[0] += .1
        driver.update([.1, .1, 1, 1, 0, 0, 0], np.zeros(6), driver.root_pose[:3], .8, 0, 1, 1/60)
        np.testing.assert_allclose(driver.goal[:3]-first[:3], [.1, 0, 0], atol=1e-9)
        self.assertFalse(driver.navigator.done)


if __name__ == '__main__':
    unittest.main()
