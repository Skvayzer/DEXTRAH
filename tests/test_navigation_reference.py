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
        self.last_feeds = feeds
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

    def test_documented_lateral_speed_requires_explicit_opt_in(self):
        context = np.tile(np.r_[0, 0, .75, 1, np.zeros(32)], (4, 1))
        with self.assertRaises(ValueError):
            velocity_inputs([0, -.4, 0], 0, context)
        value = velocity_inputs([0, -.4, 0], 0, context, speed_limit=.4)
        self.assertAlmostEqual(float(value['target_vel'][0]), .4)
        for limit in (float('nan'), -.1, .81):
            with self.assertRaises(ValueError):
                NavigationReference('fake', session=FakePlanner(), speed_limit=limit)

    def test_short_plan_does_not_restart_crossfade_at_ten_hz(self):
        class ShortPlanner(FakePlanner):
            def run(self, _, feeds):
                output, _ = super().run(_, feeds)
                return output, np.array([24], np.int32)

        planner = NavigationReference('fake', session=ShortPlanner())
        planner.reset([0, 0, .75, 1, 0, 0, 0], np.zeros(29), 0.)
        for step in range(1, 121):
            planner.reference(step/60, [.12, 0, 0], 0.)
        self.assertEqual(planner.inferences, 3)  # init, start .1, periodic 1.1

    def test_zero_yaw_rate_holds_desired_heading_despite_measured_drift(self):
        session = FakePlanner()
        planner = NavigationReference('fake', session=session)
        planner.reset([0, 0, .75, 1, 0, 0, 0], np.zeros(29), 0.)
        planner.reference(.1, [.12, 0, 0], 0.)
        for step in range(7, 121):
            planner.reference(step/60, [.12, 0, 0], .08)
        np.testing.assert_allclose(session.last_feeds['facing_direction'], [[1, 0, 0]])
        self.assertEqual(planner.inferences, 3)

    def test_commanded_yaw_rate_is_integrated_over_elapsed_time(self):
        planner = NavigationReference('fake', session=FakePlanner())
        planner.reset([0, 0, .75, 1, 0, 0, 0], np.zeros(29), 0.)
        for step in range(1, 61):
            planner.reference(step/60, [0, 0, .2], -.4)
        self.assertAlmostEqual(planner.desired_heading, .2)
        self.assertGreater(planner.heading, .15)

    def test_invalid_command_rejected_between_planner_ticks(self):
        planner = NavigationReference('fake', session=FakePlanner())
        planner.reset([0, 0, .75, 1, 0, 0, 0], np.zeros(29), 0.)
        with self.assertRaises(ValueError):
            planner.reference(.01, [1, 0, 0], 0.)

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

    def test_requested_cruise_does_not_taper_into_slow_gait(self):
        nav = WaypointVelocity([[0., 0.]], 0., speed=.4)
        command = nav.update([-.14, 0, .75, 1, 0, 0, 0], np.zeros(6), 1/60)
        np.testing.assert_allclose(command, [.4, 0, 0])

    def test_predictive_stop_waits_for_actual_stop_and_retries_overshoot(self):
        nav = WaypointVelocity([[0., 0.]], 0., speed=.4)
        root = np.array([-.15, 0, .75, 1, 0, 0, 0.])
        np.testing.assert_allclose(nav.update(root, [.35, 0, 0, 0, 0, 0], 1/60), 0.)
        self.assertTrue(nav.stopping)
        root[0] = .2
        for _ in range(61):
            nav.update(root, np.zeros(6), 1/60)
        self.assertFalse(nav.done)
        np.testing.assert_allclose(nav.update(root, np.zeros(6), 1/60), [-.4, 0, 0])

    def test_contact_hold_cannot_complete_arrival(self):
        nav = WaypointVelocity([[0, 0]], 0., speed=.4)
        for _ in range(120):
            nav.update([0, 0, .75, 1, 0, 0, 0], np.zeros(6), 1/60, hold=True)
        self.assertFalse(nav.done)

    def test_waypoints_reached_with_lagged_velocity_feedback(self):
        # Kinematic unit fixture only, not evidence of physical SONIC walking.
        nav = WaypointVelocity([[0, .66], [-.625, .66], [-.625, .46]], 0., speed=.4)
        root = np.array([0, .42, .75, 1, 0, 0, 0.])
        velocity = np.zeros(6)
        for _ in range(2400):
            command = nav.update(root, velocity, 1/60)
            velocity[:2] += (command[:2]*.85-velocity[:2])/(60*.25)
            root[:2] += velocity[:2]/60
            if nav.done:
                break
        self.assertTrue(nav.done)
        self.assertLess(np.linalg.norm(root[:2]-[-.625, .46]), .12)

    def test_invalid_routes_and_speeds(self):
        for route, speed in (([], .4), ([[0, 0]], 0), ([[0, 0]], float('nan')), ([[0, 0]], .81)):
            with self.assertRaises(ValueError):
                WaypointVelocity(route, 0., speed=speed)

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

    def test_transfer_uses_requested_speed_and_retains_route(self):
        driver = NavigationTransferDriver([[-.02, -.01, -.01], [.02, .01, .01]], navigation_speed=.4)
        driver.reset([0, .1, 1, 1, 0, 0, 0], [-.625, 0, .65, 1, 0, 0, 0], [0, .42, .75])
        for _ in range(40):
            driver.update([0, .1, 1, 1, 0, 0, 0], np.zeros(6), driver.root_pose[:3], .8, 0, 1, 1/60)
        self.assertAlmostEqual(np.linalg.norm(driver.command[:2]), .4)
        self.assertEqual(driver.report()['navigation_speed_limit_m_s'], .4)
        self.assertEqual(len(driver.state['navigation_route_m']), 3)


if __name__ == '__main__':
    unittest.main()
