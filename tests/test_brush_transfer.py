import unittest
import numpy as np
from dextrah_lab.wholebody.brush_transfer import BrushTransferDriver, TABLE_SIZE, RECEIVER_OFFSET, move_goal


class BrushTransferTests(unittest.TestCase):
    def driver(self):
        driver = BrushTransferDriver(np.array([[-.02, -.01, -.01], [.02, .01, .01]]))
        driver.reset([0., .1, 1., 1., 0., 0., 0.], [-.625, 0., .65, 1., 0., 0., 0.], [0., .42, .75])
        return driver

    def test_tables_have_gap(self):
        self.assertAlmostEqual(abs(RECEIVER_OFFSET[0])-TABLE_SIZE[0], .15)

    def test_goal_speed_lag_and_no_mutation(self):
        current = np.zeros(3)
        np.testing.assert_allclose(move_goal(current, [1, 0, 0], current, .1), [.0035, 0, 0])
        np.testing.assert_allclose(current, 0)
        np.testing.assert_allclose(move_goal(current, [1, 0, 0], [.2, 0, 0], .1), current)

    def test_floating_object_does_not_count_as_grasp(self):
        driver = self.driver()
        for _ in range(60):
            driver.update([0, .1, 1., 1, 0, 0, 0], np.zeros(6), [0, .42, .75], .8, 0, 0, 1/60)
        self.assertFalse(driver.state['grasped'])

    def test_carry_is_not_placement_or_release(self):
        driver = self.driver()
        for _ in range(40):
            driver.update([0, .1, 1., 1, 0, 0, 0], np.zeros(6), [0, .42, .75], .8, 0, 1., 1/60)
        self.assertTrue(driver.state['grasped'])
        driver.update([-.625, 0, 1., 1, 0, 0, 0], np.zeros(6), [0, .42, .75], .8, 0, 1., 1/60)
        self.assertTrue(driver.state['carried_over_receiver'])
        self.assertFalse(driver.state['placed_released'])
        for _ in range(90):
            driver.update([-.625, 0, .81, 1, 0, 0, 0], np.zeros(6), [0, .42, .75], .8, .5, 1., 1/60)
        self.assertFalse(driver.state['placed_released'])
        for _ in range(61):
            driver.update([-.625, 0, .81, 1, 0, 0, 0], np.zeros(6), [0, .42, .75], .8, .5, 0., 1/60)
        self.assertTrue(driver.state['placed_released'])

    def test_reset_retains_failed_attempt(self):
        driver = self.driver()
        driver.reset([0, .1, 1, 1, 0, 0, 0], [-.625, 0, .65, 1, 0, 0, 0], [0, .42, .75])
        self.assertEqual(len(driver.report()['attempts']), 2)
        self.assertEqual(driver.report()['counts']['placed_released'], 0)
