import unittest
import numpy as np
from dextrah_lab.wholebody.walking_diagnostic import cases, command_at, metrics


class WalkingDiagnosticTests(unittest.TestCase):
    def test_predeclared_independent_commands_and_stop(self):
        all_cases = cases()
        self.assertEqual(len({c['name'] for c in all_cases}), 9)
        case = next(c for c in all_cases if c['name'] == 'right_040_yaw90')
        np.testing.assert_array_equal(command_at(case, 1.99), [0, 0, 0])
        np.testing.assert_array_equal(command_at(case, 2.), [0, -.4, 0])
        np.testing.assert_array_equal(command_at(case, 10.), [0, 0, 0])

    def test_metrics_respect_starting_heading_and_report_falls(self):
        t = np.arange(350)/25
        case = next(c for c in cases() if c['name'] == 'right_040_yaw90')
        velocity = np.zeros((len(t), 6))
        velocity[(t >= 2) & (t < 10), 1] = -.4
        root = np.zeros((len(t), 7)); root[:, 2] = .76; root[:, 3] = 1
        root[:, 0] = -.4*np.clip(t-2, 0, 8)
        trace = dict(time_s=t, velocity_body=velocity, root=root, upright=np.ones(len(t)),
                     fell=np.zeros(len(t), bool), joint_q=np.zeros((len(t), 29)),
                     reference_q=np.zeros((len(t), 10, 29)))
        result = metrics(trace, case)
        self.assertAlmostEqual(result['direction_cosine'], 1.)
        np.testing.assert_allclose(result['command_window_displacement_start_body_m'], [0, -3.2], atol=1e-7)
        self.assertAlmostEqual(result['velocity_rmse_m_s'], 0.)
        self.assertIsNone(result['first_fall_s'])
        trace['fell'][100] = True
        self.assertEqual(metrics(trace, case)['first_fall_s'], 4.)


if __name__ == '__main__':
    unittest.main()
