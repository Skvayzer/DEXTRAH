import unittest

from scripts.recording_selection import select_candidate, summarize_candidate


class RecordingSelectionTests(unittest.TestCase):
    def test_resets_break_lifts_and_latched_flag_is_not_used(self):
        steps = [dict(step=i + 1, lift_height_m=h, lifted=True, goal_hit=i == 4,
                      done=i == 1, reward=1.)
                 for i, h in enumerate([.2, .2, .2, .2, .2, .05])]
        summary = summarize_candidate(steps, .1, .15)
        self.assertAlmostEqual(summary["longest_lift_seconds"], .3)
        self.assertEqual(summary["longest_lift_interval_s"], [.2, .5])
        self.assertEqual(summary["first_goal_seconds"], .5)
        self.assertEqual(summary["resets"], 1)

    def test_goal_count_beats_shaped_reward(self):
        summaries = [dict(goal_hits=0, longest_lift_seconds=5., total_reward=999.),
                     dict(goal_hits=1, longest_lift_seconds=1., total_reward=2.)]
        self.assertEqual(select_candidate(summaries), 1)

    def test_no_success_is_not_invented(self):
        summary = summarize_candidate([dict(step=1, lift_height_m=.05,
            goal_hit=False, done=False, reward=999.)], .1, .15)
        self.assertIsNone(summary["first_goal_seconds"])
        self.assertIsNone(summary["longest_lift_interval_s"])
        self.assertEqual(summary["longest_lift_seconds"], 0.)
        self.assertEqual(select_candidate([summary, summary]), 0)
