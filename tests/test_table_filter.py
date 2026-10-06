import json
import unittest
from pathlib import Path
from dextrah_lab.wholebody.source_scene import table_filtered_links

NAMES = ['pelvis', 'right_knee_link', 'torso_link', 'left_wrist_yaw_link', 'left_index_tip_Link',
         'right_shoulder_pitch_link', 'right_elbow_link', 'right_wrist_yaw_link', 'right_base_link',
         'right_index_distal_link', 'right_thumb_tip', 'head_link']


class TableFilterTests(unittest.TestCase):
    def test_body_filtered_right_forearm_hand_kept(self):
        filtered = table_filtered_links(NAMES)
        self.assertEqual(sorted(set(NAMES)-set(filtered)), sorted([
            'right_elbow_link', 'right_wrist_yaw_link', 'right_base_link', 'right_index_distal_link', 'right_thumb_tip']))
        for name in ('pelvis', 'right_knee_link', 'torso_link', 'left_index_tip_Link', 'right_shoulder_pitch_link'):
            self.assertIn(name, filtered)


if __name__ == '__main__':
    unittest.main()
