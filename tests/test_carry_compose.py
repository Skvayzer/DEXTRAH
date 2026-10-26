"""Carry expert composition pieces (dextrah_lab/wholebody/carry_compose.py), CPU only."""
import unittest

import torch

from dextrah_lab.wholebody.carry import WalkingClips
from dextrah_lab.wholebody.carry_compose import (LEVELS, CarrySettings, ClipSequencer, Curriculum, body_joint_masks,
                                                 carry_reward_v2, clips_up_to, compose_body, finger_command, hand_command)
from dextrah_lab.wholebody.contract import BODY_JOINTS


def toy_clips():
    names = ['idle', 'forward@0.1', 'backward@0.1', 'left@0.2', 'backward@0.2', 'forward@0.4']
    C, T = len(names), 501
    q = torch.zeros(C, T, 29)
    quat = torch.zeros(C, T, 4); quat[..., 0] = 1.
    vel = torch.zeros(C, T, 3)
    for i, n in enumerate(names):
        if '@' in n:
            vel[i, 50:350, 0] = float(n.split('@')[1])
    return WalkingClips(q, quat, vel, vel.clone(), torch.full((C,), T), names)


class CarryComposeTests(unittest.TestCase):
    def test_hand_command_inverts_source_mapping(self):
        lower, upper = torch.tensor([0., -1., .2]), torch.tensor([1., 1., 1.6])
        a = torch.tensor([[-.5, .3, .9]])
        raw = lower+.5*(a+1)*(upper-lower)             # play2perfect action_utils hand mapping
        self.assertTrue(torch.allclose(hand_command(raw, lower, upper), a, atol=1e-6))

    def test_finger_stabilizer_starts_at_the_hold_and_is_bounded(self):
        q_ref = torch.tensor([[.2, -.4, 0., .9, -1., .5]])
        self.assertTrue(torch.equal(finger_command(q_ref, torch.zeros(1, 6), .3), q_ref))
        big = finger_command(q_ref, torch.full((1, 6), 50.), .3)
        self.assertTrue(((big-q_ref) <= .3+1e-6).all() and (big <= 1).all())

    def test_ownership_masks(self):
        mask = body_joint_masks(BODY_JOINTS)
        for name, owned in zip(BODY_JOINTS, mask.tolist()):
            self.assertEqual(owned, any(p in name for p in ('shoulder', 'elbow', 'wrist')), name)
        self.assertFalse(any(m for n, m in zip(BODY_JOINTS, mask.tolist()) if 'waist' in n))
        self.assertTrue(any(m for n, m in zip(BODY_JOINTS, body_joint_masks(BODY_JOINTS, 'arms_waist').tolist()) if 'waist' in n))
        bare, res = torch.zeros(2, 29), torch.ones(2, 29)
        out = compose_body(bare, res, mask)
        self.assertTrue(torch.equal(out[0], mask.float()))

    def test_levels(self):
        clips = toy_clips()
        self.assertEqual([clips.names[i] for i in clips_up_to(clips, 0)], ['idle'])
        self.assertEqual([clips.names[i] for i in clips_up_to(clips, 1)], ['forward@0.1', 'backward@0.1'])
        self.assertEqual(len(clips_up_to(clips, len(LEVELS)-1)), 5)

    def test_sequencer_phases_and_backoff(self):
        s = CarrySettings(level=2, cut_probability=0.)
        seq = ClipSequencer(toy_clips(), s, 3, 'cpu', torch.Generator().manual_seed(0))
        seq.reset(torch.arange(3), torch.zeros(3), torch.tensor([True, False, False]))
        yaw = torch.zeros(3)
        self.assertEqual(seq.update(torch.full((3,), 1.), yaw).numel(), 0)       # settle: idle
        started = seq.update(torch.full((3,), s.warmup_s+s.settle_s), yaw)
        self.assertEqual(sorted(started.tolist()), [0, 1, 2])
        self.assertTrue(seq.clips.names[int(seq.clip[0])].startswith('backward@'))
        self.assertTrue(all('@' in seq.clips.names[int(c)] for c in seq.clip))
        stopped = seq.update(torch.full((3,), s.episode_s-s.final_stand_s+.01), yaw)
        self.assertEqual(sorted(stopped.tolist()), [0, 1, 2])
        self.assertTrue(bool((seq.clip == seq.idle).all()))
        p = seq.phase(torch.tensor([0., 1., 5., 19.]))
        self.assertEqual(p.tolist(), [ClipSequencer.WARMUP, ClipSequencer.SETTLE, ClipSequencer.WALK, ClipSequencer.FINAL])

    def test_level_zero_stands(self):
        s = CarrySettings(level=0)
        seq = ClipSequencer(toy_clips(), s, 2, 'cpu')
        seq.reset(torch.arange(2), torch.zeros(2), torch.tensor([True, False]))
        seq.update(torch.full((2,), 5.), torch.zeros(2))
        self.assertTrue(bool((seq.clip == seq.idle).all()))

    def test_curriculum_needs_enough_episodes(self):
        s = CarrySettings(promote_window=8, promote_success=.75)
        c = Curriculum(s, 'cpu', num_envs=10)          # 2 x 10 episodes before any promotion
        self.assertFalse(c.record(torch.tensor([True]*19)))
        self.assertTrue(c.record(torch.tensor([True])))

    def test_curriculum_promotion(self):
        s = CarrySettings(promote_window=8, promote_success=.75)
        c = Curriculum(s, 'cpu')
        self.assertFalse(c.record(torch.tensor([True]*5+[False]*2)))
        self.assertTrue(c.record(torch.tensor([True])))
        self.assertEqual(c.level, 1)
        state = c.state_dict()
        c2 = Curriculum(s, 'cpu'); c2.load_state_dict(state)
        self.assertEqual(c2.level, 1)
        self.assertEqual(c2.count, 0)           # a resume restarts the promotion window

    def test_reward_warmup_and_hold(self):
        s = CarrySettings()
        n = 2
        z3 = torch.zeros(n, 3)
        r, terms = carry_reward_v2(s, offset_error=torch.zeros(n), rel_linear=z3, rel_angular=z3, body_velocity=z3,
                                   planned_velocity=z3, finger_residual=torch.zeros(n, 6), latent_residual=torch.zeros(n, 64),
                                   normal_n=torch.zeros(n, 5), dropped=torch.tensor([False, True]),
                                   fell=torch.zeros(n, dtype=torch.bool), table_contact=torch.zeros(n, dtype=torch.bool),
                                   active=torch.tensor([True, False]))
        self.assertAlmostEqual(float(r[0]), s.w_hold+s.w_track_lin+s.w_track_yaw, places=5)
        self.assertEqual(float(r[1]), 0.)          # warm-up: nothing counts, not even the drop


if __name__ == '__main__':
    unittest.main()
