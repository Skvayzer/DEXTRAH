"""Carry expert composition pieces (dextrah_lab/wholebody/carry_compose.py), CPU only."""
import unittest

import torch

from dextrah_lab.wholebody.carry import WalkingClips
from dextrah_lab.wholebody.carry_compose import (LEVELS, RIGHT_ARM_JOINTS, CarrySettings, ClipSequencer, Curriculum,
                                                 StartWindowRate, arm_delta_step, body_joint_masks, carry_arm_reference,
                                                 carry_context_v2, carry_reward_v2, carry_reward_v3, clips_up_to, compose_body,
                                                 finger_command, hand_command, hand_support, region_distance,
                                                 NetRotation, rotation_angle)
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

    def test_start_window_rate_is_unbiased_by_completion_order(self):
        L = 100
        r = StartWindowRate(L, 'cpu', bucket=10)
        # 10 episodes start at step 0: 4 fail early (finish at 20), 6 succeed at 100.
        r.add(torch.zeros(4, dtype=torch.long), torch.zeros(4, dtype=torch.bool))
        r.add(torch.zeros(6, dtype=torch.long), torch.ones(6, dtype=torch.bool))
        self.assertEqual(r.rate(90)[1], 0)                   # starts in [-110, -10): none complete yet
        rate, n = r.rate(150)                                # starts in [-50, 50): all finished by step 100
        self.assertEqual(n, 10)
        self.assertAlmostEqual(rate, .6)
        # Bucket 42 (start 420) reuses the slot of bucket 0 (42 slots): the step-0 starts are dropped.
        r.add(torch.full((3,), 420, dtype=torch.long), torch.ones(3, dtype=torch.bool))
        self.assertEqual(r.rate(600), (1., 3.))
        self.assertEqual(r.rate(150)[1], 0)

    def test_exact_curriculum_promotion(self):
        s = CarrySettings(promote_success=.75)
        c = Curriculum(s, 'cpu', num_envs=4, episode_steps=100)
        start = torch.zeros(8, dtype=torch.long)
        self.assertFalse(c.record_exact(start, torch.ones(8, dtype=torch.bool), 150))   # too early (< 2L at level)
        self.assertTrue(c.record_exact(torch.full((8,), 60, dtype=torch.long), torch.ones(8, dtype=torch.bool), 210))
        self.assertEqual(c.level, 1)
        c2 = Curriculum(s, 'cpu', num_envs=4, episode_steps=100)
        self.assertFalse(c2.record_exact(start, torch.tensor([True]*5+[False]*3), 210))  # 62.5% < 75%

    def test_exact_curriculum_min_starts(self):
        s = CarrySettings(promote_success=.75)
        c = Curriculum(s, 'cpu', num_envs=64, episode_steps=100, min_starts=8)     # uniform gate: eval envs only
        start = torch.zeros(8, dtype=torch.long)
        self.assertTrue(c.record_exact(start, torch.ones(8, dtype=torch.bool), 210))
        c2 = Curriculum(s, 'cpu', num_envs=64, episode_steps=100)
        self.assertFalse(c2.record_exact(start, torch.ones(8, dtype=torch.bool), 210))

    def test_v2_asymmetric_fingers(self):
        q_ref = torch.zeros(1, 2)
        out = finger_command(q_ref, torch.tensor([[50., -50.]]), .3, close=.6, open_=.3)
        self.assertTrue(torch.allclose(out, torch.tensor([[.6, -.3]]), atol=1e-4))
        self.assertTrue(torch.equal(finger_command(q_ref, torch.zeros(1, 2), .3, close=.6, open_=.3), q_ref))

    def test_v2_variable_episodes_and_stops(self):
        s = CarrySettings(level=2, walk_s=(4., 10.), stop_probability=1., stop_s=(1., 1.), cut_probability=0.)
        seq = ClipSequencer(toy_clips(), s, 2, 'cpu', torch.Generator().manual_seed(0))
        end = torch.tensor([s.warmup_s+s.settle_s+4.+s.final_stand_s, s.max_episode_s()])
        seq.reset(torch.arange(2), torch.zeros(2), torch.zeros(2, dtype=torch.bool), end=end)
        yaw = torch.zeros(2)
        t0 = s.warmup_s+s.settle_s
        seq.update(torch.full((2,), t0), yaw)                     # p_stop = 1: both start with a stop
        self.assertTrue(bool((seq.clip == seq.idle).all()) and bool(seq.stopping.all()))
        seq.update(torch.full((2,), t0+.5), yaw)                  # still stopping
        self.assertTrue(bool(seq.stopping.all()))
        seq.update(torch.full((2,), t0+1.01), yaw)                # stop over -> a new clip (here a stop again)
        p = seq.phase(torch.full((2,), t0+5.))
        self.assertEqual(p.tolist(), [ClipSequencer.FINAL, ClipSequencer.WALK])   # env 0 walks 4 s, env 1 10 s
        self.assertEqual(seq.phase(torch.tensor([t0+5.]), ids=torch.tensor([1])).tolist(), [ClipSequencer.WALK])

    def test_v2_reward(self):
        s = CarrySettings(version=2, hold_mode='support')
        n = 3
        z3 = torch.zeros(n, 3)
        r, terms = carry_reward_v3(s, supported=torch.tensor([True, False, True]), rel_linear=z3, rel_angular=z3,
            grasp_opposition=torch.tensor([1., 0., 0.]), contacts=torch.tensor([3, 0, 1]),
            region_d=region_distance(torch.tensor([[.3, -.15, 0.], [1., 0., 0.], [.3, -.15, 0.]]), s.region_lo, s.region_hi),
            secure=torch.tensor([False, False, True]), body_velocity=z3, planned_velocity=z3, normal_n=torch.zeros(n, 5),
            action_delta=torch.zeros(n, 13), arm_delta=torch.zeros(n, 7), table_contact=torch.zeros(n, dtype=torch.bool),
            dropped=torch.tensor([False, True, False]), fell=torch.zeros(n, dtype=torch.bool), active=torch.ones(n, dtype=torch.bool))
        track = s.w_track_lin+s.w_track_yaw
        self.assertAlmostEqual(float(r[0]), s.w_support+s.w_grasp+s.w_region+track, places=5)      # held, inside the region
        self.assertAlmostEqual(float(terms['hold'][2]), 0.)                                          # secure phase: no hold
        self.assertAlmostEqual(float(terms['region'][2]), 2*s.w_region, places=5)
        self.assertAlmostEqual(float(terms['drop'][1]), -s.drop_penalty)

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

    def test_v2_shared_helpers(self):
        s = CarrySettings(version=2, arm_channel=True, postures='unused.npz')
        qg, qc = torch.zeros(2, 7), torch.ones(2, 7)
        D = torch.full((2, 7), .01)
        t = torch.tensor([s.warmup_s, s.warmup_s+s.settle_s])           # secure phase start / end
        q = carry_arm_reference(qg, qc, D, t, s.warmup_s, s.settle_s)
        self.assertTrue(torch.allclose(q[0], qg[0]+.01) and torch.allclose(q[1], qc[1]+.01))
        mid = carry_arm_reference(qg, qc, 0*D, torch.tensor([s.warmup_s+s.settle_s/2]*2), s.warmup_s, s.settle_s)
        self.assertTrue(torch.allclose(mid, torch.full_like(mid, .5)))  # minimum jerk is symmetric
        D1 = arm_delta_step(torch.zeros(1, 7), torch.ones(1, 7), s)
        self.assertTrue(torch.allclose(D1, torch.full_like(D1, s.arm_filter*s.arm_delta_rad)))
        fingers = torch.tensor([[0., .6, 0., 0., 0.], [0., 0., 0., 0., 0.], [1., 1., 0., 0., 0.]])
        palm = torch.tensor([0., 0., .0])
        flags, count, ok = hand_support(fingers, palm, torch.tensor([0., 0., .2]), torch.zeros(3), torch.tensor([.05, .05, .05]))
        self.assertEqual(count.tolist(), [1, 0, 2])
        self.assertEqual(ok.tolist(), [True, False, False])            # no contact / touching the table
        _, _, far = hand_support(fingers[:1], palm[:1], torch.zeros(1), torch.zeros(1), torch.tensor([.2]))
        self.assertFalse(bool(far[0]))
        ctx = carry_context_v2(torch.ones(2, 3), torch.ones(2, 3), q, D, torch.zeros(2, 6), torch.zeros(2, 2))
        self.assertEqual(ctx.shape[-1], 25)
        self.assertEqual(len(RIGHT_ARM_JOINTS), 7)
        self.assertTrue(all(j in BODY_JOINTS for j in RIGHT_ARM_JOINTS))

    def test_net_rotation_ignores_vibration(self):
        def about_z(angle):
            a = torch.as_tensor(angle, dtype=torch.float32)/2
            return torch.stack((a.cos(), torch.zeros_like(a), torch.zeros_like(a), a.sin()), -1)
        dt, W = 1/60, 30
        net = NetRotation(2, W, dt, 'cpu')
        rates = []
        for k in range(90):
            # env 0: steady 0.3 rad/s; env 1: +-0.05 rad vibration at 30 Hz (path ~6 rad/s, no net rotation)
            q = torch.stack((about_z(.3*k*dt), about_z(.05*(-1)**k)))
            rates.append(net.step(q))
        self.assertEqual(rates[0].tolist(), [0., 0.])
        self.assertAlmostEqual(float(rates[-1][0]), .3, places=3)
        self.assertLess(float(rates[-1][1]), .2)         # 0.1 rad / 0.5 s at most, vs a 6 rad/s path
        self.assertAlmostEqual(float(rates[5][0]), .3, places=3)   # before the window fills: time since the start
        net.reset(torch.tensor([0]))
        r = net.step(torch.stack((about_z(1.), about_z(0.))))
        self.assertEqual(float(r[0]), 0.)                 # fresh history after a reset
        self.assertAlmostEqual(float(rotation_angle(about_z(.4), about_z(-.2))), .6, places=5)
        with self.assertRaises(ValueError):
            CarrySettings(slip_angular='raw')


if __name__ == '__main__':
    unittest.main()
