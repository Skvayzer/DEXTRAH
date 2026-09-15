# Two-table brush experiment with 0.4 m/s navigation

Requested follow-up to the empty-floor speed comparison. This is inference
with the existing frozen-SONIC + SAPG BPS-128/touch checkpoint, not new training.
Training job 560 and its source snapshot remain unchanged.

## Predeclared experiment

- One brush, first assigned brush environment before observing any outcome.
- Snapshot the current best complete checkpoint; strict load and hash audit.
- 120 seconds, all attempts, resets, falls and drops retained.
- Same trained 60 Hz policy, 120 Hz physics, 70 Hz tactile acquisition. Do not
  silently substitute the 50/200 Hz empty-floor diagnostic timing.
- Confirm grasp using sustained lift and physical robot-object contact.
- Back away, translate laterally behind the tables, approach the receiver.
  The receiving table is offset 0.625 m, with a 0.15 m edge gap. This is a
  scripted known-map navigation stand-in, not autonomous obstacle perception.
- Explicit `--navigation-speed .4` passes through the recorder, environment,
  waypoint driver and planner speed validation. Commands are body-frame;
  movement directions come from measured root position/yaw.
- Cruise at 0.4 m/s, then true IDLE with predictive braking. Proportional
  slowing near a waypoint would revisit the observed low-speed stall. Each
  waypoint needs measured speed below 0.06 m/s for one second and position
  within 0.12 m. Outside that tolerance, issue another correction segment.
  These braking parameters are provisional, not hardware-calibrated.
- Keep the measured grasp offset as a body-relative XY object goal, at fixed
  world height. SAPG fingers and latent residual remain active, original SONIC
  remains frozen. Walking references retain the trained arm-reference baseline;
  physical arm joints are not frozen. No object attachment, root-state writes,
  external forces, or second controller writing leg targets.
- Lost contact commands idle. At arrival, lower the object goal to the receiver;
  do not force release. Carried-over, supported and released are distinct metrics.

## Validation before physics

20 navigation tests and 6 contact/transfer tests pass, covering command frames,
speed opt-in, no slow-gait taper, stopping/overshoot correction, lost-contact
hold, route completion with a lagged kinematic fixture, carry-goal tracking and
contact-based release. The fixture is not evidence of physical success.

The standing-trained adapter can oppose walking or lose the object. Success
of bare SONIC at 50/200 Hz does not establish success of this combined 60/120 Hz
experiment. Actual outcome, checkpoint provenance and video are reported below
only after running it.

## Completed physical test: transfer failed

Capture: `outputs/brush_navigation040_560_20260915`, Slurm step **560.20**,
source `5a9a6d0f3003d2b4cace9287d9ab7569c15579f7`. Best-checkpoint snapshot:
epoch **17,616**, frame **2,595,256,320**, SHA-256
`b4300cd2de7a72cf7c5f5a1cea36bc574bf0ca7227d04e92ba1304cbaf6da887`.
The best checkpoint predates the current training progress; this is not a
claim that the latest rolling checkpoint was used.

For the selected brush environment alone, over 120 s / 3,600 video frames:

| Measurement | Result |
|---|---:|
| Contact-confirmed grasps | 13 |
| Navigation arrivals | 0 |
| Carried over receiving table | 0 |
| Receiving-table support / placed-and-released | 0 / 0 |
| Completed attempts / ongoing final attempt | 15 / 1 |
| Robot-fall resets / object-fall resets | 7 / 8 |
| Numerical failures | 0 |

No attempt progressed past the first backing-away waypoint. Translational
command magnitude was exactly 0.4 m/s whenever nonzero (66.1 s of captured
time); contact-loss holds occupied 1.9 s. Measured body velocity projected onto
the commanded direction averaged **-0.143 m/s** during nonzero commands, with
**57.0%** of those samples opposite the command. These summaries include gait
transients and falls; they are not steady-state speed estimates. Root
displacement reached 2.12 m in a failed attempt, away from the intended route.

The first grasp/navigation transition was at approximately 1.53 s. The first
attempt ended at 6.87 s with an object-fall reset; the next ended at 11.82 s
with a robot fall. All attempts remain in the video. Existing reposing counters
also register hits while following the scripted carry goal; their 39 hits and
73.3% episode-any-goal figure **must not be presented as transfer success**.

## Interpretation and next diagnostic

The command channel is active and uses the requested speed, but this combined
controller does not successfully track it while manipulating. This does not
undo the earlier bare-SONIC 0.4 m/s walking result and is not evidence that a
global command-sign flip is needed. It also does not yet isolate the cause:
the combined test adds the learned 64-dimensional residual, fixed trained arm
reference baseline, physical grasp/table contacts and source randomization,
and uses 60/120 Hz rather than the bare diagnostic's 50/200 Hz.

There were **359 planner inferences** during the rollout, including reset,
start/stop and direction-triggered replans. Nonzero command magnitude is fixed,
but its direction is recomputed toward the nearby waypoint using measured root
position. Body sway changes that direction and can trigger replans between the
nominal one-second periodic updates. Thus this is not the same constant-command
reference stream as the successful bare walking test. Test direction-update
hysteresis/rate limiting (while retaining immediate contact-loss/arrival stops)
before attributing the failure to learned weights.

An additional code-audit concern is that the source actor includes `palm_pos`
relative to the fixed environment origin, not the moving robot base
(`play2perfect/.../utils/obs_utils.py`, `_build_clean_observation_dict`; source
`obs_list` explicitly includes it). The carry goal moves with the base, but
that does **not** make all policy observations translation-invariant. This is
a plausible distribution-shift mechanism, not a demonstrated causal result.

Before fine-tuning or discarding the checkpoint, isolate the command/replan
cadence above and 0.4 m/s bare SONIC at
the trained 60/120 Hz clocks, then the trained arm-reference baseline, then the
learned residual. Also test a separately labelled, checkpoint-compatible
moving-frame observation transform. If those interfaces pass but grasped
walking still fails, reuse the trained weights for walking-conditioned carry
fine-tuning. None of these additional tests or training changes was performed
as part of this recording.

The capture made **zero optimizer updates**, checked all 55 frozen SONIC tensors
unchanged, and exited normally. Training advanced from epoch 18,854 /
2,777,806,848 frames to epoch 19,118 / 2,816,735,232 frames during capture; it
was never stopped. The seven recording-contract tests also pass, for 33 focused
passing tests in total (navigation, transfer checks and recording contracts).

The selected-robot JSON summary and diagnostic PNG/PDF plots can be reproduced
with `python scripts/analyze_g1_navigation_transfer.py outputs/brush_navigation040_560_20260915/capture/brush`.
They include all recorded attempts, separate transfer checks from reposing hits,
and do not smooth the measured velocity or join paths across resets.
