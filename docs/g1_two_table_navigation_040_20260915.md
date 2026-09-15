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
