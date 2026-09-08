# Revo2 tactile and arm-torque option (no training launched)

Feature branch: `feature/g1-revo2-tactile-torque`.

This is a separate G1 Play2Perfect/SAPG + BPS-128 environment. It preserves the
13 actions (7 arm, 6 hand), original rewards and terminations, and has no
geometric fabrics or PCA prior. The existing BPS training task is unchanged.

## Observation contract

Task: `G1-Revo2-SimToolReal-Repose-BPS128-Touch`.

| Configuration | Actor | Critic | Additional inputs |
|---|---:|---:|---|
| Both off | 224 | 246 | None |
| `touch.enabled=True` | 249 | 271 | 25 tactile features |
| `touch.arm_torques=True`, tactile off | 231 | 253 | 7 motor torque estimates |
| Both on | 256 | 278 | 25 tactile + 7 motor torque estimates |

The SAPG coefficient embedding is separate: 32 learned dimensions appended by
RL-Games, not counted above. Never insert new inputs after that embedding.

Each fingertip contributes `[Fn / 25, Fx / 25, Fy / 25, valid, age_seconds]`,
in thumb/index/middle/ring/pinky order. Compression is positive; x points toward
the fingertip and y clockwise when viewed from outside the pad. This anatomical
CAD convention is shared by simulation and viewer, but has not been verified
against controlled hardware shear in every direction.

The arm channels are the canonical seven right-arm joints, normalized by 40 Nm.
They are Isaac Lab's **approximate clipped PD actuator efforts** for the implicit
actuators, not exact solver drive forces, external-contact torques, or a wrist
force/torque sensor. The intended real counterpart is motor torque estimation
(`tau_est`); its sign, scale, filtering and timing must be checked before real
deployment. Tactile timing does not rate-limit these separate torque channels.

## Physical contact and material scope

Normal-contact points and friction anchors are read separately from PhysX with
the physics timestep (120 Hz); outputs are already newtons. Their independent
index buffers must not be zipped together. Both are masked to the CAD tactile
window on each real distal collider, summed across contact partners, then
projected into the sensor frame. The actor gets no contact positions, object IDs,
partner-specific forces, ground-truth slip flags or simulated pressure images.

This PhysX version requires one rigid body per environment per filter pattern.
The task enumerates non-robot rigid bodies (object/table/body-collision assets),
with extra probe filters only in the bench. Reconstructed normal forces are
checked against unfiltered body forces so missing partners fail visibly. Robot
self-collisions remain disabled, as in the source task; enabling them is rejected
until explicit robot-pair sensing is implemented. Ground contacts outside the
registered partners are not silently treated as zero sensing.

Material assignment explicitly targets `*_distal_link`, not the small `*_tip`
markers. Default trial parameters are static friction 1.0, dynamic friction 0.8,
compliant stiffness 10000 N/m, damping 10 Ns/m, and multiply friction combination.
**These are provisional engineering parameters, not identified BrainCo values.**
The material covers the whole existing distal collider; the sensor mask covers
only the tactile footprint. It does not yet model a separate soft pad over a
rigid internal core. `touch.compliant=False` permits a rigid-contact comparison.

## Sensor response

The sequence is physical contact -> 70 Hz acquisition -> configurable latency
-> 10 Hz publication/sample-and-hold, matching the recorded ROS delivery rate.
Sampling is quantized to physics ticks; it does not pretend to generate more
physical updates than the simulator produces. Force magnitude is quantized to
0.01 N and direction to 1 degree. Cartesian shear avoids the 359/0 discontinuity.
Above-25 N values remain observable; 25 N is a scale, not a hard clip.

Optional filter time constant, noise, per-episode gain/bias, latency and dropped
samples are exposed. Unmeasured effects default to identity/zero, not fabricated
calibration. Configure actual deployment rates before transfer. Partial resets
clear all sensor history, validity and age for only the reset environments.
Terminal critic observations are captured before the reset clears that history.

## Validation and live viewer

`scripts/validate_revo2_touch_sim.py --headless --validate-only` runs observation,
terminal/reset, and contact checks. It does not construct an optimizer or load a
policy. Outputs go to `outputs/revo2_touch_live/`.

Without `--validate-only`, the same script serves Viser on 127.0.0.1:8091.
Use `--no-tactile` / `--no-arm-torques` with validation mode to check other input
layouts. The viewer requires tactile enabled.

The hand is the actual simulated articulated G1 right-hand asset. A dynamic
spherical indenter with rotation constrained by a test-fixture joint is moved
by forces, not by replayed tactile samples. Its rotation is constrained to
separate sliding from rolling. Tests: press/release on each finger; lateral
force under stationary contact; forward/reverse sliding; backside rejection.
The full G1, task object and table are hidden/parked for clarity. This is a
contact bench, not an autonomous grasp demonstration or training rollout.

Blue vectors show compression; orange vectors show shear. The table compares
instantaneous force with rate-limited sensor output and sample age. Probe drive
forces are only fixture inputs: displayed tactile forces come from PhysX.

## Future training option — not executed

The normal training entry point recognizes the new task. Example **for a future
explicitly authorized training run**, inside a Slurm allocation:

```sh
python dextrah_lab/rl_games/train.py --headless \
  --task G1-Revo2-SimToolReal-Repose-BPS128-Touch --num_envs 24576 \
  env.touch.enabled=True env.touch.arm_torques=True
```

Set `PLAY2PERFECT_ROOT`/`PYTHONPATH` as in the existing BPS launch, and set
`env.touch.touch_description` to the Revo2 Touch CAD package. This example starts
from scratch, not from the old checkpoint. Old checkpoint dimensions do not
match. Direct resume and the BPS-only warm-start switch must not be misused;
weights-only expansion requires a separate checked conversion. The generic
expansion helper is unit-tested for the new dimensions and SAPG embedding, but
a full trained-checkpoint tactile migration is not claimed here.

There has been no tactile training, optimizer update, or full-scale 24k tactile
throughput/VRAM benchmark. Small-scene validation does not certify 24k performance
or calibrated sim-to-real force accuracy.
