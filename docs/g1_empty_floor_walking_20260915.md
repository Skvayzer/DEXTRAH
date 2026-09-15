# Empty-floor SONIC walking comparison — 2026-09-15

## Result

The low-speed walking failure reproduces on both the original NVIDIA G1 asset
and the G1/Revo2 source asset. Changing the lateral command from 0.12 to
0.4 m/s produces real, correctly directed locomotion with the **same frozen
controller weights**. A control-sign reversal is not supported by these tests.

This establishes empty-hand locomotion in the controlled harness. It does not
establish grasp retention, table-to-table placement, precise navigation, or
correct operation of the standing-trained SAPG adapter while walking.

Mean measured body-frame velocity during seconds 4–10, excluding onset:

| Command | Original G1 | G1 + Revo2 |
|---|---:|---:|
| Forward +0.20 m/s | +0.056 | +0.008 |
| Forward +0.40 m/s | +0.369 | +0.310 |
| Backward −0.20 m/s | −0.036 | −0.066 |
| Backward −0.40 m/s | −0.259 | −0.251 |
| Left +0.40 m/s | +0.411 | +0.435 |
| Right −0.40 m/s, yaw 0° | −0.396 | −0.376 |
| Right −0.12 m/s, yaw −90° | +0.0005 | −0.0037 |
| Right −0.40 m/s, yaw −90° | −0.403 | −0.388 |
| Turn left +0.20 rad/s | +0.124 rad/s | +0.117 rad/s |
| Turn right −0.20 rad/s | −0.133 rad/s | −0.116 rad/s |

Units are m/s except the two turning rows. **All 22 trials completed with no
falls and no within-trial resets**, including the two idle trials. These are
single-seed diagnostics, not a statistically estimated success rate.

At yaw −90°, the Revo2 robot travelled approximately 2.77 m rightward during
the 8-second 0.4 m/s command, with 0.49 m backward drift. Its mean horizontal
speed during seconds 12–14 after the stop command was 0.006 m/s. Thus this
is useful locomotion, but not accurate open-loop path following. Navigation
must use measured root position/heading feedback. In the 0.12 m/s comparison,
the same robot travelled only 0.061 m rightward and 0.204 m backward.

## Exact test boundary

- Original hash-pinned GRAIL `sonic_manipulation_base`, SHA-256
  `62f3e336cc11cbfb7517fee1f053ed92c81d44accd3757faa0febf421d0a2cd4`.
- Original `G1_CYLINDER_MODEL_12_DEX_CFG` versus the actual source Revo2
  URDF importer/baker and body/hand actuator configuration. The source has
  51 joints; the original G1 configuration has 29 actuated body joints.
- No SAPG network/residual, task observation, reward, object, table, contact
  sensor, domain randomization, external push, root servo, or optimizer.
- Common nominal body reset pose and floor material; source table-clearance
  arm initialization is deliberately not used. Source touch-pad material
  overrides are not applied in this empty-hand test.
- Native SONIC timing: 200 Hz physics, 50 Hz controller; captured at 25 fps.
  **The running manipulation training remains at 120/60 Hz and touch 70 Hz.**
- Released planner with its pinned hash, 50 Hz reference resampling, ten
  future frames at 0.1 s spacing. Shared CPU ONNX session, independent planner
  states per trial. Initial state is written once; subsequent movement is physics.
- 14 seconds per trial: idle 0–2 s, command 2–10 s, idle 10–14 s.
- Joint axes and joint-frame transforms shared by the original and Revo2
  URDFs matched in the separate XML comparison. Earlier CPU tests also showed
  correct planner command signs at yaw 0° and −90°.

This is **not an untouched C++/MuJoCo deployment validation**. That runtime and
its deployment models were not installed. The controlled IsaacLab harness
uses upstream robot configurations and the exact existing frozen PyTorch
controller wrapper, so it cannot independently validate every shared wrapper
assumption. It is nevertheless a physical A/B test of the speed hypothesis,
including the exact failed 0.12 m/s condition on both robot assets.

Original G1 LFS meshes (68 files) were fetched from the pinned repository.
The staged upstream URDF changes only mesh filenames from ROS package URIs
to resolved paths. No upstream robot inertias, geometry, joints or gains were
modified. Resolved simulator configs and asset masses accompany the captures.

## Provenance and artifacts

Workstation root: `/data1/users/konstantin.smirnov/DEXTRAH-G1-WholeBody`.

1. `outputs/empty_floor_v4_560_20260915`, source `41169ea`, step `560.16`:
   nine predeclared commands × two assets; all 18 physical trials completed.
2. `outputs/empty_floor_forward040_560_20260915`, source `fd4892a`, step
   `560.18`: follow-up forward/backward 0.4 m/s × two assets; all four completed.

Both runs contain `metadata.json`, `configurations.json`, per-trial
`trajectory.npz` and `metrics.json`. Measurements include body poses,
velocities, reference root/joints, actual joints, decoded actions and targets.
The recorder saves traces incrementally; no success-based episode selection.

The processes hung in Kit teardown **after saving completed captures** and
their individual Slurm steps were terminated. The training batch was not
terminated. Scheduler step exit status must not be confused with the saved
physical trial result. Earlier launch failures remain under `empty_floor*`
v1–v3; these are not counted as completed trials. They exposed missing optional
imports, unhydrated LFS assets and a crash during a periodic stack dump.

Videos compare original G1 and Revo2 side-by-side. Solid meshes follow measured
PhysX body poses; orange pelvis/knee/foot markers are explicitly kinematic
planner references, not a second simulated robot. Each clip is an uncut
14-second trial. Files are frame-count checked and SHA-256 verified on transfer.

Laptop raw captures are in the matching project `outputs/` directories.
Laptop summary plot:
`outputs/empty_floor_v4_560_20260915/speed_comparison.png` (also PDF).
Videos are copied to `~/Desktop/research_recordings/` with dated names.

## Interpretation and next step

NVIDIA documents slow walk at 0.2–0.8 m/s and recommends approximately 0.4 m/s
for strafing. The prior 0.12 m/s command assumed slower would be easier; these
measurements contradict that assumption. The exact internal reason for the
low-speed reference/controller mismatch is not proven, but its practical
effect is now measured, including a successful higher-speed counterexample.

- [Official keyboard/planner guidance](https://nvlabs.github.io/GR00T-WholeBodyControl/tutorials/keyboard.html).
- [SONIC local tracking and root-position drift](https://github.com/NVlabs/GR00T-WholeBodyControl/issues/44#issuecomment-3964152680).

Do not reverse command signs, swap controller weights, or restart training.
Next validate the same gait commands through the manipulation integration
and its clocks, then use measured-position-feedback navigation with an explicit
idle/stopping phase. Only then enable the existing SAPG grasp-preserving
adapter and test carrying. Its response to walking references remains untested.
Do not assume 0.4 m/s is safe in a tight tabletop route simply because it works
on an empty floor; clearance, stopping distance and cross-track error matter.

All experiment code is on `feature/g1-wholebody-sapg`. The 0.4 m/s speed cap is
an explicit diagnostic opt-in; default carry commands and running training
source snapshots are unchanged. No optimizer updates were made by this work.
