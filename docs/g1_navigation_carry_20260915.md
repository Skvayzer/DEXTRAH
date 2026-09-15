# Navigation-commanded brush carrying — 2026-09-15

## Intent

The preceding transfer probe moved only the object goal, leaving SONIC's
reference at standing. That encouraged reaching/leaning and did not constitute
navigation. This opt-in experiment gives navigation its own command channel:

`known-map waypoints -> body-frame (vx, vy, yaw-rate) -> NVIDIA motion planner`

`planner body reference + learned SAPG latent residual -> frozen SONIC -> 29 body targets`

`existing SAPG finger outputs -> existing Revo2 target processing`

SONIC remains the single body-action generator. No root teleportation, root
velocity write, external wrench, physical joint freezing, or second leg-target
writer is introduced. Training job 560 and its source snapshot are not changed.

## Why a separate planner is necessary

The pinned manipulation SONIC checkpoint consumes a sequence of reference joint
positions/velocities and orientations, not a raw `cmd_vel` tensor. NVIDIA's
released kinematic planner generates that sequence from movement speed,
movement direction, facing direction and recent planned motion. The command
bridge is therefore a real planner integration, not a velocity tensor appended
to a policy that was never trained to interpret it.

Sources:

- [NVIDIA planner interface](https://github.com/NVlabs/GR00T-WholeBodyControl/blob/main/docs/source/references/planner_onnx.md).
- [NVIDIA deployment planner implementation](https://github.com/NVlabs/GR00T-WholeBodyControl/blob/main/gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/localmotion_kplanner.hpp).
- [Released planner model](https://huggingface.co/nvidia/GEAR-SONIC/blob/6733128a3d8a523b1418b06bca3cdf61c8b0987f/planner_sonic.onnx).

Model revision `6733128a3d8a523b1418b06bca3cdf61c8b0987f`, SHA-256
`39b553e197f62f077975ba38512bc04781a3fc37c2af7c6756e04629f760edea`.
Only the planner is added; the original GRAIL manipulation SONIC weights remain
hash-pinned. ONNX Runtime 1.22.1 is isolated in a planner-specific dependency
directory, outside the running training environment's installed packages.

## Implemented command/reference handling

- ROS-style local X-forward, Y-left, yaw-rate commands are rotated into world
  movement directions. For the current robot facing world -Y, local rightward
  velocity means world -X, not world +X.
- Zero translational velocity selects **IDLE**: upstream `target_vel=0` means
  use the mode's default speed, and does not by itself mean stop.
- Actual ONNX input/output names, shapes, numeric types and model hash are
  checked. Only valid output frames are consumed, never the padded tail.
- Planner output is 30 Hz, resampled to 50 Hz with quaternion slerp. Ten SONIC
  future reference frames are sampled at 100 ms spacing; the existing policy
  and tactile clocks stay at 60 Hz and 70 Hz.
- Replanning is checked at 10 Hz using previous planned motion, 40 ms lookahead
  and a 160 ms reference crossfade. Constant walking commands replan at 1 Hz,
  as in native deployment, rather than restarting the blend at every check.
  Desired facing integrates commanded yaw-rate separately from measured yaw.
  Joint order is explicitly converted from
  MuJoCo to the pinned IsaacLab order.
- During carrying, arm reference positions retain their trained standing
  baseline while legs/waist receive the planner reference. **Physical arms are
  not frozen**: SAPG's residual plus SONIC still generate all 29 body targets.
  This avoids injecting the walking clip's default arm swing over a grasp.

## Carry and navigation sequence

1. Use the existing checkpoint to grasp and sustain contact-confirmed lift.
2. Save the current object offset in the robot's yaw-aligned base frame.
3. Keep that object target with the moving base in XY and at fixed world
   height. It is not attached physically and no constraint holds the brush.
4. Command a conservative known-map route: back away from the source table,
   move laterally behind the tables, then approach the receiver. Speed is
   limited to 0.12 m/s and changes are acceleration-limited. This is a simple
   navigation stand-in, not perception, SLAM or a collision-planning system.
5. Stop navigation on lost object contact. Arrival requires proximity and low
   measured base speed for one second, not just elapsed command time.
6. After arrival, hold idle locomotion and switch the object goal to the
   receiving tabletop for lowering. There is no forced release action.

The camera/telemetry overlay shows commanded and measured base velocities,
waypoint index and arrival, alongside the original reset/fall/transfer counters.

## Validation status and limitations

Thirteen pure navigation tests and six existing transfer tests pass. Tests cover
frames/signs, stopping, joint order, quaternion interpolation, planned reference
dimensions, speed limits, waypoint arrival, non-mutating commands and the
body-relative carry target. An actual ONNX test generated rightward movement
and a stopping reference; maximum reference joint speed was 4.83 rad/s. That
test validates reference generation, **not physics or successful carrying**.

The standing-trained SAPG residual was not trained on walking references. It
may oppose locomotion or lose the grasp. The first live check therefore uses
zero residual/finger means for a short locomotion-only diagnostic, before
testing the original learned residual and fingers together. Neither diagnostic
does optimizer updates. A reliable navigation-plus-grasp controller may need
later fine-tuning across walking references; that is not silently included here.

### First physics check: locomotion tracking NOT passed

`outputs/brush_navigation_walkcheck_560_20260915`, source `7e87c37`, Slurm
step 560.8: 14 seconds, zero manipulation residual/finger means for the selected
robot, 2 seconds idle, 8 seconds local rightward velocity -0.12 m/s, then idle.
There were zero robot falls, zero resets and zero numerical failures, but this
is **not a successful commanded-walk test**. The root ended at approximately
(-0.174, 1.079, 0.788) from (0, 0.420, 0.750): most displacement was backward
in world +Y rather than commanded world -X. Mean measured body lateral speed
over 3–9 s was -0.043 m/s, and it largely stopped moving during the latter part
of the walking command. Mean speed after 12 s was 0.017 m/s.

The raw first check is preserved, including all body-pose trajectories. A
repeat adds reference joint positions/velocities, actual joint state, decoded
SONIC action and delivered motor targets to isolate the failure. Combining
this with grasping before validating locomotion would confound the diagnosis.
No training restart or reward change was made.

The instrumented 8-second repeat (`brush_navigation_trace_560_20260915`, source
`e7a7d61`, step 560.9) reproduced the failure without falls. Delivered motor
targets matched the intended target buffer exactly, and all 29 body joints were
included in the writer: legs were not frozen or omitted. The reference instead
drifted into crouching with 63 planner calls in 8 seconds. The bridge's
short-horizon check forced replanning every 100 ms, repeatedly interrupting a
160 ms blend. This is a bridge defect, not evidence that SONIC cannot walk.
That check has been removed to match upstream's 1-second periodic walking
replan; desired facing also no longer follows measured yaw drift at zero
commanded yaw-rate. Regression tests cover both. Physical retesting is required
before attributing the failure solely to these defects or claiming a fix.

The corrected 14-second test (`brush_navigation_cadence_560_20260915`, source
`4dfd280`, step 560.10) used 18 planner calls, not 86. It still did not track the
sideways command: large backward displacement followed by stalling. No robot
falls occurred; a hand-far-object reset happened at 5.7 s. It is therefore not
a clean 2-second idle / 8-second walk / 4-second stop comparison: the episode
clock restarted. Later empty-hand checks explicitly disable that one reset for
the selected robot only, while retaining robot/object falls and numerical safety.

`--navigation-native-timing` is an additional **empty-hand diagnostic only**:
200 Hz physics / 50 Hz control / 25 fps capture, matching SONIC's pinned timing
defaults instead of the manipulation task's 120/60/30. It cannot be enabled for
combined SAPG carrying, and does not change training. Original checkpoint task
contracts remain in metadata separately from the explicitly reported actual
capture clocks. This isolates whether timing compatibility needs work; it is
not a claim of successful navigation or a silent checkpoint migration.

### Native-timing result: still NOT a successful walk

`outputs/brush_navigation_native_560_20260915`, source `b1001e1`, step 560.11:
14 seconds at 50 Hz control / 200 Hz physics / 25 fps, zero selected-robot
manipulation residual and zero finger action means. There were **zero robot
falls and zero numerical failures**. The brush fell and triggered an ordinary
object-fall reset at 2.58 s; both attempts remain in the video. After that reset,
the walker moved mainly backward in world +Y, reaching approximately
(0.14, 2.03) from (0, 0.42), rather than walking sideways in world -X.
During 7–12 s, mean measured body-frame lateral velocity was +0.0014 m/s
versus the requested -0.12 m/s. There were 15 planner calls in total. These are
tracking failures, not navigation successes, and no brush-carry trial followed.

The video is an empty-hand controller diagnostic, **not a manipulation-policy
evaluation**. Generic aggregate reposing metrics include the five other
simulation environments and must not be interpreted as navigation performance.
The selected robot's events, raw velocity traces and physical poses are saved
alongside the video. Its zero finger action means are mid-range normalized
commands, not a guarantee of physically open fingers or absence of table contact.

### Remaining work

The requested architecture and inference switches are implemented and pushed,
but the current G1/Revo2 SONIC integration has not demonstrated commanded
locomotion. This failure already occurs with SAPG's manipulation residual
disabled, so it cannot be attributed to that residual fighting navigation.
Fixing replanning cadence and matching native timing did not resolve it.

Before changing learned weights, isolate the same planner reference on the
released SONIC robot/environment and on this G1/Revo2 scene. Audit delivered
reference features and the physical differences (robot/table contacts, collision
geometry, inertia, solver settings and motor response); record planned root/foot
paths alongside measured ones. This is the next comparison, **not completed
evidence identifying a cause**. After base locomotion passes, enable the existing
SAPG carry residual, measure grasp retention and waypoint arrival, and only then
test lowering/release. Any later fine-tuning is a separate task; job 560 remains
unchanged and no optimizer updates were made by these experiments.
