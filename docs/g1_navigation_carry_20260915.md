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
  and a 160 ms reference crossfade. Joint order is explicitly converted from
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

Nine pure navigation tests and six existing transfer tests pass. Tests cover
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
