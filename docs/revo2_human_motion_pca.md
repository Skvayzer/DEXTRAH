# Human-motion PCA for Unitree G1 + BrainCo Revo2

This branch adapts the offline hand-action-space construction from DextrAH-G
Appendix D to the six actuators and five fingertips of the BrainCo Revo2 right
hand. It is deliberately separate from PPO: DexYCB retargeting and PCA happen
once, and PPO later consumes a frozen five-dimensional hand task map.

## What is reproduced

- Palm-fixed human fingertip trajectories from DexYCB ground-truth `joint_3d`
  labels, using one camera per physical capture rather than duplicating the same
  trial eight times.
- Differentiable Revo2 FK parsed from the official BrainCo URDF, including the
  five coupled distal joints.
- Per-frame Adam optimization with the Appendix-D human imitation, closure, and
  regularization terms.
- The bounded parameterization
  `q_robot = lower + 0.5 * (tanh(q) + 1) * (upper - lower)`.
- The human-to-closure gamma schedule, separate power and precision-tripod
  datasets, PCA fitting, and a frozen five-dimensional `x = A q_hand` task map.
- Exact normalized-policy target scaling and the G1/Revo2 FABRICS matrix
  `[zeros(5, 7), A]`.

DextrAH-G does not publish the Adam learning rate, iteration count,
regularization weight, closure-point coordinates, demonstration IDs, or a
Revo2 regularization posture. These values are explicit CLI/config parameters
here. The default gamma schedule follows the appendix prose (first frame 1,
last frame 0); `--gamma-schedule paper_literal` preserves its printed
off-by-one equation.

The published posture semantics are preserved: Revo2 power regularization is
`[1.0, 0.75, 1.0, 1.0, 1.0, 1.0]` radians; precision regularization opposes the
thumb at `[1.0, 0.75]` while keeping the four finger commands at their lower
limits. Distal flexions follow each URDF's mimic ratios.

The paper's `alpha=1.6` is specific to its human/Allegro size mismatch. In
`--scale auto` mode this implementation records a deterministic grid search and
chooses the Revo2 scale with the lowest pure-imitation IK error on representative
pre-grasp frames. A numeric `--scale` bypasses that calibration.

## Dataset

DexYCB is CC BY-NC 4.0 and the complete release is about 119 GB. Its official
release contains 1,000 grasping sequences from 10 subjects and 20 YCB objects.
The loader uses the 500 right-hand sequences when all subject archives are
present. Download and extract them with:

```bash
scripts/download_dexycb_retargeting_data.sh \
  /data2/users/konstantin.smirnov/dex-ycb \
  /path/to/conda/env/bin/gdown
```

The object models are not required for fingertip retargeting. Calibration and
models are only needed to overlay camera images or object meshes.

## Build the frozen artifact

Run a small pilot first:

```bash
python scripts/retarget_dexycb_revo2.py \
  --dataset-root /data2/users/konstantin.smirnov/dex-ycb/data \
  --output-dir /data2/users/konstantin.smirnov/dex-ycb/revo2-pilot \
  --sequence-limit 3 \
  --max-frames-per-sequence 20 \
  --modes power,precision_tripod \
  --pca-components 5
```

Then process all available right-hand captures:

```bash
python scripts/retarget_dexycb_revo2.py \
  --dataset-root /data2/users/konstantin.smirnov/dex-ycb/data \
  --output-dir /data2/users/konstantin.smirnov/dex-ycb/revo2-full-v1 \
  --modes power,precision_tripod \
  --iterations 250 \
  --minimum-iterations 40 \
  --pca-components 5 \
  --device cpu
```

For the exact G1 setup already used by Play2Perfect, pass its combined URDF
instead of the standalone default:

```bash
  --urdf /data1/users/konstantin.smirnov/play2perfect/unitree_ros/robots/\
g1_with_brainco_hand/g1_29dof_mode_15_brainco_hand.urdf
```

This matters: the G1-integrated file has different thumb transforms, mimic
ratios, and limits from the latest standalone BrainCo file. The artifact stores
the selected URDF's SHA-256 checksum so the two profiles cannot be confused.

The optimizer batches all frames in a trajectory. This preserves independent
per-frame Adam variables and is much faster than Python-level sequential warm
starts. Use `--optimizer-execution sequential` to compare against warm-started
optimization. For this six-variable problem CPU is normally faster than paying
GPU kernel-launch overhead.

The run emits one pickle-free NPZ per trajectory/mode, `summary.json`, and
`revo2_human_motion_pca.npz`. The artifact includes joint names, the matrix,
mean, all explained-variance ratios, coordinate bounds, joint limits, run
provenance, and URDF checksum.

## Validate and inspect

```bash
python scripts/validate_revo2_retargeting.py \
  --run-dir /data2/users/konstantin.smirnov/dex-ycb/revo2-full-v1
```

Validation independently recomputes FK, fingertip errors, joint and PCA bounds,
orthonormality, retained variance, and PCA reconstruction error for every saved
trajectory. It writes `validation.json` and fails nonzero on a consistency
error.

Start Viser on the server:

When the artifact was built from the combined G1 URDF, first extract its exact
hand subtree and resolve its mesh paths:

```bash
python scripts/extract_revo2_hand_urdf.py \
  --g1-urdf /path/to/g1_29dof_mode_15_brainco_hand.urdf \
  --revo2-package-root /path/to/play2perfect/assets/urdf/revo2_description \
  --output /tmp/g1_revo2_right_hand.urdf
```

```bash
python scripts/visualize_revo2_retargeting.py \
  --trajectory /path/to/run/trajectories/TRAJECTORY.npz \
  --pca-artifact /path/to/run/revo2_human_motion_pca.npz \
  --urdf /tmp/g1_revo2_right_hand.urdf \
  --viser-host 127.0.0.1 \
  --viser-port 8089
```

From the laptop, forward the port with
`ssh -N -L 8089:127.0.0.1:8089 konstantin.smirnov@tl-server-0`, then open
`http://127.0.0.1:8089`. The viewer can switch between the Adam solution and
its five-component PCA projection.

## G1 + Revo2 policy/fabric integration

The required combined configuration order is:

```text
right_shoulder_pitch_joint, right_shoulder_roll_joint,
right_shoulder_yaw_joint, right_elbow_joint,
right_wrist_roll_joint, right_wrist_pitch_joint, right_wrist_yaw_joint,
right_thumb_metacarpal_joint, right_thumb_proximal_joint,
right_index_proximal_joint, right_middle_proximal_joint,
right_ring_proximal_joint, right_pinky_proximal_joint
```

Load the fixed task-map matrix for a FABRICS `LinearMap` with:

```python
from dextrah_lab.retargeting import (
    REVO2_RIGHT_ACTUATED_JOINTS,
    FrozenPCAHandActionMap,
)

hand_map = FrozenPCAHandActionMap.from_file(
    "/path/to/revo2_human_motion_pca.npz",
    expected_joint_names=REVO2_RIGHT_ACTUATED_JOINTS,
    device="cuda:0",
)
fabric_matrix = hand_map.fabric_taskmap_matrix(arm_dof=7)
pca_target = hand_map.normalized_to_target(policy_action[..., 6:11])
```

As in DextrAH-G, the policy action is 11-dimensional: six absolute palm-pose
targets plus five PCA hand targets. `fabric_matrix` maps the 13-dimensional G1
right-arm/Revo2 configuration into the current five-dimensional hand state.
The geometric fabric—not an inverse PCA decoder—must combine the palm and hand
attractors with joint-limit, collision, damping, acceleration, and integration
terms before issuing 13 joint targets. `reconstruct_clipped()` exists only for
visual diagnostics or a temporary position-control baseline.

The PCA artifact solves the hand representation. A complete G1 controller still
requires G1/Revo2 collision spheres, palm frames, fabric parameters, and the
simulation/real-hardware PD interface; those are robot-specific safety pieces
and must not be inferred from the Allegro controller unchanged.
