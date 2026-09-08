# G1/Revo2: direct SAPG control and object-shape experiment

## Implemented control baseline

Task: `G1-Revo2-SimToolReal-Repose-Direct`.

The task inherits `PlayEnv` directly, not the fabric environment. It therefore
uses the original seven arm delta actions, six absolute hand targets, action
delay/smoothing, physical actuation, observation builder, rewards and task
termination. No fabric solver, collision adapter, PCA artifact, virtual-motion
limits, velocity-reference feedforward or fabric-state observations are created.
The physical table, robot collision geometry and normal joint/actuator limits
remain. Isaac Sim's similarly named Fabric I/O backend remains enabled; it is
unrelated to the geometric controller and should not be disabled.

Expected runtime dimensions: 13 actions, 92 actor observations, 114 critic
observations, before SAPG coefficient embedding. Terminal `ever_lifted` and
`any_success` diagnostics are added without changing the reward.

The original fabric task and failed run are preserved. The new launcher uses a
separate run name, starts from scratch, does not auto-resume, and labels W&B with
`direct-control`, `no-fabrics`, `no-pca`, `no-shape-encoder`.

```bash
sbatch scripts/slurm/smoke_g1_direct_sapg.sbatch
# Explicit launch only after deciding whether the next run is this control
# or a separate shape-aware variant:
sbatch scripts/slurm/train_g1_direct_sapg_single.sbatch
```

The smoke checks inherited action methods, absence of controller objects,
full-range nominal hand targets, 650 simulation steps, resets, observation
dimensions and two SAPG updates. It does not demonstrate learned grasp success.
The full launcher defaults to one RTX 6000 Ada, 24,576 environments, 1,200
procedural objects and 8B frames, with W&B enabled.

## What the reference papers actually use

- **GRAIL:** Appendix B, Table 5 gives a 10-dimensional BPS descriptor. Its
  [released preprocessing code](https://github.com/NVlabs/GRAIL/blob/main/grail/retargeting/compute_bps.py)
  defaults to ten Fibonacci-sphere query points. It centers and radius-normalizes
  object geometry and records nearest-point distances. The 2,048 surface-sample
  setting is not the BPS dimension; small meshes instead use their vertices.
- **ADEPT:** Appendix A.4, Tables 6–7 list 64 XYZ object points, or 192 values,
  for its state-based teachers. Its deployed students replace that geometry
  input with RGB, plus tactile inputs on Sharpa. The paper does not justify
  assuming a particular PointNet architecture or exact point sampling/order.
  [Paper](https://arxiv.org/html/2608.19182v1#A4).

GRAIL's manipulation adaptor is trained with reference motions and binary hand
primitives on a frozen whole-body controller. Its success is not a controlled
demonstration that ten BPS values suffice for free finger-level reposing.
[GRAIL paper](https://arxiv.org/html/2606.05160v1#B1).

## Shape-aware variant

Start with **128 scalar BPS distances**, not GRAIL's exact ten-point setting.
This is an experimental design choice, not a reproduced paper hyperparameter or
a proven optimum. Compare against the direct-control baseline; a 64-point XYZ
variant is a useful alternative and is closer to ADEPT's stated input.

For a fixed common basis `b_i` and object surface samples `p_j`, scalar BPS is
`d_i = min_j ||b_i - p_j||`. It is an observation, not an avoidance controller.
There is no collision sphere, repulsive force, action restriction or hand-motion
PCA implied by adding these values.

The proposed design requirements are:

1. Compose geometry in the actual object-root frame, respecting mesh transforms,
   primitive dimensions and any scaling. Preserve centroid offset and metric
   size when normalizing. Object axes must be consistent with the pose input;
   BPS is not automatically rotation-invariant.
2. Precompute a surface sample bank and common-basis descriptors per actual
   generated object. Save generator seed, basis, preprocessing version and asset
   hashes. Avoid copying a USD vertex extractor that ignores mesh transforms.
3. Cache descriptors and gather by the exact environment-to-object mapping.
   Shape identity must follow shuffled/factorized assets correctly. Handle
   geometry changes and anisotropic scale explicitly rather than assuming a
   cached normalized descriptor covers every deformation.
4. Append geometry to both actor and critic, retaining current pose/goal inputs
   and the original fixed-size pose-keypoint reward geometry. Existing scale observations are not a
   substitute for a detailed surface descriptor. Do not change rewards or the
   control interface in the same shape ablation.
5. Visualize sampled geometry, query points and nearest-point links before
   training. Test scale/frame correctness, order invariance of distances, distinct
   shapes with similar bounds, and correct asset lookup.
6. Evaluate on held-out generated shapes. Compare correct, zeroed and shuffled
   shape inputs to establish whether the policy uses geometry. Report lift and
   reposing success, not reward alone.

A scalar BPS vector has fixed, shared feature slots and fits the existing
flat-input SAPG network. A raw point set needs a consistent ordering convention
or a permutation-invariant encoder; adding such an encoder would be our design,
not a claim about ADEPT's unreleased code. Neither method reconstructs arbitrary
geometry perfectly. Ten distant queries can miss useful handle detail; 64
sampled points can also undersample thin parts.

For 24,576 environments and horizon 16, one float32 rollout tensor costs 192 MiB
for 128 BPS values or 288 MiB for 64 XYZ points. These are arithmetic storage
figures, not whole-job VRAM forecasts: actor/critic copies, activations and other
buffers add memory. Both candidates deserve a one-GPU memory/throughput check.

## Real-world inputs

The first simulation experiment can use complete mesh geometry. For known real
tools, scan once, compute the same canonical descriptor, and estimate object pose
online. For unseen tools, RGB-D segmentation and multi-view reconstruction can
provide geometry, but a partial scan is not equivalent to the complete training
mesh. Consistent scale/origin and training with occlusion/noise are necessary.

Alternatively, use the shape-aware policy as a privileged teacher and distill a
camera-based student. This separates geometry-assisted RL from deployment
perception. Neither BPS nor raw points alone eliminates the perception gap or
guarantees faster learning.

## Status

Validation on 2026-09-08: Slurm job 350 completed with exit code 0. Three static
contract tests passed; the actual 96-environment simulation completed 650 steps
and 121 resets, then two SAPG updates completed. The two-update checkpoint has
no episode-return estimate because that separate training smoke ended before an
episode completed; it is not a trained policy. The first attempt, job 349,
failed in the validator's expected-value calculation because it omitted the
original target safety clamp; the corrected test passed without changing control.

The no-fabrics/no-PCA control task, CPU-only BPS-128 preprocessor and
[Viser preview](object_bps_viewer.md) are implemented. The separately registered
shape-aware task and checkpoint expansion are documented in
[G1 BPS-128 SAPG warm start](g1_bps128_training.md). The BPS experiment does not
resume a fabric checkpoint.
