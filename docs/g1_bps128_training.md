# G1/Revo2 SAPG with BPS-128

Task: `G1-Revo2-SimToolReal-Repose-BPS128`.
Branch: `g1-sapg-bps128`. This is an object-observation experiment on the
original Play2Perfect task, not a geometric-controller experiment.

## What changes and what does not

The policy and asymmetric critic receive 128 scalar GRAIL-style BPS distances,
plus the normalization centroid (three meters-valued coordinates) and radius
(one meters-valued coordinate): **132 added values**. Actual actor observations
grow from 92 to 224 and critic observations from 114 to 246, before the SAPG
coefficient identifier / learned 32-dimensional embedding. Actions remain
seven arm deltas and six full-range absolute hand positions.

No fabrics, PCA prior, tactile features, repulsion spheres, action-space
projection or learned shape encoder are enabled. Original action delays,
smoothing, actuators, physical collision geometry, rewards, randomization,
resets, and success definitions are inherited unchanged. The G1 reduced/fixed
body physics configuration also remains unchanged; this does not introduce
whole-body control. Isaac Sim's Fabric I/O option is unrelated and stays on.

The original reward has four fixed-size pose keypoints (not eight observed
mesh corners). They describe pose error, not object surface geometry. Existing
pose, goal and scale inputs remain present alongside BPS.

## Geometry contract

The exact 1,200 generated object URDFs are encoded, covering 12 distributions
across hammer, screwdriver, marker, spatula, eraser and brush families. Collision
box/cylinder origins, rotations and metric dimensions are composed in the
object-root frame. Cylinders use a 64-section surface approximation; this is not
an exact reproduction of PhysX's cooked contact hull. Each object's triangle
surface is area-sampled at 16,384 points with a local seed of 42, then centered
and normalized using the pinned GRAIL functions. All objects share the same
128-point Fibonacci-sphere basis. Distances are unsigned nearest-sample norms.

Object rotation is supplied separately through the original observations;
canonical BPS does not change when the object rotates in the world. Centroid
and radius preserve metric size/origin information removed by normalization.
This is not a world-frame distance-to-obstacle representation.

URDF content, preprocessing version and NumPy/trimesh versions key an on-disk
cache. The run saves every URDF hash, descriptor-bank hash, object bounds and
environment-to-asset index. An independent check compares every environment's
composed USD asset reference to the URDF selected for its descriptor, including
shuffled asset order. A mismatch stops initialization.

Descriptors are computed once and gathered on GPU, not recomputed with nearest
neighbors during rollouts. Physical shapes do not change across resets in this
task. Existing object-scale observation randomization does not physically resize
the meshes. A future physical rescaling/deformation feature must rebuild BPS.
Both ordinary and pre-reset terminal critic observations are augmented.

## Warm start: learned behavior first, shape learning next

Source is the successful original Play2Perfect SAPG checkpoint from
`outputs/2026-08-20/10-49-23`, experiment ending in
`sapg_24576_scratch_20260820_retry1`, checkpoint SHA256:

`dc4454d1db23c90cab0b7bfa44e9c95f5ed2061530f480d3bfd1d7a4948ba5b2`.

Its saved epoch is 146,869 and frame count is 57,751,240,704. We preserve
actor/critic weights, original observation and value normalization statistics,
all six learned exploration embeddings and coefficient-conditioned sigmas.
New observation columns in the actor's LSTM input and critic's first layer
start at zero. The old SAPG embedding columns move AFTER the new BPS inputs;
blindly padding the right side would put them in the wrong place.

New-feature normalization statistics are initialized using the current
per-environment descriptor distribution. The shared old running-stat count is
large, so leaving uncalibrated default statistics would adapt very slowly.
All optimizer state, rollout buffers, counters and environment recurrent state
are fresh. This is **weights warm-start fine-tuning**, not an exact run resume.

The source checkpoint has `env_state=None`, so its curriculum tolerance cannot
be recovered exactly. We explicitly restart the original tolerance curriculum
at 0.075, down to 0.01. Early success therefore must not be compared directly
against the mature old run's likely stricter tolerance. The source MDP config
is checked against current action/reward/reset/termination/randomization/asset/
observation fields and simulation timing before training can proceed.

Before the first rollout, the implementation evaluates old and expanded models
on 48 real saved source observations covering all six groups, with nonzero
source recurrent states, for eight recurrent steps. It asserts close action
means, sigmas, recurrent states and critic values, then saves the measured
errors and a weights-only initialization snapshot in the run's `bps/` folder.
This tests conversion correctness; it does not prove BPS will improve learning.

Actor fine-tuning uses constant LR 1e-5 (old run started at adaptive 1e-4).
Critic LR stays 1e-4. Original SAPG mixed-exploration experience sharing,
six groups, horizon/sequence length 16, two mini-epochs, clipping and reward
scaling are retained. New shape input weights are trainable immediately.

## Launch and validation

Use an isolated worktree with `PYTHONPATH` preferring this checkout over the
older editable installation, and the existing Play2Perfect fork at commit
`70e79b5e53f912ef04af294ff8f61ac1c7f42160` (Isaac Sim 5.0 / Isaac Lab).
Do not run Isaac Sim or heavy preprocessing on the login shell.

```bash
sbatch scripts/slurm/smoke_g1_bps128_sapg.sbatch
# Only after the smoke passes:
sbatch scripts/slurm/train_g1_bps128_sapg.sbatch
```

The smoke uses 1,200 environments and the FULL 1,200-object bank, tests 650
simulation steps including terminations/resets, then runs 50 SAPG updates from
the source checkpoint. Full training defaults to **24,576 environments on one
RTX 6000 Ada**, six groups of 4,096, 8 billion additional frames, W&B enabled,
periodic checkpoint saving every 250 updates, and a seven-day Slurm limit.
CPU geometry, cache, lookup and checkpoint-column tests are also included.

Validated on 2026-09-08 in Slurm job **353**, completed with exit code 0:

- 24 CPU tests passed in the workstation's training environment.
- 1,200-object/1,200-environment runtime test: 650 steps, 1,343 resets,
  76 terminal-observation steps; every USD reference identity check passed.
- Expanded actor means, sigmas and recurrent states matched the old model
  exactly in the recurrent validation; maximum critic value difference was
  `1.6689300537109375e-06`.
- 50 SAPG updates / 960,000 transitions completed. A separate allocated-CPU
  checkpoint check found every actor/critic tensor finite. Newly added actor
  and critic input-column norms grew from zero to 0.08182 and 0.34696.

This establishes a working implementation and trainable shape inputs, not an
improvement over the old policy. Smoke episode statistics cover a small,
early-completing subset at the easy initial tolerance, so they are not a
benchmark.

Full Slurm job **355** launched from commit `9709349` with 24,576 environments
on physical GPU 2 (UUID `GPU-4a10e4cf-0808-67e3-468e-3e883b4f7ca1`, PCI
`75:00.0`). The job-local GPU index is zero. Startup completed, the full asset
mapping and warm-start checks passed, and the job passed 13 million transitions
while saving checkpoints. Initial throughput was approximately 127k–136k total
transitions/s at approximately 31,600 MiB VRAM; these are startup observations,
not a guaranteed sustained rate. An 8B-frame budget is roughly 17 hours at
130k/s, excluding initialization and later throughput changes.

[Live W&B run](https://wandb.ai/skvayzer/adept/runs/unique_id_0_g1_bps128_warmstart_seed_42_355).
Online API verification confirmed running state, finite actor/critic losses,
and nonzero learned shape-input weights. Longer-run benefit is not established.

Known inherited logging issue: a few startup NaNs appeared ONLY in
`auxiliary_stats/off_on_grad_similarity`. The original SAPG fork computes this
diagnostic from gradient copies taken before AMP unscaling; with detailed
off-policy gradient logging disabled, its off-policy vector is also just zero.
It is not a meaningful manipulation metric in this configuration. The epoch-22
checkpoint was independently checked on CPU: all actor and critic tensors were
finite, and no non-finite loss/return metrics were found in the scanned events.
The shared Play2Perfect dependency was not modified to suppress these warnings.

## What to watch in W&B

Use completed leader episodes, not maximum reward, as the primary view:

- `episode_final/ever_lifted_rate`: fraction that lifted at least once.
- `episode_final/any_goal_success_rate`: fraction that reached at least one goal.
- `episode_final/goals_reached_count`: mean goals reached per completed episode.
- `episode_final/all_goals_completed_rate`: fraction reaching the 50-goal cap.
- `shape/actor_input_weight_norm` and `shape/critic_input_weight_norm`: new BPS
  connections should move away from zero. Nonzero weights alone do not establish
  useful shape understanding.

The old metrics are retained for compatibility. `successes` is a goal count,
not a percentage; `successes_max` is not a success rate. Main terminal metrics
are aggregated over the 4,096 leader environments by the original P2P observer.
Always examine the current success tolerance alongside success counts.

A useful later evaluation is matched-seed, fixed-tolerance success on held-out
objects, comparing correct versus shuffled/zeroed BPS and the original policy.
The baseline and expanded initialization should agree; improvements only after
fine-tuning, and sensitivity to correct shape, are the evidence to seek.
Complete simulation geometry remains a privileged input: real deployment needs
consistent scanned/CAD geometry plus object pose, or perception/distillation.
