# G1 SAPG + BPS-128 reposing evaluation

This is frozen-policy Isaac/PhysX evaluation, not training. Use the final
8,000,372,736-transition checkpoint from job 355, not the early easy-curriculum
`best/model.pth`. Training rewards, original four pose keypoints, arm/hand action
mapping, delays, observation noise and external wrench disturbances are retained.
The final tolerance is pinned explicitly. There is no fabric, PCA, or tactile path.

## Protocol

- 1,200 environments: one per original training-bank object, seed 42.
- 120 simulated seconds per environment, 60 Hz control. This is a training-set
  evaluation, not evidence of held-out-object or real-hardware generalization.
- Deterministic SAPG leader (coefficient ID zero), with the saved observation
  normalizer and recurrent policy, including recurrent-state resets.
- Preselect the first hammer, spatula, brush and eraser-family assets before
  rollout. Record their first 60 seconds at 30 FPS, without cutting failed grasps
  or hiding resets. All 1,200 objects contribute to the separate aggregate report.
- Original fall, hand-distance, goal timeout and 50-goal episode-cap terminations
  remain active. Only PhysX buffer capacities are reduced for the smaller batch;
  solver settings are unchanged.

## Success definitions

A goal is reached when the maximum distance between corresponding original
four pose keypoints is at most 0.015 m for ten accumulated control steps. The
saved task does **not** require these ten steps to be consecutive. The config
tolerance is 0.01, multiplied by its 1.5 keypoint scale. Success advances to a
new goal; it does not normally reset the episode.

The evaluator reads terminal statistics **before** Isaac auto-reset:

| Metric | Denominator |
| --- | --- |
| Episode success | Completed episodes with at least one reached goal / all completed episodes |
| Resolved-goal success | Reached goals / (reached goals + attempts ended by episode termination without a hit that step) |
| Goals per simulated minute | Reached goals / total simulated environment minutes, including grasping/reset/unfinished-attempt time |
| Mean goals per completed episode | Goals belonging to completed episodes / completed episodes |
| Episode lift rate | Completed episodes that ever lifted the object / completed episodes |
| Object coverage | Objects reaching at least one goal within 120 s / 1,200 objects |

Ongoing episodes and unresolved goal attempts are excluded from the relevant
success fractions, not declared failures. These are correlated sequential
attempts, not independently sampled Bernoulli trials. A terminal-step success
does not create an imaginary failed next attempt. `evaluation.json` preserves
raw counts, elapsed time, family breakdowns and censoring notes.

## Faithful videos and BPS inset

The renderer replays recorded simulator link/object/table/goal poses, with no
interpolation or synthetic success animation. It checks URDF forward kinematics
against measured PhysX link poses on every frame. The full G1 body is static
visual context; only the trained right arm and hand are active.

The inset uses the actual object's primitive collision mesh, the exact seeded
16,384-point surface sample, normalization and 128 fixed spherical basis queries
used by the training bank. All 132 policy features (128 distances, centroid XYZ,
radius) are checked against the environment. Amber points are fixed queries,
mint points their nearest sampled surface locations, and lines are the unsigned
distances. The entire inset rotates only for presentation: the canonical shape
descriptor remains constant when the physical object changes pose. This is not
contact sensing or collision avoidance.

## Reproduce on the workstation

Use a separate staged source directory. Do not modify the training checkout.
Set `RECORDING_REPO` and optionally `RECORDING_OUTPUT`, then submit
`scripts/slurm/evaluate_g1_bps_reposing.sbatch` through Slurm. It runs 23 helper
tests before capture. After successful capture, submit
`scripts/slurm/render_g1_bps_reposing.sbatch` with `RENDER_SMOKE=1` first to inspect
the hammer preview; then submit with `RENDER_SMOKE=0` for all four full videos.
Both jobs request just one RTX 6000 Ada, and refuse to overwrite their outputs.

Each clip has trajectory arrays, metadata, reset/goal events, exact BPS geometry,
preview frames and an encoding/pose-validation JSON. Keep those audit artifacts
outside Git; copy only final MP4s to `Desktop/research_recordings`.

## Measured results, 2026-09-09

Capture job **363** completed the protocol above. Checkpoint SHA-256:
`53c4b009cdd341b4a0e007111c4009da57898dcb00d264c12c81b18817637fec`.
Training-bank feature SHA-256:
`69a73d99a05749459fd4339f9341d3cce142ece0f7965283c04cb6410a2f3328`.

| Metric | Result | Raw numerator / denominator |
| --- | --- | --- |
| Completed-episode success | **66.49%** | 2,823 / 4,246 episodes |
| Resolved-goal success | **89.92%** | 37,827 / 42,068 resolved attempts |
| Goal throughput | **15.76 / simulated minute** | 37,827 goals / 2,400 environment-minutes |
| Goals per completed episode | **6.26** | 26,590 / 4,246 |
| Completed-episode lift rate | **89.19%** | 3,787 / 4,246 |
| Completed episodes reaching all 50 goals | **0.118%** | 5 / 4,246 |
| Object coverage within 120 s | **100%** | 1,200 / 1,200 training objects |

The coverage number is **not** a 100% episode success rate. Each object had
multiple opportunities. Per-object goals over 120 seconds ranged from 11 to 57,
with median 31. At the cutoff, 11,237 reached goals belonged to still-running
episodes; they count toward throughput and resolved-goal success, but not the
completed-episode mean. Thus this finite-horizon test also censors some long,
successful episodes. Do not present the 89.92% goal-attempt figure as the chance
that a fresh episode will succeed.

| Family | Objects | Completed-episode success | Goals / minute |
| --- | ---: | ---: | ---: |
| Brush | 400 | 66.72% | 15.95 |
| Eraser | 100 | 61.01% | 15.04 |
| Hammer | 200 | 63.89% | 15.59 |
| Marker | 100 | 66.67% | 15.85 |
| Screwdriver | 200 | 68.33% | 16.00 |
| Spatula | 200 | 69.74% | 15.64 |

The preselected 60-second clips contain hammer **10 goals / 2 resets**, spatula
**14 / 1**, brush **23 / 1**, and eraser **15 / 1**. Their first successes occur
at 9.23, 5.40, 3.05, and 4.18 seconds, respectively. These are procedural training
shapes and reposing goals, not demonstrations of functional hammering or other
tool-use tasks.

All per-object count sums were checked against the aggregate JSON, and video
goal/reset counters were checked against the corresponding event logs. This is
one seeded, deterministic-policy evaluation with retained training disturbances;
it is not a matched baseline comparison or evidence that BPS alone caused the
improvement in the training curves.

Render job **366** completed successfully. All four final MP4s were checked on
the workstation and again with macOS AVFoundation: **60.0 seconds, 1,800 frames,
1920 x 1080, 30 FPS**. Their local SHA-256 hashes match the workstation copies.
Across all 7,200 rendered frames, the largest URDF/PhysX position discrepancy
was **0.0173 mm**, angular discrepancy below **2.7e-6 rad**, and BPS display
distance error was zero. Preview frames were visually inspected for every shape.

Final filenames in `Desktop/research_recordings`:

- `sapg-bps128-g1-hammer-60s.mp4`
- `sapg-bps128-g1-spatula-60s.mp4`
- `sapg-bps128-g1-brush-60s.mp4`
- `sapg-bps128-g1-eraser-60s.mp4`

Only these four MP4s were added there. Trajectories, JSON reports and PNG previews
are retained separately under `outputs/bps_reposing_20260909`. No training was
started, and evaluation/rendering allocations were released after completion.
