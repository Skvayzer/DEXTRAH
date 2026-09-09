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
