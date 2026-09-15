# Updated frozen-SONIC checkpoint recordings — 2026-09-15

Requested: repeat the hammer, brush and spatula videos without stopping training.

## Checkpoint and recording protocol

- Training job: **560**, unchanged process/source/optimizer state.
- Selected current best-return checkpoint: epoch **14,856**, **2,188,277,760**
  cumulative transitions; not the continuously changing latest checkpoint.
- Snapshot SHA-256:
  `d37ed661f291d2fc107cf3f80222d87e6ca3fac5e0d0ed43f3e64a73e298cf0b`.
- Workstation artifacts: `outputs/sonic_video_560_20260915/`.
- Capture source: `acd15dafe1bd28aae99891d89d8de528dc338512`.
- Corrected renderer: `584a1471e65a98ebe145a7050fe08dcd036b30a4`.
- Same seed 42, 120 environments, first assigned hammer/brush/spatula in
  environments 1/0/6; deterministic leader, no outcome filtering.
- 60 simulated seconds, 30 fps, measured PhysX link poses, all resets retained;
  full-body and hand-close-up views. No new viewer or periodic evaluation.
- Capture performed zero optimizer updates and verified all 55 pretrained
  SONIC tensors unchanged. Training continued in its original batch step.

Capture used an overlapping Slurm step in the existing one-GPU allocation,
with the existing memory guards. It took 526.94 seconds after scene startup.
Rendering reuses the captured trajectories; it does not rerun the policy.

## Observed results

| Clip | Reposing goals | Robot falls | Total resets |
| --- | ---: | ---: | ---: |
| Hammer | 35 | 2 | 6 |
| Brush | 56 | 0 | 1 (50-goal success) |
| Spatula | 42 | 0 | 0 |

Yesterday's corresponding counts were 27/21/23 goals and zero robot falls in
all three clips. These are short diagnostic rollouts of training objects, not
a held-out success-rate estimate. The new checkpoint achieves more goals in
these clips, but the hammer rollout is less stable.

Across all 120 environments, the new recording reached 5,911 goals versus
3,552 previously, and recorded 56 robot-fall terminations versus six. There
were zero numerical-failure resets in either recording. Do not describe this
as an across-the-board improvement, or interpret a best-return checkpoint as
necessarily the best-balance checkpoint. Completed-episode fractions exclude
episodes still ongoing at the end of the minute.

## Caption correction

The old renderer incorrectly hard-coded **1.5 cm** in its footer. The actual
saved source task uses **1 cm**, ten accumulated near-goal steps, and a
50-goal episode limit. This was a caption error, not a reward or simulation
change. New captures save the termination configuration explicitly; legacy
captures recover it from the saved source-run configuration. Unknown varying
curricula are rejected instead of inventing a fixed tolerance.

Only the unfinished old-caption renderer was terminated. Its partial hammer
export/previews remain recoverable in `capture/hammer/superseded_caption/`.
The original capture and training were preserved. Rendering resumed in a new
overlapping step using `--render-only`; seven recording-contract tests passed
locally and in the allocated workstation step. The prior-day videos were not
overwritten or retroactively modified.

## Delivery and final validation

All three corrected exports passed 1,800-frame / 60-second validation at
1600x900, with no omitted physical link visuals. Each downloaded MP4 matched
the workstation SHA-256. They are on the laptop in
`/Users/konstantinsmirnov/Desktop/research_recordings/` as
`g1-frozen-sonic-sapg-560-{hammer,brush,spatula}-60s-20260915.mp4`.

The completed recording provenance reports training advancing from
2,231,924,736 to 2,315,532,288 frames, with pretrained SONIC unchanged and
18,312 MiB GPU memory free after rendering. A subsequent scheduler check
confirmed job 560 still RUNNING and only its batch step remaining.
