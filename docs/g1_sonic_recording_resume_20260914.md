# Checkpoint recording and continuation — 14 September 2026

## What the reported rates mean

Run 547's final W&B window is the last 3,000 completed **leader-group training
episodes**, not a held-out evaluation. The source observer selects the leader
environment block and averages each terminal flag independently.

- **90.97% ever lifted:** the object crossed the source task's lift threshold
  at least once. The source formula is `0.05 + object_z - object_init_z > 0.15`,
  i.e. a rise of more than 10 cm. The flag stays true for the rest of that
  episode. It does not require a stable grasp, retained possession, or a goal.
- **67.00% robot fall:** the robot reached the body-fall condition, pelvis
  height below 0.4 m or upright cosine below 0.5. This is separate from the
  original task's object-fall flag.
- **0% any goal success:** no completed reposing goal in those episodes. The
  goal test retains all four source pose keypoints: maximum position error
  below 1.5 cm for ten accumulated qualifying control steps. These steps are
  not required to be consecutive in this configuration.

The flags overlap: lifting then falling counts toward both lifting and robot
fall. Do not add the percentages or describe the lift rate as manipulation
success. The exported run 547 history contains no nonzero logged
`episode_final/any_goal_success_rate`.

## Recording protocol

Job 554 records run 547's **best-return** checkpoint, epoch 8,207 /
1,216,462,848 frames. It strict-loads the full 35-action actor and its trained
normalizers, runs deterministic zero-coefficient leader commands, and makes
zero optimizer updates. The training task, noise, object bank, 60/120 Hz
control/physics, 70 Hz touch and termination rules remain enabled.

The first assigned hammer, brush and spatula are selected by family before
rollout, with seed 42. Each clip covers the same uncut 60 simulated seconds,
with all resets retained. The videos show full-body and wrist close-up views,
the target pose, pose error, goal count and robot-fall/reset counters. Every
physical link is placed from its measured PhysX pose, not generated hand FK.
The close-up's translucent table is only a rendering choice.

Scripts: `record_g1_sonic_checkpoint.py`, `render_g1_sonic_checkpoint.py`, and
`scripts/slurm/record_g1_sonic_checkpoint.sbatch`. Captures, event logs and
video validation metadata are kept under
`outputs/sonic_checkpoint_video_554`. This is a diagnostic on training objects;
its rates need not match the stochastic training window.

The initial 1,200-environment attempt (550) and 120-environment retry (552)
stalled in native simulation startup before any checkpoint action and were
canceled. The allocated GPU passed six CUDA/CPU checkpoint and GC tests during
552. Native debugger attachment was denied; no privileged access was attempted.
A worker-limit launch (553) exited at command-line parsing and was corrected.

Job 554 passed startup with 16 allocated CPUs and both Carb/TBB worker pools
explicitly limited to eight. Effective settings were logged: both pools eight,
asynchronous rendering disabled. SimulationApp resets the PXR environment limit
to 32 internally, so do not claim that environment variable itself remained
eight. Worker controls follow the [Isaac Sim 5.0 performance handbook](https://docs.isaacsim.omniverse.nvidia.com/5.0.0/reference_material/sim_performance_optimization_handbook.html).
This successful retry supports a runtime mitigation, not a proven root cause.

Recording uses 120 environments, while still constructing the original
1,200-object descriptor bank. This does not reduce the training environment
count or change the three-family, predeclared selection rule.

Capture completed with zero optimizer updates: 3,774 completed episodes,
3,503 ever lifted (92.82%), 2,609 robot-fall terminations (69.13%), and zero
goal hits. Two finite numerical-outlier terminations were recorded. These
120-environment diagnostic rates are distinct from the earlier 3,000-episode
training window. Selected environment IDs are hammer 1, brush 0, spatula 6.
The captured checkpoint SHA-256 is
`0d7ac549857859bffaf4b7f1c8b11b3806f952b414ede4b961a7905af64bba69`.

All three videos passed frame-count/duration validation (1,800 frames,
30 fps, 60 seconds), and each laptop copy matches its workstation SHA-256:

- Hammer: `034fdee2c155d37c76d32a59371afaa0ac19fdae138aabb98c2ea95608016a70`
- Brush: `24db25adedaaaf69424414a980867a9a68100983914d81a23f65aa63b6ac946d`
- Spatula: `808c700a786d49ff899369b4a081ed6a72d69c6b90ebf43889a2a3ad4a5d2188`

Reviewed previews show both lifting attempts and loss of body balance. No
physical visual links were omitted. Videos are published to
`/Users/konstantinsmirnov/Desktop/research_recordings`.

## Memory mitigation before continuation

Run 547 failed with CUDA OOM, not a normal training completion. Inspection of
its stage-level memory trace found live allocations dropping from roughly
24–28 GiB to 3.7 GiB as the full-GC generation counter reset. This is stronger
evidence of garbage-collection-related retention than the earlier, low-memory
one-off probe in run 520, which had reclaimed nothing.

An optional `--boundary-gc-interval 16` now requests full cyclic garbage
collection **before** every sixteenth training update. It never clears live
rollout data, flushes the CUDA allocator, changes gradients/rewards, or resets
the simulator. Logs report actual reclaimed tensor bytes, object counts and
cleanup time. It is a bounded-retention mitigation; the particular producer
of the cycles has not been established.

CPU checks: eight passed, two CUDA-only checks skipped. The continuation's
GPU preflight additionally checks reclamation of an unreachable CUDA tensor
cycle and exact equality of the next Adam update, plus existing checkpoint
optimizer-placement parity. Longer-running memory telemetry is still needed;
these checks alone do not prove that long training will fit.

The continuation also exposes `SIM_WORKER_THREADS=8` through supported Kit
arguments, recording the effective pool sizes in its task contract. This bounds
host worker pools without changing the physics timestep, solver settings,
reward, policy architecture or optimizer settings.

Continuation **555 is running**, after recording/render job 554 completed
successfully (`afterok:554`). It preserves 9,216 environments / six SAPG groups and
both optimizers from the newest saved `latest.pth` in run 547, epoch 8,256 /
1,223,688,192 frames. It starts fresh physics episodes, not fresh weights.
Online W&B, original best-return saving and atomic latest saves remain on;
periodic evaluation and frame/epoch caps remain off.
The Slurm allocation is 48 hours on one RTX 6000 Ada; the time-limit signal
requests an atomic checkpoint at an optimizer boundary.

## Verified live continuation

At the 13:56 Dubai check, run 555 had reached epoch **8,718 /
1,291,812,864 frames**, 462 new updates and 68,124,672 new transitions after
resuming. Code snapshot: `fb4fd97d98c797ce1653a302e4671dc16eec728a`.
Resume SHA-256: `9336d411526f7e36d2adecfbb88b8db848f83cac5623f2f6ea6ceee42222c17a`.
Both optimizers were restored, with 35 actor and 11 critic Adam counters on
CPU. All seven GPU preflight tests passed. Native physics startup took 24.33 s.

The allocated device is physical GPU 0, RTX 6000 Ada UUID
`GPU-449a592b-661e-ce1f-1602-bbcbd964661d`. Device use was **28.69 GiB**,
live PyTorch tensors 3.74 GiB, reserved 15.89 GiB and peak live 12.10 GiB.
One utilization sample was 99%; this is not a claim of sustained saturation.
Twenty-nine boundary-GC calls took 8.03 s total and reclaimed about 0.55 GiB
cumulatively. Most calls freed no CUDA bytes. Long-run OOM resolution is not
proven by this initial interval.

The [live W&B run](https://wandb.ai/skvayzer/adept/runs/unique_id_0_sonic_sapg_touch_555)
was independently checked via its API: state `running`,
`experiment_status=training`, optimizer updates enabled, and epoch 8,720 /
1,292,107,776 frames uploaded. The checked return was 321.36; lifting 92%,
robot falls 69.43%, reposing-goal success 0%. Actor/critic losses were finite.
Both rolling latest and best-return checkpoint files had refreshed.

The run remains active. Numerical outlier resets and poor balance/reposing
performance remain unresolved; continuation is not a claim that the task has
been learned or that physics is now fully stable.
