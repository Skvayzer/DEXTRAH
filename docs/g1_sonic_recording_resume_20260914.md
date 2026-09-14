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

The planned continuation preserves 9,216 environments / six SAPG groups and
both optimizers from the newest saved `latest.pth` in run 547, epoch 8,256 /
1,223,688,192 frames. It starts fresh physics episodes, not fresh weights.
Online W&B, original best-return saving and atomic latest saves remain on;
periodic evaluation and frame/epoch caps remain off.
