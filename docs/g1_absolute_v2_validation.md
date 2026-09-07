# G1 absolute-target restart — 2026-09-07

Runtime commit: `d9475e0` on `g1-adept-sapg-simtoolreal`.
Previous plateaued run: Slurm 306, stopped without deleting its checkpoints.
Fresh run: Slurm 319, `0_g1_adept_sapg_simtoolreal_absolute_v2_seed_42`.

## Changes under evaluation

- Reuse the pinned Play2Perfect action pipeline: seven arm increments and six
  absolute hand targets with the original delay and smoothing.
- Keep nominal action targets independent of the downstream controller; expose
  them through the existing previous-target observation.
- Apply the frozen DexYCB/Revo2 PCA as a 5% correction to each nominal hand
  target. Do not accumulate the correction across steps.
- Integrate the fabric twice at 1/120 second per 1/60-second policy step.
- Use P2P's SAPG-aware statistics observer and add explicit episode lift and
  success fractions. Copy terminal clearance before resetting its accumulator.

Reward functions, SAPG hyperparameters, procedural object pool, physics/domain
randomization, and collision geometry are inherited from the preceding setup.
This is an ADEPT-inspired reduced G1 fabric combined with P2P/SAPG, not an
assertion that unpublished ADEPT implementation details have been reproduced.

## Completed checks

`pytest` on the workstation: **23 passed** across the reduced fabric,
collision geometry, PhysX adapter, and PCA action-map tests. The soft-prior
Jacobian has five singular values of 1 and one of 0.95 away from clipping,
confirming six independent local hand-control directions.

Slurm **318** ran `scripts/validate_g1_adept_control.py --headless` on one RTX
6000 Ada with 64 environments, 12 generated tools, and seed 42. All checks
reported PASS:

- Zero hand action moves nominal targets toward the midpoint of joint limits.
- Arm accumulation agrees with the P2P action semantics even with fabric lag.
- Physical open/close responses remain finite with fabric and PCA active.
- Subset resets restore nominal targets and the margin-clamped fabric state,
  without changing other environments' accumulators.
- Actor and critic observations have 131 and 153 finite components.
- Observer tests verify leader-only episode aggregation, terminal success,
  per-block summaries, and a consistent frame axis.

The scripted hand command changed from -0.6 to +0.6 and was held for three
seconds at each endpoint. Mean measured travel, as a fraction of each joint's
range, was `[0.5361, 0.4923, 0.5329, 0.5319, 0.5336, 0.5231]` in canonical hand
order. Task auto-resets were suppressed during this response test; physics and
fabric collision handling remained enabled. It demonstrates actuator response,
not successful learned grasping or collision-free operation at every pose.

## Training check

Slurm 319 starts from scratch with 24,576 environments, 1,200 procedural tools,
one RTX 6000 Ada (physical GPU 2, UUID `GPU-4a10e4cf-0808-67e3-468e-3e883b4f7ca1`),
W&B enabled, and an initial 8-billion-frame budget.

The full-scale run passed 56 epochs (21.63 million frames) with checkpoints
written and W&B reporting the new episode statistics. The last ten measured
epochs averaged approximately 134,827 frames/second, including policy inference
and learning. GPU use was approximately 38 GiB at 96% utilization in one sample.
Rewards, losses, and controller metrics were finite. One NaN occurred in
`auxiliary_stats/off_on_grad_similarity`, the same undefined diagnostic cosine
seen in the preceding SAPG run; no other scalar contained NaN or infinity.

After the first full episode horizon, the leader episode lift fraction was
about 0.0227 and any-success fraction 0.001. These are startup measurements,
not evidence of an improved learning rate. Early statistics before the full
horizon were biased by which episodes finished first. Fabric sphere constraints
still show some violations (latest constraint penetration fraction about
0.0023); this controller is not a proof of collision-free behavior.

W&B: https://wandb.ai/skvayzer/adept/runs/unique_id_0_g1_adept_sapg_simtoolreal_absolute_v2_seed_42

The learning hypothesis remains to be tested: restoring the original action
semantics should improve exploration of closing and lifting. The control tests
cannot establish that it will learn successful reposing. Compare leader episode
lift/success statistics at matched frame counts with the old P2P run, rather
than interpreting startup rewards or wall-clock time as evidence of learning.
