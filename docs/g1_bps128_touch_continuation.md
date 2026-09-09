# SAPG BPS-128 + Revo2 Touch continuation

## Experiment contract

Continue the final 8,000,372,736-transition BPS policy, not the original P2P
checkpoint and not a best-reward checkpoint from an easier curriculum stage.
SHA256: `53c4b009cdd341b4a0e007111c4009da57898dcb00d264c12c81b18817637fec`.
Parent W&B: <https://wandb.ai/skvayzer/adept/runs/unique_id_0_g1_bps128_warmstart_seed_42_355>.

Preserve the 1,200 training objects, BPS bank, four goal-alignment keypoints,
seven arm delta actions, six absolute hand actions, fixed-body dynamics,
rewards, reset rules, domain randomization, solver, and 120/60 Hz physics/policy
rates. Fabrics and PCA remain disabled. The saved easy-start tolerance is
replaced by the mature `.01` tolerance immediately (maximum keypoint error
`.015 m`, ten accumulated successful steps, not consecutive steps). No reward
change or curriculum restart is part of this experiment.

Only observation dimensionality changes: actor 224 → 249, critic 246 → 271.
The 32-dimensional SAPG coefficient embedding follows these features and is
preserved. Five fingertips each provide `[Fn/25, Fx/25, Fy/25, valid, age_s]`.
Arm torque observations are **off**. Contact identity/partner labels are not
provided to the actor. PhysX normal contacts and independently indexed friction
anchors are cropped to each CAD pad and transformed to its local sensor frame.
The measured recording's 70 Hz acquisition / 10 Hz publication is modeled.
The CAD shear axes remain provisional until physical direction calibration;
these forces are not claimed to be calibrated hardware-accurate measurements.

`touch.material_override=False` is essential: the earlier contact fixture
option changed distal materials, friction, and compliance. This continuation
enables contact reporting only and makes no new material bindings or material
property writes. The optional compliant material remains available for separate
future experiments, not mixed into this one.

## Warm start and normalization

Copy all old actor/critic parameters and observation/value statistics. Insert
25 zero-weight columns before the learned SAPG embedding. Verify actor means,
sigmas, LSTM states and critic values across all six coefficients and eight
recurrent steps against the original checkpoint, including nonzero saved LSTM
states. Failed equivalence aborts before the first optimizer update.

The original scalar observation-normalizer count is broadcast across the old
features. New tactile features have separate sample counts, initialized from
20 simulated seconds of the frozen BPS leader's physical grasping across 1,200
environments. This is **statistical input calibration**, not hardware force
calibration. It must observe normal loads on all five pads and nonzero shear.
New force variances have a small floor; validity is passed as a bit, invalid
forces are zero after centering, and age is scaled in 100 ms units. The old
features retain the upstream normalization exactly.

Initial optimizers, frames, episode state, and recurrent state are fresh.
Actor LR `1e-5`, critic LR `1e-4`, constant schedule; horizon/sequence 16,
two mini-epochs, six SAPG groups, minibatch `4 * environments`. This experiment
does not claim that tactile sensing improves the policy until evaluation.

## Bounded pilot and evaluations

`scripts/run_g1_touch_experiment.py` runs five approximately 100M-transition
training segments, 500M total, with checkpoints every 64 updates and at segment
boundaries. Every segment is followed by a frozen deterministic leader
evaluation: 1,200 environments covering all training objects, seed 42, 120
simulated seconds, same domain randomization and strict threshold. Processes
run sequentially on **one** Slurm-allocated RTX 6000 Ada. No viewer is needed.

Scene reconstruction between evaluations starts fresh physical episodes and
clears stale saved observations/LSTM states. Both actor and critic Adam states,
weights, normalization, and counters are retained. The external SAPG checkpoint
format omits the critic optimizer; continuation checkpoints explicitly add it.
This is an episode-boundary learning-state resume, not a bit-exact physics resume.

Baseline measured by the same evaluation protocol:

| Metric | Final BPS baseline |
| --- | ---: |
| Completed episodes with at least one goal | 66.486% |
| Resolved goal attempts successful | 89.919% |
| Goals per simulated minute per environment | 15.76125 |
| Completed episodes that ever lifted the object | 89.190% |

Ongoing episodes and unresolved attempts are censored, not counted as failures.
Throughput includes grasping, resets, failures and all simulated time. The
evaluations use training objects, so they are not held-out generalization tests.
See `docs/g1_bps_reposing_evaluation.md` for baseline raw denominators.

A coarse safety gate stops continuation if episode any-goal success falls below
56.486%, or throughput below 12.609 goals/minute (20% below baseline). These are
engineering regression thresholds, not statistical significance tests. All
checkpoints and evaluation counts remain available if the gate stops the run.

W&B uses project `adept`, group `g1-bps128-tactile-continuation`, separate run IDs
from the parent, explicit `wandb.finish()`, and a parent checkpoint hash/link.
`touch_continuation/additional_transitions` distinguishes new samples from
`touch_continuation/cumulative_transitions`, which includes the parent's 8B.
Strict evaluation metrics use `evaluation/additional_transitions` as their axis.
`experiment_status` distinguishes training, awaiting evaluation, completed pilot,
and regression stop. Never infer an evaluation rate from the ambiguous running
training `successes_max` field.

The launcher also supports `CONTROL=1`: same 500M continuation without tactile
inputs, run sequentially on the same GPU. That matched-budget comparison is
needed to separate the value of touch from the value of further training.

## Validation and launch

Source is developed on `feature/g1-bps128-touch-continuation`. Runtime files and
checkpoints are isolated in `/data1/users/konstantin.smirnov/DEXTRAH-BPS128-TOUCH`;
the completed BPS run and external P2P code are not modified.

Unit tests cover old-feature normalization equivalence, new-channel adaptation,
invalid samples, migration layout, material/MDP contract, both optimizers on
resume, and success-counter denominators. GPU smoke tests must additionally
exercise actual contact buffers, policy migration, optimizer updates, checkpoint
reload and resource usage. Unit tests and a viewer are not sufficient alone.

Launch from the remote worktree only after those checks pass:

```bash
CALIBRATION=/absolute/path/to/validated/calibration.json \
  sbatch scripts/slurm/train_g1_touch_continuation.sbatch
```

Default target is 24,576 environments; `NUM_ENVS` may be reduced only after
recording a memory/throughput reason. Each run has its own `calibration.json`,
`warmstart_validation.json`, `progress.json`, `segment_complete.json`, configs,
BPS manifest, checkpoints, and strict per-object evaluation counts.
