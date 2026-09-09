# BPS-128 + tactile continuation validation — 2026-09-09

Source/experiment contract: [g1_bps128_touch_continuation.md](g1_bps128_touch_continuation.md).
Remote worktree: `/data1/users/konstantin.smirnov/DEXTRAH-BPS128-TOUCH`.

## Passed checks

- Slurm CPU job 373: **52 tests passed**, including independent tactile
  normalization counts, source-feature equivalence, normal/shear contact
  aggregation, invalid packets, checkpoint expansion, complete optimizer resume,
  source-MDP contract, success denominators, and restoration of precision flags.
- Job 368: frozen BPS leader, 1,200 physical environments, 20 simulated seconds,
  1,440,000 sampled 25-channel observations (held sensor packets are correlated,
  not independent measurements). All five pads encountered compressive load and
  shear. No material override; contact-accounting/overflow checks remained on.
  Artifact: `outputs/0_touch_smoke_368/calibration.json`.
- Job 370: actor mean/sigma/recurrent states and critic values match exactly on
  CPU across six SAPG coefficients and eight recurrent steps.
- Job 371: GPU FP32 equivalence passed; maximum actor-mean difference
  `5.0068e-6`, sigma difference zero, critic-value difference `1.9073e-6`.
  The initial comparison had correctly stopped before learning because cuDNN
  TF32 produced shape-dependent rounding (~`2.8e-4` action error). Verification
  now disables TF32 temporarily and restores the original runtime flags.
- Job 371: **30 actual SAPG updates / 576,000 transitions** at 1,200 envs.
  Actor and critic tactile input-column norms became `0.06911` / `0.23923`;
  new channels therefore received learning updates. New input-normalizer count
  reached `2,112,000` while old-feature count remained approximately `7.671e10`.
  No fabricated resets of the mature old-feature normalization.
- Job 372: real checkpoint resume retained actor and critic Adam state and frame
  576,000, cleared obsolete physical observations/recurrent states, then reached
  633,600 transitions at epoch 33. Its evaluator exposed an IsaacLab Optional
  field-loader issue; this was fixed before the main pilot could start. The
  original pending pilot/control jobs 374/375 were canceled without training.

## Strict physical reposing check (job 376)

Frozen deterministic zero-entropy leader after the small 33-update test. Seed
42; 1,200 envs covering every training object; 120 simulated seconds each;
saved domain randomization, original resets and strict `.01` tolerance. This
compares useful policy behavior, not just tensor shapes or a contact fixture.

| Metric | Final 8B BPS baseline | BPS + touch smoke checkpoint |
| --- | ---: | ---: |
| Completed episodes with any goal | 2,823 / 4,246 = 66.486% | 2,852 / 4,272 = 66.760% |
| Resolved goal attempts successful | 37,827 / 42,068 = 89.919% | 37,794 / 42,063 = 89.851% |
| Goals per simulated minute | 15.76125 | 15.74750 |
| Completed episodes ever lifted | 89.190% | 3,796 / 4,272 = 88.858% |
| Objects reaching at least one goal over the full window | 1,200 / 1,200 | 1,200 / 1,200 |

These results support preservation of baseline behavior, **not** a claim that
tactile improves performance. A matched-budget continuation and longer training
are required to address improvement. These are training objects, not a held-out
generalization benchmark; ongoing episodes/attempts are censored.

Evaluated checkpoint SHA256:
`584859f90b6b4725bec49afb557a00523f86ca0ab3752f844e50ff0bac41cbfb`.
Raw report: `outputs/validation_376/evaluation/evaluation.json` and accompanying
`per_object_counts.npz` / metadata. The main pilot starts from the original final
8B checkpoint, **not** from these short validation updates.

## Full-scale test and pilot

Job **380 passed** at **24,576 environments**: 30 actual SAPG updates,
11,796,480 transitions in 120.68 training seconds, averaging **97,753
transitions/s**. Device memory used was **32.75 GiB**, with **14.61 GiB free**;
PyTorch peak reserved memory was 17.33 GiB. Actor/critic tactile-column norms
reached `0.06551` / `0.20769`; tactile learning and independent normalization
were active. Report: `outputs/validation_380/validation.json` (`passed: true`).

USD contact reporting is enabled on five source bodies **before** cloning,
then its API and zero reporting threshold are read back on all 122,880 distal
bodies. All inherited the configuration; fallback overrides and material
bindings were both zero. Scene construction took approximately 266 seconds;
contact readback added approximately one second. This avoids redundant schema
writes/recomposition without changing materials or the force model. Earlier
pending pilot/control jobs were canceled; no main pilot ran under those IDs.

The only nonfinite scalar in the full-scale validation's TensorBoard log was
the initial `auxiliary_stats/off_on_grad_similarity` value (NaN). Its precise
cause was not established; it is not evidence of a nonfinite policy. Primary
loss/reward metrics and the checked model parameters remained finite.

## Main pilot launch verified

- **Job 381**: running the 500M-transition tactile continuation, one RTX 6000
  Ada, 24,576 environments. It starts from the original final 8B checkpoint,
  with normal plus shear observations and arm torques/fabrics/PCA disabled.
- **Job 382**: matched-budget BPS-only control, queued with `afterany:381`,
  so it cannot use another GPU concurrently with this pilot.
- Run directory: `outputs/0_bps128_touch_pilot_381`.
- Live W&B: [BPS-128 tactile pilot 381](https://wandb.ai/skvayzer/adept/runs/unique_id_0_bps128_touch_pilot_381).
- Source implementation at launch: commit `9bf3b20`, branch
  `feature/g1-bps128-touch-continuation`.

Post-launch checks verified the original source checkpoint/BPS bank hashes,
actor/critic dimensions 249/271, strict tolerance `.01`, unchanged materials,
and FP32 policy equivalence before learning. At the status check, the local
progress report had reached **79,036,416 additional transitions** (epoch 202),
at approximately **107,000–109,000 transitions/s** steady state. Periodic
checkpoints at 25,165,824, 50,331,648 and 75,497,472 transitions existed on disk.
GPU usage was 32.91 GiB with 14.46 GiB free.

The W&B API independently confirmed `state=running`,
`experiment_status=training`, `optimizer_updates_started=true`,
`warmstart_verified=true`, and synced reward/loss/tactile metrics through
77,856,768 additional transitions. Thus this is actual learning with live
logging, not just a submitted or initializing job. The small difference from
the local counter reflects asynchronous logging.

The launcher now opens W&B before scene construction and explicitly labels
`initializing_simulation` until training starts. Five approximately 100M-step
segments alternate with strict frozen-policy evaluations; the existing
regression gates remain enabled. Estimated total pilot duration is roughly
three hours including scene reconstruction and evaluations, followed by the
separate control run. This is a launch/validation record, **not a completed
500M pilot result or evidence of improvement from tactile sensing**.
