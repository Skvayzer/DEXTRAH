# Neural simulator for dexterous carrying (feature/g1-neurosim, 2026-10-27)

Question: can a learned, differentiable model of hand-object contact (a "neural simulator") predict what
the Revo2 hand does to a held object well enough to supply policy gradients, as SuperMimic (arXiv
2610.28480) does for whole-body tracking? SuperMimic's own limitation: "No terrain or object
interaction ... is ongoing work"; pushes were dropped because "such pushes inject state discontinuities
that the deterministic world model cannot capture". Nobody has shown this for a dexterous hand.

## Stage 1 (this branch): hand-object contact model, go / no-go

Scope: the hand and the held object only. The body side (frozen SONIC + robot) is exogenous: the palm's
motion, gravity in the palm frame, the table and the external object force are logged inputs. Gradients
toward the fingers are therefore available in stage 1; gradients toward the arm channel need stage 2.

* Data (scripts/neurosim/collect_carry_transitions.py): the carry v2 expert in its training env
  (condim 4, curriculum level 2, training pushes), actions with N(0, 0.1^2) noise on the raw outputs
  (SuperMimic's collection noise) or sampled from the policy. All envs, every step; episode ends with
  their reason captured before the reset.
* Model (dextrah_lab/neurosim/model.py): deterministic MLP (3 x 1024, SiLU), state = object COM pose and
  velocity relative to the palm (palm frame; orientation 6D, changes as rotation vectors), hand joints
  (actuated + passive), plus log(1 + force) of the 5 pads, palm, object-table and object-floor contacts
  fed back as recurrent inputs (the grip state). Context: BPS shape (128), log mass, COM offset.
  Heads: state change, next forces, contact flags (> 0.5 N), table / floor contact, supported by the
  hand, lost (drop or failed grasp check; SuperMimic's termination head, BCE, calibrated: no class weight).
* Training (scripts/neurosim/train_world_model.py): SuperMimic eq. 1, k = 8 autoregressive windows
  seeded at logged states with logged actions / inputs, losses until the episode ends inside the window
  (the ending transition still trains the lost head); Adam 2e-4, clip 1.0, cosine decay. Velocities are
  down-weighted (0.25): 95% of the measured in-hand angular speed is contact vibration (job 1588), which a
  deterministic model can only predict in the mean.
* Evaluation (dextrah_lab/neurosim/evaluate.py) on objects never seen in training (10% of assets):
  open-loop error at 1 / 8 / 32 steps vs persistence (overall and on windows where the object moves
  >= 5 mm or >= 5 deg), contact / support AUC, loss of the object within 0.5 s (AUC on all held-out loss
  events + random starts; Brier vs climatology and binned reliability at the natural rate).

Go criteria (set before the run):
1. Moving windows, 8 steps: object position and rotation errors at most half of persistence's (median).
2. Loss within 0.5 s: AUC >= 0.8 on held-out objects, Brier better than climatology.
3. Pad / palm contact AUC >= 0.9 at 8 steps.
No-go on 2 ends the gradient idea for drops (the survival term would steer toward a wrong signal); a
pass on 1 and 3 alone still makes the model useful as a fast simulator for analysis.

## Stage 2 (after a go)

* Body model: SONIC + robot as the environment (state includes SONIC's 10-frame proprioception and the
  walking reference); then the arm channel gets gradients through palm motion.
* Gradient check: model d(return)/d(action) vs finite differences in MuJoCo Warp at carry states.
* Hybrid fine-tune of C at equal GPU-hours: SAPG alone vs SAPG + lambda * world-model gradient (first-order,
  H = 16-32, critic bootstrap, survival weight from the lost head), lambda gated by recent model error.
  Pushes enter as known inputs (the logged object force here; pelvis / wrist pushes through palm motion).

## Gradient check, corrected (2026-10-28): the model is action-blind; no-go for stage 2 as planned

The first checks (stage15/gradient_check_v2 / _noise / _bundled, perturbation 0.6 s after the restore) measured
a "simulator noise floor" as large as the perturbation effect. They do not test the model:

* the run config's per-step domain randomisation was on (each world draws a 0-2-step-old action every step,
  source_actions.delayed_actions with the global RNG; delayed / noised object state in the actor observation;
  observation delays; joint-velocity noise) and 20% of worlds kept the table;
* with all of that off (build_carry_env(no_randomization=True), table_keep_probability 0), restored copies are
  bit-identical, differ at the float level after one control step (MuJoCo Warp is not bitwise reproducible
  across worlds), and the hand-object system amplifies that ~x200 per 0.1 s, also with shared expert actions
  (scripts/neurosim/replica_divergence.py; object spread median 0.004 mm at 0.1 s, 0.7 mm at 0.2 s, 5 mm /
  11 deg at 0.6 s; outputs/neurosim/stage15/replicas/*_norand.json). At 0.6 s copies already differ by mm.

v5 (outputs/neurosim/stage15/gradient_check_v5_h0p1 / h0p2 / h0p5.json): randomisation off, perturbation 2
control steps after the restore, 256 states, each finger +-0.05 / +-0.1.

| Horizon | Outcome | Sim effect | Noise floor | Antisymmetry | Model / sim magnitude | Cosine | Sign agreement |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.1 s | grip force (log) | 0.11 | 0.0008 | -0.28 | 0.06 | 0.10 | 51% |
| 0.1 s | rotation | 0.15 deg | 0.004 deg | -0.25 | 0.04 | 0.02 | 51% |
| 0.1 s | displacement | 0.068 mm | 0.002 mm | -0.15 | 0.05 | -0.01 | 50% |
| 0.2 s | grip force (log) | 0.48 | 0.22 | -0.44 | 0.05 | 0.04 | 50% |
| 0.5 s | grip force (log) | 0.66 | 0.66 | -0.47 | 0.06 | -0.03 | 50% |

* Finite differences in the simulator are clean only up to ~0.1 s; at 0.2 s the noise is half the effect and
  the effect no longer grows with the perturbation; at 0.5 s it is all noise.
* At 0.1 s the response to +-0.05-0.1 is not linear (antisymmetry -0.15 to -0.28; 2x the perturbation, 1.6x
  the effect). Smaller perturbations at 0.1 s: not yet run.
* The world model's action effects are 4-6% of the simulator's and uncorrelated with them at every horizon: it
  predicts from state and body motion, not from the finger targets.
* Stage 2 as planned (first-order gradients through this model, H = 16-32) is a no-go. If continued: paired
  counterfactual data (identical restored copies, different finger commands) supervising the predicted
  difference over <= 0.1 s, re-tested at +-0.01-0.02; any gradient objective limited to ~0.1 s horizons with a
  critic beyond, or smoothed (expected) gradients.

Why action-blind (scripts/neurosim/action_sensitivity.py, stage15/action_sensitivity.json, 117k held-out
transitions of go_nogo_v2): the model's one-step contact-force Jacobian w.r.t. the finger targets has the data's
pattern and size (median Frobenius 15.8 vs 7.8 for ridge on the same inputs), but the targets are predictable from
the model inputs with R^2 0.94-1.00 for 5 of 6 joints, add 0.003 to the force R^2, and change by <= 0.012 per
step in 90% of steps. The model learned action effects only along the expert's closed-loop variations; the
check's sustained +-0.05-0.1 offsets lie far outside them. Learning the response needs independent finger-command
variation (paired counterfactuals).

Paired counterfactuals (job 1620; outputs/neurosim/counterfactual/): 61,440 groups x 8 bit-identical copies
(19,762 holding states, randomisation off, the model's held-out objects excluded), copies 1-7 with finger-command
perturbations (constant / pulse / noise) for 8 steps. Fine-tune of combined_single, 6000 iterations, difference
loss weight 1 (finetune_v1, W&B vwb7jes7): drop prediction kept (phase-matched AUC 0.982 -> 0.987), open loop about
unchanged. Held-out pair objects (pair_diagnostic.json): cosine of predicted vs simulated copy - baseline
difference, base -> fine-tuned: hand joint positions 0.54 -> 0.69, object position 0.09 -> 0.24, object rotation
0.06 -> 0.10, object velocity 0.07 -> 0.11, pad forces 0.07 -> 0.09; the model's error on the unperturbed rollout
is 1-4x the size of the object / contact effects. Held-out gradient check at 0.1 s (gradcheck_*_h0p1.json) stays at
chance (cosine -0.03-0.05, sign agreement 48-53%). Within the ~0.1 s where the simulator's action effects are
defined they are at or below a learned model's error floor; beyond it the simulator is chaotic. No usable
first-order gradient for a SuperMimic-style objective on this task; the model remains a drop-risk predictor.
