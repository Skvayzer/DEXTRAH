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
