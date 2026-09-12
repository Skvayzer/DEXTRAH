# Route 2: SAPG skill transfer into a trainable SONIC body controller

User decision, 2026-09-12. **This supersedes the frozen-decoder / 64-value
latent-adapter route.** Historical adapter experiments remain reproducible but
are not prerequisites for the new route.

## Current training configuration (supersedes historical development details below)

On September 12, after viewing the recorded attempt, the user authorized
continuous full-body SAPG training. Slurm **513** was submitted with **12,288
environments**, six groups of 2,048, one RTX 6000 Ada and a 48-hour allocation.
Startup/optimizer verification is recorded in the run's `progress.json`;
submission by itself is not evidence of successful training.

- The teacher is now the **completed BPS-128 + 70 Hz touch run 383**, checkpoint
  `complete_17364025344.pth`, SHA-256
  `ed9b26a7ba335c25f15df6623c8e0a97f6d5b3285721ae7f8263f82d6227d2cd`.
  This is an explicit reuse choice, not a tactile-ablation winner.
- Initialize the body/task projection from **touch-derived bootstrap 503**,
  `best_student.pt`, SHA-256
  `05a1f2af6dd0781bd37b548ec8671e02c7ab20acc9344f64ca0b92451ff08302`.
  It uses four original-rate object clips and a kinematic body lift plus
  standing rehearsal; offline imitation is not whole-body manipulation success.
- Copy the source LSTM, MLP, all six exploration embeddings and six finger
  output rows. The SONIC body decoder, task projection and copied task/finger
  modules all fine-tune. The SONIC reference encoder remains frozen. Fresh
  actor/critic optimizers are necessary for the 13-to-35-action migration.
- **Use the actual original Play2Perfect task**, inherited through BPS/touch,
  not the earlier `LiveClipTask`. Original rewards, goal/keypoint logic,
  tolerance, reset noise, delay/DR, 1,200-object bank and finger EMA/PD targets
  are retained. The full floating robot has 51 joints and 68 rigid bodies.
- **120 Hz physics, 60 Hz policy, 70 Hz tactile acquisition/publication**.
  Actual body history is resampled causally at SONIC's 20 ms history spacing.
  Right fingers use the original independent follower-target drives, not the
  native mimic constraints tested in older probes. No fabrics/PCA or arm-torque
  observation is added.
- Whole-body physics is necessarily different from the fixed-arm teacher:
  gravity, feet, SONIC body drives and robot-fall termination are added. The
  old fixed torso proxy is removed; with self-collision still OFF, this does
  **not preserve its arm-to-torso collision pairs**. Dynamic torso geometry
  alone does not restore internal collision avoidance. This remains a known
  fidelity limitation, not a claim of collision-identical transfer.
- Ordinary falls are episode failures. Recorded attempt 509 additionally
  showed a finite index-joint speed outlier around 1,254 rad/s. Training now
  terminates **only the affected environment** above 1,000 rad/s, logs this
  separately, and keeps the original reward function. It does not clamp or
  repair physical states. Non-finite states still abort before rewards or
  terminal critic observations. Diagnostic abort mode remains available.
  Resetting these cases enables an exploratory run; it does not prove that
  their underlying numerical cause is fixed.
- Actual six-group SAPG, source learning rates and update settings; best
  checkpoint by leader training return, plus atomic latest checkpoints every
  64 updates. No periodic evaluation orchestration, no frame/epoch cap;
  Slurm's 48-hour wall limit remains, with an advance checkpoint signal.
- Source snapshot **d53e533**; CPU job **512: 44 tests passed**, including
  source finger delay/EMA/follower parity, recurrent/normalizer checks and
  per-environment numerical termination/non-finite rejection. These tests do
  not establish learned success or long-run simulator stability.

Output: `outputs/0_sonic_sapg_touch_513` on the workstation.
[Online W&B run](https://wandb.ai/skvayzer/adept/runs/unique_id_0_sonic_sapg_touch_513).

The sections below describe the original route-2 design and earlier experiments;
their 50/200 Hz clocks, BPS-only teacher and native-coupling assumptions are
historical, not the current training defaults.

## Controller

- Initialize the 29-output dynamic body decoder from the pinned SONIC bundle.
  Its original weights are **trainable**, in a new student checkpoint.
- Preserve the reference encoder and FSQ initially, using a nominal standing
  reference. Task information enters the **body decoder's hidden layer**, not
  only its quantized motion latent. No wrist-space retargeting or per-frame
  latent optimization is required for this architecture.
- A recurrent task encoder consumes current manipulation observations (object,
  goal, BPS-128, hand state; optional 25-channel right tactile packet). A zero-
  initialized projection adds its features to SONIC's first body hidden layer.
  Before learning, every body output equals the pretrained controller. The
  current variant copies **SAPG's trained 1024-unit LSTM, LayerNorm, 512-feature
  MLP output and six-channel finger head**. It does not relearn fingers from
  random weights. The copied task modules are frozen during the first body
  bootstrap and can later fine-tune with a smaller learning rate.
- A continuous six-output finger head reads the shared SAPG task features. The resulting
  policy has **35 actions: 29 body + 6 right fingers**. All 29 body joints,
  including the right arm, have exactly one target writer. The old SAPG actor
  supplies supervision; it does not overwrite student arm actions.
- Both hands and the floating body are physically present. Left fingers stay
  neutral initially. Later bimanual policy: 29 + 12 = 41 outputs.
- No fabrics, PCA, welded base, gravity-free arm, or removed robot branches.

The initial model is

    features_t, memory_t = copied_SAPG_LSTM_and_MLP(task_observation_t, memory_(t-1))
    z_stand = frozen_SONIC_encoder_and_FSQ(nominal_standing_reference)
    x_t = concat(z_stand, actual_body_history_t)
    v_t = SiLU(W0_SONIC x_t + b0_SONIC + task_active * Wtask features_t)
    a_body = trainable_remaining_SONIC_decoder(v_t)
    a_hand = copied_continuous_finger_head(features_t)

`Wtask` starts at zero. All copied body-decoder layers may learn. This is not
an old SAPG arm controller running alongside an independent SONIC controller.
`task_active=False` explicitly disables task conditioning for standing-only
rehearsal/probes. A mean-valued task observation alone is not a reliable mode
switch: a recurrent network can evolve even on a constant input. This gate
does not freeze the legs during manipulation.

## Reusing the trained skill

The immutable SAPG teacher has 7 **delta** arm actions and 6 absolute finger
actions. Convert its arm action through the original action pipeline to the
**next applied physical joint targets**. Never compare its 13 raw outputs to
the student's 35 outputs, and never label previous applied targets as next
actions. Preserve the source normalization, recurrent history and leader
coefficient in teacher captures. New architecture means a fresh optimizer.

Initial supervised losses:

    L = L_right_arm_targets + L_right_finger_actions
        + lambda_stand L_other_22_body_commands
        + lambda_rehearsal L_all_29_commands_on_standing_states

SONIC retention is a **soft loss**, not a physical joint constraint. RL may
move the waist, legs and other arm to balance. Never apply SAPG supervision to
legs it did not control. Reject reset transitions and mask invalid samples.
Report arm error in radians and finger error in normalized action units.

A preliminary kinematic demonstration lift can populate nominal standing legs
and measured SAPG arm motion. It is explicitly an **offline bootstrap**, not
real full-body rollout data or a manipulation-success result. Its teacher
task observations stay in the source task frame; online observations must
use the same declared torso-relative frame. No future demonstrated motion is
an input to the deployed student. Frozen SONIC standing trajectories provide
separate real-physics rehearsal samples. Do not pair unrelated teacher object
states with real student body states and call that on-policy distillation.

## Execution gates

1. Exact initial SONIC equivalence; nonzero decoder gradients; frozen source
   checksums; recurrent reset and causal action tests; save a new student.
2. Supervised bootstrap from saved, checkpoint-provenanced SAPG data. Split by
   contiguous trajectory/episode, not random neighboring frames. Record both
   training and held-out losses and standing-retention errors. A low offline
   loss is not a grasping success rate.
3. New full-body object/table task with online teacher-compatible task frames,
   BPS geometry identity, right-pad normal/shear/validity/age observations.
   Physics 200 Hz, controller 50 Hz, tactile acquisition/publication 70 Hz.
   At this physics rate, actual sample intervals alternate between 10 and
   15 ms; packet timestamps/age represent the real sampling times.
4. Closed-loop loaded-contact validation using the tested palm decomposition,
   32 position iterations, native finger couplings and full self-contact.
   Check falls, feet, coupling error, arm tracking, object load transfer and
   real tactile signals. Do not auto-pass on a Slurm COMPLETED status.
5. Add SAPG updates to this same 35-action student, retaining annealed teacher
   losses and measured balance safeguards. Preserve the existing task rewards
   where transferable; document additions. Benchmark physics **plus actual
   rollouts/backward/optimizer memory**, then size a single Ada 6000 run with
   headroom. There is no requirement to retain 24k environments.
6. Save best checkpoints with W&B and explicit success denominators. Only
   after right-hand standing manipulation works, transfer to the second hand.

The development teacher is the saved 8B BPS SAPG checkpoint, not a selected
tactile winner. Comparison job 472 failed on a snapshot-relative URDF path
after the nine BPS cases. Tactile selection remains incomplete; do not claim
an ablation winner or silently substitute a source checkpoint.

Source SONIC SHA-256:
`62f3e336cc11cbfb7517fee1f053ed92c81d44accd3757faa0febf421d0a2cd4`

Source SAPG SHA-256:
`53c4b009cdd341b4a0e007111c4009da57898dcb00d264c12c81b18817637fec`

## Implementation evidence

- Commits on `feature/g1-wholebody-sapg`; source checkpoints unmodified.
- Real SONIC validation **475**, repeated in **481**: initial body output
  difference **0.0**. All **10,184,221** dynamic decoder parameters trainable;
  gradients reach its first and final layers. The reference encoder/FSQ remain
  frozen, not the body decoder.
- The first fresh-GRU bootstrap (**476**) reduced arm error but transferred
  fingers poorly. Preserved as a diagnostic, not a manipulation result.
- Pretrained-feature validation: original 60 Hz captured states/observations,
  **106 samples**, copied raw finger means exactly match the strict-loaded
  SAPG actor. Captured *player-clipped* commands differ by at most **2.50e-6**.
  SAPG coefficient IDs run from 50 to 0; the zero-entropy leader is the **last**
  embedding row. The source arm output layer is not used to command the robot.
- Corrected supervised run **479** completed 1,200 updates; best checkpoint
  selected at update **400**. Temporal holdout arm RMSE **0.07399 rad** versus
  **1.49442 rad** initially; finger RMSE **0.11389** in normalized action units;
  standing-rehearsal RMSE **0.001341 rad**. Finger error includes running the
  copied 60 Hz memory on a 50 Hz sampled stream; exact-rate equivalence is a
  different test. These are offline errors, **not grasping success rates**.
- Dataset: four 30 s source clips (brush, eraser, hammer, spatula), up to 7,200
  original samples; reset-separated temporal splits with a 50-step gap and
  burn-in. They come from the 1,200-object teacher bank but contain **four
  objects**, not 1,200 demonstrations. Separate real standing rehearsal comes
  from probe 470. Manipulation body histories are kinematic lifts of recorded
  arm states, not real full-body rollouts.
- W&B: [supervised run 479](https://wandb.ai/skvayzer/adept/runs/6x2x0xct).
  New files: `outputs/sonic_distillation_479/{initial,best,last}_student.pt`.
- Added strict student loading, standing regression, and a live one-object/
  table probe using current simulator observations and no future replay.
  It reconstructs merged palm/tip frames with locked joint angles, retains BPS,
  commands all 29 body joints through the student, and physically actuates the
  six right-hand motors through native finger couplings.
- Standing probe **480** ended in a native segmentation fault after its last
  upright sample at step 550; no final validation artifact. **Not a pass.**
  Added intermediate finite-state/progress snapshots; retry **482 passed**:
  20 s, four floating robots, minimum pelvis 0.75390 m, maximum native coupling
  error 0.01273 rad, final foot load / weight 0.99957–1.00012. This was standing
  only, not loaded manipulation. The one-off native crash remains unexplained.

### Live manipulation integration continuation

- The first live task found **four left fingertips initially inside the table**
  at SONIC's nominal rest pose. Source SAPG removed the left arm, so its
  successful recordings could not expose this. With the table frame's half
  extents `(0.2375, 0.2, 0.15)` m, left index/middle/ring/thumb tip positions
  were all inside the volume. Do not disable hand/table collisions to hide it.
- Set an explicit **left shoulder-roll rest reference of 0.6 rad**, versus
  SONIC's nominal 0.2 rad. Update the kinematic bootstrap, previous-action
  history, initial physical pose and reference encoder consistently. All 29
  body commands still come from the student. No left-arm target overwrite or
  welded joint. Initial tip/wrist point clearances are 64–155 mm; this is a
  point preflight, not a complete CAD collision proof.
- Restored the source finger **0.02-rad command margin** and EMA, with
  `alpha_50 = 1 - (1 - 0.1)^(60/50) = 0.11876647`. This preserves the source
  filter's time constant. The stochastic action delay remains off in the
  diagnostic. These are source actuator-pipeline settings, not fabrics/PCA.
- New bootstrap **485** completed 1,200 updates; best at 400. Held-out arm
  RMSE **0.072666 rad**, finger RMSE **0.113889**, other-body retention RMSE
  **0.049654 rad**, standing rehearsal RMSE **0.001438 rad**. This still uses
  four objects and the original 8B teacher, not a newly selected tactile winner.
  [W&B 485](https://wandb.ai/skvayzer/adept/runs/3kzird8t);
  `outputs/sonic_distillation_485/best_student.pt`.
- Live first-state palm and five-tip relative positions agree with source
  observations to about **3.2 micrometres**; joint positions, previous targets,
  object size and BPS agree exactly. Source qd reset noise and object noise are
  intentionally absent in this probe. Also preserved the original object
  quaternion hemisphere: matrix conversion can flip q/-q without changing
  physical orientation but does change a neural observation.
- **487 failed at control step 11 (0.22 s)**: extreme passive right-finger
  velocities preceded the rejected body command. Saved current state/actions
  in `rejected_action.npz`; do not interpret the post-explosion policy output
  as evidence of normal learned behavior. Native coupling passed slow free-
  space cycles, but that did **not** establish contact/fast-action stability.
- **489** used 400 Hz physics with unchanged 50 Hz control. It avoided the
  immediate velocity explosion, but ended in a native segmentation fault
  after the last saved state at 1.02 s. This is not a numerical-stability pass
  or a change to the agreed training defaults.
- **490**, one environment with physics replication disabled at 400 Hz,
  reached 3.92 s before the normalized-body-action guard rejected a right
  wrist-yaw command of 20.016. Joint speeds were finite (maximum 4.95 rad/s),
  unlike 487's finger explosion. The pelvis had drifted and tilted: this is
  evidence of a closed-loop control-transfer gap, not proof of another
  numerical explosion. A short one-environment result cannot establish that
  replication caused the earlier native crashes.
- Probe state checking now also runs after each physics substep. Catastrophic
  velocities/nonfinite states abort and save `rejected_physics.npz` before
  another actor invocation. No positions or velocities are reset/clamped to
  hide the failure. Four-environment repeat **491** reached 3.94 s with finite
  joint velocities (maximum 5.31 rad/s at rejection), then stopped on a body
  command of magnitude 20.182. Neither this nor 490 passed the requested 20 s
  probe. These short repeats distinguish ordinary control drift from the
  previous huge finger velocities; they do not prove long-run physics stability
  or identify the root cause of the native crashes. No GPU job remains active
  after these diagnostics.

Remaining before full-body SAPG: finish stable live loaded-contact validation;
port original action delay and observation randomization;
integrate real full-body tactile including self-contact at 70 Hz; add reset/
goal/reward/termination logic and the actual 35-action mixed-group SAPG actor,
critic and optimizer; benchmark that complete workload. The live diagnostic
is deliberately **not** labeled a training-ready task. No full-body RL updates
or standing-manipulation success have been claimed.

Do **not** require a successful grasp or perfect standing manipulation from
the bootstrap student before permitting RL: learning that behavior is the
purpose of full-body fine-tuning. The pre-RL physics gate concerns valid
contacts/couplings and recoverable finite states. Falls/control failures must
be handled by explicit task terminations and resets, not confused with a
simulator crash or hidden by altering the robot's physical constraints.
