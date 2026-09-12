# Route 2: SAPG skill transfer into a trainable SONIC body controller

User decision, 2026-09-12. **This supersedes the frozen-decoder / 64-value
latent-adapter route.** Historical adapter experiments remain reproducible but
are not prerequisites for the new route.

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
  Before learning, every body output equals the pretrained controller.
- A continuous six-output finger head shares task/body features. The resulting
  policy has **35 actions: 29 body + 6 right fingers**. All 29 body joints,
  including the right arm, have exactly one target writer. The old SAPG actor
  supplies supervision; it does not overwrite student arm actions.
- Both hands and the floating body are physically present. Left fingers stay
  neutral initially. Later bimanual policy: 29 + 12 = 41 outputs.
- No fabrics, PCA, welded base, gravity-free arm, or removed robot branches.

The initial model is

    h_t = GRU(task_observation_t, h_(t-1))
    z_stand = frozen_SONIC_encoder_and_FSQ(nominal_standing_reference)
    x_t = concat(z_stand, actual_body_history_t)
    v_t = SiLU(W0_SONIC x_t + b0_SONIC + Wtask h_t)
    a_body = trainable_remaining_SONIC_decoder(v_t)
    a_hand = continuous_finger_head(body_features, h_t)

`Wtask` starts at zero. All copied body-decoder layers may learn. This is not
an old SAPG arm controller running alongside an independent SONIC controller.

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

Status at this decision: architecture implementation starting; no trained
route-2 student or full-body SAPG run yet.
