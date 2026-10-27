# Carry expert v2: design and sources (2026-10-27)

**Why v2.** Carry v1 (DexCompose stabiliser, runs 1549-1565) freezes M's grasp: the hold reward anchors
the object to its grasp-time position in the palm, the fingers move at most +-0.3 around the grasp
command, and the arm reference stays at the grasp-time angles (often extended over the table). It
reaches 61-65% on uniform starts but only 30% of grasped objects survive the 33 s Phase-2 route.
Drops are front-loaded (5.5-6.5 %/s in the first 3 s of walking, 0.3-0.45 %/s after 30 s; 2,701
walks, runs 1568-1577) and cluster by object (screwdriver 69%, eraser 90% vs hammer 28%), i.e. weak
grasps of hard objects. Held objects spin ~2.8 rad/s relative to the palm: every geom uses MuJoCo
condim 3 (no torsional friction).

## Sources (read in full, appendices included)

| Paper | arXiv | Used for |
| --- | --- | --- |
| MOVER: Learning Generalizable Humanoid Carrying from Human Motion | 2610.14561 | posture prior, two-phase relaxation (drop position tracking, keep orientation, raise object/contact rewards; 68.6 -> 89.5% off-nominal), contact-graph reward (w 2.0), start from real hand-off states |
| HOTICE: Whole-Body Humanoid Object Transportation in Cluttered Environments | 2609.25363 | split upper/lower reward streams with separate critics (+18.4 pts, largest gain), crediting relative to the body |
| OCLO: Dataset-Free Compliant Humanoid Loco-Manipulation with Dynamic Online Posture | 2610.05678 | end-effector force training (hardware EE pull 5/10 -> 10/10), reachability posture prior, gravity-aligned target frame, force-displaced compliant hand reference |
| Workhorse: Robust Whole-Body Humanoid Loco-Manipulation from Human Data | 2610.09117 | push protocol 40 N x 1 s pelvis every 5-10 s (77 -> 64% box transport), torso-frame targets + gravity (others collapse to 1-5%), reference drift augmentation |
| RobustDexGrasp | 2504.05287 | contact reward (fingertip 6.0 vs other links 1.5), slip penalties (-15 lin vel^2, -0.2 ang vel^2, -5 displacement), friction DR 0.5-0.9, evaluation force 2.5 N random point; NO force training or ramp in the paper |
| See to Reach, Feel to Grasp (Blind Grasp Reflex) | 2609.31323 | learned grasp score that gates a consequence (Spearman 0.77 with Ferrari-Canny), q - q_cmd history observation (96% vs 60%) |
| TacDexGrasp | 2603.07040 | "no translational slip -> no rotation" only for non-collinear contacts: a pad + palm hold needs torsional friction |
| Keep on Going (selective adversarial training) | 2507.08303 | targeted perturbations at weak moments (learned adversary over sensor/action/gain noise; attacks cluster at transitions) |
| Time Limits in RL (Pardo et al., ICML 2018) | 1712.00378 | bootstrap timeouts (our value_bootstrap is correct), no remaining-time observation; long episodes add low-hazard states, more walk onsets target our hazard |

## Spec (starting values to tune; everything opt-in, recorded in the run contract)

**Physics.** First an inference-only test (scripts/mj_contact_test.py): condim 4 / 6 with torsional
friction on robot, object and world geoms. Adopt only if held rotation and hazard drop; then all
policies need a short fine-tune.

Result (job 1581; carry v1 checkpoint 1565 unchanged, 1200 worlds x 60 s per setting, level 2):

| condim / torsional | walking drop hazard | success | held rotation (median, walking) |
| --- | --- | --- | --- |
| 3 / 0.005 (default) | 2.00 %/s | 69.4% | 3.34 rad/s |
| 4 / 0.005 | 1.56 %/s | 75.0% | 2.87 rad/s |
| 4 / 0.02 | 1.77 %/s | 71.3% | 2.95 rad/s |
| 6 / 0.02 | 1.62 %/s | 74.0% | 2.87 rad/s |

Torsional friction helps (hazard -22%) but the remaining ~2.9 rad/s wobble does not respond to a 4x
coefficient, so it is not twist about the contact normal (likely soft-contact compliance or wrist
vibration). Carry v2 trains with condim 4 / 0.005 (CONTACT_CONDIM=4, MJ_NJMAX 400); M and M-place
were trained under condim 3 and need a short fine-tune before they run under condim 4.

**Actions.**
* Fingers: q = clip(q_ref + d), d = 0.6 tanh(a) closing (a >= 0), 0.3 tanh(a) opening (Revo2: + closes).
* Right arm (7 joints) through the SONIC reference: q_arm = q_ref_arm(t) + D, D <- 0.8 D + 0.2 * 0.4 tanh(a_arm),
  a_arm = the first 7 latent outputs (the adapter head rows 7..63 and their log-std frozen); latent
  residual off. q_ref_arm(t): minimum-jerk from the grasp-time angles to a per-episode carry posture
  (dextrah_lab/wholebody/carry_posture.py: object at (0.30, -0.15, +0.05) m in the torso frame +- 5 cm,
  palm orientation kept close to the grasp) over the secure phase.

**Episode.** warm-up 0.5 s (no reward) -> secure 1.5 s (arm to carry posture, fingers may tighten,
table present) -> table removed (80%) -> walk: random clips with stops (1-3 s, p 0.2) and restarts
(many walk onsets), walk length U(10, L_max) -> stand 2 s -> timeout (bootstrapped). Curriculum on
L_max (20 -> 40 -> 80 s), speed (0.1 -> 0.3 m/s) and disturbance level k (0 -> 1).

**Reward (per step after warm-up).**

| Term | Formula | Weight |
| --- | --- | --- |
| hold | s_t (object supported by the hand: >= 1 hand-object contact, within 15 cm of the palm, no table/floor contact) | 1.0 |
| slip | -w_v tanh(\|v_rel\|/0.05) - w_w tanh(\|w_rel\|/0.5), relative to rigid palm motion at the COMs, 0.1 s average | 0.4, 0.3 |
| grasp quality | 0.5 C_opp + 0.5 min(n_c, 3)/3 (InterMimicGen opposition, contact count) | 0.3 |
| carry region | exp(-(d_box/0.10)^2), object distance to the box x [0.20, 0.40], y [-0.25, -0.05], z [-0.10, +0.15] (torso frame) | 0.2 (x2 in secure) |
| velocity tracking | as v1 | 0.5 / 0.25 |
| grip force | -sum max(0, f_i - 15 N)^2 | 0.002 |
| smoothness | -\|a_t - a_{t-1}\|^2 (fingers + arm) | 0.01 |
| arm deviation | -\|D\|^2 | 0.05 |
| table contact, SONIC command | as v1 | 0.5, 0.05 |
| drop / fall | terminal: unsupported for 0.3 s or object below 0.30 m / robot fall | -20 / -50 |

hold is off in the secure phase. gamma 0.995, rollout 32 (memory permitting), 100 critic-only epochs.
Initialising from a v1 carry checkpoint resets the value normalisers (the v1 ones carry a ~1e9 sample
count and cannot follow the new return scale: job 1583 value loss 0.2 -> 11 in 28 critic-only epochs).

**Disturbances (scaled by k).** Hand/object forces via xfrc_applied (OCLO: the largest robustness
factor), wrist pushes up to 25 N, pelvis pushes (Workhorse protocol 40 N x 1 s every 5-10 s as the
ceiling, lower in training: SONIC is frozen and recovers on its own; we score object security
through the recovery), sudden stops / reversals, object/pad friction x0.6-1.2, mass and COM as v1.

**Metrics.** Walking survival curve, early (< 5 s) vs late (> 10 s) hazard per second, secure-phase
failures, recovery (holding 3 s after a push), carry-region distance, grip force; per object family.
Evaluation: Phase-2 route (scripts/mj_table_to_table.py) and a fixed 2.5 N random-point object force.

**Deferred (v2.x).** Separate arm / finger critics (HOTICE); learned grasp score gating the walk start
(Blind Grasp Reflex: the score needs a consequence); force-displaced compliant arm reference (OCLO);
reference drift augmentation (Workhorse); targeted adversarial perturbations (Keep on Going);
MOVER-style later relaxation of the carry-posture term.
