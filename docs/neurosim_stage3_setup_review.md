# Neural simulator for dexterous manipulation: setup for review

Oct 29, 2026 · @Constantine

We want a differentiable model of hand–object contact for the Unitree G1 humanoid's Revo2 hand, accurate enough to give useful policy gradients for precise manipulation, first for setting objects down upright. A black-box learned world model (stages 1–1.5) predicts drops well but is blind to the effect of finger commands, so its gradients are at chance. The replacement is a physics-structured contact model (stage 3): designed, with only its reference simulator built so far. Every detail is below; the questions for the reviewer are in the second-to-last section.

## 1. Context

The robot is a Unitree G1 humanoid with a BrainCo Revo2 right hand, trained in simulation only. Policies sit on top of a frozen, pretrained whole-body controller and learn only the manipulation layer.

**Robot and control stack**

- G1 body: 29 joints, driven by SONIC, a frozen pretrained whole-body tracking controller. Our policies never change SONIC; they feed it a 64-D latent (a residual on its reference).
- Revo2 right hand: 11 joints (thumb metacarpal, proximal, distal; proximal and distal for index, middle, ring and pinky). The policy sends 6 normalised finger commands in \[−1, 1\] (thumb metacarpal, thumb proximal, 4 finger proximals); the distal joints follow by fixed coupling: thumb distal = 1.0 × thumb proximal, finger distal = 1.155 × finger proximal.
- Each hand joint is a velocity servo emulating the PhysX drive (kp 1200, kd 25): commanded velocity = clip((kp/kv)·(q\* − q), ±vmax), torque = kv·(commanded − q̇) with kv = effort/vmax, so a blocked joint pushes with min(kp·|q\* − q|, effort). Joint armature 1e-3 kg·m². Details per joint in section 5.
- Control rate 60 Hz.

**Policies in the project**

- M, manipulation expert: grasps an object from a table and moves it to 6-D goal poses. SAPG (split-and-aggregate PPO, 6 blocks), recurrent; outputs the 64-D SONIC latent plus 6 finger commands. Inputs include a BPS-128 shape code, simulated 70 Hz fingertip touch, vectors from 17 hand-surface points to the object surface, and a 54-D table block. Trained with success-guided start sampling (SGS). Current best: about 71% success over table tops 0.70–0.90 m.
- C, carry expert: walks the robot between tables holding the object.
- P, placing expert (planned): sets objects down at a target, including solid cylinders (radius 2.5–4.5 cm, height 8–20 cm, mass 0.1–0.5 kg) that must end upright (tilt below 10°) within a 5 cm spot.
- Objects in M's training: 1,200 procedurally generated tools (hammer, screwdriver, marker, spatula, eraser, brush), built from boxes and capsules.

**Training physics (mjlab on MuJoCo Warp, GPU, float32, about 27,000 parallel worlds)**

| Setting | Value |
| --- | --- |
| Physics step / control step | 1/120 s / 1/60 s (2 substeps) |
| Integrator, solver | implicitfast, Newton |
| Friction cone, impratio | elliptic, 10 |
| Solver iterations / line-search iterations | 20 / 20 |
| Contact dimension | condim 4 (sliding + torsional friction) |
| Friction (sliding, torsional, rolling) | 0.5, 0.005, 0.0001; fingertip markers 1.0 sliding |
| solref / solimp | (0.02, 1) / (0.9, 0.95, 0.001, 0.5, 2) |
| Collision shapes | convex hulls, at most 64 vertices per mesh; robot–robot collisions off |
| Known simulator patch | vendored MuJoCo Warp fix for an elliptic-cone Hessian that produced NaNs |

**Why a neural simulator.** Precise placing has tiny success regions (upright, a 5 cm spot, gentle touchdown) that sampling-based RL reaches rarely. A model whose gradients point toward them could speed this up. SuperMimic (arXiv 2610.28480) trains whole-body tracking with gradients through a learned world model, but states that object interaction is outside its scope; we know of no demonstration for a dexterous hand.

## 2. Stages 1–1.5: the black-box world model

A SuperMimic-style deterministic network was trained to imitate MuJoCo Warp for the hand and the object it holds during carrying. It became a good drop-risk predictor but did not predict in-hand motion well enough, and section 3 shows why it cannot give action gradients.

**Scope.** Hand and held object only. The body side is exogenous: palm motion, gravity in the palm frame, the table and any external object force are logged inputs. Gradients therefore reach the finger commands, not the arm.

**Data.** About 11M transitions: 2,048 parallel worlds × 90 s at 60 Hz, every world and every step, from the carry expert C (run 1587, epoch \~660) in its own training environment (condim 4, curriculum level 2, training pushes). Actions carry N(0, 0.1²) noise on the raw policy outputs (SuperMimic's collection noise). A later collection (v2) added 30% perturbed worlds: executed-command noise σ 0.3 (realised \~0.24 within the action bounds), loosened grips and extra object pushes, for 2,615 drops.

**Model.** MLP 3 × 1024, SiLU, inputs normalised per feature.

- State: object centre-of-mass pose and velocity in the palm frame (orientation as the first two rotation-matrix columns, changes predicted as rotation vectors), hand joint positions (actuated and passive), and log(1 + force) of the 5 fingertip pads, the palm, object–table and object–floor contacts, fed back as recurrent inputs (the grip state).
- Action: 6 finger joint targets. Exogenous input (22-D, later 26-D using pre-step quantities only): palm linear and angular velocity, gravity in the palm frame, table position relative to the palm, a parked flag, external object force.
- Context (64-D projection): BPS-128 shape code, log mass, centre-of-mass offset.
- Heads: state change, next forces, contact flags (> 0.5 N), table / floor contact, hand support, and "lost" (drop or failed grasp check; binary cross-entropy, no class weight, so calibrated).

**Training.** SuperMimic's eq. 1: autoregressive windows of k = 8 steps (later 16) seeded at logged states with logged actions and inputs, losses at every step until the episode ends inside the window. Adam, learning rate 2e-4, gradient clip 1.0, cosine decay. Velocities weighted 0.25 because 95% of the measured in-hand angular speed is contact vibration, which a deterministic model can only predict on average. Later variants: chordal orientation loss, low-pass twist state (τ 0.3 s), pre-drop oversampling, a 5-member probabilistic ensemble.

**Evaluation.** On objects never seen in training (10% of assets, 92 objects), open loop from logged states with logged actions and inputs.

Go criteria were fixed before the runs: (1) on moving windows, 8-step object position and rotation errors at most half of persistence's; (2) loss of the object within 0.5 s predicted with AUC ≥ 0.8 and Brier better than climatology; (3) pad and palm contact AUC ≥ 0.9 at 8 steps.

| Criterion | v2 result | Verdict |
| --- | --- | --- |
| 1, smoothed pose, drifting windows | 8 steps: 2.80 vs 3.43 mm (−18%), 6.1 vs 7.7° (−21%); 16 steps: −12% / −19%; 32 steps: −2% / −13% | Fail |
| 2, no future information | AUC 0.948 \[0.937, 0.960\] phase-matched vs 0.886 for logistic regression; Brier 0.0068 vs 0.0075 climatology | Pass, narrowly |
| 3, contacts at 8 steps | Pads 0.92 / 0.92 / 0.90 / 0.95, palm 0.92, support 0.91 | Pass |

Drop prediction, 0.5 s horizon (first model; AUC with 95% interval):

| Predictor | Random negatives (19,631) | Phase-matched negatives (777) |
| --- | --- | --- |
| Model, logged inputs over the horizon | 0.991 \[0.984, 0.996\] | 0.986 \[0.977, 0.992\] |
| Model, only what is known at the window start | 0.960 \[0.946, 0.972\] | 0.944 \[0.928, 0.960\] |
| Logistic regression on current contacts, forces, support, table | 0.911 \[0.886, 0.933\] | 0.895 \[0.867, 0.919\] |

- About a third of 8-step windows move 5 mm or 5° even after smoothing, and the model removes only a fifth of that error: in-hand motion is largely unpredictable from the state it sees.
- Held-out grip-force error swings between evaluations (1.4–2.9) while the training loss falls: forces overfit the training objects.
- Conclusion of stage 1: usable as a drop-risk predictor; imagined rollouts of 16–32 steps would be dominated by model error.

## 3. What the simulator itself showed

Finite differences in MuJoCo Warp are meaningful only up to about 0.1 s of contact rollout, and within that window the black-box model's response to finger commands is 4–6% of the simulator's and uncorrelated with it. Paired counterfactual data did not fix this. These two findings shape stage 3.

**Identical worlds do not stay identical.** With per-step domain randomisation off, restored copies of a holding state are bit-identical in state and observation. After one control step they differ at the float level (MuJoCo Warp is not bitwise reproducible across worlds), and the hand–object system amplifies the difference about 200× per 0.1 s, even when every copy executes the same expert actions. Object spread across copies: median 0.004 mm at 0.1 s, 0.7 mm at 0.2 s, 5 mm and 11° at 0.6 s. An earlier check had been invalidated by randomisation left on (each world drew a 0–2-step-old action every step, with delayed and noised observations).

**Gradient check (v5).** 256 holding states, randomisation off, each finger command perturbed by ±0.05 or ±0.1 starting 2 control steps after the restore. "Noise floor" is the spread between unperturbed copies; "antisymmetry" compares the +δ and −δ effects (−1 = linear).

| Horizon | Outcome | Simulator effect | Noise floor | Antisymmetry | Model / simulator magnitude | Cosine | Sign agreement |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.1 s | grip force (log) | 0.11 | 0.0008 | −0.28 | 0.06 | 0.10 | 51% |
| 0.1 s | rotation | 0.15° | 0.004° | −0.25 | 0.04 | 0.02 | 51% |
| 0.1 s | displacement | 0.068 mm | 0.002 mm | −0.15 | 0.05 | −0.01 | 50% |
| 0.2 s | grip force (log) | 0.48 | 0.22 | −0.44 | 0.05 | 0.04 | 50% |
| 0.5 s | grip force (log) | 0.66 | 0.66 | −0.47 | 0.06 | −0.03 | 50% |

- Simulator finite differences are clean only up to \~0.1 s; at 0.2 s the noise is half the effect; at 0.5 s it is all noise.
- Even at 0.1 s the response is not linear: doubling the perturbation gives 1.6× the effect.

**Why the model is action-blind (117k held-out transitions).** At one step the model does respond to the finger targets, in the data's pattern: its contact-force Jacobian has median Frobenius norm 15.8, against 7.8 for a ridge regression on the same inputs. But in the expert's data the targets carry almost no independent variation. The model's other inputs predict them with R² 0.94–1.00 for 5 of 6 joints, they add 0.003 to the force R², and 90% of steps change a target by at most 0.012. The check's sustained ±0.05–0.1 offsets lie far outside anything the expert's closed loop produced. This is an identifiability problem: the data cannot separate what an action does from what the state implies.

**Paired counterfactuals.** 61,440 groups × 8 bit-identical copies of 19,762 holding states (randomisation off, the model's held-out objects excluded); copies 1–7 received finger-command perturbations (constant, pulse or noise) for 8 steps. The model was fine-tuned for 6,000 iterations with an added loss on the predicted copy-minus-baseline difference. Cosine between predicted and simulated differences on held-out objects:

| Component | Base model: cosine | Fine-tuned: cosine | Fine-tuned: magnitude ratio | Model error ÷ effect (base → fine-tuned) |
| --- | --- | --- | --- | --- |
| Hand joint positions | 0.54 | 0.69 | 0.69 | 2.1 → 1.9 |
| Object position | 0.09 | 0.24 | 0.24 | 2.3 → 4.4 |
| Object rotation | 0.06 | 0.10 | 0.10 | 1.9 → 1.9 |
| Object linear velocity | 0.07 | 0.11 | 0.10 | 1.1 → 1.1 |
| Pad contact forces | 0.07 | 0.09 | 0.12 | 1.1 → 1.1 |

- The model learns how the fingers follow their targets, but barely how the object and contacts respond; those effects are no larger than its own error on the unperturbed rollout.
- Held-out gradient check after the fine-tune (0.1 s, ±0.02–0.1 per finger): still at chance, cosine −0.03 to 0.05, sign agreement 48–53%. Drop prediction was kept (phase-matched AUC 0.982 → 0.987).
- Conclusion: within \~0.1 s the simulator's action effects are at or below a learned model's error floor; beyond it the simulator is chaotic. A first-order, SuperMimic-style objective has no usable gradient on this task.

## 4. Stage 3 design: a physics-structured contact model

Everything that is known exactly is built in, and only the contact law's few parameters plus a small bounded residual are learned. The reason is identifiability, not gradient smoothness. The functional forms below are proposals for the reviewer to judge; none is implemented yet.

**Why structure.** In the structured model the chain finger command → servo → joint motion → finger geometry → signed distance → contact force is known physics. The contact parameters are fitted to per-contact forces measured in the reference simulator, and those carry information however correlated the expert's actions are. The black box had to discover the action pathway from data in which actions are almost a function of the state, and could not.

**What the gradient should match.** A policy needs the sensitivity of an outcome (object pose, grip force, slip) to its finger commands over a short horizon, not the instantaneous force field. Contact rollouts are chaotic beyond \~0.1 s, so the target is the gradient of the expected outcome under small noise of scale σ (actuator noise plus perception error), over at most 8 control steps (0.13 s).

**Known exactly (taken from the training robot model):** hand kinematics (joint axes, link frames, coupling), link masses and inertias, joint armature 1e-3, servo gains and limits, object mass and inertia (solid cylinder), gravity, and the palm's prescribed motion.

**State and input.** State: finger joints q, q̇ (11 each); object pose and twist; per contact pair k a 2-D tangential bristle state z\_k and a scalar torsional state θ\_k. Input: finger targets (6 commands plus coupling, or 11 targets) and the palm pose trajectory.

**Normal contact (proposed).** A smooth version of MuJoCo's soft contact: a spring-damper in penetration, gated smoothly at d = 0, with learned stiffness κ, damping b and sharpness β.

```latex
f_{n,k} = \kappa\,\mathrm{softplus}_{\beta}(-d_k) + b\,\mathrm{softplus}_{\beta}(-\dot d_k)\,\sigma(-\beta d_k)
```

**Long-range tail, backward pass only (proposed).** A pushing-only force ε·exp(−d/ℓ) for d > 0 gives non-zero gradients before contact. It enters only the gradient, never the forward value: f = f\_contact + (f\_tail − stopgrad(f\_tail)). So forward rollouts match the reference, and the tail cannot create phantom forces that a policy could exploit in the model.

**Friction with internal state (LuGre / Dahl, per pair).** The bristle state stretches while sticking and saturates at the friction limit, so static holding is exact and the moment sticking ends is an explicit, predictable variable:

```latex
\dot z_k = v_{t,k} - \frac{\sigma_0 \lVert v_{t,k} \rVert}{\mu f_{n,k}}\, z_k, \qquad f_{t,k} = -\left(\sigma_0 z_k + \sigma_1 \dot z_k + \sigma_2 v_{t,k}\right)
```

Torsional friction (condim 4) uses the same law on a scalar state with coefficient μ\_tors (training value 0.005 m). Note that MuJoCo itself uses a regularised elliptic cone, not a bristle model; the fit must absorb the difference.

**Dynamics.**

- Object: Newton–Euler with the summed contact forces and torques, torsional moments and gravity.
- Fingers: M(q)q̈ + c(q, q̇) = τ\_servo + Σ J\_k^T f\_k − palm-acceleration terms, with τ\_servo = kv·(clip((kp/kv)(q\* − q), ±vmax) − q̇). The fingers must yield under contact: grip force comes from servo tracking error, so they cannot be treated as kinematic.
- Palm: prescribed (input).
- Integration: semi-implicit Euler with N substeps per 1/120 s physics step (N chosen for stability at the fitted stiffness; to be determined).

**Two temperatures.** Forward rollouts use the fitted, stiff law (large β, small bristle compliance). The backward pass uses a smoothed copy (smaller β, tail on), annealed from smooth to stiff during policy training: smooth early to discover contacts, stiff later for precision.

**Residual.** A small network conditioned on mass, centre of mass and friction adds a tanh-bounded correction to the object and finger accelerations, with an L2 penalty, for what the structure misses (soft-contact details, solver effects).

**Learned parameters (about 10):** κ, b, β, μ (sliding; possibly separate for the fingertip markers, 1.0 in training), μ\_tors, σ₀, σ₁, σ₂, optionally effective joint inertias; plus the residual. ε, ℓ and the backward β are chosen by the gradient check, not fitted.

## 5. Reference testbed as built

The reference simulator is a standalone C MuJoCo scene (CPU, float64) with the training physics: the training robot's Revo2 right hand on a palm with prescribed motion, one solid cylinder and the table. It is built and smoke-tested (commit 28da10a, `dextrah_lab/neurosim/testbed.py`) but not yet used to generate data.

**Physics.** Identical to training (section 1 table): 1/120 s physics step, 2 substeps per 60 Hz control step, implicitfast, Newton, elliptic cones, impratio 10, 20 / 20 iterations, tolerance 1e-8, condim 4, the training friction, solref and solimp, hulls of at most 64 vertices. The hand is cut from the training robot specification, so geometry, masses, inertias, armature and joint limits are the training ones.

**Hand joints (velocity servos as in training).** Commanded velocity = clip(gain·(q\* − q), ±vmax), torque = kv·(commanded − q̇); gain = kp/kv with kp = 1200. Joint damping and friction loss 0.

| Joint | Range (rad) | Effort (N·m) | vmax (rad/s) | kv | Gain (1/s) |
| --- | --- | --- | --- | --- | --- |
| Thumb metacarpal | 0–1.518 | 0.5 | 2.618 | 0.191 | 6,282 |
| Thumb proximal | 0–1.047 | 1.1 | 2.53 | 0.435 | 2,760 |
| Thumb distal | 0–1.047 | 1.1 | 2.53 | 0.435 | 2,760 |
| Index, middle, ring, pinky proximal | 0–1.466 | 2.0 | 2.268 | 0.882 | 1,361 |
| Index, middle, ring, pinky distal | 0–1.693 | 2.0 | 2.268 | 0.882 | 1,361 |

Link masses: hand base 238 g; thumb metacarpal 10 g, proximal 50 g, distal 10 g; each finger link 9 g; each fingertip marker 1 g.

**Hand geometry, measured.** In the hand-base frame the fingers point along (0.31, 0, 0.95) and flex toward (0.95, 0, −0.31); the thumb is on +y. The palm hull spans x −0.020 to 0.021 m, y −0.037 to 0.039 m, z 0 to 0.094 m; the finger roots sit at z ≈ 0.081 m and the fingers are about 65 mm long. Index fingertip in the base frame (x, z) as the targets close 0 / 25 / 50 / 75 / 100%: (0.020, 0.146), (0.047, 0.124), (0.056, 0.093), (0.047, 0.064), (0.025, 0.048) m. So the fingers reach at most \~35 mm in front of the palm surface.

**Palm: prescribed motion.** The palm is a free body whose pose and velocity are set at every physics substep, following the commanded trajectory at constant linear and angular velocity within a control step. Its 6 degrees of freedom carry an armature of 1e3, so contacts cannot move it within a step (an infinitely stiff arm), and gravity compensation is on for the palm and hand base; the fingers feel gravity as in training. Two alternatives failed:

- A mocap body has zero velocity in MuJoCo's contact solve, so a carried object would be dragged only through penetration, not friction.
- A weld from a free palm to a mocap target was either soft or unstable at dt = 1/120 s: with stiffness/damping given directly, k·dt² ≤ 0.7 sagged 12.5 mm under the fingers' weight and lagged 18 mm at 14 cm/s, and k·dt² ≥ 1 diverged; a time-constant setting is floored at 2·dt and sagged 39–55 mm.

**Object and table.** One solid cylinder with a free joint, uniform density (mass sets the inertia), world contact parameters, spawned standing at the table centre. The table is a mocap box of 0.475 × 0.4 × 0.3 m with its top at 0.80 m by default (settable, 0.70–0.90 m in the project), plus a ground plane. Collision masks as in training: hand–object, hand–table, object–table and object–ground collide; finger self-collisions are off.

**Interface.** `reset(palm pose, finger joints, object pose, table top)`; `step(11 finger targets, palm pose, next palm pose)` runs one control step (servo command recomputed at each substep, as in training); `contacts()` lists every hand–object, object–table and hand–table contact with the hand geom, distance, position, contact frame and 6-D local force (normal, two tangential, torsional, rolling).

**Two finger-geometry variants.** (b) 'mesh': the training convex hulls, unchanged. (a) 'capsule': per link the tighter (by hull volume / primitive volume) of an enclosing capsule along the principal axis or an oriented bounding box, so distances are analytic in the model. The primitives enclose the hulls, so in (a) the fingers are fatter than in training, but the reference and the model use the same shapes.

| Link | Primitive | Size (mm) | Hull fill (other primitive) |
| --- | --- | --- | --- |
| Palm (hand base) | box | half 47.5 × 37.7 × 20.7 | 0.56 (capsule 0.36) |
| Thumb metacarpal | box | half 17.1 × 12.4 × 11.0 | 0.35 (0.28) |
| Thumb proximal | box | half 30.8 × 11.9 × 10.5 | 0.56 (0.55) |
| Thumb distal | capsule | radius 9.7, length 13.6 | 0.60 (box 0.48) |
| Index proximal | box | half 21.0 × 7.8 × 8.3 | 0.65 (0.57) |
| Index distal | capsule | radius 7.8, length 29.9 | 0.67 (0.63) |
| Middle proximal | box | half 23.6 × 7.8 × 8.4 | 0.66 (0.58) |
| Middle distal | capsule | radius 8.2, length 33.9 | 0.66 (0.62) |
| Ring proximal | box | half 22.6 × 7.8 × 8.1 | 0.67 (0.58) |
| Ring distal | capsule | radius 7.9, length 33.3 | 0.68 (0.65) |
| Pinky proximal | box | half 19.4 × 7.8 × 8.5 | 0.63 (0.56) |
| Pinky distal | capsule | radius 7.8, length 26.3 | 0.67 (0.64) |
| Fingertip markers (5) | sphere | radius 0.25 | — |

The model side of (b), signed-distance grids of the hulls, is not built yet.

## 6. Testbed status, gaps from training, open problems

The testbed reproduces resting contact, finger tracking and exact palm motion, but scripted grasps do not hold yet: 0 of 72 attempts lifted the cylinder. Several deliberate differences from the training simulator are listed so the reviewer can judge whether the reference is the right one.

**Verified (smoke tests, both geometry variants).**

- A cylinder placed on the table settles at the expected height (within 0.1 mm).
- Fingers reach their targets within 0.02 rad in 1 s.
- The palm follows commanded translations and rotations exactly (30° yaw over 0.5 s ends at 30.00°).
- Enclosing primitives: compiled and simulated without instability.

**Open problem: scripted grasps fail (0 / 72).** Side power grasp of an upright cylinder (hand +y up, palm facing the cylinder axis), swept over palm gap (2–8 mm), grasp point along the fingers (z 0.10–0.13 m in the hand frame), grasp height (5–7 cm above the table), three sizes (r 2.5 / 3.5 / 4.5 cm, h 10 / 14 / 20 cm, 0.1 / 0.3 / 0.5 kg) and both geometry variants. An earlier sweep with the cylinder in front of the palm instead of the fingers also failed (0 / 32). Trace of one attempt: while the targets close over 25 control steps, the thumb makes first contact (8–17 N) and pushes the cylinder 43 mm sideways before any finger touches it; the fingers then close on air. Planned fixes: close the fingers before the thumb, place the thumb to oppose from the far side, or record grasps from the trained policy instead of scripting them (question 11).

**The Revo2 is small for the specified cylinders.** The fingers start at the end of a 94 mm palm, are \~65 mm long, and reach at most \~35 mm in front of the palm surface. A 4.5 cm-radius cylinder (9 cm diameter) is probably beyond a power grasp, so P's radius range may need to drop toward 2.5–3.5 cm unless pinch grasps are used.

**Deliberate differences from training.**

| Aspect | Training | Testbed | Why / consequence |
| --- | --- | --- | --- |
| Simulator | MuJoCo Warp, GPU, float32, not bitwise reproducible | C MuJoCo, CPU, float64, deterministic | Clean finite differences and exact restores; collision routines and precision differ (question 1) |
| Hand base | Driven by SONIC through the arm (compliant, dynamic) | Prescribed motion, effectively infinite stiffness | Removes arm dynamics; the model takes palm motion as input |
| Finger commands | 6 commands + fixed distal coupling | 11 independent targets (the coupling map can be applied in the generator) | Independent targets give richer data; the policy test must use the coupled map |
| Objects | 1,200 tools from boxes and capsules | Solid cylinders | P's target objects; analytic distances |
| Table | Pose randomised per world | Fixed pose, settable height | Simpler; height covers 0.70–0.90 m |
| Perception, touch, policy observations | Present | Absent | Not needed for physics |
| Domain randomisation, pushes | Present | Off | Deterministic reference; noise is injected explicitly for the σ sweep |

## 7. Planned data generation and reference gradients

Scripted episodes in the testbed will cover every contact regime of placing, with finger commands that vary independently of the state by construction. Reference gradients are noise-averaged central differences at stratified states. Nothing in this section is implemented; all numbers are proposals.

**Episode script (4–6 s at 60 Hz).**

1. Setup: cylinder size from the training set, mass uniform 0.1–0.5 kg, table top uniform 0.70–0.90 m, cylinder standing at a random spot within ±10 cm of the table centre with random yaw; palm 8–15 cm from the pre-grasp pose, orientation noise up to 10°; fingers open with small random offsets.
2. Approach, 0.5–1 s: palm to the grasp pose at a random speed.
3. Close, 0.3–0.6 s: finger targets ramp to random closure levels per command, in a randomised order (fingers before thumb in most episodes).
4. Lift, 0.5–1 s: 5–15 cm up.
5. Carry, 0.5–1.5 s: random translations (up to ±10 cm) and rotations (up to ±20°), smooth and jerky.
6. Lower, 0.5–1 s: down to touchdown at a random speed.
7. Release, 0.3–0.6 s: fingers open at random rates; palm retreats.

Throughout, each finger command carries an independent Ornstein–Uhlenbeck perturbation (σ 0.05–0.15 rad, correlation time 0.1–0.3 s), so commands are not a function of the state (the fix for action-blindness at the data level). About 20% of episodes are near-misses and brushes (the palm passes close without grasping) and about 10% are pushes that tip the cylinder, to cover pre-contact and incidental contact.

**Sizes.** Training radii {2.5, 3.0, 3.5} cm × heights {8, 12, 16, 20} cm; held out: radius 3.25 cm and heights 10 and 18 cm. Larger radii only if grasps prove feasible (section 6).

**Logged per control step.** Palm pose and twist (input); finger commands (6 coupled and the 11 targets); finger q and q̇; object pose and twist; every contact at each physics substep (pair, hand link, distance, position, frame, 6-D local force); the full MuJoCo physics state for exact restores; phase labels and flags (grasped, dropped, tipped, resting on the table).

**Volume.** Target 0.5–1M control steps (about 2,000–3,500 episodes) on the CPU partition (4 cores per job). Throughput not yet measured.

**Reference gradients.**

- States: sampled from logged episodes and restored exactly, stratified into pre-contact (no hand–object contact, closest distance < 2 cm), contact forming, stable holding, lifting, lowering and touchdown, and release; about 2,000 per stratum.
- Perturbation: each of the 6 finger commands ±δ, δ ∈ {0.005, 0.01, 0.02} rad (small, because the response was already nonlinear at 0.05–0.1), held over the horizon.
- Noise (the σ sweep): i.i.d. noise on every executed finger target each step, σ ∈ {0, 0.005, 0.02, 0.05} rad, and palm pose noise of {0, 1, 3} mm and 0.5°. +δ and −δ share the same noise draws (common random numbers); 32–64 draws per state, with confidence intervals from the draw-to-draw variance.
- Outcomes, at horizons of 1, 2, 4 and 8 control steps: object position and rotation change, object velocity, per-finger normal force, total grip force, slip margin (tangential / (μ·normal)), and the time sticking ends.
- σ = 0 gives exact finite differences of the deterministic, non-smooth simulator: the stiff limit against which smoothing is measured.

## 8. Model implementation plan

The model will be a batched PyTorch simulator, differentiable end to end, whose parameters are fitted in three stages: contact forces first (the identifiable part), then accelerations, then short rollouts. Nothing here is implemented.

**Kinematics.** Forward kinematics from the training robot's model tree (palm pose → hand base → 11 joint rotations → link frames), with link Jacobians by automatic differentiation. Validation: link poses match MuJoCo to float precision.

**Distances and contact points.**

- Object and table: exact analytic signed-distance functions (solid cylinder; box).
- Variant (a): capsule = segment + radius, so its distance to the cylinder is the minimum over the segment of SDF\_cylinder minus the radius, computed by dense sampling (\~16 points) plus a short refinement. Box links: surface point samples. Both directions are evaluated (finger points in the object's SDF, object surface points in the finger's analytic SDF), keeping the closer.
- Variant (b): a signed-distance grid of each link's hull (about 1 mm resolution, trilinear interpolation), with hull vertices and face samples as the link's surface points.
- Fixed pair set: 17 hand shapes × object, 17 × table, object × table, object × ground. Fixed pairs make the per-pair friction state well defined.
- Several witness points per pair where a single point cannot carry the torque: the cylinder's flat face on the table (about 8 rim points per face) and flat finger faces. This matters for placing upright, where tilting depends on where along the rim the support sits.

**Forces and dynamics.** The section 4 laws per witness point: normal law, tangential and torsional bristle states, relative velocity at the witness point. Object Newton–Euler and finger joint dynamics with the servo law; semi-implicit Euler with N substeps per physics step; rollouts unrolled over at most 8 control steps for gradients.

**Fitting, in three stages, held-out sizes kept aside.**

1. Contact law: with positions and velocities taken from the reference (teacher-forced), predict contact forces. MuJoCo reports forces at its own contact points, which differ from the model's witness points, so the comparison is on the net force and torque per link pair (in the link frame). Huber loss. Fits κ, b, β, μ, μ\_tors and the bristle parameters.
2. Accelerations: teacher-forced one-step object and finger accelerations.
3. Rollouts: 8-step autoregressive windows on object pose, velocity and finger joints (SuperMimic-style), adding the bounded residual network (about 2 × 256 units) with an L2 penalty.

**What the policy uses.** d(outcome)/d(finger commands) by automatic differentiation through at most 8 control steps of the backward (smoothed) model, with the forward values from the stiff model.

## 9. Baselines, evaluation metrics and go/no-go

The structured model must beat two baselines on gradient quality against the noise-averaged reference, before contact and in contact, on held-out cylinder sizes. Cosine is reported together with magnitude ratio and sign agreement, because the black box's 4–6% magnitude was hidden behind cosine-style scores.

| Model | What it is | Expected |
| --- | --- | --- |
| Black box | The stage 1.5 architecture retrained on the same testbed data (independent finger variation included) | Fair test of whether data alone fixes action-blindness |
| Softened MJX | MuJoCo's own soft contact differentiated in JAX, with softened contact only in a gradient-only copy | Loses before contact (zero force at a distance); strong in contact |
| Structured model | Sections 4 and 8 | Must win before contact (long-range tail) and at least tie in contact |

**MJX prerequisites.** JAX and MJX are not installed in any of our environments; a separate environment is needed. MJX supports fewer collision pairs than C MuJoCo: cylinder–box and mesh/capsule–cylinder support must be confirmed first, or cylinders replaced by meshes. The same model file and contact parameters as the reference (condim 4, elliptic cones, impratio 10), so the comparison measures gradients rather than different physics.

**Metrics,** each reported per noise scale σ, per horizon (1, 2, 4, 8 steps) and separately for pre-contact and in-contact states:

- Cosine between the model gradient and the reference gradient, per outcome.
- Magnitude ratio (model norm ÷ reference norm).
- Per-finger sign agreement.
- Release timing: predicted versus reference time at which sticking ends, on slow lowering trajectories.
- Open-loop prediction error at 8 and 32 steps (object pose, grip force) versus persistence.
- Data needed: each metric as a function of training-set size.

The cosine depends on the reference's noise scale: a model smoothed at scale τ matches finite differences averaged at σ ≈ τ best. So the backward smoothing is chosen to match the task's σ (actuator noise plus perception error), and the whole sweep is reported, not one number.

**Go criteria (to be fixed before the run).** At the task σ on held-out sizes: cosine ≥ 0.7 both before and in contact (agreed). Proposed, not yet agreed: magnitude ratio within 0.5–2, sign agreement ≥ 80%, release timing within one control step on ≥ 80% of lowering trajectories.

## 10. Policy test, sequencing and compute constraints

Only after a go: train the placing expert P with SAPG alone versus SAPG plus model gradients at equal GPU-hours, scored on upright placement. Everything before that runs on CPUs.

**Policy test.**

- Task: set a held cylinder down upright (tilt < 10°) within a 5 cm target spot without knocking it over, from the carry expert's hand-off states.
- Hybrid objective: a short-horizon (at most 8 control steps) model-gradient term on the finger commands added to the SAPG actor loss, weight λ gated by the model's recent prediction error, with the critic bootstrapping beyond the horizon. Gradients reach the finger commands only; the arm channel would need a model of SONIC and the robot body.
- Safeguards against phantom forces: pushing-only tail, tail in the backward pass only, forward values from the stiff model, and SAPG anchored in the real simulator.
- Transfer check: the policy trains in MuJoCo Warp (float32, GPU) while the model is fitted to C MuJoCo, so the gain must be measured in the training simulator.

**Build order.**

1. Testbed (built; grasps to fix).
2. Data generator and reference-gradient harness (CPU).
3. Kinematics and geometry, variants (a) then (b) (CPU).
4. Contact law, friction, dynamics; fitting stages 1–3 (CPU; GPU only if needed and free).
5. Baselines: black box retrained on testbed data; MJX environment and support check.
6. Go/no-go report.
7. Policy test with P (GPU), after the cylinders exist in the training environment.

**Compute constraints.** One shared GPU (RTX 6000 Ada, 48 GB), and only one GPU job at a time, running or queued. GPU jobs up to 110 GB host memory; CPU jobs at most 4 cores and 19 GB. Every training run is logged to Weights & Biases.

**Project state.** M's table fine-tune finished (deterministic evaluation of an intermediate checkpoint: 71.0% success, worst height bin 67.1%, against 57.9% / 42.3% for the frozen original). Retraining C is on hold while the neural simulator is the focus.

## 11. Risks and questions for the reviewer

The main risks are that smooth gradients still fail to improve the policy in the real simulator, and that the reference (C MuJoCo) is not the simulator the policy trains in. Specific questions, roughly in order of importance:

1. **Reference simulator.** Is deterministic C MuJoCo (float64) the right reference when policies train in MuJoCo Warp (float32, not bitwise reproducible, different collision code)? Or should the reference gradients come from Warp, with replicas averaged?
2. **Gradient target.** Is the gradient of the expected outcome under noise σ, over at most 0.13 s, the right quantity for policy learning? How should σ be chosen, and is a short horizon plus a critic beyond it sound?
3. **Normal contact law.** Mirror MuJoCo's soft constraint (solref / solimp, impedance and reference acceleration) analytically, or learn a smooth spring-damper as proposed?
4. **Friction.** Is a LuGre / Dahl bristle model the right choice, given that MuJoCo uses a regularised elliptic cone with impratio 10? How should its parameters be fitted to MuJoCo's contact forces, and does it still predict the time sticking ends accurately?
5. **Backward-only long-range tail.** Gradients that include a force the forward model never applies: is that a sound estimator, or does it bias the policy toward approaching without contact? How should ε and ℓ be set?
6. **Finger dynamics.** Must the fingers be dynamic (servo law, contact torques), or would compliant kinematic fingers be accurate enough?
7. **Geometry.** The enclosing primitives fill only 35–68% of the hulls in variant (a). Acceptable for a method test? What resolution do the mesh grids in variant (b) need, and is per-pair witness-point sampling adequate for flat-face contacts?
8. **Go criteria.** Do cosine ≥ 0.7, magnitude ratio 0.5–2 and sign agreement ≥ 80% at the task σ predict a policy improvement? Is there a better test, for example improvement of a short-horizon trajectory optimisation in the true simulator?
9. **Task choice.** Is upright placing the right first task, given that the Revo2 may not power-grasp cylinders wider than \~7 cm?
10. **Hybrid training.** How should model gradients be combined with SAPG (weight λ, gating by model error, horizon length, critic coupling) without destabilising PPO's trust region?
11. **Data source.** Script grasps in the testbed, or drive the testbed hand with finger commands recorded from the trained policies (M, C) in the GPU environment, plus independent perturbations?
12. **Alternatives.** Would a differentiable rigid-contact simulator (MJX with smoothed contact, or a complementarity-based one) make the learned structure unnecessary? What would make the learned model worth it: speed, real-world calibration, or gradient quality?
13. **Anything missing** in the design, the evaluation or the controls.

## 12. Related work, code and data

The three papers with links were checked against their text on 2026-10-29; the others match their abstracts, and no numbers are quoted from them.

| Work | Relevant result |
| --- | --- |
| [ContactSDF](https://www.alphaxiv.org/abs/2408.09612) | Signed-distance contact smoothed end to end. In simulation, 31.6% lower position and 24.9% lower orientation error than a QP-based MPC contact model, with fewer than 5k environment steps. Separately, on hardware: in-hand reorientation on an Allegro hand at 50 Hz within 2 minutes of learning. |
| [Long-range contact model](https://www.alphaxiv.org/abs/2509.20917) | Properties for a contact potential: barrier, twice differentiable, pushing-only, and gradients that "remain non-zero even when objects are arbitrarily far apart". |
| [PA-RL (potential-field actions)](https://www.alphaxiv.org/abs/2609.21609) | The policy outputs an energy-like field; in simulation 100% evaluation success on peg-in-hole versus 92.6% for the best baseline; real insertions without fine-tuning. |
| SuperMimic (arXiv 2610.28480) | Whole-body tracking with gradients through a learned world model; object interaction explicitly out of scope. |
| NEXUS | Dynamics as learned scalar energy and dissipation terms; forces by differentiation. |
| Differentiable contact dynamics | Smoothed gradient in the backward pass, exact contact solve forward. |
| Smoothing-error tubes | Bound and compensate the mismatch between smoothed and true hybrid dynamics. |
| ContactNets | Smooth, distance-like contact functions learned from data. |

**Code.** Repository github.com/Skvayzer/DEXTRAH, branch `feature/g1-neurosim` (testbed at commit 28da10a).

- `dextrah_lab/neurosim/testbed.py`: the stage 3 reference testbed (section 5).
- `dextrah_lab/neurosim/model.py`, `ensemble.py`, `data.py`, `evaluate.py`, `pairs.py`, `env_io.py`, `geometry.py`: the stage 1–1.5 black-box model, data handling, evaluation and paired-counterfactual tools.
- `scripts/neurosim/`: `collect_carry_transitions.py`, `train_world_model.py`, `evaluate_world_model.py`, `gradient_check.py`, `replica_divergence.py`, `action_sensitivity.py`, `collect_counterfactual_pairs.py`, `pair_diagnostic.py`.
- `docs/neurosim_design.md`: the running design note, stages 1–3.
- Robot, contact parameters and servo law: `dextrah_lab/mj/robot.py` (shared with the training code on branch `feature/g1-wholebody-sapg`).

**Data and results.** Stored on the training workstation under `outputs/neurosim/` (not in the repository): stage 1 datasets and checkpoints, `stage15/` (gradient checks, replica divergence, action sensitivity), `counterfactual/` (paired data and diagnostics). Training runs are logged to Weights & Biases (project skvayzer/adept).
