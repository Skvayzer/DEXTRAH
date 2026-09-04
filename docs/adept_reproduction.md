# ADEPT reproduction roadmap

This branch reconstructs the KUKA iiwa7 + Allegro simulation pipeline described
in **ADEPT: Accelerating Dexterity via Pre-Training and Post-Training using
Reinforcement Learning** (arXiv:2608.19182v1). It is an independent
reimplementation on top of the public DextrAH and NVIDIA FABRICS codebases; it
does not contain unreleased ADEPT code or checkpoints.

## Reproduction boundary

The first target is a testable simulation reproduction, followed by stereo-RGB
distillation and real-robot deployment. Numerical parity with the paper is not
claimed until all milestones and seed-based evaluations below are complete.

The implementation keeps three kinds of settings visibly separate:

1. `paper`: values explicitly reported in ADEPT v1;
2. `upstream`: behavior inherited from DextrAH/FABRICS;
3. `inferred`: choices required where the paper does not specify an executable
   value.

Every inferred choice must be documented next to its configuration field.

## Milestones

### M0 — reproducible baseline

- Pin and smoke-test DextrAH with Isaac Lab v2.2.1 and NVIDIA FABRICS.
- Preserve the original 11-D palm-pose + hand-PCA task as an upstream baseline.
- Add shape and action-contract tests that do not require launching Isaac Sim.

### M1 — full configuration-space fabric

- Replace the 11-D action with 23 relative joint deltas.
- Map actions using `q_target = clamp(q_fabric + 0.1 * action, limits)`.
- Drive separate arm and hand C-space forcing attractors while retaining
  geometric posture, collision avoidance, joint limits, damping, and speed
  control.
- Keep the controller at 60 Hz with two fabric integration steps.

### M2 — reposing pre-training MDP

- Use 16 primitive cuboid, cylinder, sphere, and cone assets with randomized
  scale and physical properties.
- Represent object geometry with 64 surface points.
- Implement the paper's 8-keypoint pose error, contact-gated reward, four-second
  episodes, and 50-step ADR curriculum.
- Train an asymmetric PPO actor-critic; add decentralized PBT only after a
  deterministic single-policy run passes.

### M3 — FMB post-training

- Add the star and square/round FMB peg-and-board scenes.
- Define the downstream observation projection explicitly instead of assuming
  that it is a superset of the pre-training observation.
- Implement the three transfer phases as independently resumable commands:
  1. 40,000-iteration actor behavior-cloning warm start;
  2. 20-iteration frozen-actor critic warm-up;
  3. PPO with actor LR `1e-5`, critic LR `5e-5`, and clip `0.05`.
- Evaluate scratch, direct fine-tuning, low-LR-only, and full-recipe baselines
  with identical seeds and environment budgets.

### M4 — stereo-RGB student

- Reuse DextrAH's online DAgger infrastructure and stereo encoder.
- Pre-train perception on peg lift/reposing, then distill the FMB teacher.
- Replace privileged object inputs with two RGB images and supervise an
  auxiliary 8-keypoint pose head.
- Record the complete visual and physics randomization configuration.

### M5 — deployment parity

- Run the same C-space fabric parameters in simulation and deployment.
- Validate rate hierarchy, joint ordering, limits, and emergency-stop behavior
  before commanding hardware.
- Treat real-world deployment as a separate safety review and never infer that
  a simulation checkpoint is hardware-safe.

## Validation gates

Each milestone must land as one or more focused commits and pass its local unit
tests before long Slurm training starts. Expensive experiments are launched from
clean, pushed commits; output checkpoints and logs are not committed.

The first scientific gate is not final insertion success. It is preservation of
the pretrained reposing success rate during the first post-training updates.
Only after that gate passes do we spend compute on full ADR progression,
multi-seed comparisons, or vision distillation.

## Primary references

- ADEPT v1: https://arxiv.org/abs/2608.19182v1
- DextrAH-G: https://arxiv.org/abs/2407.02274
- DextrAH-RGB: https://arxiv.org/abs/2412.01791
- NVIDIA FABRICS: https://github.com/NVlabs/FABRICS
- Play2Perfect: https://arxiv.org/abs/2606.26428
- SimToolReal: https://arxiv.org/abs/2602.16863
- Functional Manipulation Benchmark: https://github.com/rail-berkeley/fmb
