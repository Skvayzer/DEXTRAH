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

## Current runnable target

The Gym task `Adept-Kuka-Allegro-Repose` uses DextrAH's KUKA-Allegro simulator
base with ADEPT's reposing MDP, 23-D relative C-space interface, asymmetric
observations, contact gate, ADR, and Appendix-B fabric. The original
`Dextrah-Kuka-Allegro` task remains available as the 11-D upstream baseline.

On the TL server, create the isolated environment and hydrate Git LFS assets:

```bash
cd ~/data1/DEXTRAH-ADEPT
./scripts/setup_adept_env.sh
```

Do not install FABRICS into another Isaac environment: its legacy `urdfpy`
metadata requests dependency versions that conflict with Isaac Sim 5. The
setup script retains Isaac Sim's NumPy 1.26, NetworkX 3.3, and PyCollada 0.9.3;
the controller contains the one runtime compatibility shim urdfpy needs.

Run the physics/contact validator, then the inexpensive PPO smoke test:

```bash
cd ~/data1/DEXTRAH-ADEPT
sbatch scripts/slurm/validate_adept_repose.sbatch
sbatch scripts/slurm/smoke_adept_repose.sbatch
```

Both launchers request exactly one `rtx_6000_ada`. They intentionally use eager
FABRICS evaluation. CUDA-graph replay fails on the current Isaac Sim 5 / Warp /
driver stack in both unmodified DextrAH and this branch, so it is not a valid
training option on this workstation.

## Training sequence

Run each stage only after its smoke/validation gate passes. Checkpoint paths are
explicit environment variables so a downstream job cannot silently select the
wrong upstream run.

```bash
# Stage 1 on this workstation: one 4096-environment PPO policy
RUN_NAME=0_adept_repose_seed42 \
  sbatch scripts/slurm/train_adept_repose_single.sbatch

# Paper-reference path only: requires 16 simultaneous GPUs/policies
# sbatch scripts/slurm/train_adept_pbt.sbatch

# Stage 2a: new FMB actor, supervised by the selected Stage-1 actor
TEACHER_CHECKPOINT=/absolute/path/repose.pth \
  sbatch scripts/slurm/train_adept_fmb_bc.sbatch

# Stage 2b/c: fresh critic warm-up for 20 epochs, then conservative PPO
BC_CHECKPOINT="$PWD/logs/adept_fmb_bc.pth" \
  sbatch scripts/slurm/train_adept_fmb_posttrain.sbatch

# Stage 3a: perception pretraining from the reposing teacher
TEACHER_STAGE=pretraining \
TEACHER_CHECKPOINT=/absolute/path/repose.pth \
OUTPUT_CHECKPOINT="$PWD/logs/adept_rgb_stage1.pth" \
  sbatch scripts/slurm/train_adept_vision_dagger.sbatch

# Stage 3b: FMB teacher distillation, initializing only the visual encoder
TEACHER_STAGE=downstream \
TEACHER_CHECKPOINT=/absolute/path/fmb_posttrained.pth \
STAGE1_STUDENT="$PWD/logs/adept_rgb_stage1.pth" \
  sbatch scripts/slurm/train_adept_vision_dagger.sbatch
```

The single-policy launcher enables Weights & Biases by default and uses the
stable run name as the W&B resume ID. It stops at 8 billion environment frames,
saves every 50 epochs, restores PPO and ADR state from the newest complete
checkpoint, and requests one RTX 6000 Ada. Re-submit the same command after a
node interruption or time limit; `AUTO_RESUME=true` is the default. Set
`WANDB_ACTIVATE=false` only for an offline run.

Useful short run:

```bash
MAX_ITERATIONS=100 RUN_NAME=0_adept_repose_pilot_seed42 \
  sbatch scripts/slurm/train_adept_repose_single.sbatch
```

The full-scale gate measured about 21,557 total environment frames/s after
startup. At that rate, 100 epochs take about 10 minutes of training plus the
roughly 11-minute cold scene build; 8 billion frames take about 4.3 days. Treat
that as a planning estimate, because contacts, ADR progression, filesystem
load, and preemption can change throughput.

One GPU cannot reproduce ADEPT's decentralized 16-policy PBT search. The
single-GPU run fixes the reported Table-5 PPO hyperparameters and reproduces
the policy, MDP, fabric, and frame budget as closely as the released artifacts
permit. Its learning curve is therefore not expected to be numerically
identical to the paper's selected PBT worker.

Set `TASK=Adept-Kuka-Allegro-FMB-SquareRound` for the square/round teacher
stages and append `-Vision` for its student stages. ADEPT trains each geometry
and embodiment independently; these launchers intentionally do not mix them.

## Milestones

### M0 — reproducible baseline

- [x] Isolate DextrAH with Isaac Lab v2.2.1 and NVIDIA FABRICS.
- [x] Preserve the original 11-D palm-pose + hand-PCA task as an upstream baseline.
- [x] Add shape and action-contract tests that do not require launching Isaac Sim.
- [x] Pass the one-epoch GPU smoke test (Slurm job 217, 2026-09-04).

### M1 — full configuration-space fabric

- [x] Replace the 11-D action with 23 relative joint deltas.
- [x] Map actions using `q_target = clamp(q_fabric + 0.1 * action, limits)`.
- [x] Drive separate arm and hand C-space forcing attractors while retaining
  geometric posture, collision avoidance, joint limits, damping, and speed
  control.
- [x] Keep the controller at 60 Hz with two fabric integration steps.
- [x] Validate eager rollouts for finite states and joint limits.
- [x] Establish that CUDA-graph replay is unavailable on this workstation by
  reproducing the same failure on the pre-ADEPT DextrAH controller.

### M2 — reposing pre-training MDP

- [x] Use 16 primitive cuboid, capsule, sphere, and cone assets with randomized
  scale and physical properties.
- [x] Represent object geometry with 64 surface points.
- [x] Implement the paper's 8-keypoint pose error, contact-gated reward, four-second
  episodes, and 50-step ADR curriculum.
- [x] Configure the asymmetric PPO actor-critic and 16-worker decentralized PBT.
- [x] Validate two PPO updates with 4,096 environments on one RTX 6000 Ada.
- [ ] Complete the 8-billion-frame single-policy run and evaluate multiple seeds.

### M3 — FMB post-training

- [x] Add the star and square/round FMB peg-and-board scenes.
- [x] Define the downstream observation projection explicitly instead of assuming
  that it is a superset of the pre-training observation.
- [x] Implement the three transfer phases as independently resumable commands:
  1. 40,000-iteration actor behavior-cloning warm start;
  2. 20-iteration frozen-actor critic warm-up;
  3. PPO with actor LR `1e-5`, critic LR `5e-5`, and clip `0.05`.
- Evaluate scratch, direct fine-tuning, low-LR-only, and full-recipe baselines
  with identical seeds and environment budgets.

### M4 — stereo-RGB student

- [x] Add online student-rollout DAgger and the Appendix-H cross-attention encoder.
- [x] Pre-train perception on peg lift/reposing, then initialize the downstream
  visual encoder from that checkpoint.
- [x] Replace privileged object inputs with two RGB images and supervise an
  auxiliary 8-keypoint pose head.
- [x] Record the complete visual and physics randomization configuration.

### M5 — deployment parity

- Run the same C-space fabric parameters in simulation and deployment.
- Validate rate hierarchy, joint ordering, limits, and emergency-stop behavior
  before commanding hardware.
- Treat real-world deployment as a separate safety review and never infer that
  a simulation checkpoint is hardware-safe.

## Validation gates

Each milestone lands as one or more focused commits and passes its local unit
tests before long Slurm training starts. Expensive experiments are launched from
clean, pushed commits; output checkpoints and logs are not committed.

The Stage-1 implementation passed these workstation gates on 2026-09-04/05:

- 51 CPU unit tests;
- job 264: distal-link + BioTac contact aggregation, index and thumb both above
  1 N, finite observations/rewards/fabric states, and positive joint margin;
- jobs 229/231: checkpoint creation and continuation with restored ADR state;
- job 232: an accelerated end-to-end ADR transition from level 0 to 1;
- job 265: 4,096 environments and two PPO updates on one RTX 6000 Ada,
  approximately 21,557 total frames/s and 22 GB host peak RSS.

These gates establish execution readiness, not learned-task success. Success
requires the long optimization run and checkpoint evaluation.

The first scientific gate is not final insertion success. It is preservation of
the pretrained reposing success rate during the first post-training updates.
Only after that gate passes do we spend compute on full ADR progression,
multi-seed comparisons, or vision distillation.

## Appendix implementation ledger

| ADEPT v1 section | Implementation | Fidelity / remaining validation |
|---|---|---|
| A.1 reposing MDP | `adept_mdp.py`, `adept_repose_env.py`, generated 16-object set | Reported reward, distal/BioTac contact gate, 4 s episodes, 64-point cloud and 50 ADR steps are implemented and GPU-validated. Goal sampling box is not numerically published and is marked inferred. Object scale is sampled uniformly in 0.5--1.0 per environment when USDs are spawned; PhysX tensor simulation cannot rescale active rigid bodies at episode reset, so scale does not advance dynamically with ADR. |
| A.2 downstream FMB | `adept_fmb_mdp.py`, `adept_fmb_env.py` | L-shaped ADR goal, final insertion switch, reward and Table-6 observations are implemented. ADEPT simulation CAD/dimensions are not released; proxy dimensions and board pose are visibly marked inferred. |
| A.3 / D post-training | `actor_bc.py`, `post_training_observer.py`, `adept_*bc.py`, Slurm launchers | 40k alternating mixed-policy BC at ADR 20; fresh critic; 20 frozen-actor epochs at final goal and log-std -2; then separate-trunk PPO with the reported optimizer settings. |
| A.4 observations | Stage-specific builders in the two environments | KUKA flat dimensions are asserted by construction: 391/438, 392/280 and 206+images. Flexiv/Sharpa is architecture-only and is not registered as a simulator task in this KUKA-first branch. |
| B C-space Fabric | `adept_cspace_fabric.py`, `fabric_math.py` | Full 23-D target interface, 31 spheres, per-sphere metric normalization/budgeting, normalized smooth joint-limit barriers, acceleration/jerk caps, separate geometric/forcing channels, 75/25 energy allocation, damping and speed control are implemented on NVIDIA FABRICS. ADEPT omits arm/hand attractor gains, collision metric budgets and joint-limit gate constants; inherited/inferred values are marked in code. Moving-obstacle velocity is unsupported by the public FABRICS query but is not needed for the static workspace/self-collision field used here. |
| C PBT | `pbt.py`, `pbt_observer.py`, 16-worker Slurm array | Population, interval, rank fractions, parameter mutations, true-objective ranking and recipient ADR preservation are implemented. Approximate-frame tolerance is exposed because no numeric tolerance is published. The one-GPU training route deliberately does not activate PBT. |
| E efficiency | Experiment concern | Paper runtime/throughput figures are evaluation targets, not algorithm settings. No parity claim before full runs. |
| F ablations | Experiment concern | Scratch/direct/low-LR/full recipes remain to be run with matched seeds and steps. |
| G losses | `post_training.py` | Mahalanobis distribution loss, tight 8-keypoint auxiliary loss, weights 1/20 and z-mask 0.08/0.02/0.1 implemented. |
| H student | `student.py`, `vision_dagger.py` | Shared ResNet-style encoder, bidirectional cross-attention, optional 5-finger tactile CNN+FiLM, 512x512 fusion and 1024 LSTM implemented. ADEPT does not publish image resolution, exact ResNet checkpoint, total DAgger iterations, or LR; exposed defaults are inherited/inferred. |
| I randomization | `randomization.py` plus ADEPT configs | Published physics, layout, Gaussian observation noise, colors, and ±3 cm/±3° camera jitter encoded. Runtime ranges still require distribution sampling audits in Isaac. |
| J qualitative grasps | Evaluation concern | Requires trained checkpoints and rollout capture; no code-only parity claim. |

Generated FMB files are intentionally ignored and reproducible from
`scripts/generate_adept_fmb_assets.py`. Replace them with measured FMB CAD before
claiming downstream geometry or contact-dynamics parity. ADEPT's unreleased
controller gains/checkpoints, dynamic object-scale schedule, exact goal box,
and 16-worker PBT selection remain explicit reproduction boundaries rather
than silently invented parity claims.

## Primary references

- ADEPT v1: https://arxiv.org/abs/2608.19182v1
- DextrAH-G: https://arxiv.org/abs/2407.02274
- DextrAH-RGB: https://arxiv.org/abs/2412.01791
- NVIDIA FABRICS: https://github.com/NVlabs/FABRICS
- Play2Perfect: https://arxiv.org/abs/2606.26428
- SimToolReal: https://arxiv.org/abs/2602.16863
- Functional Manipulation Benchmark: https://github.com/rail-berkeley/fmb
