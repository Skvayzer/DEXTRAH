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

The Gym task `Adept-Kuka-Allegro-Repose` keeps DextrAH's reposing scene and
reward but replaces its policy/controller boundary with ADEPT's 23-D relative
C-space interface. The original `Dextrah-Kuka-Allegro` task remains unchanged
as the 11-D baseline.

On the TL server, create the isolated environment and hydrate Git LFS assets:

```bash
cd ~/data1/DEXTRAH-ADEPT
./scripts/setup_adept_env.sh
```

Do not install FABRICS into another Isaac environment: its legacy `urdfpy`
metadata requests dependency versions that conflict with Isaac Sim 5. The
setup script retains Isaac Sim's NumPy 1.26, NetworkX 3.3, and PyCollada 0.9.3;
the controller contains the one runtime compatibility shim urdfpy needs.

Submit the one-epoch, 16-environment smoke test through Slurm:

```bash
cd ~/data1/DEXTRAH-ADEPT
sbatch scripts/slurm/smoke_adept_repose.sbatch
```

The smoke test intentionally disables CUDA graph capture. Enable it only after
the eager controller and observation/action contracts pass.

## Milestones

### M0 — reproducible baseline

- [x] Isolate DextrAH with Isaac Lab v2.2.1 and NVIDIA FABRICS.
- [x] Preserve the original 11-D palm-pose + hand-PCA task as an upstream baseline.
- [x] Add shape and action-contract tests that do not require launching Isaac Sim.
- [ ] Pass the one-epoch GPU smoke test.

### M1 — full configuration-space fabric

- [x] Replace the 11-D action with 23 relative joint deltas.
- [x] Map actions using `q_target = clamp(q_fabric + 0.1 * action, limits)`.
- [x] Drive separate arm and hand C-space forcing attractors while retaining
  geometric posture, collision avoidance, joint limits, damping, and speed
  control.
- [x] Keep the controller at 60 Hz with two fabric integration steps.
- [ ] Validate eager and CUDA-graph rollouts for finite states and joint limits.

### M2 — reposing pre-training MDP

- [x] Use 16 primitive cuboid, capsule, sphere, and cone assets with randomized
  scale and physical properties.
- [x] Represent object geometry with 64 surface points.
- [x] Implement the paper's 8-keypoint pose error, contact-gated reward, four-second
  episodes, and 50-step ADR curriculum.
- [x] Configure the asymmetric PPO actor-critic and 16-worker decentralized PBT.
- [ ] Train an asymmetric PPO actor-critic; enable PBT only after a
  deterministic single-policy run passes.

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

Each milestone must land as one or more focused commits and pass its local unit
tests before long Slurm training starts. Expensive experiments are launched from
clean, pushed commits; output checkpoints and logs are not committed.

The first scientific gate is not final insertion success. It is preservation of
the pretrained reposing success rate during the first post-training updates.
Only after that gate passes do we spend compute on full ADR progression,
multi-seed comparisons, or vision distillation.

## Appendix implementation ledger

| ADEPT v1 section | Implementation | Fidelity / remaining validation |
|---|---|---|
| A.1 reposing MDP | `adept_mdp.py`, `adept_repose_env.py`, generated 16-object set | Reported reward, contact gate, 4 s episodes, 64-point cloud and 50 ADR steps are implemented. Goal sampling box is not numerically published and is marked inferred. Runtime object-scale annealing remains constrained by DextrAH's per-environment USD spawn scale. |
| A.2 downstream FMB | `adept_fmb_mdp.py`, `adept_fmb_env.py` | L-shaped ADR goal, final insertion switch, reward and Table-6 observations are implemented. ADEPT simulation CAD/dimensions are not released; proxy dimensions and board pose are visibly marked inferred. |
| A.3 / D post-training | `actor_bc.py`, `post_training_observer.py`, `adept_*bc.py`, Slurm launchers | 40k alternating mixed-policy BC at ADR 20; fresh critic; 20 frozen-actor epochs at final goal and log-std -2; then separate-trunk PPO with the reported optimizer settings. |
| A.4 observations | Stage-specific builders in the two environments | KUKA flat dimensions are asserted by construction: 391/438, 392/280 and 206+images. Flexiv/Sharpa is architecture-only and is not registered as a simulator task in this KUKA-first branch. |
| B C-space Fabric | `adept_cspace_fabric.py` | Full 23-D target interface and published metric/component constants implemented on NVIDIA FABRICS. Attractor gains omitted by the paper are marked inherited/inferred. |
| C PBT | `pbt.py`, `pbt_observer.py`, 16-worker Slurm array | Population, interval, rank fractions, parameter mutations, true-objective ranking and recipient ADR preservation implemented. Approximate-frame tolerance is exposed because no numeric tolerance is published. |
| E efficiency | Experiment concern | Paper runtime/throughput figures are evaluation targets, not algorithm settings. No parity claim before full runs. |
| F ablations | Experiment concern | Scratch/direct/low-LR/full recipes remain to be run with matched seeds and steps. |
| G losses | `post_training.py` | Mahalanobis distribution loss, tight 8-keypoint auxiliary loss, weights 1/20 and z-mask 0.08/0.02/0.1 implemented. |
| H student | `student.py`, `vision_dagger.py` | Shared ResNet-style encoder, bidirectional cross-attention, optional 5-finger tactile CNN+FiLM, 512x512 fusion and 1024 LSTM implemented. ADEPT does not publish image resolution, exact ResNet checkpoint, total DAgger iterations, or LR; exposed defaults are inherited/inferred. |
| I randomization | `randomization.py` plus ADEPT configs | Published physics, layout, Gaussian observation noise, colors, and ±3 cm/±3° camera jitter encoded. Runtime ranges still require distribution sampling audits in Isaac. |
| J qualitative grasps | Evaluation concern | Requires trained checkpoints and rollout capture; no code-only parity claim. |

Generated FMB files are intentionally ignored and reproducible from
`scripts/generate_adept_fmb_assets.py`. Replace them with measured FMB CAD before
claiming geometry or contact-dynamics parity. The GPU smoke must also pass before
starting long training; unit tests do not validate USD/PhysX behavior.

## Primary references

- ADEPT v1: https://arxiv.org/abs/2608.19182v1
- DextrAH-G: https://arxiv.org/abs/2407.02274
- DextrAH-RGB: https://arxiv.org/abs/2412.01791
- NVIDIA FABRICS: https://github.com/NVlabs/FABRICS
- Play2Perfect: https://arxiv.org/abs/2606.26428
- SimToolReal: https://arxiv.org/abs/2602.16863
- Functional Manipulation Benchmark: https://github.com/rail-berkeley/fmb
