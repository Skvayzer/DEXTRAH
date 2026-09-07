# G1 + Revo2 ADEPT/SAPG reposing

This task combines three pieces without pretending unpublished ADEPT code is
available:

- Play2Perfect's working G1 + BrainCo/Revo2 dynamics, reposing MDP, SAPG fork,
  and factorized large-scale scene construction;
- an independent ADEPT-style reduced geometric fabric for the seven right-arm
  and six independent Revo2 joints; and
- a frozen five-component human-motion PCA prior retargeted from DexYCB.

The registered task is `Adept-G1-Revo2-SimToolReal-Repose`. It is designed for
one RTX 6000 Ada and defaults to 24,576 environments.

## Pinned inputs

- Play2Perfect: `70e79b5e53f912ef04af294ff8f61ac1c7f42160`
- SimToolReal reference checkout: `313d5aea1f507c6cfe097b672b62945d7b0bbff5`
- Revo2 PCA SHA-256:
  `8cea2fe7602958bbefec826fd331a100c15145c7dd837c68406a07895a2237b3`

These are the procedural training objects from SimToolReal, not its 12 scanned
DexToolBench evaluation objects. The Play2Perfect generator and distribution
tables used here are code-identical to the official Isaac Sim implementation
apart from module names, documentation, and the temporary-directory default.
The generator has 12 size/shape distributions across six semantic families:
hammer 2, screwdriver 2, marker 1, spatula 2, eraser 1, and brush 4.
At `num_assets_per_type=100`, the full pool is therefore **1,200 generated
URDFs**, distributed across the environments by factorized PhysX replication.
The environment checks this count at startup and aborts on a mismatch.

## Control and learning setup

SAPG emits 13 actions: seven arm increments and six absolute finger targets.
The pinned Play2Perfect action pipeline applies its original delay, arm target
accumulation, and moving-average smoothing. Its nominal targets remain separate
from the applied fabric targets and are observed through `prev_action_targets`.
This avoids accumulating fabric or PCA corrections back into the policy target.

The hand target gets a fixed 5% residual correction toward the frozen PCA:
`q_target = q_nominal + 0.05 * (project_PCA(q_nominal) - q_nominal)`.
All six finger outputs remain trainable. Away from clipping, the five retained
directions have unit gain and the sixth has gain 0.95. The PCA does not force
every grasp to follow the recorded human motions. Configured weights must stay
below one.

At every 60 Hz policy step, the reduced fabric performs two 120 Hz integration
steps; total integrated time must equal the physical policy step.
Its terms cover target attraction, joint limits, damping, velocity/acceleration/
jerk limits, table avoidance, and selected robot self-collision pairs. Measured
PhysX link Jacobians are mapped into the 13 independent coordinates, including
the five Revo2 mimic joints. The manipulated object is deliberately excluded
from fabric avoidance so the hand can make contact. The physical torso collider
remains active as a simulation safety backstop.

The policy receives the original 92-dimensional Play observation plus fabric
position, velocity, and acceleration (39 values), for 131 actor inputs. The
asymmetric critic has 153 inputs. Action and observation delay, physics/domain
randomization, reset sampling, and the original reposing rewards remain active.

An older Play2Perfect checkpoint cannot be loaded strictly: although both
policies have 13 actions, this policy has 39 additional observations. Start the
first run from scratch unless an explicit observation-layer migration is added.

## Validation and launch

On the workstation:

```bash
cd /home/konstantin.smirnov/data1/DEXTRAH-ADEPT
bash scripts/setup_g1_adept_sapg.sh
sbatch scripts/slurm/smoke_g1_adept_sapg.sbatch
sbatch scripts/slurm/train_g1_adept_sapg_single.sbatch
```

The setup check runs controller/geometry/adapter/PCA tests and a kinematic
validation with finite-difference Jacobians. GPU smoke testing has also covered
environment construction, one SAPG update, a full 10-second episode, reset,
checkpoint writing, and actor/critic dimensions. Those checks reduce launch
risk but do not prove that the learned policy will succeed; learning curves and
evaluation rollouts are still required.

`scripts/validate_g1_adept_control.py --headless` additionally exercises the
actual PhysX hand with open/close commands, nominal target accumulation, subset
reset behavior, and SAPG observer leader/terminal-stat handling. Run it inside
a GPU allocation. This scripted response is not a learned grasp evaluation.

The long job defaults to an 8-billion-frame ceiling, periodic checkpoints, and
W&B enabled. Version 2 uses a fresh `absolute_v2` run name: the previous relative
action checkpoint is not automatically resumed. Same-run auto-resume remains
available for Slurm requeues. Do not pass a v1 checkpoint to the v2 controller:
matching tensor dimensions do not imply matching action/observation semantics.
Useful W&B series include:

- `rewards/step`, episode reward, and policy/value losses for optimization;
- `episode_final/ever_lifted` and `episode_final/any_success` for the fraction
  of completed leader episodes with a lift or at least one reposing success;
- `episode_final/successes` for goals reached per completed leader episode;
- `episode_cumulative/lifting_rew` and `episode_cumulative/lifting_bonus` for
  comparison with the original Play2Perfect run;
- `fabric/min_clearance_m`, `fabric/active_constraint_fraction`, and
  `fabric/penetration_fraction` for collision safety;
- `fabric/tracking_error_rad` for tracking the applied fabric target;
- `control/nominal_tracking_error_rad` and nominal/measured hand fractions to
  distinguish upstream command demand from physical movement; and
- `fabric/pca_prior_weight` and `fabric/pca_correction_rad` for prior strength.

The P2P `EnvStatsAlgoObserver` now handles SAPG's leader boundary, terminal
statistics, and per-block success summaries. Environment series all use frame
steps (W&B `global_step`), with no competing `/iter` or `/time` aliases. Fabric
scalar diagnostics aggregate all environments; episode metrics use the leader
block, matching the original P2P observer. Terminal clearance is copied before
auto-reset so resetting the accumulator cannot turn logged values into infinity.

Sparse `auxiliary_stats/off_on_grad_similarity` NaNs can occur when an SAPG
update has no off-policy gradient. This is an undefined diagnostic cosine, not
by itself evidence of a non-finite environment or controller state. Treat NaNs
in rewards, losses, observations, fabric metrics, or parameters as failures.

For a geometry-only audit, use:

```bash
python scripts/visualize_g1_adept_fabric.py --validate-only \
  --urdf /home/konstantin.smirnov/data1/play2perfect/unitree_ros/robots/g1_with_brainco_hand/g1_29dof_mode_15_brainco_hand.urdf
```
