# Success Guided Sampling (SGS) for the fixed-body G1 + Revo2 task

Paper: O. Zhang, M. Guaman Castro, P. Yin, I. Dagnino, A. Gupta, R. Scalise, B. Boots,
"A Balanced Data Diet: Addressing the Exploration Bottleneck in Mega-Scale RL for
Robot Control", CoRL 2026, arXiv:2610.12465, https://sgs-rl.github.io (code not released).
Read from the arXiv LaTeX source (all sections and appendices).

SGS is an additional, opt-in reset sampler (`--sgs`, sbatch `SGS=1`). PPO/SAPG, the
reward, observations, actions and physics are unchanged; without `--sgs` nothing
differs from before. Code: `dextrah_lab/sgs/sampler.py` (the algorithm),
`dextrah_lab/sgs/mj_task.py` (task configurations on the mjlab fixed-body task),
three hooks in `dextrah_lab/mj/env.py`, flags in `scripts/train_g1_touch_continuation.py`.

## Algorithm (paper -> code)

| Paper | Implementation |
|---|---|
| Fixed set of N task configurations tau = (s0, g, e), sampled before training (Sec. 3.2) | `num_configs` (default 32768, the manipulation value); reaching configurations drawn at startup; held states collected at the start of the run, then frozen (see below) |
| Last H Boolean outcomes per configuration in a circular buffer, p_i = their mean (Sec. 3.2, App. B.2) | `SuccessGuidedSampler.outcomes` (N x H bool), H = 100. Window starts as H failures (project page: "0 of 64 successes" at iteration 0) |
| w_i = (p_i+eps)^(kappa t) (1-p_i+eps)^(kappa(1-t)), Beta(1+kappa t, 1+kappa(1-t)) mode form (Eq. 1) | `sgs_logits` |
| w~_i = max(w_i, eps), l_i = log(w~_i + eps) (App. B.2) | `sgs_logits` |
| p(i) = softmax(l_i / T), T_eff = max(T, 1) (Eq. 2, App. B.2) | `SuccessGuidedSampler.probabilities` |
| Next configuration sampled i.i.d. at every episode end (Sec. 3.2) | `torch.multinomial(..., replacement=True)` in `before_reset` |
| Manipulation hyperparameters: N = 32768, H = 100, T = 2, t = 0.5, kappa = 1, eps = 1e-4 (Sec. 3.2) | defaults (`--sgs-*`); locomotion values in `sampler.LOCOMOTION` |
| Boolean task success at the end of the episode | at least one goal reached in the episode (`_successes > 0`, the existing `any_goal_success_rate`) |
| Reset buffer balanced across Reaching / Stable Grasp / Near-Goal (App. A.2) | three equal blocks of the configuration set (`--sgs-families`) |
| Evaluation on freshly sampled resets from the training distribution (Sec. 4.3); simulation success reported from Reaching (Table 2) | every 16th env always resets from a fresh sample of rho with a uniformly random object: `sgs/eval_*` (compare these with runs 1254/1277 `episode_final/*`) |

Check: `tests/test_sgs_sampler.py` reproduces the project page's displayed weights
(0.14 / 0.99 / 0.73 at 0/64, 27/64, 59/64 successes) from these equations exactly.

## Task configurations here

tau = (object asset e; robot joint state, table height, object state s0; first goal g).
mjlab stores each world's object as per-world model fields (geom sizes/poses/bounds,
mass, inertia), so the object is part of the configuration and is rewritten when an
episode starts with another asset (`MjSgsTask.assign_objects`, with the task's
per-env object buffers: COM offset, mass, scale, keypoints, BPS features).
Derived solver constants (invweight0, subtree mass) stay shared from world 0 exactly as
in the fixed assignment (startup check refuses to run otherwise). The tables are
verified against the scene at startup and every 2000 steps.

* `reaching`: our unchanged rho (object dropped 10 cm above the table at a random
  pose, robot near its default pose, absolute goal). Drawn by running the task's own
  `reset_utils` sampling functions with their writes captured; assets balanced.
* `stable_grasp`: object held in the hand above the table; goal = a fresh sample of
  rho's goal distribution.
* `near_goal`: object held in the hand inside the goal volume; goal = the object pose
  moved by f ~ U[0, 1.1] of one of the task's own goal steps (position offset up to
  f x 0.1 m per axis, rotation up to f x 90 deg about a random axis, clamped to the
  goal volume). Mirrors the paper's Near-Goal fractional range [0, 1.1] along the path
  (0 = at the goal).

Held = object lifted (task latch), thumb pad and at least one other pad with > 0.25 N,
object origin > 5 cm above the table, COM speed < 0.5 m/s, spin < 5 rad/s, not
terminating this step. Up to 32 states per family per step, at most
2 x (family size / assets) per asset; the full MuJoCo qpos/qvel of the world, drive
targets and setpoints are stored, so a restored grasp is the exact physical state.
A held family gets sampling probability once it is complete (or after
`--sgs-harvest-max-steps`, default 6000 env steps, with what was collected).

## Deviations from the paper (all forced by the task or setup)

1. Stable Grasp / Near-Goal states: the paper places a parallel-jaw gripper with IK
   around task grasp points; the dexterous Revo2 hand has no such generator, so held
   states come from the policy's own rollouts at the start of the run.
2. "Near goal": our goal is a free pose (not an assembly path); defined as a fraction
   of the task's own goal step, as above.
3. SAPG (six policies with different exploration) instead of plain PPO; all blocks
   share one set of success estimates.
4. Evaluation envs (6.25%) run online inside training and their transitions are also
   used by PPO; the paper evaluates separately.
5. The reward is the task's existing shaped reward (the paper's shared sparse reward is
   a separate design choice, not part of SGS); no gravity curriculum (the paper's own ablation
   found it unnecessary).

## Metrics (W&B / TensorBoard)

* `sgs/eval_any_goal_success_rate`, `sgs/eval_goals_reached_count`,
  `sgs/eval_ever_lifted_rate`: fresh rho (Reaching) episodes, rolling 3000. Use these to
  compare with non-SGS runs; `episode_final/*` now mixes the three families.
* `sgs/<family>/train_any_goal_success_rate`, `.../train_fall_rate`,
  `.../sampling_share`, `.../success_rate_mean`, `.../configs`.
* `sgs/effective_configs_fraction` (1 / (sum p(i)^2 N)), `sgs/configs_p0_fraction`,
  `sgs/configs_p1_fraction`, `sgs/harvest_done`.

## Checkpoints

Each network checkpoint `X.pth` gets `X.sgs.pt` (configuration set + outcome windows).
`--resume X.pth --sgs` continues them when `X.sgs.pt` exists (window shape and family
layout must match; kernel hyperparameters may be retuned).

## Best checkpoint

Under SGS the training reward describes the reset mix and falls when SGS moves to harder
configurations (run 1301: best 2261 at hour 2, reward 1577 at hour 8 while eval success
rose 5.5% -> 7%). `--best-metric sgs_eval_goals` (sbatch `BEST_METRIC=sgs_eval_goals`;
or `sgs_eval_success`) disables rl_games' reward best and saves `nn/<name>.pth` +
`best/model.pth` when the full 3000-episode window of the evaluation envs reaches a new
maximum, after `--best-delay-epochs`; `best/best_eval.json` records the values
(`dextrah_lab/sgs/best.py`).
