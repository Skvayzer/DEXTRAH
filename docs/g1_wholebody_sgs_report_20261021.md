# Whole-body G1 + Revo2 reposing with SGS — report, 21 October 2026

Unitree G1 (floating, frozen pretrained SONIC whole-body controller) with a BrainCo
Revo2 right hand reposes 1200 procedural tool-like objects (6 types, SimToolReal /
Play2Perfect generator) to random 6-D goals. Simulator: mjlab (MuJoCo Warp) with
per-world object physics. All numbers are simulation results on training objects;
nothing here is a held-out or real-robot result.

## Status in one paragraph

A fixed-body BPS+touch teacher (run 1369, ~20% evaluation success) initialized the
frozen-SONIC policy (run 1377, the run 558/560 recipe: copied recurrent task
features, finger head, value head, normalizers and critic; zero-initialized 64-D
SONIC latent adapter). Trained 48 h with Success Guided Sampling (SGS, ported to
the whole body for this run) and spawn-validated starts, it reached **57.5%
success (deterministic policy, first episode per env, 12 000 episodes)** with 7.4
goals per episode on average, versus ~20% for the fixed-body teacher on the same
task. Failures concentrate on specific object geometries (thin handles, thin
markers, erasers) and are mostly drops after lifting.

## 1. Setup (run 1377)

| Item | Value |
|---|---|
| Robot / control | G1 floating base, frozen SONIC; policy outputs 64 latent residuals (x0.1 before FSQ) + 6 Revo2 finger commands at 60 Hz; physics 120 Hz; touch 70 Hz |
| Initialization | teacher `0_bps128_touch70_fixed_mjlab_sgs_1369/nn/complete_55651467264.pth` (`--teacher-checkpoint`), task from run 1225 (`--task-run`) |
| Objects | 1200 (hammer 200, screwdriver 200, marker 100, spatula 200, eraser 100, brush 400), per-world geometry, mass, inertia |
| Starts | SGS (reaching / stable grasp / near goal; 32 768 configurations), spawn-validated reaching starts; every 16th env evaluates on fresh validated starts (`sgs/eval_*`) |
| Optimization | SAPG (6 blocks), 22 080 envs, minibatch 35 328 (10 per epoch), adapter LR 3e-4, task LR 1e-5 |
| Scale | 14 934 epochs, 5.28 B transitions, 48 h, ~31k transitions/s, 0 numerical failures |
| Checkpoints | `outputs/0_mjlab_wb_sgs_teacher1369_1377/nn/complete_5275883520.pth` (+ `.sgs.pt`), best eval `best/model.pth` |
| W&B | `skvayzer/adept`, run `unique_id_0_mjlab_wb_sgs_teacher1369_1377` |

Code: `docs/sgs_success_guided_sampling.md` (SGS incl. whole-body port),
`scripts/train_g1_sonic_sapg.py` (`--teacher-checkpoint`, `--task-run`, `--sgs*`,
`--best-metric`, `--minibatch-size`), `scripts/slurm/train_g1_sonic_sapg.sbatch`.

## 2. Training curve (evaluation envs: fresh validated starts, stochastic actions)

| Hours | 2 | 4 | 8 | 12 | 16 | 24 | 32 | 40 | 47.5 |
|---|---|---|---|---|---|---|---|---|---|
| Success (>= 1 goal) | 1.5% | 12.5% | 20.0% | 29.5% | 34.6% | 39.1% | 42.7% | 46.6% | 50.2% |

Still rising at the end (~0.5 points per hour over the last 8 h).

## 3. Evaluation of the final checkpoint

`scripts/eval_wholebody_checkpoint.py` (job 1391; `outputs/eval_wb_1377_ep14934/`):
the run's task and simulator settings (domain randomization as in training), every
episode from a fresh spawn-validated start with a uniformly random object, 12 000
envs, first episode per env as the headline (episodes still running at the 100 s
cap have reached a goal and count as successes; their goal counts are lower bounds).

| Policy mode | Success | Goals / episode (lower bound) | Lifted | Object dropped | Robot fell |
|---|---|---|---|---|---|
| Deterministic SAPG leader (deployable) | **57.5%** | 7.4 (median 2) | 87.9% | 43.4% | 0.08% |
| Leader, sampled actions | 52.6% | 5.7 | 84.7% | 52.1% | 0.06% |
| Sampled actions, per SAPG block (= training metric) | 51.5% | 5.3 | 84.6% | 52.1% | 0.03% |

Per object type, deterministic leader:

| Type | Success | Goals / episode | Lifted | Dropped |
|---|---|---|---|---|
| hammer | 71.1% | 8.4 | 92.0% | 23.6% |
| screwdriver | 72.0% | 8.9 | 89.0% | 62.8% |
| spatula | 62.0% | 8.4 | 90.6% | 36.2% |
| brush | 58.1% | 8.0 | 87.5% | 38.6% |
| marker | 40.7% | 5.0 | 77.8% | 81.6% |
| eraser | 8.8% | 0.3 | 83.5% | 40.6% |

109 of the 1200 objects never succeeded in this evaluation (about 10 episodes each).
The last row of the first table agrees with the training curve's end (50.2%),
which cross-checks the evaluation.

Videos (domain randomization off, deterministic leader, starts from scratch, 2 objects
per type, 30 s, full body + two hand close-ups): `scripts/record_wholebody_videos.py`,
`outputs/wb_videos_1377_ep14934_nodr/videos/`. Successful episodes chain 10-17 goals;
the failures in the clips are drops after lifting (thin marker, thin brush, eraser).

## 4. Where the policy loses: object geometry

From the SGS state (last 100 outcomes of each of the 10 923 reaching configurations):

* Object features alone (handle/head dimensions, mass, centre-of-mass offset, type)
  predict each object's success with R^2 = 0.87 (5-fold cross-validated random
  forest). The object explains 64% of the variance across configurations; the rest is
  the start pose.
* By type (reaching success, share of objects below 5%): hammer 61% / 1%,
  screwdriver 55% / 1%, spatula 44% / 15%, brush 43% / 14%, marker 31% / 33%,
  eraser 2% / 89%.
* Thin handles lose: < 1.6 cm thick 19% vs 2.0-2.9 cm 56-58%; markers < 1.8 cm 0-8%;
  handles shorter than 9 cm 31%. Erasers (flat boxes, no head) fail at every size.
* Not a capacity limit: compared with the fixed-body teacher the whole-body policy
  solved 76 objects the teacher never solved and regressed on 1; about 184 objects
  (thin markers, erasers, thin brushes) are near zero for both.

## 5. What SGS does here

* It produced the intended reverse curriculum: near-goal starts 61% success at 2 h,
  held-grasp starts 53% and from-scratch starts 25% at 4 h, while evaluation went
  0 -> 12.5%.
* It kept training at the frontier: from-scratch training success stayed at 57-61%
  from 16 h on while the uniform evaluation rose from 35% to 50%; never-solved
  configurations fell from 97% to 12%.
* Its causal share of the result is not measured (no no-SGS control on the whole body).

Weak points found (no change made yet; candidate opt-in fixes):

1. Held starts were harvested once, ~45 min into the run (env step 3 135), from a
   policy that lifted in 2-26% of episodes, then frozen. 258 objects (22%) have none;
   they reach 24% from scratch vs 49% for the rest. Fix: periodic re-harvest with
   per-object quotas for objects without held starts.
2. The zero-filled 100-outcome window is long for whole-body throughput: ~3 tries per
   configuration per hour, so windows needed 30+ h to fill (32% still had < 100 tries
   at the end) and success was underestimated for most of the run. Fix: success =
   successes / tries (up to 100).
3. Objects below 5% success (18% of objects) receive 3.9% of samples; objects without
   held starts (22%) 6.5% - by the kernel's design, but with no held-start bridge they
   stay stuck.
4. About half of all training episodes end with a dropped object, even from near-goal
   starts (52%); SGS does not target drops.

## 6. Robot body representation (new, opt-in)

What the policy already gets from the robot model: palm-centre pose and the five
fingertip pad positions relative to the palm (link poses, i.e. URDF forward
kinematics), object keypoints relative to the palm, raw joint state, and SONIC's
proprioceptive history. Missing: where the hand is relative to the object's surface -
shape enters only as BPS-128 in the object's own frame and as the handle box.

Added `--hand-surface-obs` (`dextrah_lab/wholebody/hand_surface.py`, commit 207f0df):
for 17 hand points (palm centre, 5 fingertip pads, 11 finger and thumb link origins)
the vector to the nearest point on the object's surface (per-world primitives; boxes
and capsules in closed form), in decimetres with the norm clipped at 30 cm: 51 inputs
for actor and critic, with their own running normalizer. The actor uses the same
delayed and noised object pose as its other object observations; the critic the true
pose. On hardware the robot side is encoders + URDF; the object side needs a mesh and
pose (as BPS already does; SimToolReal's SAM 3D + FoundationPose pipeline provides both).

The inputs enter the task LSTM and the critic's first layer through zero-initialized
columns, so a policy expanded with them acts exactly as before; `--init-weights`
from run 1377 expands its checkpoint and verifies at startup that actor means, actor
value and critic value do not change with the new inputs. Related work: RobustDexGrasp
(finger-to-surface vectors), InterMimicGen (link-to-surface vectors on G1).

Smoke test (job 1392, 40 epochs, 21 120 envs): 1294 actor / 1316 critic inputs,
run 1377 loaded into the expanded network, new inputs inert at initialization
(max output difference 0.0), SGS state continued, checkpoint + `.sgs.pt` saved,
5.9 GB GPU memory free.

Planned comparison (not run yet): fine-tune run 1377's final checkpoint with and
without `--hand-surface-obs`, same fresh optimizers, same SGS state (`--sgs-state`),
same hours; compare evaluation success overall and on thin-handle objects.

## 7. Comparison with SimToolReal (summary)

Our reposing task is SimToolReal's: identical procedural generator and size
distributions, identical reward code and coefficients, the same grasp-region input
(handle box as `object_scales`, object frame at the handle centroid). Neither marks
the head non-graspable. SimToolReal trains on primitives and uses scanned meshes only
for DexToolBench evaluation twins (12 tools, CoACD collision) and for pose tracking.
Differences: robot (KUKA + 22-DoF Sharpa hand vs G1 whole body + 6-DoF Revo2 via
SONIC), our extra inputs (BPS-128, touch, SONIC body history), our evaluation (random
goals on procedural objects, simulation only) vs theirs (human-video goal sequences,
real robot).

## 8. Next steps

1. Run the hand-surface comparison (two fine-tunes from run 1377).
2. Opt-in SGS fixes 1-2 above (re-harvest with per-object quotas, count-based success).
3. Evaluate on the 12 DexToolBench meshes (needs mesh objects in the mjlab scene).
4. Precision budget of the object interface (pose noise, latency, dropouts, coarse
   shape) on the final checkpoint - simulation only, no training.
