# Port plan: Isaac Lab (PhysX) → mjlab (MuJoCo Warp), 2026-10-13

## Goal

Run the current G1 + Revo2 frozen-SONIC SAPG reposing task in mjlab with the **same setup**:

- Same rewards, goals, terminations, observations (1243 actor / 1265 critic), action layout (64 latent + 6 fingers) and 70 Hz touch model.
- Same robot (our URDF `g1_29dof_mode_15_brainco_hand.urdf`), table, 1200-object bank and BPS128.
- Same rates (120 Hz physics / 60 Hz policy), PD gains, effort limits and delays.

Then test whether it is faster at many envs, and whether the physics is close enough to continue from the Isaac checkpoints.

Isaac stays untouched as the reference. Training was stopped at run 1105 `complete_17485953024.pth` (epoch 118,600); best is at epoch 117,926.

## What cannot be "the same"

The contact solver differs:

| Aspect | PhysX | MuJoCo |
|---|---|---|
| Contact | Rigid, TGS 8 iterations, contactOffset 2 mm / restOffset 0 | Soft (solref/solimp), small penetration |
| Friction combine | Average | Max |
| Noslip | — | No noslip solver in MuJoCo Warp |
| Velocity limit | Joint velocity clamp | No joint velocity clamp |

Tactile **post-processing can be identical** (pad masks, frames, 70 Hz quantized model), but the raw contact forces will differ. The parity tests below measure by how much. The checkpoint is expected to need fine-tuning in mjlab.

## Design decisions

1. **Software stack.**
   - Separate venv `venvs/mjlab`: mjlab 1.6.0 (pins mujoco/mujoco-warp 3.11, torch ≥ 2.14).
   - The driver is 580.178 (verified), so the cu130 torch wheels work.
   - The Isaac env (torch 2.7) is not modified.
2. **Direct env, not managers.**
   - Our env is DirectRLEnv-style, and mjlab's own benchmarks show more than 50% manager overhead.
   - Build `G1SonicTouchMjEnv` directly on mjlab `Scene`/`Simulation`.
   - Reuse every simulator-free module unchanged: SONIC, timed history, source actions, touch observation model, contact geometry, BPS bank, body termination, SAPG networks.
   - P2P reward/termination utils are pure torch. Obs/reset utils use only isaaclab math; MuJoCo uses the same wxyz quaternion convention, so these get thin shims.
3. **Robot asset.**
   - Compile **our** URDF (not mjlab's capsule G1), keeping convex-hull mesh colliders like PhysX.
   - `fusestatic` off for the 10 `*_tip` bodies (fingertip observations), or replace them with sites plus tiny boxes. Decide after the parity check.
   - Finger coupling is target-level in our env (`native_finger_coupling=False`), so **no mimic/equality constraints** are needed.
   - Self-collision off.
4. **Gains.** `BuiltinPositionActuatorCfg` with the exact kp/kd/armature/effort per joint group from `wholebody/actuators.py`:
   - Hand: kp 1200, kd 25.
   - Effort from the URDF.
5. **Friction parity.**
   - Base friction 0.5 everywhere.
   - PhysX averages the fingertip-marker 1.5 against the object's 0.5, giving 1.0. In MuJoCo, set the marker geoms to 1.0 so the effective value matches.
6. **Object bank, per env.** One floating body with 4 primitive slots: handle-box, handle-capsule, head-box, head-capsule.
   - Primitive geom types are global in MuJoCo Warp; sizes, positions, mass and inertia can be set per env.
   - Unused slots shrink to a tiny box/capsule enclosed in the used handle. Never size 0.
   - Mass, COM and inertia are written from the generator's exact values, followed by `set_const`.
   - Same `i % 1200` assignment.
   - `VariantEntityCfg` is **not** used: it silently keeps variant-0 primitive sizes.
7. **Table.** A box with no table-ground collision; height ±1 cm per reset via mocap or per-env `geom_pos`.
8. **Tactile.** One custom pass over the MuJoCo Warp contact buffer per physics step (`d.contact` + `mjwarp.contact_force`) produces:
   - Per-contact normal and friction force plus contact point, for the 5 right distal links × {object, table, ground}. These are fed into the existing `aggregate_pad_*` masks and the `TouchObservationModel`.
   - Per-link force for the 30 monitored links against the table.

   Native mjlab ContactSensors cost O(sensors × contact pool) and their history drops contact points, so they are used only as a cross-check in tests.
9. **RL.** An rl_games vecenv wrapper with the same obs/states/extras keys (`episode_final/*`, `body_table/*`), the same SAPG fork, W&B, checkpointing and the `RESET_BEST` logic.

## Test stages

Each stage must pass before the next, and runs one GPU job at a time.

| Stage | What | GPU | Pass criterion |
|---|---|---|---|
| 0. Stack | venv; mjlab G1 demo for a few seconds; CUDA graphs on; SONIC and SAPG model forward under torch 2.14 vs 2.7 on saved inputs | tiny | Graphs enabled; SONIC/policy outputs match to ~1e-5 |
| 1. Assets | URDF→MJCF; compare body/joint names and order, limits, masses, inertias and collider counts with the Isaac articulation; object slots vs generator (mass, COM, inertia, extents) for all 1200 | CPU | Exact match (inertia to float tolerance) |
| 2. Dynamics | Per-joint-group PD step responses (legs, waist, arms, wrist, fingers), same targets in Isaac and mjlab, small env count; SONIC zero-residual standing at the table for 30 s, plus walking | small | Rise time and steady-state error within ~10%; no falls standing |
| 3. Contact & touch | (a) object drop and settle on table; (b) scripted static grasp of a fixed object: per-pad Fn/shear; (c) slip under gravity and shake; sweep cone (pyramidal/elliptic), impratio, condim 3/4, solref. (d) **Zero-shot**: 1105 best policy, 60 envs × 60 s, same metrics as the Isaac recording (goals/min, lifted rate, falls, per-finger Fn/shear histograms, contact fraction) | small | Pick the contact settings closest to Isaac; report the gap honestly |
| 4. Scale | 1024 / 4096 / 8192 / 9216 / 12288 envs: physics-only and full steps/s (with SONIC and obs), VRAM, startup time; nconmax/njmax/CCD sweep; overflow flags must stay 0 | full | Report vs Isaac: 38k frames/s total, 59k env-step, ~35 min startup, 28 GB |
| 5. Training | Short fine-tune (few hours) from 1105 `complete` in mjlab with W&B and `RESET_BEST`; best checkpoint verified after tracking start | full | Return and goals recover toward Isaac levels; no crashes or overflow |

**Decision after stage 4:** switch only if mjlab is clearly faster at our scale and stage 3 shows grasping and touch are usable. Otherwise keep Isaac and use mjlab as a sim-to-sim robustness check.

## Known risks (from source reading)

- Contact capacity: overflow silently drops contacts. nconmax, njmax and the CCD scratch must be sized explicitly. Others report scaling problems above about 12k G1 envs.
- Box-box contacts go through GJK/EPA (float32 precision issues are open upstream).
- Ada-specific crash history: mujoco_warp #1543, fixed in 3.10.0.3.
- No determinism.
- Python overhead plus SONIC inference every step.
- PhysX velocity clamp and friction-combine differences.

## Status, 2026-10-13 (late)

Isaac backend unchanged; `--backend mjlab` / `BACKEND=mjlab` selects the port.

* Asset parity vs the Isaac dump (job 1112): 0 issues (`scripts/mj_asset_parity.py`).
* Contact capacity (job 1126): <= 49 contacts and <= 150 constraint rows per
  world, no overflow at 9216 envs; defaults nconmax 96 / njmax 600.
* Spawn fix (user decision, opt-in `--spawn-on-table`, both backends): no
  table intersection, oriented footprint 1 cm inside the table. Isaac passive
  untouched falls 32% -> 17.7% (marker/screwdriver ~32%: round handles roll).
* Object launches on mjlab, root causes (single-world mujoco_warp replay of
  captured steps, `outputs/diag/`):
  1. Free joint at the URDF link origin couples a fast spin into translation
     -> object body frame at the COM; adapter keeps the link pose API.
  2. Objects spawn overlapping the open hand; MuJoCo solves contacts against
     the explicit Revo2 servo force on zero-armature 9 g finger links
     (60-160 N) -> `ContactParams.hand_armature = 1e-3` (MuJoCo-only).
  Spawn-window peak object speed > 5 m/s: 8.7% -> 0% (Isaac 0.77%).
* Zero-shot run-1105 policy on mjlab (600 envs x 30 s, deterministic, 1 cm
  tolerance): ever lifted 74% (Isaac training 75%), any goal 8% (Isaac 33%),
  0.17 goals/episode (Isaac 2.9). Remaining gap is precision manipulation;
  needs fine-tuning on mjlab.
* Training path works (smoke run 1151, 9216 envs, resume 1105 + RESET_BEST):
  57k env-steps/s step+inference (Isaac 52.6k), best checkpoints saved after
  the 200-epoch delay.
* Open: Stage-2 PD step-response protocol; one unreproduced segfault at env
  build (job 1146).
