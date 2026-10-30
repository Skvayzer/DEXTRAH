# M: frozen-SONIC SAPG manipulation expert, run 1454, epoch 12,662

Grasps one of 1,200 procedurally generated tools (hammer, screwdriver, marker, spatula, eraser, brush) from a table
and moves it through 6-D goal poses with the G1 humanoid's Revo2 right hand. SAPG (6 blocks), recurrent; outputs a
64-D latent into the frozen pretrained SONIC whole-body controller plus 6 finger commands; 60 Hz; trained in mjlab
(MuJoCo Warp) at condim 3 with the table top at 0.80 m.

## Deterministic evaluation (leader policy, 12,000 worlds, first episode per world)

| Setting | Success (≥ 1 goal) | Lifted | Robot fall |
| --- | --- | --- | --- |
| condim 3, table top 0.80 m (as trained) | 72.0% ± 0.8 | 92.8% | 0.17% |
| condim 4, table top 0.80 m | 72.6% ± 0.8 | 93.0% | 0.19% |
| condim 4, table top 0.70–0.90 m (arm follows the table) | 57.9% ± 0.9 | 78.7% | 0.38% |

Full summaries: `eval_*.json`. Training-time best-checkpoint record: `best_eval.json`.

## Contents

- `best/m_policy_weights.pth` (52 MB): the actor (with its input / value normalisers) and the asymmetric critic,
  `{0: {model, assymetric_vf_nets, scaler, central_val_stats, epoch, frame, last_mean_rewards}}`. Every tensor is
  identical to the training checkpoint `outputs/0_mjlab_wb_imgen_dualclip_1454/best/best_epoch12662.pth`
  (604 MB, sha256 b0ce5417…4ee7), from which the resume-only state was dropped: optimizer moments, recurrent states
  and observations of the training worlds, trackers. So it serves inference and initial weights, not an exact resume.
- `task_contract.json`, `params/env_resolved.yaml`, `params/agent.yaml`: the run's task, environment and agent
  configuration (needed to rebuild the environment and the network).

## Dependencies (not in this folder)

- The network is rebuilt from the fixed-body teacher named in `task_contract.json` (`source_checkpoint`: run 1369,
  `complete_55651467264.pth`, sha256-checked: 697f75e6…d8b7), see `scripts/eval_wholebody_checkpoint.py:build_policy`
  and `dextrah_lab/wholebody/sapg_network.py:load_touch_sources`.
- Pretrained SONIC models (`G1-SONIC-models/checkpoint/SONIC/models/sonic_manipulation_base`, sha256 in the contract).
- The evaluation script also reads the success-guided sampler state `best_epoch12662.sgs.pt` (197 MB, the start
  pool and 21,845 holding states) next to the checkpoint; it is not included here.
