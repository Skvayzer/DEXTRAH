# Whole-body SAPG memory investigation — 2026-09-13

## Evidence and changes

- **517 failed**, not completed training: CUDA OOM at epoch 152 after the
  last completed update 151 / 29,687,808 cumulative transitions. SAPG's
  `augment_batch_for_mixed_expl -> filter_leader -> torch.cat` requested
  another 1.06 GiB with only 392 MiB free. Process usage was 46.90 GiB,
  including 29.67 GiB live PyTorch allocations and 2.75 GiB reserved but
  unallocated. The remainder was simulator/runtime memory. This failure was
  not the earlier missing-floor tactile assertion.
- Full-body actor/critic observations add 994 values each to the original
  249/271-value task observations. At 12,288 environments and horizon 16,
  those additions alone require about 1.46 GiB extra raw rollout storage.
  The unchanged SAPG implementation makes additional temporary copies when
  combining exploration groups and filtering leader/follower data. The
  first successful optimizer updates did not establish worst-case capacity.
- The old memory trace available in TensorBoard showed reserved/device
  memory rising, then a plateau over roughly epochs 31–101, then further
  increases before failure. This is not enough to identify a steady leak.
  Added stage-level live/reserved/active/split-block memory measurements,
  allocation retries and OOM counts in `memory_trace.jsonl`; no tensors are
  retained by the tracer. Detailed sampling is limited after 512 updates.
- **520** resumed 517's latest checkpoint at epoch 128 / 25,165,824 frames,
  with **9,216 environments** (six groups of 1,536), unchanged horizon,
  minibatch-to-environment ratio, rewards, observations, physics and touch.
  It completed 92 new updates and was deliberately checkpoint-stopped at
  epoch 220 / **38,731,776 frames**, not crashed. All progress was saved to
  `outputs/0_sonic_sapg_touch_520/nn/complete_38731776.pth`.
- During that measured interval, total usage settled near **27.43 GiB**,
  PyTorch reserved 14.70 GiB and live post-update allocations were around
  3.7–3.9 GiB. A one-time garbage-collection probe at epoch 160 released
  **zero GPU tensor bytes**. This does not support unreachable Python tensors
  as the explanation at that point, but does not prove every future workload
  has the same memory peak. No allocator workaround or periodic cache flush
  was used.
- Printed rollout+update throughput was approximately **40–41k transitions/s**.
  The complete measured continuation, including between-update logging,
  checkpointing and shutdown save, was **33,654 transitions/s**. Do not quote
  the former as fully end-to-end throughput. A 20-second utilization sample
  ranged from 0–94%; the GPU was not continuously saturated.
- Found a separate **resume inefficiency**: `torch.load(map_location='cuda')`
  also moved ordinary Adam's scalar step counters onto CUDA. Installed
  PyTorch preserves that placement for non-capturable/non-fused Adam, and
  its bias-correction code reads each counter using `.item()`. Runtime stack
  samples entered that path. Resume now loads through CPU, lets
  `load_state_dict` place model/momentum tensors on their parameter devices,
  and explicitly preserves CPU step counters for ordinary Adam (device
  counters remain supported for capturable/fused optimizers).
- This is **not an optimizer reset**. Both momentum buffers, counter values,
  learning rates and model parameters are inherited. Old episode buffers are
  discarded as before. Resume provenance now records the checkpoint path,
  SHA-256 and optimizer-counter devices, and does not label a resume as fresh
  architecture-migration optimizers.

## Validation and active continuation

- CPU **519**: 47 tests passed (memory tracing included).
- CPU **524**: 49 passed, one CUDA case skipped in the CPU allocation.
- GPU preflight in **527 and 528**: all three optimizer-placement tests passed,
  including exact next-update parameter equality on CUDA.
- Commits: `b83064d`, `bfbbaf2` (telemetry/tests), `69d640b` (resume counter
  placement), `2b88332` (exact provenance). External pinned Play2Perfect code
  and original teacher checkpoints remain unchanged.
- **527** did not reach checkpoint loading or training. It stalled inside
  Isaac's native simulation `play/reset` during scene initialization and was
  canceled after ten minutes. The corresponding stage took about 24 seconds
  in 520. Its native startup cause is not established; there was no model or
  optimizer update to lose. It must not be described as an active trainer.
- **528** is the bounded retry, resuming the complete 520 checkpoint with
  9,216 environments, source snapshot `321032e`, one RTX 6000 Ada (physical
  GPU 2, UUID `GPU-4a10e4cf-0808-67e3-468e-3e883b4f7ca1`), 48-hour allocation, online
  W&B, no epoch/frame cap or periodic evaluations. Atomic latest saves and
  source best-return selection remain enabled. The one-off GC experiment is
  disabled in this continuation; memory telemetry stays enabled.

- **528 has passed startup and is making real updates.** At the first live
  check it reached epoch 234 / 40,796,160 cumulative transitions. Online W&B
  reports `experiment_status=training`, finite actor/critic losses, and both
  best and latest checkpoints exist. The resume report confirms epoch 220 /
  38,731,776 transitions, both optimizers restored, 35 actor and 11 critic
  step counters on CPU. Full checkpoint SHA-256:
  `866e5e43125005385090b9b871464324aa901d608f2263db6ccc2b6d66ebda4a`.
  Initial total device usage is 26.80 GiB; this is not a claim of long-run
  maximum usage. Weight-change measurements continue changing across updates,
  rather than merely differing from the pre-resume initialization.

[W&B continuation 528](https://wandb.ai/skvayzer/adept/runs/unique_id_0_sonic_sapg_touch_528).
Runtime evidence: `outputs/0_sonic_sapg_touch_528/{progress.json,memory_trace.jsonl,warmstart_validation.json}`.
Handoff check: **epoch 266 / 45,514,752 cumulative transitions**, 46 new
updates since this process resumed. Device usage remained **26.80 GiB**,
PyTorch reserved 14.03 GiB, peak live tensors 11.61 GiB, with **zero allocator
retries and zero OOMs**. The periodic latest checkpoint refreshed at epoch
256 and best-return saving also refreshed. W&B is online, with a small
ingestion delay relative to the local progress file. The measured interval
is about **31k transitions/s end to end** versus about 40k/s in the printed
rollout-plus-update timer. No demonstrated speedup from the optimizer-counter
fix or continuous GPU saturation is claimed.

Training is left running, not stopped for an evaluation. This is an early
observed interval, not proof of multi-hour memory stability: 528 has not yet
completed as many fresh updates as the 141-update 517 continuation. Reposing
successes were zero in the checked W&B summaries. Ordinary falls and rare
finite-speed outlier resets remain present (23 numerical resets in 6.78M
new transitions at the handoff check); the numerical cause is not resolved.
Optimizer updates and a rising training return are not evidence of learned
whole-body manipulation success.
