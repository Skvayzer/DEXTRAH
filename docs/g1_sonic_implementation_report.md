# SONIC + SAPG implementation report — 14 September 2026

[Read the implementation report](G1_SONIC_SAPG_Implementation_Report.pdf).
The previous `G1_SONIC_SAPG_Proposal.pdf` filename now points to the same report;
the original one-page proposal is archived as
`G1_SONIC_SAPG_Proposal_20260912.pdf`.

This report supersedes the proposal's implementation-status claims. It documents
the actual code and experiments, not an assertion that the proposed research is
complete. No training was started or changed to produce the report.

## Scope and result

- A single 35-action actor combines the copied SAPG task network/right-finger
  head with a trainable, task-conditioned SONIC body decoder. The trained
  BPS-128 + 70 Hz touch checkpoint from run 383 supplied the manipulation skill;
  supervised bootstrap 503 preceded full-body SAPG updates.
- The original manipulation reward, goals, object bank and task observations
  are retained. Whole-body physics, body observations/actions and robot-fall
  termination are explicit changes. Self-collision is disabled; fabrics and
  the PCA prior are not enabled. There is no online teacher-imitation loss.
- Run 547 used 9,216 environments on one RTX 6000 Ada, resumed both optimizers,
  and reached 1,232,535,552 cumulative transitions before CUDA OOM after nearly
  ten hours. It did not finish successfully. No user training job was active
  at the report's inspection time.
- The final logged training window showed about 91% ever-lifted episodes,
  67% robot-fall terminations and 0% goal success. These are training statistics,
  not a held-out evaluation. No nonzero goal-success rate was found anywhere
  in the exported run history. Lifting is not evidence of successful reposing.
- Purposeful stepping, stable reposing, bimanual manipulation and sustained
  one-GPU memory fit remain unvalidated or unimplemented, as detailed in the PDF.

## Evidence and reproduction

The report audits training code snapshot
`8a5c1b5c025cd4b1cf342a36ba9e24805f2f0f46`, the saved task contract,
bootstrap validation logs, full TensorBoard scalars, W&B summary and memory
telemetry. The source Play2Perfect revision is
`70e79b5e53f912ef04af294ff8f61ac1c7f42160`.

Raw exports are retained locally under
`outputs/implementation_report_20260914/data/{training_547,bootstrap_503}`.
The source runs are under the workstation repository's
`outputs/0_sonic_sapg_touch_547` and `outputs/sonic_distillation_503`.
Raw logs and checkpoints are not committed to Git.

Use `scripts/build_g1_wholebody_implementation_report.py`, not the historical
one-page proposal generator. Dependencies: NumPy, Matplotlib, ReportLab,
PyMuPDF and TensorBoard. The local `.venv_report` contains these dependencies.

```sh
.venv_report/bin/python scripts/build_g1_wholebody_implementation_report.py \
  --assets-output /Users/konstantinsmirnov/research/slides_assets/sonic_sapg_implementation_20260914 \
  --pdf docs/G1_SONIC_SAPG_Implementation_Report.pdf \
  --research-directory /Users/konstantinsmirnov/research
```

The generator verifies the raw event-file checksum, checks the source teacher
identity and task contract, exports plotted CSV/JSON and provenance hashes,
creates editable SVG/vector PDF figures, and renders page previews. Automated
checks cover page count, required claims and text bounds; all six pages were
also visually inspected. The research-directory option publishes both PDF
filenames and preserves the dated original before replacing the old alias.

Figures and audit data are in the `--assets-output` directory, including
`report_evidence.json` and `document_checks.json`. The live run record is
[W&B run 547](https://wandb.ai/skvayzer/adept/runs/unique_id_0_sonic_sapg_touch_547).
