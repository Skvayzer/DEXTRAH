#!/usr/bin/env python3
"""CPU-only post-smoke checks; run inside a Slurm CPU allocation."""
import argparse
import json
from pathlib import Path
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("run_dir", type=Path)
args = parser.parse_args()
checkpoints = list((args.run_dir / "nn").glob("last_*.pth"))
if not checkpoints:
    raise RuntimeError("No completed smoke checkpoint")
checkpoint = max(checkpoints, key=lambda path: path.stat().st_mtime)
payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
state = payload[0] if 0 in payload else payload
for group in ("model", "assymetric_vf_nets"):
    for key, value in state[group].items():
        assert torch.isfinite(value).all(), f"Non-finite trained tensor: {key}"
actor = state["model"]["a2c_network.rnn.rnn.weight_ih_l0"][:, 92:224]
critic = state["assymetric_vf_nets"]["model.a2c_network.actor_mlp.0.weight"][:, 114:246]
assert actor.shape == (4096, 132) and critic.shape == (1024, 132)
assert actor.norm() > 0 and critic.norm() > 0, "Shape inputs did not receive learned updates"
events = EventAccumulator(str(args.run_dir / "summaries"), size_guidance={"scalars": 1})
events.Reload()
tags = events.Tags()["scalars"]
selected = [tag for tag in tags if tag.startswith("episode_final/") or tag.startswith("shape/")
            or tag in ("rewards/step", "current_success_tolerance")]
metrics = {tag: events.Scalars(tag)[-1].value for tag in selected}
report = dict(checkpoint=str(checkpoint), epoch=state["epoch"], frames=state["frame"],
              actor_shape_weight_norm=float(actor.norm()), critic_shape_weight_norm=float(critic.norm()),
              all_model_tensors_finite=True, latest_metrics=metrics)
(args.run_dir / "bps" / "smoke_training_validation.json").write_text(json.dumps(report, indent=2)+"\n")
print(json.dumps(report, indent=2))
