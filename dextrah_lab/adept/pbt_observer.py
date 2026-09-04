"""RL-Games bridge for ADEPT's decentralized PBT protocol."""

from __future__ import annotations

from collections import deque
import os
from pathlib import Path
import random
from typing import Any

import torch

from .pbt import (
    PBTConfig,
    atomic_write_metadata,
    decide_exploit_explore,
    new_metadata,
    pbt_check_due,
    read_population,
)


ALGO_TO_METADATA = {
    "learning_rate": "learning_rate",
    "grad_norm": "grad_norm",
    "entropy_coef": "entropy_coef",
    "critic_coef": "critic_coef",
    "bounds_loss_coef": "bounds_loss_coef",
    "kl_threshold": "kl_threshold",
    "e_clip": "e_clip",
    "mini_epochs_num": "mini_epochs",
    "gamma": "gamma",
    "tau": "tau",
}


class AdeptPBTObserver:
    """Publish, exploit, and explore at appendix-defined frame boundaries."""

    def __init__(
        self,
        worker_id: int,
        shared_dir: str | Path,
        cfg: PBTConfig | None = None,
        seed: int = 0,
        objective_window: int = 100,
    ):
        self.worker_id = worker_id
        self.shared_dir = Path(shared_dir)
        self.cfg = cfg or PBTConfig()
        if not 0 <= worker_id < self.cfg.population_size:
            raise ValueError("worker_id must lie within the configured population")
        self.rng = random.Random(seed + worker_id)
        self.objectives: deque[float] = deque(maxlen=objective_window)
        self.adr_level = 0
        self.last_check_frames = 0
        self.algo = None

    def before_init(self, *_args, **_kwargs):
        pass

    def after_init(self, algo):
        self.algo = algo

    def process_infos(self, infos: dict[str, Any], _done_indices):
        objective = infos.get("true_objective")
        if objective is not None:
            value = objective.detach().float().mean().item() if torch.is_tensor(objective) else float(objective)
            self.objectives.append(value)
        adr_level = infos.get("adr_level")
        if adr_level is not None:
            self.adr_level = int(adr_level.detach().max().item() if torch.is_tensor(adr_level) else adr_level)

    def after_steps(self):
        pass

    def after_clear_stats(self):
        pass

    def _hparams(self) -> dict[str, float | int]:
        return {
            metadata_name: getattr(self.algo, algo_name)
            for algo_name, metadata_name in ALGO_TO_METADATA.items()
        }

    def _apply_hparams(self, hparams: dict[str, float | int]) -> None:
        for algo_name, metadata_name in ALGO_TO_METADATA.items():
            value = hparams[metadata_name]
            setattr(self.algo, algo_name, value)
        self.algo.learning_rate = float(hparams["learning_rate"])
        self.algo.last_lr = self.algo.learning_rate
        for param_group in self.algo.optimizer.param_groups:
            param_group["lr"] = self.algo.learning_rate
        if getattr(self.algo, "is_adaptive_lr", False):
            self.algo.scheduler.kl_threshold = float(hparams["kl_threshold"])

    def _save_checkpoint(self, frame: int) -> Path:
        checkpoint_dir = self.shared_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = checkpoint_dir / f"worker_{self.worker_id:02d}_{frame}.pth"
        temporary = checkpoint.with_suffix(".tmp")
        self.algo.save(str(temporary))
        os.replace(temporary, checkpoint)
        return checkpoint

    def _load_policy_only(self, checkpoint: str) -> None:
        from rl_games.algos_torch import torch_ext

        payload = torch_ext.load_checkpoint(checkpoint)
        rank = getattr(self.algo, "global_rank", 0)
        weights = payload.get(rank, payload.get(0, payload))
        # Do not call set_full_state_weights: frame count, optimizer, rollout
        # buffers, and most importantly the receiving worker's ADR state stay local.
        self.algo.set_weights(weights)

    def after_print_stats(self, frame: int, _epoch_num: int, _total_time: float):
        if getattr(self.algo, "global_rank", 0) != 0:
            return
        if not pbt_check_due(frame, self.last_check_frames, self.cfg):
            return
        checkpoint = self._save_checkpoint(frame)
        objective = sum(self.objectives) / len(self.objectives) if self.objectives else 0.0
        current = new_metadata(
            self.worker_id,
            frame,
            objective,
            self.adr_level,
            str(checkpoint),
            self._hparams(),
        )
        atomic_write_metadata(
            self.shared_dir / f"worker_{self.worker_id:02d}.json", current
        )
        population = read_population(self.shared_dir)
        decision = decide_exploit_explore(current, population, self.cfg, self.rng)
        if not decision.mutate:
            self.last_check_frames = frame
            return
        if decision.donor is not None:
            self._load_policy_only(decision.donor.checkpoint)
        self._apply_hparams(decision.hparams)
        self.last_check_frames = frame

