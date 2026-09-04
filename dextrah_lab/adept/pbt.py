"""Decentralized population-based training used by ADEPT pretraining.

This module deliberately has no Isaac Lab dependency so every worker can run
the exploit/explore decision before launching (or while checkpointing) a
simulator process.  It implements Appendix C; thresholds that the paper does
not quantify are explicit configuration values rather than hidden constants.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import random
import tempfile
import time
from typing import Any, Mapping


FLOAT_HPARAMS = (
    "learning_rate",
    "grad_norm",
    "entropy_coef",
    "critic_coef",
    "bounds_loss_coef",
    "kl_threshold",
)


@dataclass(frozen=True)
class PBTConfig:
    population_size: int = 16
    start_frames: int = 200_000_000
    interval_frames: int = 200_000_000
    replace_fraction: float = 0.4
    mutation_probability: float = 0.25
    mutation_factor_min: float = 1.1
    mutation_factor_max: float = 2.0
    epsilon_bounds: tuple[float, float] = (0.01, 0.3)
    mini_epoch_bounds: tuple[int, int] = (1, 12)
    mini_epoch_step_bounds: tuple[int, int] = (1, 3)

    # Appendix C says comparisons use "approximately" matched frames and
    # describes two leader-distance thresholds without publishing values.
    # Keep both knobs explicit and overrideable in experiment provenance.
    comparable_frame_fraction: float = 0.10
    near_leader_std_fraction: float = 0.25
    near_leader_absolute: float = 0.02


@dataclass(frozen=True)
class WorkerMetadata:
    worker_id: int
    frames: int
    objective: float
    adr_level: int
    checkpoint: str
    hparams: dict[str, float | int]
    timestamp: float

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "WorkerMetadata":
        return cls(
            worker_id=int(payload["worker_id"]),
            frames=int(payload["frames"]),
            objective=float(payload["objective"]),
            adr_level=int(payload["adr_level"]),
            checkpoint=str(payload["checkpoint"]),
            hparams=dict(payload["hparams"]),
            timestamp=float(payload["timestamp"]),
        )


@dataclass(frozen=True)
class PBTDecision:
    mutate: bool
    donor: WorkerMetadata | None
    hparams: dict[str, float | int]
    preserve_adr_level: int
    reason: str


def atomic_write_metadata(path: str | Path, metadata: WorkerMetadata) -> None:
    """Publish one complete JSON record using an atomic same-dir rename."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(asdict(metadata), stream, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def read_population(metadata_dir: str | Path) -> list[WorkerMetadata]:
    records = []
    for path in sorted(Path(metadata_dir).glob("worker_*.json")):
        try:
            with path.open(encoding="utf-8") as stream:
                records.append(WorkerMetadata.from_dict(json.load(stream)))
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            # A worker must never fail because another worker left a stale or
            # incompatible sidecar. Atomic writes prevent partial new records.
            continue
    return records


def pbt_check_due(frames: int, last_check_frames: int, cfg: PBTConfig) -> bool:
    return frames >= cfg.start_frames and frames - last_check_frames >= cfg.interval_frames


def comparable_population(
    current: WorkerMetadata,
    population: list[WorkerMetadata],
    cfg: PBTConfig,
) -> list[WorkerMetadata]:
    tolerance = max(1, round(current.frames * cfg.comparable_frame_fraction))
    matched = [record for record in population if abs(record.frames - current.frames) <= tolerance]
    if all(record.worker_id != current.worker_id for record in matched):
        matched.append(current)
    return matched


def _mutate_float(value: float, cfg: PBTConfig, rng: random.Random) -> float:
    factor = rng.uniform(cfg.mutation_factor_min, cfg.mutation_factor_max)
    return value * factor if rng.random() < 0.5 else value / factor


def mutate_hparams(
    hparams: Mapping[str, float | int],
    cfg: PBTConfig,
    rng: random.Random,
) -> dict[str, float | int]:
    """Apply every mutation rule listed in ADEPT Appendix C."""

    result = dict(hparams)
    for name in FLOAT_HPARAMS:
        if name in result and rng.random() < cfg.mutation_probability:
            result[name] = _mutate_float(float(result[name]), cfg, rng)

    if "e_clip" in result and rng.random() < cfg.mutation_probability:
        value = _mutate_float(float(result["e_clip"]), cfg, rng)
        result["e_clip"] = min(cfg.epsilon_bounds[1], max(cfg.epsilon_bounds[0], value))

    if "mini_epochs" in result and rng.random() < cfg.mutation_probability:
        magnitude = rng.randint(*cfg.mini_epoch_step_bounds)
        direction = 1 if rng.random() < 0.5 else -1
        value = int(result["mini_epochs"]) + direction * magnitude
        result["mini_epochs"] = min(
            cfg.mini_epoch_bounds[1], max(cfg.mini_epoch_bounds[0], value)
        )

    for name in ("gamma", "tau"):
        if name in result and rng.random() < cfg.mutation_probability:
            complement = _mutate_float(1.0 - float(result[name]), cfg, rng)
            result[name] = min(1.0 - 1e-8, max(0.0, 1.0 - complement))
    return result


def decide_exploit_explore(
    current: WorkerMetadata,
    population: list[WorkerMetadata],
    cfg: PBTConfig,
    rng: random.Random,
) -> PBTDecision:
    """Rank by true objective, optionally exploit, then explore.

    Local ADR is returned independently of the selected donor, enforcing the
    paper's requirement that curriculum progress is never copied.
    """

    matched = comparable_population(current, population, cfg)
    if len(matched) < 2:
        return PBTDecision(False, None, dict(current.hparams), current.adr_level, "insufficient_peers")

    ranked = sorted(matched, key=lambda record: record.objective, reverse=True)
    group_size = max(1, math.floor(len(ranked) * cfg.replace_fraction))
    bottom_ids = {record.worker_id for record in ranked[-group_size:]}
    if current.worker_id not in bottom_ids:
        return PBTDecision(False, None, dict(current.hparams), current.adr_level, "not_in_bottom_fraction")

    objectives = [record.objective for record in ranked]
    mean = sum(objectives) / len(objectives)
    std = math.sqrt(sum((value - mean) ** 2 for value in objectives) / len(objectives))
    gap = ranked[0].objective - current.objective
    near_leader = gap <= max(cfg.near_leader_absolute, cfg.near_leader_std_fraction * std)

    donor = None if near_leader else rng.choice(ranked[:group_size])
    base_hparams = current.hparams if donor is None else donor.hparams
    return PBTDecision(
        mutate=True,
        donor=donor,
        hparams=mutate_hparams(base_hparams, cfg, rng),
        preserve_adr_level=current.adr_level,
        reason="mutate_only_near_leader" if near_leader else "replace_and_mutate",
    )


def new_metadata(
    worker_id: int,
    frames: int,
    objective: float,
    adr_level: int,
    checkpoint: str,
    hparams: Mapping[str, float | int],
) -> WorkerMetadata:
    return WorkerMetadata(
        worker_id=worker_id,
        frames=frames,
        objective=objective,
        adr_level=adr_level,
        checkpoint=checkpoint,
        hparams=dict(hparams),
        timestamp=time.time(),
    )

