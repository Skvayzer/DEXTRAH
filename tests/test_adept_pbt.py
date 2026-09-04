import json
import random

import pytest

from dextrah_lab.adept.pbt import (
    PBTConfig,
    WorkerMetadata,
    atomic_write_metadata,
    decide_exploit_explore,
    mutate_hparams,
    pbt_check_due,
    read_population,
)


BASE_HPARAMS = {
    "learning_rate": 1e-3,
    "grad_norm": 1.0,
    "entropy_coef": 1e-3,
    "critic_coef": 4.0,
    "bounds_loss_coef": 1e-4,
    "kl_threshold": 0.01,
    "e_clip": 0.2,
    "mini_epochs": 5,
    "gamma": 0.998,
    "tau": 0.95,
}


def metadata(worker_id, objective, *, frames=400_000_000, adr=None):
    adr = worker_id if adr is None else adr
    return WorkerMetadata(worker_id, frames, objective, adr, f"{worker_id}.pth", dict(BASE_HPARAMS), 1.0)


def test_pbt_cadence_starts_and_repeats_at_200m_frames():
    cfg = PBTConfig()
    assert not pbt_check_due(199_999_999, 0, cfg)
    assert pbt_check_due(200_000_000, 0, cfg)
    assert not pbt_check_due(399_999_999, 200_000_000, cfg)
    assert pbt_check_due(400_000_000, 200_000_000, cfg)


def test_all_appendix_mutations_and_clamps():
    cfg = PBTConfig(mutation_probability=1.0)
    result = mutate_hparams(BASE_HPARAMS, cfg, random.Random(3))
    for name in ("learning_rate", "grad_norm", "entropy_coef", "critic_coef", "bounds_loss_coef", "kl_threshold"):
        ratio = result[name] / BASE_HPARAMS[name]
        assert 0.5 <= ratio <= 2.0
        assert ratio != 1.0
    assert 0.01 <= result["e_clip"] <= 0.3
    assert 1 <= result["mini_epochs"] <= 12
    assert result["gamma"] != BASE_HPARAMS["gamma"]
    assert result["tau"] != BASE_HPARAMS["tau"]


def test_bottom_worker_copies_top_weights_but_preserves_local_adr():
    population = [metadata(i, float(16 - i)) for i in range(16)]
    current = population[-1]
    decision = decide_exploit_explore(
        current,
        population,
        PBTConfig(mutation_probability=0.0),
        random.Random(0),
    )
    assert decision.donor in population[:6]
    assert decision.preserve_adr_level == current.adr_level
    assert decision.hparams == decision.donor.hparams


def test_close_bottom_worker_mutates_without_replacement():
    population = [metadata(i, 0.50 - i * 0.001) for i in range(16)]
    current = population[-1]
    decision = decide_exploit_explore(
        current,
        population,
        PBTConfig(mutation_probability=0.0, near_leader_absolute=0.02),
        random.Random(0),
    )
    assert decision.mutate
    assert decision.donor is None


def test_atomic_metadata_ignores_invalid_peers(tmp_path):
    record = metadata(3, 0.42)
    atomic_write_metadata(tmp_path / "worker_03.json", record)
    (tmp_path / "worker_04.json").write_text("{")
    loaded = read_population(tmp_path)
    assert loaded == [record]
    assert json.loads((tmp_path / "worker_03.json").read_text())["adr_level"] == 3
