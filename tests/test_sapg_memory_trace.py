import json
from types import SimpleNamespace as NS
import pytest
import torch
from dextrah_lab.wholebody.memory_diagnostics import install_memory_trace


@pytest.fixture
def cuda_stats(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'memory_stats', lambda: {})
    monkeypatch.setattr(torch.cuda, 'mem_get_info', lambda: (10*2**30, 48*2**30))
    for key in ('memory_allocated', 'memory_reserved', 'max_memory_allocated', 'max_memory_reserved'):
        monkeypatch.setattr(torch.cuda, key, lambda: 1024)
    monkeypatch.setattr(torch.cuda, 'memory_summary', lambda: 'diagnostic summary')


def algo():
    return NS(epoch_num=128, frame=25165824, play_steps=lambda x: x,
              augment_batch_for_mixed_expl=lambda x: x,
              train_central_value=lambda: None, train_epoch=lambda: {'same': True})


def test_trace_preserves_outputs_and_rng(tmp_path, cuda_stats):
    a = algo()
    before = torch.get_rng_state().clone()
    install_memory_trace(a, tmp_path)
    x = torch.tensor([3.])
    assert a.play_steps(x) is x
    assert a.augment_batch_for_mixed_expl(x) is x
    assert a.train_epoch() == {'same': True}
    torch.testing.assert_close(before, torch.get_rng_state(), rtol=0, atol=0)
    rows = [json.loads(s) for s in (tmp_path/'memory_trace.jsonl').read_text().splitlines()]
    assert rows[-1]['stage'] == 'train_epoch:after'
    assert all(r['frame'] == 25165824 for r in rows)


def test_gc_probe_runs_only_once(tmp_path, cuda_stats, monkeypatch):
    calls = []
    monkeypatch.setattr('dextrah_lab.wholebody.memory_diagnostics.gc.collect', lambda: calls.append(1) or 42)
    a = algo()
    install_memory_trace(a, tmp_path, gc_probe_epoch=129)
    a.train_epoch()
    assert not calls
    a.epoch_num = 129
    a.train_epoch()
    a.train_epoch()
    assert calls == [1]


def test_error_is_saved_and_rethrown(tmp_path, cuda_stats):
    a = algo()
    def fail():
        raise RuntimeError('original failure')
    a.train_epoch = fail
    install_memory_trace(a, tmp_path)
    with pytest.raises(RuntimeError, match='original failure'):
        a.train_epoch()
    assert (tmp_path/'memory_error_summary.txt').read_text() == 'diagnostic summary'
