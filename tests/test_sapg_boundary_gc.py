import gc
import json
from types import SimpleNamespace
import weakref

import pytest
import torch

from dextrah_lab.wholebody.memory_cleanup import install_boundary_gc


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_reclaims_cycles_and_preserves_next_optimizer_update(tmp_path, monkeypatch, device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA allocation required')
    # Deterministic next-update parity, with intentionally unreachable storage.
    torch.manual_seed(31)
    reference = torch.nn.Linear(4, 3).to(device)
    model = torch.nn.Linear(4, 3).to(device)
    model.load_state_dict(reference.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    expected_optimizer = torch.optim.Adam(reference.parameters(), lr=1e-3)
    x = torch.randn(8, 4, device=device)
    def update(net, opt):
        opt.zero_grad(set_to_none=True)
        loss = net(x).square().mean()
        loss.backward()
        opt.step()
        return float(loss.detach())
    expected = update(reference, expected_optimizer)
    logged = []
    algo = SimpleNamespace(epoch_num=50, frame=1234,
        writer=SimpleNamespace(add_scalar=lambda *items: logged.append(items)),
        train_epoch=lambda: update(model, optimizer))
    if device == 'cpu':
        monkeypatch.setattr(torch.cuda, 'memory_allocated', lambda: 0)
    gc.collect(2)
    enabled = gc.isenabled()
    gc.disable()
    try:
        tensor = torch.zeros(8*1024*1024, device=device)
        ref = weakref.ref(tensor)
        cycle = [tensor]
        cycle.append(cycle)
        del tensor, cycle
        assert ref() is not None
        report = install_boundary_gc(algo, tmp_path, interval=4)
        algo.epoch_num = 51
        assert algo.train_epoch() == expected
        assert ref() is None
        for a, b in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
        assert report['calls'] == 1
        if device == 'cuda':
            assert report['released_tensor_bytes'] >= 32*1024*1024
        for epoch in (52, 53, 54):
            algo.epoch_num = epoch
            algo.train_epoch()
        assert report['calls'] == 1
        algo.epoch_num = 55
        algo.train_epoch()
        assert report['calls'] == 2
        assert len((tmp_path/'boundary_gc.jsonl').read_text().splitlines()) == 2
        assert json.loads((tmp_path/'boundary_gc_summary.json').read_text())['calls'] == 2
        assert len(logged) == 8
    finally:
        if enabled:
            gc.enable()


def test_rejects_disabled_interval(tmp_path):
    with pytest.raises(ValueError):
        install_boundary_gc(None, tmp_path, interval=0)
