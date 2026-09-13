from copy import deepcopy
import pytest
import torch
from dextrah_lab.g1_adept.touch_continuation import optimizer_step_devices


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_resume_step_placement_preserves_adam_update(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA regression runs in the training allocation before launch')
    torch.manual_seed(12)
    original = torch.nn.Linear(3, 2).to(device)
    optimizer = torch.optim.Adam(original.parameters(), lr=1e-5, foreach=True)
    x = torch.randn(8, 3, device=device)
    for _ in range(3):
        optimizer.zero_grad()
        original(x).square().mean().backward()
        optimizer.step()
    resumed = deepcopy(original)
    resumed_optimizer = torch.optim.Adam(resumed.parameters(), lr=1e-5, foreach=True)
    saved = deepcopy(optimizer.state_dict())
    # Reproduce the former map_location=cuda behavior, then fix placement.
    for state in saved['state'].values():
        state['step'] = state['step'].to(device)
    resumed_optimizer.load_state_dict(saved)
    momenta = {id(p): (s['exp_avg'].clone(), s['exp_avg_sq'].clone(), s['step'].item())
               for p, s in resumed_optimizer.state.items()}
    counts = optimizer_step_devices(resumed_optimizer)
    assert counts == {'cpu': 2}
    for p, s in resumed_optimizer.state.items():
        assert s['step'].item() == momenta[id(p)][2]
        assert s['exp_avg'].device == p.device
        torch.testing.assert_close(s['exp_avg'], momenta[id(p)][0], rtol=0, atol=0)
        torch.testing.assert_close(s['exp_avg_sq'], momenta[id(p)][1], rtol=0, atol=0)
    for model, opt in ((original, optimizer), (resumed, resumed_optimizer)):
        opt.zero_grad()
        model(x).square().mean().backward()
        opt.step()
    for expected, actual in zip(original.parameters(), resumed.parameters()):
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)


def test_capturable_step_placement_rule():
    p = torch.nn.Parameter(torch.ones(2))
    optimizer = torch.optim.Adam([p], capturable=True)
    optimizer.state[p]['step'] = torch.tensor(7.)
    optimizer_step_devices(optimizer)
    assert optimizer.state[p]['step'].device == p.device
    assert optimizer.state[p]['step'].item() == 7.
