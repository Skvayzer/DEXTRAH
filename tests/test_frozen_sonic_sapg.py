"""Authoritative teacher/SONIC regressions; run inside a Slurm allocation."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch
from torch import nn
from rl_games.algos_torch import model_builder

from dextrah_lab.wholebody.frozen_sapg import (
    FrozenControllerAudit, install_latent_action_bounds, install_latent_optimizer)
from dextrah_lab.wholebody.sapg_network import load_touch_sources, register_sonic_models
from dextrah_lab.wholebody.task_contract import TOUCH_CHECKPOINT, BODY_EXTRA_DIM


@pytest.fixture(scope='module')
def sources():
    if not Path(TOUCH_CHECKPOINT).exists():
        pytest.skip('Authoritative checkpoints are on the workstation')
    torch.set_num_threads(4)
    return load_touch_sources()


@pytest.fixture
def models(sources):
    source, critic = sources
    torch.manual_seed(42)
    register_sonic_models(source, critic, None, None, None, frozen=True)

    def build(dim, name, network):
        return model_builder.ModelBuilder().load(dict(model=dict(name=name), network=dict(name=network))).build(
            dict(actions_num=70, input_shape=(dim+BODY_EXTRA_DIM+32,), num_seqs=6, value_size=1,
                normalize_value=True, normalize_input=True, type='extra_param',
                coef_ids=source.a2c_network.param_ids, coef_id_idx=dim+BODY_EXTRA_DIM)).eval()

    return build(249, 'sonic_sapg_logstd', 'sonic_sapg_actor'), build(271, 'sonic_sapg_value', 'sonic_sapg_critic')


def observations(source, dim, steps=4):
    n = 6*steps
    raw = torch.randn(n, dim).clamp(-10, 10)
    raw[:, -25:].reshape(n, 5, 5)[..., 3] = 1
    raw[:, -25:].reshape(n, 5, 5)[..., 4] = .01
    ids = source.a2c_network.param_ids.repeat_interleave(steps)[:, None]
    return torch.cat((raw, ids), -1), torch.cat((raw, torch.randn(n, BODY_EXTRA_DIM), ids), -1)


@pytest.mark.parametrize('with_dones', [False, True])
def test_frozen_adapter_preserves_all_six_teacher_finger_distributions_and_states(sources, models, with_dones):
    source, _ = sources
    actor, _ = models
    task, whole = observations(source, 249)
    state = tuple(torch.randn(1, 6, 1024) for _ in range(2))
    common = dict(is_train=True, rnn_states=state, seq_length=4)
    if with_dones:
        common['dones'] = torch.tensor([0., 1., 0., 0.]*6)
    with torch.no_grad():
        expected = source(dict(obs=task, prev_actions=torch.zeros(24, 13), **common))
        actual = actor(dict(obs=whole, prev_actions=torch.zeros(24, 70), **common))
    assert not actual['mus'][:, :64].any()
    for name in ('mus', 'sigmas'):
        torch.testing.assert_close(actual[name][:, 64:], expected[name][:, 7:13], rtol=0, atol=0)
    for a, b in zip(actual['rnn_states'], expected['rnn_states']):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert all('decoder' not in k for k in actor.state_dict())


def test_frozen_adapter_critic_warmstart(sources, models):
    _, source = sources
    _, critic = models
    task, whole = observations(source, 271)
    with torch.no_grad():
        expected = source(dict(obs=task, is_train=False))
        actual = critic(dict(obs=whole, is_train=False))
    torch.testing.assert_close(actual['values'], expected['values'], rtol=1e-5, atol=1e-4)


def test_adapter_and_fingers_receive_sapg_likelihood_gradients(sources, models):
    source, _ = sources
    actor, _ = models
    _, obs = observations(source, 249, steps=1)
    output = actor(dict(obs=obs, is_train=True, prev_actions=torch.randn(6, 70),
                        rnn_states=actor.get_default_rnn_state(), seq_length=1))
    # A real Gaussian score-function gradient in meta-action space; do not
    # differentiate the reward through a decoder or through the simulator.
    distribution = torch.distributions.Normal(output['mus'], output['sigmas'])
    targets = torch.randn_like(output['mus'])
    loss = -distribution.log_prob(targets).sum(-1).mean()
    loss.backward()
    net = actor.a2c_network
    for p in (net.latent_adapter[-1].weight, net.source.mu.weight, net.logstd):
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
    assert net.source.sigma.grad is None
    before = deepcopy(net.latent_adapter.state_dict())
    torch.optim.Adam(actor.parameters(), lr=1e-4).step()
    assert not torch.equal(net.latent_adapter[-1].weight, before['6.weight'])


def test_frozen_audit_checks_buffers_and_optimizer_membership():
    sonic = nn.Module()
    sonic.actor = nn.Linear(4, 3)
    sonic.register_buffer('marker', torch.tensor(1.))
    sonic.eval().requires_grad_(False)
    audit = FrozenControllerAudit(sonic)
    assert audit.check()['pretrained_sonic_trainable_parameters'] == 0
    with pytest.raises(RuntimeError, match='optimizer'):
        audit.check(torch.optim.Adam(sonic.parameters()))
    sonic.marker.add_(1)
    with pytest.raises(RuntimeError, match='changed'):
        audit.check()


def test_no_artificial_latent_bounds_and_separate_adapter_learning_rate(sources, models):
    actor, _ = models
    algo = NS(model=actor, actions_num=70, clip_actions=False, last_lr=1e-5, weight_decay=0.,
              optimizer=torch.optim.Adam(actor.parameters(), lr=1e-5),
              bound_loss=lambda x: (x.abs()-1.1).clamp_min(0).square().sum(-1))

    def base_update(lr):
        for group in algo.optimizer.param_groups:
            group['lr'] = lr

    algo.update_lr = base_update
    install_latent_action_bounds(algo)
    means = torch.full((2, 70), 50.)
    means[:, 64:] = 0
    assert not algo.bound_loss(means).any()
    means[:, 64:] = 2
    torch.testing.assert_close(algo.bound_loss(means), torch.full((2,), 6*.9**2))
    install_latent_optimizer(algo, 3e-4)
    algo.update_lr(1e-5)
    assert [g['lr'] for g in algo.optimizer.param_groups] == pytest.approx([1e-5, 3e-4])
    algo.update_lr(5e-6)
    assert [g['lr'] for g in algo.optimizer.param_groups] == pytest.approx([5e-6, 1.5e-4])
    group_ids = [{id(p) for p in g['params']} for g in algo.optimizer.param_groups]
    assert not group_ids[0] & group_ids[1]
    assert group_ids[0] | group_ids[1] == {id(p) for p in actor.parameters() if p.requires_grad}


def test_reject_finetuned_bootstrap_in_frozen_mode(sources):
    with pytest.raises(ValueError, match='fine-tuned'):
        register_sonic_models(*sources, None, None, None, student=object(), frozen=True)


def test_real_pretrained_sonic_zero_residual_and_gradients_only_into_latent():
    from dextrah_lab.wholebody.sonic import FrozenSonic
    from dextrah_lab.wholebody.contract import nominal_body_pose
    root = Path('/data1/users/konstantin.smirnov')
    if not (root/'GRAIL').exists():
        pytest.skip('Pinned SONIC bundle is on the workstation')
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    sonic = FrozenSonic(root/'GRAIL', root/'G1-SONIC-models/checkpoint/SONIC/models/sonic_manipulation_base', device)
    audit = FrozenControllerAudit(sonic)
    torch.manual_seed(42)
    q = torch.as_tensor(nominal_body_pose(), device=device)[None, None].repeat(4, 10, 1)
    qd = torch.zeros_like(q)
    ori = q.new_tensor([1., 0., 0., 1., 0., 0.])[None, None].repeat(4, 10, 1)
    proprio = q.new_zeros(4, 930)
    proprio[:, 900:] = q.new_tensor([0., 0., -1.]).repeat(10)
    baseline = sonic(proprio, q, qd, ori)
    zero = sonic(proprio, q, qd, ori, q.new_zeros(4, 64))
    torch.testing.assert_close(baseline, zero, rtol=1e-6, atol=1e-6)
    latent = torch.randn(4, 64, device=device, requires_grad=True)
    action = sonic.decode_for_imitation(proprio, q, qd, ori, latent)
    assert (action.detach()-baseline).abs().max() > 1e-5
    action.square().mean().backward()
    assert latent.grad is not None and torch.isfinite(latent.grad).all() and latent.grad.abs().sum() > 0
    torch.optim.Adam([latent], lr=.01).step()
    audit.check()
