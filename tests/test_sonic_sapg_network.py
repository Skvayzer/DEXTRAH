"""Real SAPG checkpoint regression; CPU Slurm, no simulator required."""
from pathlib import Path
from types import SimpleNamespace
from copy import deepcopy
import pytest
import torch
from torch import nn
from rl_games.algos_torch import model_builder
from dextrah_lab.wholebody.sapg_network import load_touch_sources, register_sonic_models
from dextrah_lab.wholebody.task_contract import TOUCH_CHECKPOINT, BODY_EXTRA_DIM


@pytest.fixture(scope='module')
def sources():
    if not Path(TOUCH_CHECKPOINT).exists():
        pytest.skip('Authoritative checkpoint is on the workstation')
    torch.set_num_threads(4)
    return load_touch_sources()


@pytest.fixture
def models(sources):
    actor, critic = sources
    torch.manual_seed(20)
    activation = nn.SiLU()
    decoder = nn.Sequential(nn.Linear(994, 64), activation, nn.Linear(64, 32), activation, nn.Linear(32, 29))
    sonic = SimpleNamespace(actor=SimpleNamespace(actor_module=SimpleNamespace(
        decoders={'g1_dyn': SimpleNamespace(module=decoder.requires_grad_(False))})))
    lower, upper = torch.full((29,), -2.), torch.full((29,), 2.)
    register_sonic_models(actor, critic, sonic, lower, upper)
    def build(task_dim, name, network):
        return model_builder.ModelBuilder().load(dict(model=dict(name=name), network=dict(name=network))).build(
            dict(actions_num=35, input_shape=(task_dim+BODY_EXTRA_DIM+32,), num_seqs=6, value_size=1,
                normalize_value=True, normalize_input=True, type='extra_param',
                coef_ids=actor.a2c_network.param_ids, coef_id_idx=task_dim+BODY_EXTRA_DIM)).eval()
    return build(249, 'sonic_sapg_logstd', 'sonic_sapg_actor'), build(271, 'sonic_sapg_value', 'sonic_sapg_critic'), decoder


def inputs(source, task_dim, steps=4):
    n = 6*steps
    raw = torch.randn(n, task_dim).clamp(-10, 10)
    raw[:, -25:].reshape(n, 5, 5)[..., 3] = 1
    raw[:, -25:].reshape(n, 5, 5)[..., 4] = .01
    coefficients = source.a2c_network.param_ids.repeat_interleave(steps)[:, None]
    extra = torch.randn(n, BODY_EXTRA_DIM)
    task = torch.cat((raw, coefficients), -1)
    whole = torch.cat((raw, extra, coefficients), -1)
    return task, whole, extra


@pytest.mark.parametrize('with_dones', [False, True])
def test_all_six_source_groups_fingers_and_recurrent_states(sources, models, with_dones):
    source, _ = sources
    model, _, decoder = models
    task, whole, extra = inputs(source, 249)
    state = tuple(torch.randn(1, 6, 1024) for _ in range(2))
    common = dict(is_train=True, rnn_states=state, seq_length=4)
    if with_dones:
        common['dones'] = torch.tensor([0., 1., 0., 0.]*6)
    with torch.no_grad():
        expected = source(dict(obs=task.clone(), prev_actions=torch.zeros(24, 13), **common))
        actual = model(dict(obs=whole.clone(), prev_actions=torch.zeros(24, 35), **common))
    torch.testing.assert_close(actual['mus'][:, 29:], expected['mus'][:, 7:13], rtol=0, atol=0)
    torch.testing.assert_close(actual['sigmas'][:, 29:], expected['sigmas'][:, 7:13], rtol=0, atol=0)
    for a, b in zip(actual['rnn_states'], expected['rnn_states']):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    network = model.a2c_network
    expected_q = network.body_nominal + network.body_scales*decoder(torch.cat((extra[:, 930:], extra[:, :930]), -1))
    actual_q = network.body_center + network.body_half_range*actual['mus'][:, :29]
    torch.testing.assert_close(actual_q, expected_q, atol=1e-7, rtol=1e-6)


def test_critic_and_normalizers_preserved(sources, models):
    _, source = sources
    _, model, _ = models
    task, whole, _ = inputs(source, 271)
    with torch.no_grad():
        expected = source(dict(obs=task, is_train=False))
        actual = model(dict(obs=whole, is_train=False))
    torch.testing.assert_close(actual['values'], expected['values'], rtol=1e-5, atol=1e-4)
    for key, value in source.running_mean_std.state_dict().items():
        torch.testing.assert_close(model.running_mean_std.task.state_dict()[key], value, rtol=0, atol=0)
    for key, value in source.value_mean_std.state_dict().items():
        torch.testing.assert_close(model.value_mean_std.state_dict()[key], value, rtol=0, atol=0)


def test_body_decoder_fingers_and_new_critic_columns_learn(sources, models):
    source_actor, source_critic = sources
    actor, critic, frozen_decoder = models
    original = deepcopy(frozen_decoder.state_dict())
    _, whole, _ = inputs(source_actor, 249, steps=1)
    output = actor(dict(obs=whole, is_train=True, prev_actions=torch.zeros(6, 35),
                        rnn_states=actor.get_default_rnn_state(), seq_length=1))
    (output['mus'].square().mean() + output['sigmas'].mean()).backward()
    network = actor.a2c_network
    for parameter in (network.decoder[0].weight, network.decoder[-1].weight,
                      network.task_to_body.weight, network.source.mu.weight, network.logstd):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0
    optimizer = torch.optim.Adam(actor.parameters(), lr=1e-5)
    optimizer.step()
    for key, value in frozen_decoder.state_dict().items():
        torch.testing.assert_close(value, original[key], rtol=0, atol=0)
    _, whole, _ = inputs(source_critic, 271, steps=1)
    critic(dict(obs=whole, is_train=True))['values'].square().mean().backward()
    gradient = critic.a2c_network.actor_mlp[0].weight.grad[:, 271:271+BODY_EXTRA_DIM]
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0


def test_body_observations_not_clipped_like_legacy_task(sources, models):
    source, _ = sources
    model, _, _ = models
    _, whole, _ = inputs(source, 249, steps=1)
    whole[:, 249:1243] = 23
    normalized = model.norm_obs(whole)
    assert torch.all(normalized[:, 249:1243] == 23)
