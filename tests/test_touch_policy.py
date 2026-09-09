import pytest
import torch
from rl_games.algos_torch.running_mean_std import RunningMeanStd
from dextrah_lab.g1_adept.touch_policy import TouchRunningMeanStd, expand_touch_weights


def test_old_statistics_and_outputs_preserved_while_new_channels_adapt():
    torch.manual_seed(7)
    old, new = RunningMeanStd((224,)), TouchRunningMeanStd(224)
    old.running_mean.copy_(torch.randn(224))
    old.running_var.copy_(torch.rand(224) + .5)
    old.count.fill_(1e10)
    new.running_mean[:224].copy_(old.running_mean)
    new.running_var[:224].copy_(old.running_var)
    new.count[:224].fill_(1e10)
    x, touch = torch.randn(48, 224), torch.ones(48, 25)
    a, b = old(x), new(torch.cat((x, touch), -1))
    torch.testing.assert_close(a, b[:, :224], rtol=0, atol=0)
    torch.testing.assert_close(old.running_mean, new.running_mean[:224], rtol=0, atol=0)
    assert new.running_mean[224] > .9
    assert new.count[224] == 49 and new.count[0] == old.count


def test_validity_age_missing_force_and_state_dict_roundtrip():
    model = TouchRunningMeanStd(246).eval()
    model.running_mean.fill_(.3)
    x = torch.zeros(2, 271)
    extra = x[:, 246:].reshape(2, 5, 5)
    extra[1, :, 3] = 1
    extra[1, :, 4] = .08
    y = model(x)[:, 246:].reshape(2, 5, 5)
    assert not y[0].count_nonzero()
    torch.testing.assert_close(y[1, :, 3], torch.ones(5))
    torch.testing.assert_close(y[1, :, 4], torch.full((5,), .8))
    restored = TouchRunningMeanStd(246).eval()
    restored.load_state_dict(model.state_dict(), strict=True)
    torch.testing.assert_close(restored(x), model(x), rtol=0, atol=0)


@pytest.mark.parametrize('base', [224, 246])
def test_expansion_keeps_learned_bps_and_embedding(base):
    prefix = '' if base == 224 else 'model.'
    matrix = prefix + ('a2c_network.rnn.rnn.weight_ih_l0' if base == 224 else 'a2c_network.actor_mlp.0.weight')
    source = {prefix+'running_mean_std.running_mean': torch.randn(base),
              prefix+'running_mean_std.running_var': torch.ones(base),
              prefix+'running_mean_std.count': torch.tensor(1e9, dtype=torch.float64),
              matrix: torch.randn(8, base+32)}
    target = {prefix+'running_mean_std.running_mean': torch.zeros(base+25),
              prefix+'running_mean_std.running_var': torch.ones(base+25),
              prefix+'running_mean_std.count': torch.ones(base+25),
              matrix: torch.zeros(8, base+25+32)}
    result = expand_touch_weights(source, target, base, torch.zeros(25), torch.ones(25), 4000)
    assert not result[matrix][:, base:base+25].count_nonzero()
    torch.testing.assert_close(result[matrix][:, :base], source[matrix][:, :base])
    torch.testing.assert_close(result[matrix][:, -32:], source[matrix][:, -32:])
    count = result[prefix+'running_mean_std.count']
    assert (count[:base] == 1e9).all() and (count[base:] == 4000).all()
    with pytest.raises(ValueError, match='calibrat'):
        expand_touch_weights(source, target, base, torch.zeros(25), torch.ones(25), 0)
