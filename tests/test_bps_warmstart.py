import pytest
import torch
from dextrah_lab.object_shape.warmstart import expand_state_dict


@pytest.mark.parametrize("base,prefix,matrix", [
    (92, "", "a2c_network.rnn.rnn.weight_ih_l0"),
    (114, "model.", "a2c_network.actor_mlp.0.weight")])
def test_expansion_preserves_old_features_embedding_and_all_other_weights(base, prefix, matrix):
    source = {prefix+"running_mean_std.running_mean": torch.randn(base),
              prefix+"running_mean_std.running_var": torch.rand(base)+1,
              prefix+"running_mean_std.count": torch.tensor(1000.),
              prefix+matrix: torch.randn(16, base+32),
              prefix+"unchanged": torch.randn(5, 7)}
    target = {k: v.clone() for k, v in source.items()}
    for name in ("running_mean", "running_var"):
        target[prefix+"running_mean_std."+name] = torch.zeros(base+132)
    target[prefix+matrix] = torch.empty(16, base+132+32)
    mean, var = torch.randn(132), torch.rand(132)+.1
    expanded = expand_state_dict(source, target, base, mean, var)
    weights = expanded[prefix+matrix].requires_grad_()
    torch.testing.assert_close(weights[:, :base], source[prefix+matrix][:, :base])
    torch.testing.assert_close(weights[:, -32:], source[prefix+matrix][:, -32:])
    assert not weights[:, base:base+132].count_nonzero()
    x, shape = torch.randn(10, base+32), torch.randn(10, 132)
    xx = torch.cat([x[:, :base], shape, x[:, base:]], -1)
    torch.testing.assert_close(x @ source[prefix+matrix].T, xx @ weights.T, atol=1e-5, rtol=1e-5)
    (xx @ weights.T).square().mean().backward()
    assert weights.grad[:, base:base+132].norm() > 0
    torch.testing.assert_close(expanded[prefix+"running_mean_std.running_mean"][-132:], mean)
    torch.testing.assert_close(expanded[prefix+"running_mean_std.running_var"][-132:], var)
    torch.testing.assert_close(expanded[prefix+"unchanged"], source[prefix+"unchanged"])


def test_unknown_mismatch_fails_closed():
    with pytest.raises(ValueError, match="Unexpected"):
        expand_state_dict({"unknown": torch.zeros(1)}, {"unknown": torch.zeros(2)}, 92,
                          torch.zeros(132), torch.ones(132))
    with pytest.raises(ValueError, match="key mismatch"):
        expand_state_dict({"a": torch.zeros(1)}, {"b": torch.zeros(1)}, 92,
                          torch.zeros(132), torch.ones(132))
