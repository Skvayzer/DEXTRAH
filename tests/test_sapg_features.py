from types import SimpleNamespace
import pytest
import torch
from torch import nn
from dextrah_lab.wholebody.sapg_features import SapgTaskFeatures


def source():
    return SimpleNamespace(rnn=SimpleNamespace(rnn=nn.LSTM(52,16)),
        layer_norm=nn.LayerNorm(16),actor_mlp=nn.Sequential(nn.Linear(16,12),nn.ELU()),
        param_ids=torch.linspace(50,0,6),extra_params=torch.arange(192.).reshape(6,32)/192)


def test_leader_is_last_and_recurrence_matches_source():
    torch.manual_seed(42)
    net=source();features=SapgTaskFeatures(net)
    x=torch.randn(2,4,20)
    state=(torch.randn(1,2,16),torch.randn(1,2,16))
    out,hidden=features(x,state)
    combined=torch.cat((x,net.extra_params[-1].expand(2,4,-1)),-1)
    expected,expected_hidden=net.rnn.rnn(combined.transpose(0,1),state)
    expected=net.actor_mlp(net.layer_norm(expected.transpose(0,1)))
    torch.testing.assert_close(out,expected,rtol=0,atol=0)
    for a,b in zip(hidden,expected_hidden):
        torch.testing.assert_close(a,b,rtol=0,atol=0)
    assert features.rnn.weight_ih_l0.data_ptr()!=net.rnn.rnn.weight_ih_l0.data_ptr()


def test_copied_lstm_resets_and_rejects_gru_state():
    features=SapgTaskFeatures(source())
    x=torch.randn(2,4,20);starts=torch.zeros(2,4,dtype=torch.bool);starts[1,2]=True
    out,_=features(x,episode_starts=starts)
    fresh,_=features(x[1:,2:])
    torch.testing.assert_close(out[1:,2:],fresh)
    with pytest.raises(ValueError,match='h/c'):
        features(x,torch.zeros(1,2,16))
