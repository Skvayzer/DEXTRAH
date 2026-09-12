import pytest
import torch
from torch import nn

from dextrah_lab.wholebody.student import (
    ACTION_JOINTS, FixedTaskNormalizer, SonicManipulationStudent, direct_imitation_loss)


def student():
    torch.manual_seed(12)
    # Match the released MLP builder's shared activation object. children()
    # deduplicates it; iterating Sequential must not omit later activations.
    activation=nn.SiLU()
    decoder=nn.Sequential(nn.Linear(994,32),activation,nn.Linear(32,16),activation,nn.Linear(16,29))
    decoder.requires_grad_(False)
    return decoder,SonicManipulationStudent(decoder,torch.zeros(20),torch.ones(20),hidden_dim=16)


def inputs():
    return torch.randn(2,4,930),torch.randn(2,4,20),torch.randn(2,4,64)


def test_exact_function_preserving_initialization_and_independent_weights():
    source,model=student()
    p,o,z=inputs()
    action,_=model(p,o,z)
    torch.testing.assert_close(action[...,:29],source(torch.cat((z,p),-1)),rtol=0,atol=0)
    assert not action[...,29:].any()
    for source_p,student_p in zip(source.parameters(),model.decoder.parameters()):
        assert source_p.data_ptr()!=student_p.data_ptr()
        assert student_p.requires_grad and not source_p.requires_grad
    assert len(ACTION_JOINTS)==len(set(ACTION_JOINTS))==35
    with pytest.raises(ValueError,match='35 actions'):
        model.body_targets(torch.zeros(2,13))


def test_body_decoder_gets_gradients_and_original_stays_unchanged():
    source,model=student()
    before={k:v.clone() for k,v in source.state_dict().items()}
    p,o,z=inputs()
    action,_=model(p,o,z)
    target=model.body_targets(action.detach())[...,model.arm_indices]+.15
    valid=torch.ones(2,4,dtype=torch.bool)
    teacher=action[...,:29].detach()
    loss,metrics=direct_imitation_loss(model,action,target,torch.ones(2,4,6)*.3,teacher,valid)
    loss.backward()
    assert metrics['arm_rmse_rad']==pytest.approx(.15)
    assert model.decoder[0].weight.grad.abs().sum()>0
    assert model.decoder[-1].weight.grad.abs().sum()>0
    assert model.task_to_body.weight.grad.abs().sum()>0
    assert model.fingers.weight.grad.abs().sum()>0
    optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    optimizer.step()
    for k,v in source.state_dict().items():
        torch.testing.assert_close(v,before[k],rtol=0,atol=0)
    # Zero projections deliberately shield the encoder on step zero; after an
    # update it must also receive gradients, not remain a permanently dead path.
    optimizer.zero_grad()
    action,_=model(p,o,z)
    loss,_=direct_imitation_loss(model,action,target,torch.ones(2,4,6)*.3,teacher,valid)
    loss.backward()
    assert model.task_rnn.weight_ih_l0.grad.abs().sum()>0


def test_recurrent_reset_and_sequence_equivalence():
    _,model=student()
    with torch.no_grad():
        model.fingers.weight.normal_(0,.01)
    p,o,z=inputs()
    starts=torch.zeros(2,4,dtype=torch.bool)
    starts[1,2]=True
    action,_=model(p,o,z,episode_starts=starts)
    fresh,_=model(p[1:,2:],o[1:,2:],z[1:,2:])
    torch.testing.assert_close(action[1:,2:],fresh)
    hidden=None
    pieces=[]
    for t in range(4):
        a,hidden=model(p[:,t:t+1],o[:,t:t+1],z[:,t:t+1],hidden,starts[:,t:t+1])
        pieces.append(a)
    torch.testing.assert_close(torch.cat(pieces,1),action)


def test_invalid_labels_and_action_sliced_standing_loss():
    _,model=student()
    p,o,z=inputs()
    action,_=model(p,o,z)
    mask=torch.zeros(2,4,dtype=torch.bool)
    standing=~mask
    nan_arm=torch.full((2,4,7),float('nan'))
    nan_finger=torch.full((2,4,6),float('nan'))
    loss,metrics=direct_imitation_loss(model,action,nan_arm,nan_finger,
        action[...,:29].detach()+.1,mask,standing)
    assert torch.isfinite(loss) and metrics['manipulation_samples']==0
    loss.backward()
    assert not model.fingers.weight.grad.any()
    with pytest.raises(ValueError,match='disjoint'):
        direct_imitation_loss(model,action,nan_arm,nan_finger,action[...,:29],standing,standing)


def test_tactile_invalid_force_and_age_are_not_standardized_as_forces():
    norm=FixedTaskNormalizer(torch.ones(249),torch.ones(249))
    obs=torch.zeros(2,249)
    pad=obs[:,224:].view(2,5,5)
    pad[...,4]=.02
    pad[0,0,3]=1
    out=norm(obs)[:,224:].view(2,5,5)
    assert not out[1,:,:3].any()
    assert out[0,0,3]==1
    torch.testing.assert_close(out[...,4],torch.full((2,5),.2))


def test_shape_and_source_decoder_contract_fail_closed():
    _,model=student()
    p,o,z=inputs()
    with pytest.raises(ValueError,match='batch/time'):
        model(p,o,z[:,:,:63])
    with pytest.raises(ValueError,match='recurrent-state'):
        model(p,o,z,hidden=torch.zeros(1,2,5))
    with pytest.raises(ValueError,match='930'):
        SonicManipulationStudent(nn.Sequential(nn.Linear(13,29)),torch.zeros(20),torch.ones(20))


def test_inactive_task_is_not_a_mean_observation_heuristic():
    _,model=student()
    with torch.no_grad():
        model.task_to_body.weight.normal_(0,.1)
    p,o,z=inputs()
    first,_=model(p,o,z,task_active=False)
    second,_=model(p,o+2,z,task_active=False)
    torch.testing.assert_close(first[...,:29],second[...,:29],rtol=0,atol=0)
    active,_=model(p,o,z)
    assert not torch.allclose(active[...,:29],first[...,:29])
