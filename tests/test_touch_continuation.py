import copy
from types import SimpleNamespace
import pytest
import torch
from dextrah_lab.g1_adept.touch_continuation import (
    install_complete_checkpoint, resume_at_episode_boundary, verify_source_mdp, equivalence_precision)


def test_equivalence_precision_restores_runtime_flags_even_on_failure():
    previous=(torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32)
    with pytest.raises(RuntimeError,match='intentional'):
        with equivalence_precision():
            assert not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32
            raise RuntimeError('intentional')
    assert previous == (torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32)


def test_resume_preserves_both_optimizers_but_discards_stale_physics(tmp_path):
    class FakeAlgo:
        def __init__(self):
            self.ppo_device = 'cpu'
            self.model = torch.nn.Linear(2,1)
            self.optimizer = torch.optim.Adam(self.model.parameters())
            critic = torch.nn.Linear(2,1)
            self.central_value_net = SimpleNamespace(model=critic,
                optimizer=torch.optim.Adam(critic.parameters()),epoch_num=9)
            for model,opt in ((self.model,self.optimizer),(critic,self.central_value_net.optimizer)):
                model(torch.ones(2,2)).sum().backward(); opt.step()
        def get_full_state_weights(self):
            return dict(model=self.model.state_dict(),optimizer=self.optimizer.state_dict(),frame=400,epoch=9)
        def set_full_state_weights(self,state):
            self.model.load_state_dict(state['model'])
            self.optimizer.load_state_dict(state['optimizer'])
            self.frame,self.epoch_num=state['frame'],state['epoch']
            self.obs=torch.ones(2,2)
    old = FakeAlgo()
    install_complete_checkpoint(old)
    path = tmp_path/'checkpoint.pth'
    torch.save({0:old.get_full_state_weights()},path)
    new = FakeAlgo()
    new.central_value_net.optimizer.state.clear()
    report=resume_at_episode_boundary(new,path)
    assert report['actor_optimizer_restored'] and report['critic_optimizer_restored']
    assert report['fresh_episodes'] and report['frame']==400
    assert new.central_value_net.epoch_num == 9
    for key in ('obs','rnn_states','current_rewards','current_lengths'):
        assert getattr(new,key) is None
    for a,b in zip(old.central_value_net.optimizer.state.values(),new.central_value_net.optimizer.state.values()):
        for k in a:
            torch.testing.assert_close(a[k],b[k],rtol=0,atol=0)


def test_source_contract_allows_tolerance_and_capacity_not_physics_changes():
    saved={k:{} for k in ('action','reward','reset','domain_randomization','assets','obs')}
    saved.update(decimation=2,episode_length_s=90.,is_finite_horizon=False,
        termination=dict(success_tolerance=.1,target_success_tolerance=.01,eval_success_tolerance=.01,
                         success_steps=10),
        sim=dict(dt=1/120,gravity=(0,0,-9.81),physx=dict(solver_type=1,gpu_heap_capacity=100)))
    current=copy.deepcopy(saved)
    current['termination']['success_tolerance']=.01
    current['sim']['physx']['gpu_heap_capacity']=200
    cfg=SimpleNamespace(to_dict=lambda:current,touch=SimpleNamespace(material_override=False,arm_torques=False,enabled=True))
    verify_source_mdp(saved,cfg)
    cfg.touch.material_override=True
    with pytest.raises(ValueError,match='sensing-only'):
        verify_source_mdp(saved,cfg)
    cfg.touch.material_override=False
    current['sim']['physx']['solver_type']=0
    with pytest.raises(ValueError,match='solver'):
        verify_source_mdp(saved,cfg)
