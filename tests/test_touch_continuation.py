import copy
import json
from types import SimpleNamespace
import pytest
import torch
from dextrah_lab.g1_adept.touch_continuation import (
    install_complete_checkpoint, resume_at_episode_boundary, verify_source_mdp, equivalence_precision,
    install_atomic_checkpoint_saves, inherit_resume_artifacts, SOURCE_SHA, BANK_SHA)


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


def test_atomic_checkpoint_keeps_previous_on_failed_write(tmp_path):
    target = tmp_path/'latest.pth'
    target.write_bytes(b'previous complete checkpoint')
    def fail(filename, state):
        from pathlib import Path
        Path(filename+'.pth').write_bytes(b'incomplete')
        raise OSError('disk write failed')
    algo = SimpleNamespace(save=fail)
    install_atomic_checkpoint_saves(algo)
    with pytest.raises(OSError):
        algo.save(str(tmp_path/'latest'))
    assert target.read_bytes() == b'previous complete checkpoint'
    def success(filename, state):
        from pathlib import Path
        Path(filename+'.pth').write_bytes(b'new complete checkpoint')
    algo = SimpleNamespace(save=success)
    install_atomic_checkpoint_saves(algo)
    algo.save(str(tmp_path/'latest'))
    assert target.read_bytes() == b'new complete checkpoint'
    assert not (tmp_path/'latest.pending.pth').exists()


def test_resume_provenance_preserves_original_calibration_without_refitting(tmp_path):
    parent, output = tmp_path/'parent', tmp_path/'new_run'
    (parent/'nn').mkdir(parents=True)
    checkpoint = parent/'nn/complete_500.pth'
    checkpoint.write_bytes(b'test checkpoint')
    report = dict(source_sha256=SOURCE_SHA,bank_sha256=BANK_SHA,control=False)
    (parent/'warmstart_validation.json').write_text(json.dumps(report))
    original = json.dumps(dict(sensor_hz=70,publish_hz=10,count=123))
    (parent/'calibration.json').write_text(original)
    provenance = inherit_resume_artifacts(output,checkpoint)
    assert (output/'calibration.json').read_text() == original
    assert 'not recalibrated' in provenance['normalization']
    assert len(provenance['checkpoint_sha256']) == 64
    with pytest.raises(FileExistsError):
        inherit_resume_artifacts(output,checkpoint)
    with pytest.raises(ValueError,match='provenance'):
        inherit_resume_artifacts(tmp_path/'wrong_type',checkpoint,control=True)


def test_continuous_checkpoints_do_not_stop_training_without_signal(tmp_path, monkeypatch):
    from pathlib import Path
    from dextrah_lab.g1_adept.touch_continuation import TouchContinuationObserver
    for name in ('memory_allocated','max_memory_reserved'):
        monkeypatch.setattr(torch.cuda,name,lambda: 0)
    monkeypatch.setattr(torch.cuda,'mem_get_info',lambda: (100,200))
    env = SimpleNamespace(_current_success_tolerance=.01)
    observer = TouchContinuationObserver(env,tmp_path,tmp_path/'parent.pth',tmp_path,
                                         resume=True,control=True,continuous=True)
    observer.logged_update = True
    def save(filename):
        p=Path(filename+'.pth'); p.parent.mkdir(parents=True,exist_ok=True)
        p.write_bytes(b'complete')
    observer.algo = SimpleNamespace(frame=500_170_752,max_frames=-1,save=save,
                                    writer=SimpleNamespace(add_scalar=lambda *args: None))
    observer.after_print_stats(500_170_752,1273,0.)
    assert (tmp_path/'nn/latest.pth').exists()
    assert observer.algo.max_frames == -1
    observer.algo.frame += 393216
    observer.after_print_stats(observer.algo.frame,2048,0.)
    assert (tmp_path/'nn'/f'snapshot_{observer.algo.frame}.pth').exists()
    assert not list((tmp_path/'nn').glob('periodic_*'))
    assert observer.algo.max_frames == -1
    observer.stop_requested = 'SIGUSR1'
    observer.after_print_stats(observer.algo.frame,2049,0.)
    assert observer.algo.max_frames == observer.algo.frame
    assert json.loads((tmp_path/'latest_checkpoint.json').read_text())['stop_requested'] == 'SIGUSR1'
