"""Checked weights-only migration from the final BPS policy; no MDP redesign."""
import copy
from contextlib import contextmanager
import json
from pathlib import Path
import time

import torch
import yaml

from dextrah_lab.object_shape.warmstart import sha256_file
from .touch_policy import expand_touch_weights

SOURCE_SHA = '53c4b009cdd341b4a0e007111c4009da57898dcb00d264c12c81b18817637fec'
SOURCE_FRAMES = 8000372736
BANK_SHA = '69a73d99a05749459fd4339f9341d3cce142ece0f7965283c04cb6410a2f3328'


@contextmanager
def equivalence_precision():
    """Compare changed GEMM shapes without TF32 rounding; restore training flags."""
    matmul, cudnn = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = matmul
        torch.backends.cudnn.allow_tf32 = cudnn


def install_complete_checkpoint(algo):
    """The upstream SAPG save omits the central critic's Adam state."""
    original = algo.get_full_state_weights
    def complete():
        state = original()
        state['continuation_critic_optimizer'] = algo.central_value_net.optimizer.state_dict()
        state['continuation_critic_epoch'] = algo.central_value_net.epoch_num
        return state
    algo.get_full_state_weights = complete


def resume_at_episode_boundary(algo, checkpoint):
    """Retain learning state, never replay observations from absent physics."""
    payload = torch.load(checkpoint, map_location=algo.ppo_device, weights_only=False)
    state = payload[0] if 0 in payload else payload
    if 'continuation_critic_optimizer' not in state:
        raise ValueError('Resume needs a complete continuation checkpoint')
    algo.set_full_state_weights(state)
    algo.central_value_net.optimizer.load_state_dict(state['continuation_critic_optimizer'])
    algo.central_value_net.epoch_num = state['continuation_critic_epoch']
    for key in ('obs', 'rnn_states', 'current_rewards', 'current_shaped_rewards', 'current_lengths'):
        setattr(algo, key, None)
    return dict(frame=int(algo.frame), epoch=int(algo.epoch_num), fresh_episodes=True,
                actor_optimizer_restored=bool(algo.optimizer.state),
                critic_optimizer_restored=bool(algo.central_value_net.optimizer.state))


def load_yaml(path):
    class Loader(yaml.SafeLoader):
        pass
    Loader.add_constructor('tag:yaml.org,2002:python/tuple',
        lambda loader, node: tuple(loader.construct_sequence(node)))
    return yaml.load(Path(path).read_text(), Loader=Loader)


def verify_source_mdp(saved, cfg):
    """Only the mature tolerance differs from saved easy-start config."""
    current = cfg.to_dict()
    def plain(x):
        if isinstance(x, (list, tuple)):
            return [plain(v) for v in x]
        if isinstance(x, dict):
            return {k: plain(v) for k, v in x.items()}
        return x
    for key in ('action', 'reward', 'reset', 'domain_randomization', 'assets', 'obs',
                'decimation', 'episode_length_s', 'is_finite_horizon'):
        if plain(saved[key]) != plain(current[key]):
            raise ValueError(f'Continuation changed source MDP field: {key}')
    for key, value in saved['termination'].items():
        expected = .01 if key in ('success_tolerance', 'target_success_tolerance', 'eval_success_tolerance') else value
        if plain(current['termination'][key]) != plain(expected):
            raise ValueError(f'Unexpected termination change: {key}')
    for key in ('dt', 'gravity'):
        if plain(saved['sim'][key]) != plain(current['sim'][key]):
            raise ValueError(f'Unexpected physics change: {key}')
    for key, value in saved['sim']['physx'].items():
        if not key.startswith('gpu_') and plain(current['sim']['physx'][key]) != plain(value):
            raise ValueError(f'Unexpected solver change: {key}')
    if hasattr(cfg, 'touch') and (cfg.touch.material_override or cfg.touch.arm_torques or not cfg.touch.enabled):
        raise ValueError('First continuation requires sensing-only tactile, no arm torques')


def build_old_models(params, source, coef_ids, n, device):
    from rl_games.algos_torch import model_builder
    actor = model_builder.ModelBuilder().load(copy.deepcopy(params)).build(dict(
        actions_num=13, input_shape=(256,), num_seqs=n, value_size=1,
        normalize_input=True, normalize_value=True, type='extra_param',
        coef_ids=coef_ids, coef_id_idx=224)).to(device).eval()
    actor.load_state_dict(source['model'], strict=True)
    critic_params = dict(model={'name': 'central_value'},
                         network=copy.deepcopy(params['config']['central_value_config']['network']))
    critic = model_builder.ModelBuilder().load(critic_params).build(dict(
        actions_num=13, input_shape=(278,), num_seqs=n, value_size=1,
        normalize_input=True, normalize_value=True, type='extra_param',
        coef_ids=coef_ids, coef_id_idx=246)).to(device).eval()
    critic.load_state_dict({k.removeprefix('model.'): v for k,v in source['assymetric_vf_nets'].items()}, strict=True)
    return actor, critic


def calibrate_touch(env, actor, steps, path):
    if steps < 600:
        raise ValueError('Calibration must include at least ten simulated seconds of real grasps')
    sums = torch.zeros(25, dtype=torch.float64, device=env.device)
    squares, peak = torch.zeros_like(sums), torch.zeros(5,3,device=env.device)
    count, contacts = 0, torch.zeros(5,device=env.device)
    obs, _ = env.reset()
    rnn = [s.to(env.device).zero_() for s in actor.get_default_rnn_state()]
    options = dict(is_train=True, prev_actions=torch.zeros(env.num_envs,13,device=env.device))
    begin = time.monotonic()
    with torch.inference_mode():
        for step in range(steps):
            raw = torch.cat((obs['policy'][:,:224].clamp(-10,10),
                             torch.zeros(env.num_envs,1,device=env.device)), -1)
            result = actor(dict(options, obs=raw, rnn_states=rnn))
            obs, reward, terminated, truncated, _ = env.step(result['mus'].clamp(-1,1))
            rnn = result['rnn_states']
            done = terminated | truncated
            for state in rnn:
                state[:,done] = 0.
            extra = obs['policy'][:,224:].double()
            if extra.shape != (env.num_envs,25) or not torch.isfinite(extra).all():
                raise ValueError('Invalid tactile calibration samples')
            sums += extra.sum(0)
            squares += extra.square().sum(0)
            count += env.num_envs
            peak = torch.maximum(peak,env.touch_raw.abs().amax(0))
            contacts += (env.touch_raw[...,0]>.01).sum(0)
            if (step+1)%300 == 0:
                print(f'TOUCH_CALIBRATION step={step+1}/{steps} wall={time.monotonic()-begin:.1f}s',flush=True)
    mean = sums/count
    var = (squares/count-mean.square()).clamp_min(1e-6)
    if not (peak[:,0]>.01).all() or peak[:,1:].max() <= .01:
        raise ValueError(f'Calibration did not observe all fingertip contacts and shear: {peak}')
    payload = dict(mean=mean.cpu().tolist(), variance=var.cpu().tolist(), count=count,
        peak_force_n=peak.cpu().tolist(), contact_fraction=(contacts/count).cpu().tolist(),
        source_sha256=SOURCE_SHA, bank_sha256=BANK_SHA, steps=steps,
        policy='frozen deterministic BPS leader', optimizer_updates=0,
        material_override=False, sensor_hz=env.cfg.touch.sensor_hz, publish_hz=env.cfg.touch.publish_hz)
    path.write_text(json.dumps(payload,indent=2))
    return payload


class TouchContinuationObserver:
    def __init__(self, env, source_run, checkpoint, output, calibration=None,
                 calibration_steps=1200, resume=False, control=False):
        self.env, self.source_run, self.checkpoint = env, Path(source_run), Path(checkpoint)
        self.output = Path(output)
        self.output.mkdir(parents=True,exist_ok=True)
        self.calibration = Path(calibration) if calibration else None
        self.calibration_steps, self.resume, self.control = calibration_steps, resume, control
        self.started = time.monotonic()
        self.logged_update = False

    def before_init(self,*args):
        pass

    def after_init(self, algo):
        self.algo = algo
        env = self.env
        assert env._bps_manifest['features_sha256'] == BANK_SHA
        assert len(env._bps_bank) == 1200 and env.cfg.action_space == 13
        assert env._current_success_tolerance == .01
        if self.resume:
            self.report = json.loads((self.output/'warmstart_validation.json').read_text())
            return
        if sha256_file(self.checkpoint) != SOURCE_SHA:
            raise ValueError('Must initialize from the checked final 8B BPS checkpoint')
        payload = torch.load(self.checkpoint,map_location='cpu',weights_only=False)
        source = payload[0] if 0 in payload else payload
        assert source['frame'] == SOURCE_FRAMES and source['epoch'] == 20346
        params = load_yaml(self.source_run/'params/agent.yaml')['params']
        coef_ids = algo.intr_reward_coef_embd[::algo.intr_coef_block_size,0]
        assert len(coef_ids) == 6
        old_actor, old_critic = build_old_models(params,source,coef_ids,env.num_envs,algo.ppo_device)
        if self.control:
            algo.model.load_state_dict(source['model'],strict=True)
            algo.central_value_net.load_state_dict(source['assymetric_vf_nets'],strict=True)
            calibration = None
        else:
            if self.calibration:
                calibration = json.loads(self.calibration.read_text())
                assert calibration['source_sha256'] == SOURCE_SHA and calibration['bank_sha256'] == BANK_SHA
                assert calibration['material_override'] is False
                assert calibration['publish_hz'] == env.cfg.touch.publish_hz
                assert calibration['sensor_hz'] == env.cfg.touch.sensor_hz
            else:
                calibration = calibrate_touch(env,old_actor,self.calibration_steps,self.output/'calibration.json')
            mean, var = (torch.tensor(calibration[k],dtype=torch.float64) for k in ('mean','variance'))
            for src,target,base in ((source['model'],algo.model,224),
                    (source['assymetric_vf_nets'],algo.central_value_net,246)):
                target.load_state_dict(expand_touch_weights(src,target.state_dict(),base,mean,var,calibration['count']),strict=True)
            (self.output/'calibration.json').write_text(json.dumps(calibration,indent=2))
        if algo.optimizer.state or algo.central_value_net.optimizer.state:
            raise RuntimeError('Initial continuation must have fresh optimizers')
        maxima = dict(actor_mean=0.,actor_sigma=0.,actor_recurrent=0.,critic_value=0.)
        ids = torch.cat([torch.arange(i*4096,i*4096+8) for i in range(6)])
        raw = source['obs']['obs'][ids].to(algo.ppo_device)
        states = source['obs']['states'][ids].to(algo.ppo_device)
        old_rnn = [v[:,ids].to(algo.ppo_device) for v in source['rnn_states']]
        new_rnn = [v.clone() for v in old_rnn]
        actor_mode,critic_mode = algo.model.training,algo.central_value_net.model.training
        algo.model.eval(); algo.central_value_net.model.eval()
        devices = [torch.device(algo.ppo_device).index or 0] if str(algo.ppo_device).startswith('cuda') else []
        print('EQUIVALENCE_RUNTIME_PRECISION '+json.dumps(dict(matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
            cudnn_tf32=torch.backends.cudnn.allow_tf32)),flush=True)
        with torch.random.fork_rng(devices=devices), torch.no_grad(), equivalence_precision():
            for step in range(8):
                tactile = torch.randn(len(ids),25,device=algo.ppo_device)
                augmented = raw if self.control else torch.cat((raw[:,:224],tactile,raw[:,224:]),-1)
                opts = dict(is_train=True,prev_actions=torch.zeros(len(ids),13,device=algo.ppo_device),seq_length=1)
                a = old_actor(dict(opts,obs=raw,rnn_states=old_rnn))
                b = algo.model(dict(opts,obs=augmented,rnn_states=new_rnn))
                for field,key in (('mus','actor_mean'),('sigmas','actor_sigma')):
                    maxima[key] = max(maxima[key],float((a[field]-b[field]).abs().max()))
                    torch.testing.assert_close(a[field],b[field],atol=5e-5,rtol=1e-5)
                for x,y in zip(a['rnn_states'],b['rnn_states']):
                    maxima['actor_recurrent'] = max(maxima['actor_recurrent'],float((x-y).abs().max()))
                    torch.testing.assert_close(x,y,atol=5e-5,rtol=1e-5)
                old_rnn,new_rnn = a['rnn_states'],b['rnn_states']
                augmented_states = states if self.control else torch.cat((states[:,:246],tactile,states[:,246:]),-1)
                av = old_critic(dict(obs=states,is_train=False))['values']
                bv = algo.central_value_net.model(dict(obs=augmented_states,is_train=False))['values']
                maxima['critic_value'] = max(maxima['critic_value'],float((av-bv).abs().max()))
                torch.testing.assert_close(av,bv,atol=5e-4,rtol=1e-5)
        algo.model.train(actor_mode); algo.central_value_net.model.train(critic_mode)
        self.report = dict(source_checkpoint=str(self.checkpoint),source_sha256=SOURCE_SHA,
            source_frames=SOURCE_FRAMES,bank_sha256=BANK_SHA,maximum_errors=maxima,
            equivalence_precision='FP32, TF32 disabled only during verification',
            actor_dim=env.cfg.observation_space,critic_dim=env.cfg.state_space,
            fresh_optimizer=True,material_override=False,strict_tolerance=.01,
            arm_torques=False,control=self.control,calibration=calibration)
        (self.output/'warmstart_validation.json').write_text(json.dumps(self.report,indent=2))
        torch.save(dict(model=algo.model.state_dict(),assymetric_vf_nets=algo.central_value_net.state_dict(),
                        continuation=self.report),self.output/'initial_weights.pth')
        print('TOUCH_WARMSTART_VALIDATED '+json.dumps({k:v for k,v in self.report.items() if k!='calibration'}),flush=True)
        del old_actor,old_critic,payload,source
        torch.cuda.empty_cache()

    def process_infos(self,*args,**kwargs):
        pass

    def after_steps(self):
        pass

    def after_clear_stats(self):
        pass

    def after_print_stats(self,frame,epoch_num,total_time):
        a,e = self.algo,self.env
        if not self.logged_update:
            import wandb
            if wandb.run:
                wandb.run.summary.update(dict(optimizer_updates_started=True,
                    first_logged_epoch=int(epoch_num),first_logged_frame=int(a.frame)))
            self.logged_update = True
        data = dict(additional_transitions=frame,cumulative_transitions=SOURCE_FRAMES+frame,
            cuda_allocated_gib=torch.cuda.memory_allocated()/2**30,
            cuda_peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
            tolerance=e._current_success_tolerance)
        free,total = torch.cuda.mem_get_info()
        data.update(device_used_gib=(total-free)/2**30,device_free_gib=free/2**30)
        if not self.control:
            data.update(actor_touch_weight_norm=float(a.model.a2c_network.rnn.rnn.weight_ih_l0[:,224:249].detach().norm()),
                critic_touch_weight_norm=float(a.central_value_net.model.a2c_network.actor_mlp[0].weight[:,246:271].detach().norm()),
                force_normal_mean_n=float(e.touch_raw[...,0].mean()),
                force_shear_mean_n=float(e.touch_raw[...,1:].norm(dim=-1).mean()),
                force_abs_max_n=float(e.touch_raw.abs().max()),
                valid_fraction=float(e.touch_model.valid.float().mean()),
                mean_sample_age_s=float(e.touch_model.age_s.mean()),
                new_normalizer_count=float(a.model.running_mean_std.count[224]),
                old_normalizer_count=float(a.model.running_mean_std.count[0]))
        for key,value in data.items():
            a.writer.add_scalar('touch_continuation/'+key,value,frame)
        if epoch_num%10 == 0:
            for model in (a.model,a.central_value_net):
                if any(not torch.isfinite(p).all() for p in model.parameters()):
                    raise RuntimeError('Nonfinite trainable weight')
        (self.output/'progress.json').write_text(json.dumps(dict(data,epoch=epoch_num,wall_seconds=time.monotonic()-self.started),indent=2))
        # Upstream save_frequency follows a sparse square-number schedule.
        # Use an explicit regular interval for this experiment instead.
        if epoch_num % 64 == 0:
            a.save(str(self.output/'nn'/f'periodic_{a.frame}'))
