#!/usr/bin/env python3
"""Run the full saved-policy expansion check without constructing Isaac scenes."""
import argparse
import copy
from pathlib import Path
from types import SimpleNamespace
import torch
from rl_games.algos_torch import model_builder
from dextrah_lab.g1_adept.touch_policy import register_models
from dextrah_lab.g1_adept.touch_continuation import TouchContinuationObserver, load_yaml, BANK_SHA


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--calibration',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cpu')
    args=p.parse_args()
    torch.set_num_threads(4)
    register_models()
    params=load_yaml(args.source_run/'params/agent.yaml')['params']
    params['model']['name']='continuous_a2c_logstd_touch'
    coefficients=torch.tensor([50.,40.,30.,20.,10.,0.],device=args.device)
    actor=model_builder.ModelBuilder().load(params).build(dict(actions_num=13,input_shape=(281,),
        num_seqs=48,value_size=1,normalize_input=True,normalize_value=True,type='extra_param',
        coef_ids=coefficients,coef_id_idx=249)).to(args.device)
    critic_params=dict(model=dict(name='central_value_touch'),network=copy.deepcopy(params['config']['central_value_config']['network']))
    critic=torch.nn.Module()
    critic.model=model_builder.ModelBuilder().load(critic_params).build(dict(actions_num=13,input_shape=(303,),
        num_seqs=48,value_size=1,normalize_input=True,normalize_value=True,type='extra_param',
        coef_ids=coefficients,coef_id_idx=271)).to(args.device)
    critic.optimizer=torch.optim.Adam(critic.parameters())
    env=SimpleNamespace(_bps_manifest=dict(features_sha256=BANK_SHA),_bps_bank=[0]*1200,
        _current_success_tolerance=.01,num_envs=48,cfg=SimpleNamespace(action_space=13,
        observation_space=249,state_space=271,touch=SimpleNamespace(sensor_hz=70.,publish_hz=10.)))
    algo=SimpleNamespace(model=actor,central_value_net=critic,optimizer=torch.optim.Adam(actor.parameters()),
        intr_reward_coef_embd=coefficients.repeat_interleave(8)[:,None],intr_coef_block_size=8,ppo_device=args.device)
    TouchContinuationObserver(env,args.source_run,args.checkpoint,args.output,args.calibration).after_init(algo)


if __name__=='__main__':
    main()
