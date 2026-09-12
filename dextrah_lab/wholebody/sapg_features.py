"""Reuse the trained SAPG recurrent features/fingers inside one SONIC student.

The old 7-arm output head is discarded from control. All 29 body commands come
from the trainable SONIC decoder conditioned by these task features. This is
not a parallel SAPG arm controller. Strict-load the complete source actor
before selecting its reusable modules; do not silently skip source tensors.
"""
from copy import deepcopy
from pathlib import Path

import torch
from torch import nn

from .student import SonicManipulationStudent
from .sonic import verify_file

ARCHITECTURE_SAPG = 'sonic_trainable_decoder_pretrained_sapg_lstm_v2'


def load_sapg_actor(checkpoint, agent_path, sha256, device='cpu'):
    import yaml
    from rl_games.algos_torch import model_builder
    verify_file(checkpoint,sha256)
    class Loader(yaml.SafeLoader):
        pass
    Loader.add_constructor('tag:yaml.org,2002:python/tuple',lambda loader,node:tuple(loader.construct_sequence(node)))
    config=yaml.load(Path(agent_path).read_text(),Loader=Loader)['params']
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    state=(payload[0] if 0 in payload else payload)['model']
    dim=state['running_mean_std.running_mean'].numel()
    if dim not in (224,249):
        raise ValueError('Only the audited G1 BPS/tactile SAPG actor interfaces are supported')
    if dim==249:
        from dextrah_lab.g1_adept.touch_policy import register_models
        register_models()
    network=config['network']
    if (network['separate'] or network['rnn'] != dict(name='lstm',units=1024,layers=1,before_mlp=True,layer_norm=True)
            or network['mlp']['units']!=[1024,1024,512,512] or network['mlp']['activation']!='elu'):
        raise ValueError('SAPG recurrent feature architecture changed')
    groups=state['a2c_network.extra_params'].shape[0]
    # Match the actual saved SAPG player's ordered IDs (50 -> 0). The zero-
    # entropy leader is the LAST row, not row zero. Preserve all group shapes;
    # this helper is not an optimizer/rollout resume implementation.
    model=model_builder.ModelBuilder().load(config).build(dict(actions_num=13,
        input_shape=(dim+32,),num_seqs=1,value_size=1,normalize_value=True,
        normalize_input=True,type='extra_param',coef_ids=torch.linspace(50.,0.,groups,device=device),coef_id_idx=dim)).to(device)
    model.load_state_dict(state,strict=True)
    model.eval().requires_grad_(False)
    return model


class SapgTaskFeatures(nn.Module):
    def __init__(self, network):
        super().__init__()
        if not isinstance(network.rnn.rnn,nn.LSTM) or network.rnn.rnn.num_layers!=1:
            raise ValueError('Expected the source one-layer torch LSTM')
        self.rnn=deepcopy(network.rnn.rnn)
        self.layer_norm=deepcopy(network.layer_norm)
        self.mlp=deepcopy(network.actor_mlp)
        leaders=torch.nonzero(network.param_ids==0,as_tuple=True)[0]
        if len(leaders)!=1:
            raise ValueError('SAPG source must have one zero-entropy leader')
        self.register_buffer('leader_embedding',network.extra_params[int(leaders[0])].detach().clone())
        self.hidden_size=self.rnn.hidden_size

    def forward(self, normalized, hidden=None, episode_starts=None):
        batch,steps=normalized.shape[:2]
        if hidden is None:
            hidden=tuple(normalized.new_zeros(1,batch,self.hidden_size) for _ in range(2))
        if not isinstance(hidden,(tuple,list)) or len(hidden)!=2 or any(x.shape!=(1,batch,self.hidden_size) for x in hidden):
            raise ValueError('Expected copied SAPG LSTM h/c states, not a GRU state')
        x=torch.cat((normalized,self.leader_embedding.expand(batch,steps,-1)),-1).transpose(0,1)
        if episode_starts is None:
            output,hidden=self.rnn(x,tuple(hidden))
        else:
            if episode_starts.shape!=(batch,steps) or episode_starts.dtype!=torch.bool:
                raise ValueError('Expected boolean episode-start mask')
            pieces=[]
            for t in range(steps):
                hidden=tuple(s*(~episode_starts[:,t])[None,:,None] for s in hidden)
                value,hidden=self.rnn(x[t:t+1],hidden)
                pieces.append(value)
            output=torch.cat(pieces)
        return self.mlp(self.layer_norm(output.transpose(0,1))),hidden


class SonicSapgStudent(SonicManipulationStudent):
    def __init__(self,sonic,sapg,task_mean,task_variance,freeze_task=True):
        super().__init__(sonic.actor.actor_module.decoders['g1_dyn'].module,task_mean,task_variance,hidden_dim=512)
        del self.task_projection,self.task_rnn
        self.task_encoder=SapgTaskFeatures(sapg.a2c_network)
        self.fingers=nn.Linear(512,6).to(sapg.a2c_network.mu.weight)
        with torch.no_grad():
            self.fingers.weight.copy_(sapg.a2c_network.mu.weight[7:13])
            self.fingers.bias.copy_(sapg.a2c_network.mu.bias[7:13])
        self.task_encoder.requires_grad_(not freeze_task)
        self.fingers.requires_grad_(not freeze_task)
        self.freeze_task=freeze_task

    def forward(self,proprio,task,tokens,hidden=None,episode_starts=None,task_active=True):
        if task.ndim!=3 or proprio.shape!=(*task.shape[:2],930) or tokens.shape!=(*task.shape[:2],64):
            raise ValueError('Expected matching batch/time body, task and token observations')
        context,hidden=self.task_encoder(self.normalizer(task),hidden,episode_starts)
        conditioning=context if task_active else torch.zeros_like(context)
        x=self.decoder[0](torch.cat((tokens,proprio),-1))+self.task_to_body(conditioning)
        for layer in self.decoder[1:-1]:
            x=layer(x)
        body=self.decoder[-1](x)
        # Preserve raw Gaussian means; the simulator clips finger execution to
        # [-1,1]. BC compares clipped physical finger commands explicitly.
        fingers=self.fingers(context)
        return torch.cat((body,fingers),-1),hidden
