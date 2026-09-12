"""Strict frozen SONIC inference using the pinned upstream implementation.

No PhysX or SAPG imports. Returns the checkpoint's normalized 29 body actions,
not radians. The simulator must apply its nominal pose/action scales unchanged.
"""
import hashlib
import math
from pathlib import Path
import subprocess
import sys

import torch
from torch import nn

from .contract import LATENT_DIM, LATENT_SCALE

GRAIL_REVISION = 'aa31d8242ac79b11545b9e3635f73014a227bdfc'
WEIGHTS_SHA256 = '62f3e336cc11cbfb7517fee1f053ed92c81d44accd3757faa0febf421d0a2cd4'
CONFIG_SHA256 = '474ba9d6b92d48f757d0af6543673517cd22c69874bde6b10a5129429a087229'


def verify_file(path, expected):
    with Path(path).open('rb') as file:
        actual = hashlib.file_digest(file,'sha256').hexdigest()
    if actual != expected:
        raise ValueError(f'Pinned SONIC checksum mismatch: {path}')


def load_export_config(path):
    import yaml
    from omegaconf import OmegaConf
    class Loader(yaml.SafeLoader):
        pass
    # The exported config includes a legacy PosixPath metadata value. Handle
    # only this exact tag; never use a general arbitrary-object YAML loader.
    Loader.add_constructor('tag:yaml.org,2002:python/object/apply:pathlib.PosixPath',
        lambda loader,node: str(Path(*loader.construct_sequence(node))))
    text = Path(path).read_text()
    for old,new in (
        ('groot.rl.trl.','gear_sonic.trl.'),
        ('groot.rl.envs.','gear_sonic.envs.'),
        ('groot.rl.utils.','gear_sonic.utils.'),
        ('groot.rl.agents.modules.modules.','gear_sonic.trl.modules.base_module.'),
        ('groot.rl.agents.','gear_sonic.trl.'),
    ):
        text = text.replace(old,new)
    return OmegaConf.create(yaml.load(text,Loader=Loader))


class FrozenSonic(nn.Module):
    def __init__(self, grail, bundle, device='cpu'):
        super().__init__()
        grail, bundle = Path(grail).resolve(), Path(bundle).resolve()
        revision = subprocess.check_output(['git','-C',str(grail),'rev-parse','HEAD'],text=True).strip()
        if revision != GRAIL_REVISION:
            raise ValueError('GRAIL code revision differs from pinned controller')
        sys.path.insert(0,str(grail/'imports/SONIC'))
        checkpoint = bundle/'last.pt'
        config = bundle/'model_config.yaml'
        verify_file(checkpoint,WEIGHTS_SHA256)
        verify_file(config,CONFIG_SHA256)
        self.config = load_export_config(config)
        env, algo = self.config.env_config, self.config.algo_config
        if (env.robot.actions_dim,env.obs.obs_dims.actor_obs,env.sim_dt,env.decimation) != (29,930,.005,4):
            raise ValueError('Unexpected SONIC timing/body interface')
        from gear_sonic.trl.utils.common import custom_instantiate
        # Auxiliary training losses are not needed for frozen inference. They
        # import motion-reconstruction dependencies; retain all actor networks
        # and strict-load ALL saved actor tensors so omissions cannot be hidden.
        algo.actor.backbone.aux_loss_func = {}
        algo.actor.backbone.aux_loss_coef = {}
        self.actor = custom_instantiate(algo.actor,env_config=env,algo_config=algo,_resolve=False)
        # Compatibility path used by the released GRAIL loader for older TRL
        # checkpoint metadata. This is the REAL upstream class, not a stub.
        import trl.trainer.utils
        if not hasattr(trl.trainer.utils,'OnlineTrainerState'):
            from trl.experimental.ppo.ppo_trainer import OnlineTrainerState, exact_div
            trl.trainer.utils.OnlineTrainerState = OnlineTrainerState
            trl.trainer.utils.exact_div = exact_div
        saved = torch.load(checkpoint,map_location='cpu',weights_only=False)
        self.actor.load_state_dict(saved['policy_state_dict'],strict=True)
        if not all(torch.isfinite(v).all() for v in self.actor.state_dict().values()):
            raise ValueError('Nonfinite pretrained actor parameter')
        self.actor.requires_grad_(False)
        self.actor.eval()
        self.to(device)
        module = self.actor.actor_module
        if (module.token_total_dim,module.max_num_tokens,module.token_dim) != (64,2,32):
            raise ValueError('Unexpected SONIC latent layout')
        self.tokenizer_names = tuple(module.tokenizer_obs_names)
        self.tokenizer_dims = {k:tuple(v) for k,v in module.tokenizer_obs_dims.items()}
        self.encoder_names = tuple(module.encoder_sample_probs.keys())
        self.eval()

    def train(self, mode=True):
        # This submodule must remain frozen even when a parent adapter trains.
        return super().train(False)

    def observations(self, proprio, reference_q, reference_qd, reference_ori6):
        n = proprio.shape[0]
        expected = ((proprio,(n,930)),(reference_q,(n,10,29)),
                    (reference_qd,(n,10,29)),(reference_ori6,(n,10,6)))
        if any(value.shape != shape or not torch.isfinite(value).all() for value,shape in expected):
            raise ValueError('Invalid SONIC observation/reference shape or nonfinite values')
        # Match released commands.py -> observations.py packing, not interleaved
        # q_t/qd_t frames. See contract.checkpoint_reference_command.
        command = torch.cat((reference_q.flatten(1),reference_qd.flatten(1)),-1).reshape(n,10,58)
        encoder_index = proprio.new_zeros(n,len(self.encoder_names))
        encoder_index[:,self.encoder_names.index('g1')] = 1
        values = dict(encoder_index=encoder_index,command_multi_future_nonflat=command,
                      motion_anchor_ori_heading_mf_nonflat=reference_ori6)
        terms=[]
        for name in self.tokenizer_names:
            value = values.get(name)
            if value is None:
                value = proprio.new_zeros(n,*self.tokenizer_dims[name])
            terms.append(value.reshape(n,-1))
        tokenizer = torch.cat(terms,-1)
        if tokenizer.shape[-1] != self.config.env_config.obs.obs_dims.tokenizer:
            raise ValueError('Tokenizer packing mismatch')
        return dict(actor_obs=proprio[:,None,:],tokenizer=tokenizer[:,None,:])

    @torch.no_grad()
    def forward(self, proprio, reference_q, reference_qd, reference_ori6, residual=None):
        obs = self.observations(proprio,reference_q,reference_qd,reference_ori6)
        kwargs = {}
        if residual is not None:
            if residual.shape != (proprio.shape[0],LATENT_DIM) or not torch.isfinite(residual).all():
                raise ValueError('Expected finite 64-dimensional latent correction')
            kwargs.update(latent_residual=LATENT_SCALE*residual,latent_residual_mode='pre_quantization')
        action = self.actor(obs,is_training=False,**kwargs)[:,0]
        if action.shape != (proprio.shape[0],29) or not torch.isfinite(action).all():
            raise ValueError('Invalid SONIC body command')
        return action


class SonicHistory:
    """Ten control samples, per-term oldest-to-newest (Isaac observation order)."""
    DIMENSIONS = (3,29,29,29,3)  # base omega, q-q0, qd, last normalized action, gravity

    def __init__(self,n,device='cpu'):
        self.buffers = [torch.zeros(n,10,d,device=device) for d in self.DIMENSIONS]
        self.fresh = torch.ones(n,dtype=torch.bool,device=device)

    def reset(self,ids=None):
        ids = slice(None) if ids is None else ids
        self.fresh[ids] = True
        for buffer in self.buffers:
            buffer[ids] = 0

    def push(self,*terms):
        if len(terms) != len(self.buffers):
            raise ValueError('Expected five proprioception terms')
        for buffer,term,d in zip(self.buffers,terms,self.DIMENSIONS):
            if term.shape != (len(self.fresh),d) or not torch.isfinite(term).all():
                raise ValueError('Invalid proprioception term')
            buffer[:,:-1] = buffer[:,1:].clone()
            buffer[:,-1] = term
            buffer[self.fresh] = term[self.fresh,None,:]
        self.fresh[:] = False
        return torch.cat([b.flatten(1) for b in self.buffers],-1)
