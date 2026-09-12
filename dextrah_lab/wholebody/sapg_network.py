"""SONIC body decoder inside the EXISTING six-group SAPG trainer.

Reuse the trained task LSTM, all six exploration embeddings, finger means and
finger log standard deviations. A separate copied critic gains zero-initialized
body input columns. No teacher writes simulator joints; one student owns them.
"""
from copy import deepcopy
import torch
from torch import nn
from rl_games.algos_torch import models, model_builder
from rl_games.algos_torch.running_mean_std import RunningMeanStd
from .actuators import body_motors
from .contract import nominal_body_pose
from .task_contract import TASK_ACTOR_DIM, TASK_CRITIC_DIM, BODY_EXTRA_DIM, TOUCH_SHA256


class TaskBodyNormalizer(nn.Module):
    """Keep exact source tactile normalization and unclipped SONIC inputs."""
    def __init__(self, source, task_dim, normalize_body=False):
        super().__init__()
        self.task = deepcopy(source)
        self.task_dim = task_dim
        self.body = RunningMeanStd((BODY_EXTRA_DIM,)) if normalize_body else nn.Identity()

    def forward(self, x, **kwargs):
        # Original VecEnv clips only the original observation to +/-10 before
        # updating its normalizer. Never clip SONIC velocity/history at 10.
        task = self.task(x[:, :self.task_dim].clamp(-10, 10), **kwargs)
        return torch.cat((task, self.body(x[:, self.task_dim:])), -1)


def recurrent_task_features(source, observation, states, sequence_length=1, dones=None, bptt_len=0):
    """The pinned shared A2CBuilder forward, up to its trained 512-D features.

    Use the upstream recurrent wrapper itself for done handling and truncated
    BPTT. Regression tests compare means AND h/c for all six coefficient IDs.
    """
    ids = (observation[:, source.pid_idx, None] == source.param_ids).float().argmax(1)
    x = torch.cat((observation[:, :source.pid_idx], source.extra_params[ids]), -1)
    x = source.actor_cnn(x).flatten(1)
    batch = x.shape[0] // sequence_length
    x = x.reshape(batch, sequence_length, -1).transpose(0, 1)
    if dones is not None:
        dones = dones.reshape(batch, sequence_length, -1).transpose(0, 1)
    x, states = source.rnn(x, states, dones, bptt_len)
    x = x.transpose(0, 1).contiguous().reshape(batch*sequence_length, -1)
    x = source.layer_norm(x)
    return source.actor_mlp(x), states, ids


class SonicSapgNetwork(nn.Module):
    def __init__(self, source, sonic, lower, upper, num_seqs, student=None):
        super().__init__()
        self.source = deepcopy(source.a2c_network).requires_grad_(True)
        if (self.source.separate or not self.source.has_rnn or not self.source.is_rnn_before_mlp
                or self.source.rnn_name != 'lstm' or not self.source.rnn_ln
                or self.source.pid_idx != TASK_ACTOR_DIM or self.source.fixed_sigma != 'coef_cond'):
            raise ValueError('Unsupported source recurrent SAPG interface')
        self.source.num_seqs = num_seqs
        self.source.sigma.requires_grad_(False)  # replaced by the 35-action distribution below
        self.decoder = deepcopy(sonic.actor.actor_module.decoders['g1_dyn'].module).requires_grad_(True)
        self.task_to_body = nn.Linear(512, self.decoder[0].out_features, bias=False).to(self.source.mu.weight)
        nn.init.zeros_(self.task_to_body.weight)
        lower = torch.as_tensor(lower).to(self.source.mu.weight)
        upper = torch.as_tensor(upper).to(self.source.mu.weight)
        if lower.shape != (29,) or upper.shape != (29,) or not (upper > lower).all():
            raise ValueError('Require all 29 imported body target limits')
        self.register_buffer('body_center', (upper+lower)/2)
        self.register_buffer('body_half_range', (upper-lower)/2)
        self.register_buffer('body_nominal', torch.as_tensor(nominal_body_pose()).to(lower))
        self.register_buffer('body_scales', lower.new_tensor([m.action_scale for m in body_motors().values()]))
        leader = int(torch.nonzero(self.source.param_ids == 0, as_tuple=True)[0].item())
        # New body exploration starts at 0.025 rad, modulated by the relative
        # source group scales. Finger distributions are copied without change.
        old_sigma = self.source.sigma.detach()
        group_offset = (old_sigma[:, :7].mean(1)-old_sigma[leader, :7].mean()).clamp(-.693147, .693147)
        body_logstd = (.025/self.body_half_range).log()[None] + group_offset[:, None]
        self.logstd = nn.Parameter(torch.cat((body_logstd, old_sigma[:, 7:13]), -1))
        if student is not None:
            self.import_bootstrap(student)

    def import_bootstrap(self, student):
        if student.task_dim != TASK_ACTOR_DIM:
            raise ValueError('Cannot initialize this touch transfer with the older BPS-only student')
        # BC freezes these modules. Assert that they are still the original
        # source, then transfer only the decoder/conditioning weights it learnt.
        for copied, original in ((student.task_encoder.rnn, self.source.rnn.rnn),
                                 (student.task_encoder.layer_norm, self.source.layer_norm),
                                 (student.task_encoder.mlp, self.source.actor_mlp)):
            for key, value in copied.state_dict().items():
                torch.testing.assert_close(value, original.state_dict()[key], rtol=0, atol=0)
        torch.testing.assert_close(student.fingers.weight, self.source.mu.weight[7:13], rtol=0, atol=0)
        torch.testing.assert_close(student.fingers.bias, self.source.mu.bias[7:13], rtol=0, atol=0)
        self.decoder.load_state_dict(student.decoder.state_dict(), strict=True)
        self.task_to_body.load_state_dict(student.task_to_body.state_dict(), strict=True)

    def is_rnn(self):
        return True

    def get_default_rnn_state(self):
        return self.source.get_default_rnn_state()

    def get_value_layer(self):
        return self.source.value

    def forward(self, inputs):
        obs = inputs['obs']
        task = torch.cat((obs[:, :TASK_ACTOR_DIM], obs[:, TASK_ACTOR_DIM+BODY_EXTRA_DIM:]), -1)
        features, states, ids = recurrent_task_features(self.source, task, inputs.get('rnn_states'),
            inputs.get('seq_length', 1), inputs.get('dones'), inputs.get('bptt_len', 0))
        extra = obs[:, TASK_ACTOR_DIM:TASK_ACTOR_DIM+BODY_EXTRA_DIM]
        # Sequential iteration preserves SONIC's repeated shared SiLU calls.
        x = self.decoder[0](torch.cat((extra[:, 930:], extra[:, :930]), -1)) + self.task_to_body(features)
        for layer in self.decoder[1:]:
            x = layer(x)
        physical_targets = self.body_nominal + self.body_scales*x
        # Affine unit conversion only: keep SAPG's existing [-1,1] Gaussian
        # action/clip/bounds handling. Environment converts back to radians;
        # executed-action history is converted back to SONIC units there.
        body = (physical_targets-self.body_center)/self.body_half_range
        fingers = self.source.mu_act(self.source.mu(features))[:, 7:13]
        value = self.source.value_act(self.source.value(features))
        return torch.cat((body, fingers), -1), self.logstd[ids], value, states


def expanded_critic(source, num_seqs):
    network = deepcopy(source.a2c_network).requires_grad_(True)
    if network.has_rnn or network.pid_idx != TASK_CRITIC_DIM or not network.central_value:
        raise ValueError('Expected the original feedforward asymmetric critic')
    old = network.actor_mlp[0]
    if not isinstance(old, nn.Linear) or old.in_features != TASK_CRITIC_DIM+32:
        raise ValueError('Unexpected critic embedding/input layout')
    new = nn.Linear(old.in_features+BODY_EXTRA_DIM, old.out_features).to(old.weight)
    with torch.no_grad():
        new.weight.zero_()
        new.weight[:, :TASK_CRITIC_DIM] = old.weight[:, :TASK_CRITIC_DIM]
        new.weight[:, -32:] = old.weight[:, -32:]
        new.bias.copy_(old.bias)
    network.actor_mlp[0] = new
    network.pid_idx = TASK_CRITIC_DIM+BODY_EXTRA_DIM
    network.num_seqs = num_seqs
    return network


def register_sonic_models(source_actor, source_critic, sonic, lower, upper, student=None):
    """Register builders for the unchanged rl_games/SAPG Runner, once per run."""
    def validate(config, task_dim):
        if config.get('coef_id_idx') != task_dim+BODY_EXTRA_DIM or config.get('type') != 'extra_param':
            raise ValueError('Require task+body observations and six mixed SAPG groups')
        torch.testing.assert_close(config['coef_ids'].to(source_actor.a2c_network.param_ids),
                                   source_actor.a2c_network.param_ids, rtol=0, atol=0)
        if config['actions_num'] != 35:
            raise ValueError('Require 29 body + 6 right finger actions')

    class ActorBuilder:
        def load(self, params):
            self.params = params

        def build(self, name, **config):
            validate(config, TASK_ACTOR_DIM)
            return SonicSapgNetwork(source_actor, sonic, lower, upper, config['num_seqs'], student)

    class CriticBuilder(ActorBuilder):
        def build(self, name, **config):
            validate(config, TASK_CRITIC_DIM)
            return expanded_critic(source_critic, config['num_seqs'])

    class ActorModel(models.ModelA2CContinuousLogStd):
        def build(self, config):
            model = super().build(config)
            model.running_mean_std = TaskBodyNormalizer(source_actor.running_mean_std, TASK_ACTOR_DIM)
            model.value_mean_std = deepcopy(source_actor.value_mean_std)
            return model

    class CriticModel(models.ModelCentralValue):
        def build(self, config):
            model = super().build(config)
            model.running_mean_std = TaskBodyNormalizer(source_critic.running_mean_std, TASK_CRITIC_DIM, True)
            model.value_mean_std = deepcopy(source_critic.value_mean_std)
            return model

    model_builder.register_network('sonic_sapg_actor', ActorBuilder)
    model_builder.register_network('sonic_sapg_critic', CriticBuilder)
    model_builder.register_model('sonic_sapg_logstd', ActorModel)
    model_builder.register_model('sonic_sapg_value', CriticModel)


def load_touch_sources(device='cpu'):
    from pathlib import Path
    from dextrah_lab.g1_adept.touch_continuation import load_yaml
    from .sapg_features import load_sapg_actor
    from .task_contract import TOUCH_CHECKPOINT, TOUCH_RUN
    actor = load_sapg_actor(TOUCH_CHECKPOINT, Path(TOUCH_RUN)/'params/agent.yaml', TOUCH_SHA256, device)
    saved = torch.load(TOUCH_CHECKPOINT, map_location='cpu', weights_only=False)
    saved = saved[0] if 0 in saved else saved
    params = load_yaml(Path(TOUCH_RUN)/'params/agent.yaml')['params']
    cv = params['config']['central_value_config']
    critic = model_builder.ModelBuilder().load(dict(model=cv['model'], network=cv['network'])).build(
        dict(actions_num=13, input_shape=(TASK_CRITIC_DIM+32,), num_seqs=1, value_size=1,
             normalize_value=True, normalize_input=True, type='extra_param',
             coef_ids=actor.a2c_network.param_ids, coef_id_idx=TASK_CRITIC_DIM)).to(device)
    critic.load_state_dict({k.removeprefix('model.'): v for k, v in saved['assymetric_vf_nets'].items()}, strict=True)
    return actor, critic.eval().requires_grad_(False)
