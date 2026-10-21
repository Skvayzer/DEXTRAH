"""Portable SAPG models with independent new-channel normalization counts.

No changes to the external RL-Games checkout. Original input/value normalizers,
architecture, action distributions and SAPG embeddings keep their semantics.
"""
import torch
from rl_games.algos_torch import models, model_builder
from rl_games.algos_torch.running_mean_std import RunningMeanStd


class TouchRunningMeanStd(RunningMeanStd):
    def __init__(self, base_dim, extra_dim=0):
        # extra_dim: opt-in hand-to-surface inputs after the 25 touch channels
        # (dextrah_lab/wholebody/hand_surface.py), plain per-channel normalization.
        super().__init__((base_dim + 25 + extra_dim,))
        self.base_dim, self.extra_dim = base_dim, extra_dim
        # The original scalar count is broadcast onto OLD features at migration.
        # New channels have their own calibration count and adapt immediately.
        self.count = torch.ones(base_dim + 25 + extra_dim, dtype=torch.float64)

    def _update_mean_var_count_from_moments(self, *args):
        mean, var, count = super()._update_mean_var_count_from_moments(*args)
        var = torch.cat((var[:self.base_dim], var[self.base_dim:].clamp_min(1e-6)))
        return mean, var, count

    def forward(self, input, denorm=False, mask=None):
        if denorm:
            raise ValueError('Touch normalization is for observations, not values')
        output = super().forward(input, mask=mask)
        touch = slice(self.base_dim, self.base_dim+25)
        raw = input[:, touch].reshape(-1, 5, 5)
        extra = output[:, touch].reshape(-1, 5, 5)
        # Preserve validity as a bit; invalid forces stay zero after centering.
        # Fixed 100 ms UNITS, not the publication period. Preserve checkpoint
        # semantics when sensor rates change; do not rescale learned inputs.
        extra[..., :3] = torch.where(raw[..., 3:4] > .5, extra[..., :3], 0.)
        extra[..., 3] = raw[..., 3]
        extra[..., 4] = (raw[..., 4] / .1).clamp(0., 10.)
        return output


HAND_DIM = 0     # set by register_models (opt-in hand-to-surface inputs)


class TouchActor(models.ModelA2CContinuousLogStd):
    def build(self, config):
        if config.get('coef_id_idx') != 249+HAND_DIM:
            raise ValueError(f'Touch continuation actor requires 224 + 25 (+ {HAND_DIM}) observations')
        model = super().build(config)
        model.running_mean_std = TouchRunningMeanStd(224, HAND_DIM)
        return model


class TouchCritic(models.ModelCentralValue):
    def build(self, config):
        if config.get('coef_id_idx') != 271+HAND_DIM:
            raise ValueError(f'Touch continuation critic requires 246 + 25 (+ {HAND_DIM}) observations')
        model = super().build(config)
        model.running_mean_std = TouchRunningMeanStd(246, HAND_DIM)
        return model


def register_models(hand_dim=0):
    global HAND_DIM
    HAND_DIM = hand_dim
    model_builder.register_model('continuous_a2c_logstd_touch', TouchActor)
    model_builder.register_model('central_value_touch', TouchCritic)


def expand_touch_weights(source, target, base_dim, mean, var, calibration_count):
    from dextrah_lab.object_shape.warmstart import expand_state_dict
    prefix = '' if base_dim == 224 else 'model.'
    count_key = prefix + 'running_mean_std.count'
    old = dict(source)
    if old[count_key].numel() != 1 or target[count_key].shape != (base_dim + 25,):
        raise ValueError('Expected original scalar count and expanded per-channel counts')
    if mean.shape != (25,) or var.shape != (25,) or calibration_count < 1:
        raise ValueError('Require calibrated 25-channel tactile statistics')
    if not torch.isfinite(mean).all() or not torch.isfinite(var).all() or (var <= 0).any():
        raise ValueError('Invalid tactile calibration')
    old[count_key] = torch.cat((old[count_key].expand(base_dim),
        old[count_key].new_full((25,), float(calibration_count))))
    return expand_state_dict(old, target, base_dim, mean, var)
