"""Revo2-facing observation model; no simulator, rewards, or training side effects."""
from dataclasses import dataclass
import math
import torch


@dataclass
class TouchObservationConfig:
    enabled: bool = True
    arm_torques: bool = False
    sensor_hz: float = 70.0
    publish_hz: float = 10.0  # observed ROS stream, NOT sensor's internal rate
    latency_s: float = 0.0  # unmeasured: configurable, not falsely calibrated
    filter_tau_s: float = 0.0
    noise_std_n: float = 0.0
    gain_range: tuple = (1.0, 1.0)
    bias_std_n: float = 0.0
    dropout_probability: float = 0.0
    force_scale_n: float = 25.0
    torque_scale_nm: float = 40.0

    @property
    def dimension(self):
        # Per finger: Fn, shear_x, shear_y, validity, sample age (seconds).
        return 25 * int(self.enabled) + 7 * int(self.arm_torques)

    def validate(self, dt):
        values = (dt, self.sensor_hz, self.publish_hz, self.force_scale_n,
                  self.torque_scale_nm)
        if not all(math.isfinite(x) and x > 0 for x in values):
            raise ValueError("rates, timestep and scales must be finite and positive")
        if self.publish_hz > self.sensor_hz or self.sensor_hz > 1 / dt + 1e-6:
            raise ValueError("require publish_hz <= sensor_hz <= physics_hz")
        if not 0 <= self.dropout_probability <= 1:
            raise ValueError("invalid dropout probability")
        if not all(math.isfinite(x) and x >= 0 for x in
                   (self.latency_s, self.filter_tau_s, self.noise_std_n, self.bias_std_n)):
            raise ValueError("invalid sensor response parameter")
        if not 0 < self.gain_range[0] <= self.gain_range[1]:
            raise ValueError("invalid gain interval")


class TouchObservationModel:
    """Physics -> acquisition -> latency -> ROS sample-and-hold.

    Force/direction quantization follows the packet interface; uncertain noise,
    filtering and calibration parameters default to identity. No invented raw
    capacitive channels, contact positions, object identities or slip labels.
    Torque channels are a separate instantaneous motor-estimate input.
    """
    def __init__(self, n_envs, dt, device, cfg=None):
        self.cfg = cfg or TouchObservationConfig()
        self.cfg.validate(dt)
        self.dt, self.steps = dt, 0
        self.filtered = torch.zeros(n_envs, 5, 3, device=device)
        self.force = torch.zeros_like(self.filtered)
        self.valid = torch.zeros(n_envs, 5, dtype=torch.bool, device=device)
        self.age_s = torch.zeros(n_envs, 5, device=device)
        self.gain = torch.ones_like(self.filtered)
        self.bias = torch.zeros_like(self.filtered)
        self._slots = math.ceil(self.cfg.latency_s / dt) + 2
        self._history = torch.zeros(self._slots, n_envs, 5, 3, device=device)
        self._valid_history = torch.zeros(self._slots, n_envs, 5, dtype=torch.bool, device=device)
        self._stamp_history = torch.full((self._slots, n_envs, 5), -1., device=device)
        self._acquired = torch.zeros_like(self.filtered)
        self._acquired_valid = torch.zeros_like(self.valid)
        self._acquired_stamp = torch.full_like(self.age_s, -1.)
        self._published_stamp = torch.full_like(self.age_s, -1.)
        self.reset()

    def reset(self, ids=None):
        ids = slice(None) if ids is None else ids
        for array in (self.filtered, self.force, self.age_s, self._acquired):
            array[ids] = 0
        for array in (self.valid, self._acquired_valid):
            array[ids] = False
        self._history[:, ids] = 0
        self._valid_history[:, ids] = False
        self._stamp_history[:, ids] = -1
        self._acquired_stamp[ids] = -1
        self._published_stamp[ids] = -1
        if self.cfg.gain_range[0] == self.cfg.gain_range[1]:
            self.gain[ids] = self.cfg.gain_range[0]
        else:
            self.gain[ids] = torch.empty_like(self.gain[ids]).uniform_(*self.cfg.gain_range)
        self.bias[ids] = (torch.randn_like(self.bias[ids]) * self.cfg.bias_std_n
                          if self.cfg.bias_std_n else 0.)

    def advance(self, forces, valid=None):
        if forces.shape != self.force.shape:
            raise ValueError("expected forces (env, five fingers, three components)")
        self.steps += 1
        t = self.steps * self.dt
        good = torch.isfinite(forces).all(-1)
        if valid is not None:
            good &= valid
        raw = torch.where(good[..., None], forces, 0.) * self.gain + self.bias
        a = 1. if self.cfg.filter_tau_s == 0 else -math.expm1(-self.dt / self.cfg.filter_tau_s)
        self.filtered += a * (raw - self.filtered)
        if math.floor(t * self.cfg.sensor_hz + 1e-8) > math.floor((t-self.dt) * self.cfg.sensor_hz + 1e-8):
            noisy = self.filtered + (torch.randn_like(raw) * self.cfg.noise_std_n
                                      if self.cfg.noise_std_n else 0.)
            normal = (noisy[..., 0].clamp_min(0) * 100).round() / 100
            magnitude = (noisy[..., 1:].norm(dim=-1) * 100).round() / 100
            angle = torch.atan2(noisy[..., 2], noisy[..., 1])
            angle = torch.rad2deg(angle).round().remainder(360) * (math.pi / 180)
            self._acquired = torch.stack((normal, magnitude*angle.cos(), magnitude*angle.sin()), -1)
            self._acquired_valid = good.clone()
            self._acquired_stamp.fill_(t)
        slot = self.steps % self._slots
        self._history[slot] = self._acquired
        self._valid_history[slot] = self._acquired_valid
        self._stamp_history[slot] = self._acquired_stamp
        if math.floor(t * self.cfg.publish_hz + 1e-8) > math.floor((t-self.dt) * self.cfg.publish_hz + 1e-8):
            delayed = (self.steps - math.ceil(self.cfg.latency_s / self.dt)) % self._slots
            self.force.copy_(self._history[delayed])
            self.valid.copy_(self._valid_history[delayed])
            if self.cfg.dropout_probability:
                self.valid &= torch.rand_like(self.age_s) >= self.cfg.dropout_probability
            self._published_stamp.copy_(self._stamp_history[delayed])
        self.age_s = torch.where(self._published_stamp >= 0, t-self._published_stamp, 0.)

    def observation(self, arm_torques=None):
        parts = []
        if self.cfg.enabled:
            force = torch.where(self.valid[..., None], self.force, 0.) / self.cfg.force_scale_n
            parts.append(torch.cat((force, self.valid[..., None].float(), self.age_s[..., None]), -1).flatten(1))
        if self.cfg.arm_torques:
            if arm_torques is None or arm_torques.shape != (self.force.shape[0], 7):
                raise ValueError("seven motor torque estimates are required")
            if not torch.isfinite(arm_torques).all():
                raise ValueError("nonfinite arm motor torque estimate")
            parts.append(arm_torques / self.cfg.torque_scale_nm)
        return torch.cat(parts, -1) if parts else self.force.new_zeros(self.force.shape[0], 0)
