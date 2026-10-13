"""rl_games registration for the mjlab env, with Isaac Lab's wrapper semantics.

Mirrors `isaaclab_rl.rl_games.RlGamesVecEnvWrapper` as used by Play2Perfect's
`register_rlgames_env`: obs {'obs': policy, 'states': critic}, actions and
observations clipped to the given bounds, dones = terminated | truncated,
`extras['time_outs'] = truncated` (infinite-horizon task).
"""
import numpy as np
import torch
# rl_games allocates its buffers only for `type(space) is gym.spaces.Box` of
# the legacy gym package it imports; gymnasium spaces leave 'obses' = None.
from rl_games.common.experience import gym


class MjRlGamesVecEnv:
    def __init__(self, env, rl_device, clip_obs, clip_actions):
        self.env, self.unwrapped = env, env
        self._rl_device, self._sim_device = rl_device, env.device
        self._clip_obs, self._clip_actions = clip_obs, clip_actions
        self.num_envs = env.num_envs
        obs = env.single_observation_space['policy'].shape
        states = env.single_observation_space['critic'].shape
        self.observation_space = gym.spaces.Box(-clip_obs, clip_obs, obs)
        self.state_space = gym.spaces.Box(-clip_obs, clip_obs, states)
        self.rlg_num_states = states[0]
        low, high = env.single_action_space.low, env.single_action_space.high
        self.action_space = gym.spaces.Box(np.maximum(low, -clip_actions), np.minimum(high, clip_actions))

    @property
    def device(self):   # RlGamesVecEnvWrapper.device; the P2P rl_games fork moves actions there
        return self._sim_device

    def get_number_of_agents(self):
        return 1

    def get_env_info(self):
        return dict(observation_space=self.observation_space, action_space=self.action_space,
                    state_space=self.state_space)

    def seed(self, seed=-1):
        return seed

    def _process_obs(self, obs):
        out = dict(obs=torch.clamp(obs['policy'], -self._clip_obs, self._clip_obs).to(self._rl_device).clone())
        out['states'] = torch.clamp(obs['critic'], -self._clip_obs, self._clip_obs).to(self._rl_device).clone()
        return out

    def reset(self):
        obs, _ = self.env.reset()
        return self._process_obs(obs)

    def step(self, actions):
        actions = torch.clamp(actions.detach().clone().to(self._sim_device), -self._clip_actions, self._clip_actions)
        obs, rew, terminated, truncated, extras = self.env.step(actions)
        extras['time_outs'] = truncated.to(self._rl_device)
        out = self._process_obs(obs)
        dones = (terminated | truncated).to(self._rl_device)
        extras = {k: v.to(self._rl_device, non_blocking=True) if hasattr(v, 'to') else v for k, v in extras.items()}
        if 'log' in extras:
            extras['episode'] = extras.pop('log')
        return out, rew.to(self._rl_device), dones, extras

    def close(self):
        pass


def register_mj_rlgames_env(env, rl_device, clip_obs, clip_actions, name='rlgpu'):
    from rl_games.common import env_configurations, vecenv
    wrapped = MjRlGamesVecEnv(env, rl_device, clip_obs, clip_actions)

    class _Gpu(vecenv.IVecEnv):
        def __init__(self, config_name, num_actors, **kwargs):
            self.env = env_configurations.configurations[config_name]['env_creator'](**kwargs)

        def step(self, action):
            return self.env.step(action)

        def reset(self):
            return self.env.reset()

        def get_number_of_agents(self):
            return self.env.get_number_of_agents()

        def get_env_info(self):
            return self.env.get_env_info()

    vecenv.register('IsaacRlgWrapper', lambda config_name, num_actors, **kw: _Gpu(config_name, num_actors, **kw))
    env_configurations.register(name, dict(vecenv_type='IsaacRlgWrapper', env_creator=lambda **kw: wrapped))
    return wrapped
