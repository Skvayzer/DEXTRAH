"""Exact, lower-peak-memory SAPG batch handling (opt-in: --lean-sapg-batches).

rl_games' leader-follower ('lf') augment_batch_for_mixed_expl concatenates k full
copies of every batch tensor (obs, critic states, ...) and then keeps the original
plus one SAPG block of each other copy. ``lean_augment`` builds the kept rows
directly (same values, same row order, same random draws, same critic calls), and
empties the consumed rollout dict (train_epoch does not use it afterwards). The
previous epoch's PPO datasets are released before the next rollout instead of after
it (they are replaced by prepare_dataset anyway). Only peak GPU memory changes;
tests/test_sapg_memory.py compares against the original method.
"""
import numpy as np
import torch
from torch import distributed as dist

from rl_games.common.custom_utils import filter_leader, swap_and_flatten01


def _keep(val, repeat_idxs, bsize, dim=0):
    """= filter_leader(torch.cat([val]*k, dim), len, repeat_idxs, num_blocks) without the k copies."""
    parts = [val if idx == 0 else val.narrow(dim, (idx-1)*bsize, bsize) for idx in repeat_idxs]
    return torch.cat(parts, dim)


def lean_augment(self, batch_dict, extras, repeat_idxs=None):
    if self.use_others_experience != 'lf':
        return type(self).augment_batch_for_mixed_expl(self, batch_dict, extras, repeat_idxs)
    new_batch_dict = {}
    num_blocks = self.num_actors // self.intr_coef_block_size
    if repeat_idxs is None:
        num_repeat = min(num_blocks, int(self.config['off_policy_ratio']) + 1)
        repeat_idxs = [0] + [int(x) for x in np.random.choice(range(1, self.num_actors // self.intr_coef_block_size), num_repeat-1, replace=False)]
        if self.multi_gpu:
            dist.broadcast_object_list(repeat_idxs, 0)
    embd = self.intr_reward_coef_embd
    E = embd.shape[-1]

    def coef_rows(n_rows):
        """The per-row SAPG embedding the original writes into the k-fold copy, kept rows only."""
        bsize = n_rows // num_blocks
        rows = [torch.roll(embd, self.intr_coef_block_size*idx, dims=0).repeat_interleave(self.horizon_length, dim=0)
                for idx in repeat_idxs]
        return torch.cat([r if idx == 0 else r[(idx-1)*bsize:idx*bsize] for r, idx in zip(rows, repeat_idxs)], 0)

    for key, val in batch_dict.items():
        if key in ['played_frames', 'step_time']:
            new_batch_dict[key] = val
        elif key in ('obses', 'states'):
            bsize = len(val) // num_blocks
            out = _keep(val, repeat_idxs, bsize)
            out[:, -E:] = coef_rows(len(val))
            new_batch_dict[key] = out
            if key == 'obses':
                mask = torch.ones(len(out), dtype=torch.bool, device=out.device)
                mask[:len(val)] = False
                new_batch_dict['off_policy_mask'] = mask
        elif key in ['values', 'returns']:
            pass  # handled below
        elif key == 'rnn_states':
            # filter_leader slices dim 1 for single-layer states (len(v) == 1); keep its
            # behaviour for multi-layer states by delegating.
            new_batch_dict[key] = None if val is None else [
                _keep(v, repeat_idxs, v.shape[1] // num_blocks, dim=1) if len(v) == 1 else
                filter_leader(torch.cat([v]*len(repeat_idxs), dim=1), v.shape[1], repeat_idxs, num_blocks) for v in val]
        else:
            new_batch_dict[key] = _keep(val, repeat_idxs, len(val) // num_blocks)

    # Off-policy returns/values: identical to the original (full critic passes per copy).
    new_returns_list = [batch_dict['returns']]
    new_values_list = [batch_dict['values']]
    for r_k in repeat_idxs[1:]:
        mb_rewards = extras['rewards']
        mb_obs = extras['obs']
        last_obs_and_states = extras['last_obs']
        last_rnn_states = extras['last_rnn_states']
        mb_states = extras['states']
        mb_rnn_states = extras['rnn_states']
        mb_obs[:, :, -E:] = torch.roll(embd, self.intr_coef_block_size*r_k, dims=0)
        last_obs_and_states['obs'][:, -E:] = torch.roll(embd, self.intr_coef_block_size*r_k, dims=0)
        flattened_rnn_states = [rnn_s.transpose(0, 1).reshape(rnn_s.transpose(0, 1).shape[0], -1, *rnn_s.shape[3:])
                                for rnn_s in mb_rnn_states] if mb_rnn_states is not None else None
        flattened_mb_obs = mb_obs.reshape(-1, *mb_obs.shape[2:])
        flattened_mb_states = mb_states.reshape(-1, *mb_states.shape[2:]) if mb_states is not None else None
        mb_values = []
        for i in range((flattened_mb_obs.shape[0] + 8191) // 8192):
            mb_values.append(self.get_values({
                'obs': flattened_mb_obs[i*8192:(i+1)*8192],
                'states': flattened_mb_states[i*8192:(i+1)*8192] if mb_states is not None else None
                }, rnn_states=[s[:, i*8192:(i+1)*8192] for s in flattened_rnn_states] if flattened_rnn_states is not None else None))
        mb_values = torch.cat(mb_values, dim=0)
        last_values = self.get_values(last_obs_and_states, last_rnn_states)
        mb_values = mb_values.reshape(*mb_obs.shape[:2], *mb_values.shape[1:])
        mb_values = torch.cat([mb_values, last_values.unsqueeze(0)], dim=0)
        mb_fdones = extras['dones']
        fdones = extras['last_dones']
        mb_fdones = torch.cat([mb_fdones, fdones.unsqueeze(0)], dim=0)
        mb_returns = mb_rewards + (torch.roll(self.intr_reward_coef, self.intr_coef_block_size*r_k, dims=0).unsqueeze(0).unsqueeze(2) * extras['mb_intr_rewards'] if extras['mb_intr_rewards'] is not None else 0) + self.gamma * mb_values[1:] * (1 - mb_fdones[1:]).unsqueeze(-1)
        new_returns_list.append(swap_and_flatten01(mb_returns))
        new_values_list.append(swap_and_flatten01(mb_values[:-1]))
    n = len(batch_dict['returns'])
    bsize = n // num_blocks
    new_batch_dict['returns'] = torch.cat([r if idx == 0 else r[(idx-1)*bsize:idx*bsize]
                                           for r, idx in zip(new_returns_list, repeat_idxs)], 0)
    new_batch_dict['values'] = torch.cat([v if idx == 0 else v[(idx-1)*bsize:idx*bsize]
                                          for v, idx in zip(new_values_list, repeat_idxs)], 0)
    extras['obs'][:, :, -E:] = embd
    extras['last_obs']['obs'][:, -E:] = embd
    batch_dict.clear()   # the rollout dict is consumed here; train_epoch uses only the result
    return new_batch_dict


def install_lean_sapg_batches(algo):
    """Bind lean_augment and release the previous datasets before each rollout."""
    if not getattr(algo, 'expl_type', '').startswith('mixed_expl') or algo.use_others_experience != 'lf':
        raise ValueError('Lean SAPG batches implement the leader-follower (lf) SAPG update only')
    algo.augment_batch_for_mixed_expl = lean_augment.__get__(algo)
    original = algo.train_epoch

    def train_epoch(*args, **kwargs):
        algo.dataset.update_values_dict(None)
        if getattr(algo, 'central_value_net', None) is not None:
            algo.central_value_net.dataset.update_values_dict(None)
        return original(*args, **kwargs)
    algo.train_epoch = train_epoch
    return dict(augment='kept rows built directly (no k-fold copies); consumed rollout dict emptied',
                datasets='previous epoch PPO datasets released before the rollout')
