"""Dual-clip PPO actor loss (opt-in: --dual-clip C), Ye et al., "Mastering Complex Control
in MOBA Games with Deep Reinforcement Learning", AAAI 2020.

rl_games' clipped surrogate max(-A r, -A clip(r, 1-e, 1+e)) is unbounded for A < 0 when the
ratio r is large. SAPG trains the leader on the followers' transitions (off-policy rows), where
r can be huge in states the policies disagree on; job 1447 had KL spikes up to 9663 and NaN
losses there. Dual clip caps the loss of A < 0 samples at -C A (C > 1); A >= 0 is unchanged.
"""
import torch


def dual_clip_actor_loss(c):
    if c <= 1.:
        raise ValueError('Dual-clip constant must exceed 1')

    def actor_loss(old_action_neglog_probs_batch, action_neglog_probs, advantage, is_ppo, curr_e_clip):
        if not is_ppo:
            return action_neglog_probs*advantage
        ratio = torch.exp(old_action_neglog_probs_batch-action_neglog_probs)
        surr1 = advantage*ratio
        surr2 = advantage*torch.clamp(ratio, 1.-curr_e_clip, 1.+curr_e_clip)
        clipped = torch.max(-surr1, -surr2)
        return torch.where(advantage < 0, torch.min(clipped, -c*advantage), clipped)
    return actor_loss


def install_dual_clip(algo, c):
    algo.actor_loss_func = dual_clip_actor_loss(c)
    return dict(dual_clip=c, reference='Ye et al. AAAI 2020')
