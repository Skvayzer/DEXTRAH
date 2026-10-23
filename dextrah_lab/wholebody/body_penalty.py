"""Opt-in whole-body penalties (--sonic-action-penalty-scale / --fall-penalty).

Policies drive frozen SONIC to command the right wrist at its joint limit (normalized
SONIC action up to ~21; SONIC's own range stays within ~9) in 19-27% of lifted steps
(scripts/diagnose_sonic_actions.py, job 1450), and robot falls rose to 0.5-1.8% of
episodes late in job 1454. Out-of-range SONIC inputs are a sim-to-real risk.

* SONIC action penalty, every policy step:
      reward -= scale * sum_j max(|a_j| - bound, 0)
  a = frozen SONIC's decoded body action (29 joints, normalized SONIC units), bound
  default 8 (just below the original harvest's maximum of ~9).
* Fall penalty: reward -= fall_penalty on the step an episode ends with a robot fall.
"""
import torch


def sonic_action_excess(action, bound):
    """(n, 29) -> (n,) summed exceedance of |a| over bound."""
    return (action.abs()-bound).clamp_min(0.).sum(-1)


class BodyPenalty:
    def __init__(self, env, action_scale=0., action_bound=8., fall_penalty=0.):
        if action_scale < 0 or fall_penalty < 0 or action_bound <= 0:
            raise ValueError('Body penalties must be non-negative with a positive bound')
        self.env, self.scale, self.bound, self.fall = env, float(action_scale), float(action_bound), float(fall_penalty)

    @torch.no_grad()
    def __call__(self):
        env = self.env
        excess = sonic_action_excess(env._last_decoded_sonic_action, self.bound)
        fell = env._body_fallen.float()
        penalty = self.scale*excess+self.fall*fell
        terms = dict(sonic_action_penalty=-self.scale*excess, fall_penalty=-self.fall*fell)
        logs = dict(sonic_action_excess=excess, sonic_action_over_bound=(excess > 0).float())
        return -penalty, terms, logs
