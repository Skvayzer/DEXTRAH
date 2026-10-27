"""Carry posture prior (carry v2): per-episode right-arm posture from a forward-kinematics library.

The library (scripts/build_carry_postures.py) holds right-arm joint configurations with their palm pose
in the torso_link frame. For each new episode the posture is chosen so that the HELD object, with its
own palm-to-object offset, lands nearest a carry target in front of the torso (randomised per episode),
while the palm keeps an orientation close to its grasp-time one (the object is not tipped) and the arm
moves little:

    cost = |p_obj - p_target|^2 / s_p^2 + w_r * angle(R_palm, R_palm_grasp)^2 / s_r^2 + w_q |q - q_grasp|^2 / s_q^2
    p_obj = p_palm + R_palm t_po          (t_po: object position in the palm frame at the start)

A posture lookup instead of online IK: exact forward kinematics, no solver failures, vectorised on GPU.
The SONIC arm reference then moves from the grasp-time angles to the chosen posture with a
minimum-jerk profile (min_jerk) during the secure phase.
"""
import numpy as np
import torch


def quat_apply(q, v):
    """Rotate v (..., 3) by unit quaternions q (..., 4) wxyz (broadcasting)."""
    w, xyz = q[..., :1], q[..., 1:]
    t = 2*torch.cross(xyz, v, dim=-1)
    return v+w*t+torch.cross(xyz, t, dim=-1)


def min_jerk(s):
    """Minimum-jerk blend 10 s^3 - 15 s^4 + 6 s^5 of s in [0, 1] (clamped)."""
    s = s.clamp(0., 1.)
    return s**3*(10-15*s+6*s*s)


class CarryPostures:
    def __init__(self, path, device, target=(.30, -.15, .05), target_jitter=.05, sigma_pos=.05, w_rot=1., sigma_rot=.5,
                 w_q=.5, sigma_q=1., chunk=128):
        data = np.load(path)
        self.q = torch.as_tensor(data['q'], device=device)                      # (N, 7)
        self.palm_pos = torch.as_tensor(data['palm_pos'], device=device)        # (N, 3) torso frame
        self.palm_quat = torch.as_tensor(data['palm_quat'], device=device)      # (N, 4) torso frame, wxyz
        self.body_joint_index = torch.as_tensor(data['body_joint_index'], device=device, dtype=torch.long)
        self.target = torch.tensor(target, device=device)
        self.jitter, self.sigma_pos, self.w_rot, self.sigma_rot = float(target_jitter), sigma_pos, w_rot, sigma_rot
        self.w_q, self.sigma_q, self.chunk = w_q, sigma_q, int(chunk)

    def sample_targets(self, n, generator=None):
        noise = (torch.rand(n, 3, device=self.q.device, generator=generator)*2-1)*self.jitter
        return self.target+noise

    @torch.no_grad()
    def select(self, t_po, palm_quat_grasp, q_grasp, target):
        """t_po (k, 3) object in the palm frame; palm_quat_grasp (k, 4) torso frame; q_grasp (k, 7);
        target (k, 3) torso frame -> (library index (k,), q (k, 7), predicted object position (k, 3))."""
        out_idx = []
        for a in range(0, len(t_po), self.chunk):
            b = slice(a, a+self.chunk)
            p_obj = self.palm_pos[None]+quat_apply(self.palm_quat[None], t_po[b, None])            # (c, N, 3)
            cost = ((p_obj-target[b, None])**2).sum(-1)/self.sigma_pos**2
            dot = (self.palm_quat[None]*palm_quat_grasp[b, None]).sum(-1).abs().clamp(max=1.)
            cost = cost+self.w_rot*(2*torch.acos(dot))**2/self.sigma_rot**2
            cost = cost+self.w_q*((self.q[None]-q_grasp[b, None])**2).sum(-1)/self.sigma_q**2
            out_idx.append(cost.argmin(1))
        idx = torch.cat(out_idx) if out_idx else torch.zeros(0, dtype=torch.long, device=self.q.device)
        p = self.palm_pos[idx]+quat_apply(self.palm_quat[idx], t_po)
        return idx, self.q[idx], p
