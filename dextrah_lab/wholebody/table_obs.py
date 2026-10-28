"""Table observation (opt-in: train_g1_sonic_sapg.py --table-obs; 2026-10-28 placement design review).

One flat box table described the way the object already is: geometry relative to the robot and the hand, never
absolute heights, so "the table is 10 cm below the palm" means the same at a 0.70 m and a 0.90 m table. Encoded
like the hand-to-object vectors (wholebody/hand_surface.py): decimetres, large values clipped at 3 (30 cm).

Block (TABLE_OBS_DIM = 54, appended after the hand-surface block; network columns start at zero):
  plane      5   table-top height relative to the pelvis and to the palm centre; table normal in the pelvis frame
  edges      7   torso -> nearest table-top edge point (torso yaw frame, 2) and its distance; table half extents
                 (2); table yaw relative to the torso (cos, sin)
  hand      17   signed height above the table top of the 17 hand points (palm centre, 5 fingertip pads, 11 finger
                 links): the "fingers touch the table first" signal
  body      21   nearest-point vectors to the table box from the elbows, the forearm ends (wrist roll links), the
                 torso and the knees (vectors, not heights: knees and torso meet the table's side)
  object     1   height of the object's support point above the table top: soft minimum (1 cm) over its keypoints
                 (smooth under rotation, unlike the hard lowest point)
  critic     3   log(1 + N): object-table force, right-hand-table force (monitored hand / wrist links), largest
                 other body-table force. Zeros in the actor's copy (actor and critic blocks have the same width).

Actor noise (the real robot estimates the plane from depth): per episode a table-height bias U(+-height_noise), an
edge / position bias U(+-edge_noise) per axis and a yaw bias U(+-yaw_noise); per step jitter N(0, step_noise) on
the height. The actor's object support point uses the object pose its other object observations use; the critic
gets exact values.
"""
import math

import torch

from .hand_surface import CLIP, HAND_SURFACE_POINTS, SCALE, encode, nearest_surface_vectors

BODY_TABLE_LINKS = ('left_elbow_link', 'right_elbow_link', 'left_wrist_roll_link', 'right_wrist_roll_link',
                    'torso_link', 'left_knee_link', 'right_knee_link')
PLANE_DIM, EDGE_DIM, CRITIC_DIM = 5, 7, 3
TABLE_OBS_DIM = PLANE_DIM+EDGE_DIM+HAND_SURFACE_POINTS+3*len(BODY_TABLE_LINKS)+1+CRITIC_DIM
SUPPORT_TAU = .01


def yaw_of_quat(q):
    """Yaw (rad) of wxyz quaternions (..., 4)."""
    w, x, y, z = q.unbind(-1)
    return torch.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))


def rotate_z(v, angle):
    """Rotate the xy part of v (..., 2 or 3) by angle (...) about z."""
    c, s = torch.cos(angle), torch.sin(angle)
    x, y = v[..., 0], v[..., 1]
    out = torch.stack((c*x-s*y, s*x+c*y), -1)
    return torch.cat((out, v[..., 2:]), -1) if v.shape[-1] > 2 else out


def yaw_matrix(yaw):
    c, s, o, i = torch.cos(yaw), torch.sin(yaw), torch.zeros_like(yaw), torch.ones_like(yaw)
    return torch.stack((c, -s, o, s, c, o, o, o, i), -1).reshape(*yaw.shape, 3, 3)


def nearest_edge(local, half):
    """Vector (table frame, xy) from points ``local`` (n, 2) to the nearest point on the boundary of the rectangle
    [-half, half] (2,): inside, to the nearest side; outside, to the clamped point."""
    clamped = torch.maximum(torch.minimum(local, half), -half)
    gap = half-local.abs()
    inside = (gap > 0).all(-1, keepdim=True)
    axis = torch.nn.functional.one_hot(gap.argmin(-1), 2).bool()
    sign = torch.where(local >= 0, 1., -1.)
    on_side = torch.where(axis, sign*half, local)
    return torch.where(inside, on_side, clamped)-local


def soft_min(z, tau=SUPPORT_TAU):
    return -tau*torch.logsumexp(-z/tau, -1)


class TableObservation:
    def __init__(self, env, hand_surface, noise=True, height_noise=.015, edge_noise=.03, yaw_noise_deg=2.,
                 step_noise=.003, keypoints_fn=None, table_size=None):
        if table_size is None:
            from dextrah_lab.mj.scene import TABLE_SIZE as table_size
        if hand_surface is None:
            raise ValueError('The table observation uses the hand-surface points (enable --hand-surface-obs)')
        self.env, self.hs, d = env, hand_surface, env.device
        names = env.robot.body_names
        missing = [b for b in BODY_TABLE_LINKS if b not in names]
        if missing:
            raise ValueError(f'Table observation links missing from the robot: {missing}')
        self.body_ids = torch.tensor([names.index(b) for b in BODY_TABLE_LINKS], device=d)
        self.torso_id = names.index('torso_link')
        self.half = torch.tensor(table_size, device=d, dtype=torch.float32)/2
        self.dim = TABLE_OBS_DIM
        if keypoints_fn is None:
            from dextrah_lab.mj.compat import p2p
            keypoints_fn = p2p('obs_utils')._keypoints_world
        self._keypoints = keypoints_fn
        n = env.num_envs
        self.noise = bool(noise)
        self.height_noise, self.edge_noise, self.yaw_noise = float(height_noise), float(edge_noise), math.radians(yaw_noise_deg)
        self.step_noise = float(step_noise)
        self.bias_h, self.bias_xy, self.bias_yaw = torch.zeros(n, device=d), torch.zeros(n, 2, device=d), torch.zeros(n, device=d)
        monitored = list(getattr(env, 'body_table_names', []))
        self.hand_cols = [i for i, m in enumerate(monitored) if m.startswith('right_') and any(k in m for k in ('wrist', 'base_link', 'palm', 'hand'))]
        self.body_cols = [i for i in range(len(monitored)) if i not in self.hand_cols]

    def reset(self, ids):
        """New per-episode estimation biases for the worlds ``ids``."""
        if not self.noise or ids.numel() == 0:
            return
        k, d = ids.numel(), self.env.device
        u = lambda *s: torch.rand(*s, device=d)*2-1
        self.bias_h[ids] = u(k)*self.height_noise
        self.bias_xy[ids] = u(k, 2)*self.edge_noise
        self.bias_yaw[ids] = u(k)*self.yaw_noise

    def table_pose(self, noisy):
        env = self.env
        pos = env.table.data.root_pos_w-env.scene.env_origins
        yaw = yaw_of_quat(env.table.data.root_quat_w)
        if noisy and self.noise:
            pos = pos.clone()
            pos[:, :2] = pos[:, :2]+self.bias_xy
            pos[:, 2] = pos[:, 2]+self.bias_h+self.step_noise*torch.randn_like(self.bias_h)
            yaw = yaw+self.bias_yaw
        return pos, yaw

    def features(self, obj_pos, obj_quat, noisy):
        env = self.env
        origin = env.scene.env_origins
        pos, yaw = self.table_pose(noisy)
        top = pos[:, 2]+self.half[2]
        state = env.robot.data.body_state_w
        root = env.robot.data.root_pos_w-origin
        points = self.hs.points()[:, :HAND_SURFACE_POINTS]                     # (n, 17, 3) env-local
        # plane: heights relative to the pelvis and the palm centre, normal in the pelvis frame
        root_q = env.robot.data.root_quat_w
        up = torch.zeros_like(root)
        up[:, 2] = 1.
        w, xyz = root_q[:, :1], root_q[:, 1:]
        t = 2*torch.cross(-xyz, up, dim=-1)                                     # inverse rotation of the table normal
        normal = up+w*t+torch.cross(-xyz, t, dim=-1)
        plane = torch.stack(((top-root[:, 2])*SCALE, (top-points[:, 0, 2])*SCALE), -1)
        # edges: torso -> nearest table-top edge, torso yaw frame
        torso = state[:, self.torso_id]
        tpos, tyaw = torso[:, :3]-origin, yaw_of_quat(torso[:, 3:7])
        local = rotate_z(tpos[:, :2]-pos[:, :2], -yaw)
        edge = rotate_z(rotate_z(nearest_edge(local, self.half[:2]), yaw), -tyaw)
        rel = yaw-tyaw
        edges = torch.cat((edge*SCALE, edge.norm(dim=-1, keepdim=True)*SCALE, self.half[:2].expand(len(pos), 2)*SCALE,
                           torch.stack((torch.cos(rel), torch.sin(rel)), -1)), -1)
        # hand points: signed height above the table top
        hand = ((points[..., 2]-top[:, None])*SCALE).clamp(-CLIP, CLIP)
        # body links: nearest-point vectors to the table box
        links = state[:, self.body_ids, :3]-origin[:, None]
        n = len(pos)
        box = nearest_surface_vectors(links, pos[:, None], yaw_matrix(yaw)[:, None], self.half.expand(n, 1, 3),
                                      torch.ones(1, dtype=torch.bool, device=pos.device),
                                      torch.ones(n, 1, dtype=torch.bool, device=pos.device))
        body = encode(box)
        # object support point: soft minimum over the keypoints
        kp = self._keypoints(obj_pos, obj_quat, env._keypoint_offsets)
        support = ((soft_min(kp[..., 2])-top)*SCALE).clamp(-CLIP, CLIP)[:, None]
        return torch.cat((plane, normal, edges, hand, body, support), -1)

    def critic_forces(self):
        env, z = self.env, torch.zeros(self.env.num_envs, device=self.env.device)
        c = getattr(env, '_last_contacts', None) or {}
        obj = c.get('object_table_force', z)
        bt = getattr(env, '_body_table_force_step', None)
        hand = bt[:, self.hand_cols].amax(-1) if bt is not None and self.hand_cols else z
        body = bt[:, self.body_cols].amax(-1) if bt is not None and self.body_cols else z
        return torch.log1p(torch.stack((obj.reshape(-1), hand, body), -1).clamp_min(0.))

    def observation(self, actor_pose, true_pose):
        """(actor, critic) blocks of TABLE_OBS_DIM: the actor with estimation noise, its own object pose and zeroed
        critic-only forces; the critic exact."""
        critic = torch.cat((self.features(*true_pose, noisy=False), self.critic_forces()), -1)
        actor = torch.cat((self.features(*actor_pose, noisy=True), torch.zeros_like(critic[:, -CRITIC_DIM:])), -1)
        return actor, critic

    def contract(self):
        return dict(dim=self.dim, layout=dict(plane=PLANE_DIM, edges=EDGE_DIM, hand=HAND_SURFACE_POINTS,
                                              body=3*len(BODY_TABLE_LINKS), object_support=1, critic_forces=CRITIC_DIM),
                    body_links=list(BODY_TABLE_LINKS), units='decimetres, clipped at 3 where noted', noise=self.noise,
                    height_noise_m=self.height_noise, edge_noise_m=self.edge_noise,
                    yaw_noise_deg=round(math.degrees(self.yaw_noise), 3), step_noise_m=self.step_noise,
                    support_point=f'soft minimum over the object keypoints (tau {SUPPORT_TAU} m)')


def install_goals_follow_table(reset_utils, nominal_table_z):
    """Goals keep their height above the table. The task samples goals in one absolute volume (sized for the
    nominal table); here every goal written by reset_utils._reset_goal_pose is raised by its world's table offset
    from the nominal height, through whatever goal writer is installed at the call (so the SGS configuration
    sampler, which intercepts the writer to record goals, records shifted goals too). Delta goals start from the
    previous goal moved back into nominal coordinates."""
    original = reset_utils._reset_goal_pose
    if getattr(original, 'follows_table', False):
        return

    def follow(env, env_ids, mode):
        viz = env.goal_viz
        writer = viz.write_root_pose_to_sim

        def shifted(pose, env_ids=None):
            pose = pose.clone()
            pose[:, 2] += env._table_z_per_env[env_ids]-nominal_table_z
            return writer(pose, env_ids=env_ids)
        if mode == 'delta':
            viz.data.root_pos_w[env_ids, 2] -= env._table_z_per_env[env_ids]-nominal_table_z
        viz.write_root_pose_to_sim = shifted
        try:
            original(env, env_ids, mode)
        finally:
            viz.write_root_pose_to_sim = writer
    follow.follows_table = True
    reset_utils._reset_goal_pose = follow


def install_own_counter(rms, start, count=1.):
    """Channels ``start:`` of an rl_games RunningMeanStd keep their own sample count (2026-10-28 review): the
    table block is appended to the hand normaliser, whose shared count (~1e9 after M's training) would leave the
    added channels at their initial mean 0 / var 1. The shared update still runs (the existing channels keep their
    statistics); the added channels' mean / var are then replaced by their own count-weighted estimate and the
    input is normalised with those. The own count is not saved with the checkpoint (a resumed run re-estimates the
    added channels from its first batches)."""
    if getattr(rms, 'own_counter_start', None) is not None:
        return
    rms.register_buffer('own_count', torch.tensor(float(count), dtype=torch.float64, device=rms.count.device), persistent=False)
    rms.own_counter_start = int(start)
    base = rms.forward

    def forward(input, denorm=False, mask=None):
        if not rms.training or denorm or mask is not None:
            return base(input, denorm, mask)
        x = input[:, start:].double()
        mean, var, n = rms._update_mean_var_count_from_moments(rms.running_mean[start:], rms.running_var[start:], rms.own_count,
                                                               x.mean(0), x.var(0), x.shape[0])
        base(input, denorm, mask)                       # shared update (all channels), output discarded
        with torch.no_grad():
            rms.running_mean[start:] = mean
            rms.running_var[start:] = var
        rms.own_count = n
        rms.train(False)
        try:
            return base(input, denorm, mask)            # normalise with the updated statistics
        finally:
            rms.train(True)
    rms.forward = forward
