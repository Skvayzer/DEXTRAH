"""Opt-in placement goals for the manipulation expert (train --place-goals P; Phase 2 hand-back).

M's goals are poses in the air (task target volume z 0.90-1.08 m, the table top at 0.80 m): frozen M
holds objects within ~1 cm of in-volume goals over table B after the carry hand-back, but at a goal
resting on the table it reached only 12/66 set-downs (scripts/mj_table_to_table.py, job 1570).
With probability P every newly sampled goal is replaced by a stable resting pose of the world's own
object on the table top: a pose from the resting-pose library (scripts/mj_rest_poses.py: settled,
untouched, table-supported poses), rotated by a uniform yaw, at a uniform xy in the placement area
(inside M's goal xy range, kept away from the table edges), height = table top + the library height.
The task reward, success test and goal sequence are the task's own: keypoint progress towards the
goal, near-goal bonus, success after the task's success steps, then the next goal (the task's delta
sampling from the placed pose goes back up into the volume: set down, then lift again).
Release stays scripted (support-gated, in the Phase 2 evaluation).
"""
import math

import numpy as np
import torch

from .compat import install, p2p
install()                                   # isaaclab.utils.math without Isaac Sim (as mj/env.py)
from isaaclab.utils.math import quat_mul  # noqa: E402

reset_utils = p2p('reset_utils')
TABLE_TOP = .15            # table box half height (scene.TABLE_SIZE[2] / 2)


class PlacementGoals:
    def __init__(self, env, library, probability, xy_mins=(-.12, .0), xy_maxs=(.12, .12), seed=0):
        if not 0. < probability <= 1.:
            raise ValueError('placement goal probability must be in (0, 1]')
        data = np.load(library)
        d = env.device
        asset = torch.as_tensor(data['asset'], dtype=torch.long, device=d)
        order = torch.argsort(asset, stable=True)
        self.height = torch.as_tensor(data['height'], dtype=torch.float32, device=d)[order]
        self.quat = torch.as_tensor(data['quat'], dtype=torch.float32, device=d)[order]
        n_assets = int(env._object_asset_index_per_env.max())+1
        self.count = torch.bincount(asset, minlength=n_assets)[:n_assets]
        self.start = torch.cumsum(self.count, 0)-self.count
        self.env, self.p = env, float(probability)
        self.xy_mins = torch.tensor(xy_mins, device=d)
        self.xy_maxs = torch.tensor(xy_maxs, device=d)
        self.g = torch.Generator(device=d).manual_seed(int(seed)+11)
        self.active = torch.zeros(env.num_envs, dtype=torch.bool, device=d)     # current goal is a placement
        self.sampled = 0
        self.successes = 0
        self.assets_covered = int((self.count > 0).sum())

    def apply(self, env_ids):
        """After the task wrote new goals for env_ids: replace a share of them by placement goals."""
        env, d = self.env, self.env.device
        ids = env_ids.reshape(-1)
        self.active[ids] = False
        if ids.numel() == 0:
            return
        asset = env._object_asset_index_per_env[ids]
        pick = (torch.rand(len(ids), device=d, generator=self.g) < self.p) & (self.count[asset] > 0)
        ids, asset = ids[pick], asset[pick]
        n = len(ids)
        if n == 0:
            return
        k = self.start[asset]+(torch.rand(n, device=d, generator=self.g)*self.count[asset]).long()
        k = torch.minimum(k, self.start[asset]+self.count[asset]-1)
        yaw = torch.rand(n, device=d, generator=self.g)*2*math.pi
        yaw_q = torch.stack(((yaw/2).cos(), torch.zeros_like(yaw), torch.zeros_like(yaw), (yaw/2).sin()), -1)
        quat = quat_mul(yaw_q, self.quat[k])
        xy = self.xy_mins+torch.rand(n, 2, device=d, generator=self.g)*(self.xy_maxs-self.xy_mins)
        z = env._table_z_per_env[ids]+TABLE_TOP+self.height[k]
        pose = torch.cat((torch.cat((xy, z[:, None]), -1)+env.scene.env_origins[ids], quat), -1)
        env.goal_viz.write_root_pose_to_sim(pose, env_ids=ids)
        self.active[ids] = True
        self.sampled += n

    def record(self, is_success):
        """Count successes at placement goals (call with the step's success mask before resampling)."""
        hit = is_success & self.active
        self.successes += int(hit.sum())
        x = self.env.extras
        x['place/goal_share'] = self.active.float().mean()
        x['place/successes_per_step'] = hit.float().mean()
        x['place/other_successes_per_step'] = (is_success & ~self.active).float().mean()
        x['place/successes_total'] = torch.tensor(float(self.successes), device=self.env.device)
        x['place/sampled_total'] = torch.tensor(float(self.sampled), device=self.env.device)

    def contract(self, library):
        return dict(probability=self.p, library=str(library), assets_with_poses=self.assets_covered,
                    poses=int(self.count.sum()), xy_mins=self.xy_mins.tolist(), xy_maxs=self.xy_maxs.tolist(),
                    yaw='uniform', definition=__doc__.strip())


def install_placement_goals(env, placement):
    """Route the task's goal sampling (reset_utils._reset_goal_pose, every new goal) through
    PlacementGoals for this env; count placement successes from the terminations."""
    original = reset_utils._reset_goal_pose
    if getattr(original, '_placement_aware', False):
        raise RuntimeError('Placement goals already installed')

    def reset_goal_pose(e, env_ids, mode):
        original(e, env_ids, mode)
        if e is env:
            placement.apply(torch.as_tensor(env_ids, device=env.device, dtype=torch.long))

    reset_goal_pose._placement_aware = True
    reset_utils._reset_goal_pose = reset_goal_pose
    termination_utils = p2p('termination_utils')
    compute = termination_utils.compute_terminations

    def compute_terminations(e):
        if e is env:
            placement.record(e._is_success)
        return compute(e)

    termination_utils.compute_terminations = compute_terminations
    env._placement = placement
