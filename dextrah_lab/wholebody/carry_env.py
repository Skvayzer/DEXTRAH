"""Carry-expert environment: hold a grasped object while frozen SONIC walks.

Phase 1 of the MoE plan. Each episode restores a grasp captured from the
reposing expert (run 560) in the same environment, i.e. with the same physical
object. The source table is teleported below the floor once the grasp is
restored, so walking commands cannot collide with it. After a standing settle
window, the environment plays precomputed released-planner walking clips.

Unchanged from the reposing task: robot/hand/touch physics, BPS-128 and touch
observations, finger action processing, object wrench DR, action delay, the
70-D latent+finger action space and frozen SONIC. Changed: no object-pose goal
(goal error is zeroed in observations), palm/object observations use a moving
body frame, carry reward, drop termination, walking references.
"""
import json
import torch
from isaaclab.utils.math import matrix_from_quat, quat_conjugate, quat_mul, yaw_quat
from isaacsimenvs.tasks.play.utils import obs_utils
from dextrah_lab.tasks.g1_revo2_adept.g1_sonic_touch_env import G1SonicTouchEnv
from .carry import (CarryRewardCfg, MovingFrame, WalkingClips, carry_reward, field_slice,
    transform_clean_observation, yaw_of)
from .contract import BODY_JOINTS

SNAPSHOT_LAYOUT = (('root', 13), ('joint_pos', 51), ('joint_vel', 51), ('targets', 51),
                   ('object', 13), ('palm_object_distance', 1))
SNAPSHOT_DIM = sum(n for _, n in SNAPSHOT_LAYOUT)
TABLE_PARKING_Z = -2.


def snapshot_slices():
    out, start = {}, 0
    for name, size in SNAPSHOT_LAYOUT:
        out[name] = slice(start, start+size)
        start += size
    return out


def palm_center_w(env):
    palm = env.robot.data.body_state_w[:, env._palm_body_id, :]
    return obs_utils._apply_local_offset(palm[:, :3], palm[:, 3:7], env._palm_center_offset, (env.num_envs,)), palm


def install_moving_frame_observations():
    """Route source clean observations through an env-provided moving frame.

    Environments without ``carry_moving_frame`` (all existing tasks) are unchanged.
    """
    original = obs_utils._build_clean_observation_dict
    if getattr(original, '_carry_aware', False):
        return

    def build(env, **kwargs):
        obs, context = original(env, **kwargs)
        hook = getattr(env, 'carry_moving_frame', None)
        frame = hook() if hook is not None else None
        return (obs, context) if frame is None else transform_clean_observation(obs, context, frame)

    build._carry_aware = True
    obs_utils._build_clean_observation_dict = build


class G1CarryEnv(G1SonicTouchEnv):
    def __init__(self, cfg, *, sonic, clips_path, grasp_bank_path, settle_s=1., idle_clip_probability=.1,
                 reward_cfg=None, **kwargs):
        self._carry_ready = False
        install_moving_frame_observations()
        super().__init__(cfg, sonic=sonic, **kwargs)
        if not self._frozen_latent:
            raise ValueError('Carry training uses the frozen pretrained SONIC latent interface')
        self.reward_cfg = reward_cfg or CarryRewardCfg()
        self.settle_s, self.idle_clip_probability = settle_s, idle_clip_probability
        self.clips = WalkingClips.load(clips_path, self.device)
        self._idle_clip = self.clips.index('idle')
        bank = torch.load(grasp_bank_path, map_location='cpu', weights_only=False)
        if bank['num_envs'] != self.num_envs:
            raise ValueError('Grasp bank was captured with a different environment count')
        if not torch.equal(bank['object_asset_index'], self._object_asset_index_per_env.cpu()):
            raise ValueError('Grasp bank object assignment differs from this scene')
        if bank['snapshot_dim'] != SNAPSHOT_DIM:
            raise ValueError('Grasp bank snapshot layout changed')
        self._bank = bank['snapshots'].to(self.device)          # (N, K, D)
        self._bank_count = bank['count'].to(self.device)        # (N,)
        self._bank_meta = {k: bank[k] for k in ('checkpoint_sha256', 'seconds', 'capture_rule')}
        self.carry_valid = self._bank_count > 0
        self._slices = snapshot_slices()
        n, d = self.num_envs, self.device
        self._standing_reference_q = self._reference_q.clone()
        self._arm_ids = torch.tensor([i for i, name in enumerate(BODY_JOINTS)
                                      if any(p in name for p in ('shoulder', 'elbow', 'wrist'))], device=d)
        self._home_pos = self.robot.data.default_root_state[:, :3].clone()
        self._home_yaw = yaw_of(self.robot.data.default_root_state[:, 3:7])
        self._clip = torch.full((n,), self._idle_clip, dtype=torch.long, device=d)
        self._clip_t0 = torch.zeros(n, device=d)
        self._clip_heading = torch.zeros(n, device=d)
        self._clip_started = torch.zeros(n, dtype=torch.bool, device=d)
        self._blend_q = self._standing_reference_q.clone()
        self._ref_quat = torch.zeros(n, 10, 4, device=d); self._ref_quat[..., 0] = 1
        self._planned_velocity = torch.zeros(n, 3, device=d)
        self._home_distance = torch.zeros(n, device=d)
        self._drop_count = torch.zeros(n, dtype=torch.long, device=d)
        self._carry_dropped = torch.zeros(n, dtype=torch.bool, device=d)
        self._reset_root_xy = torch.zeros(n, 2, device=d)
        self._prev_actions = torch.zeros(n, 70, device=d)
        self._actions_now = torch.zeros(n, 70, device=d)
        sizes = obs_utils._obs_field_sizes(13, self._num_fingertips)
        self._goal_policy = field_slice(cfg.obs.obs_list, sizes, 'keypoints_rel_goal')
        self._goal_critic = field_slice(cfg.obs.state_list, sizes, 'keypoints_rel_goal')
        self._carry_ready = True
        print('CARRY_ENV_READY '+json.dumps(dict(envs=n, bank_coverage=float(self.carry_valid.float().mean()),
            snapshots=int(self._bank_count.sum()), clips=self.clips.names, settle_s=settle_s)), flush=True)

    # ---------------------------------------------------------------- frames
    def carry_moving_frame(self):
        if not getattr(self, '_carry_ready', False):
            return None
        data = self.robot.data
        return MovingFrame(root_pos=data.root_pos_w-self.scene.env_origins, root_yaw=yaw_of(data.root_quat_w),
                           home_pos=self._home_pos, home_yaw=self._home_yaw)

    def _get_observations(self):
        obs = super()._get_observations()
        if self._carry_ready:
            obs['policy'][:, self._goal_policy] = 0
            obs['critic'][:, self._goal_critic] = 0
        return obs

    # ------------------------------------------------------- walking reference
    def _elapsed(self):
        return self.episode_length_buf.float()*self.step_dt

    def _start_clips(self, ids, elapsed):
        if ids.numel() == 0:
            return
        walking = [i for i in range(len(self.clips.names)) if i != self._idle_clip]
        choice = torch.tensor(walking, device=self.device)[torch.randint(len(walking), (ids.numel(),), device=self.device)]
        idle = torch.rand(ids.numel(), device=self.device) < self.idle_clip_probability
        self._blend_q[ids] = self._reference_q[ids]
        self._clip[ids] = torch.where(idle, torch.full_like(choice, self._idle_clip), choice)
        self._clip_t0[ids] = elapsed[ids]
        self._clip_heading[ids] = yaw_of(self.robot.data.root_quat_w[ids])
        self._clip_started[ids] = True

    def _update_walking_reference(self):
        elapsed = self._elapsed()
        clip_time = elapsed-self._clip_t0
        settle_done = ~self._clip_started & (elapsed >= self.settle_s)
        finished = self._clip_started & (clip_time > self.clips.duration(self._clip))
        self._start_clips(torch.nonzero(settle_done | finished).flatten(), elapsed)
        clip_time = elapsed-self._clip_t0
        q, qd, root, velocity, _ = self.clips.reference(self._clip, clip_time, self._clip_heading)
        w = (clip_time/.2).clamp(0, 1)[:, None, None]
        q = self._blend_q+(q-self._blend_q)*w
        qd = qd*w
        q[:, :, self._arm_ids] = self._standing_reference_q[:, :, self._arm_ids]
        qd[:, :, self._arm_ids] = 0
        self._reference_q, self._reference_qd = q, qd
        self._ref_quat, self._planned_velocity = root, velocity

    @torch.no_grad()
    def _wholebody_observation(self):
        if not self._carry_ready:
            return super()._wholebody_observation()
        if self._body_extra_step != self._sim_step_counter:
            self._update_walking_reference()
            proprio = self._body_history.push(*self._body_terms())
        else:
            self._body_history.initialize_fresh(*self._body_terms())
            proprio = self._body_history.value()
        robot_yaw = yaw_quat(self.robot.data.root_quat_w)
        delta = quat_mul(quat_conjugate(robot_yaw)[:, None].expand(-1, 10, -1), self._ref_quat)
        self._reference_ori6 = matrix_from_quat(delta)[..., :2].reshape(self.num_envs, 10, 6)
        tokens = self._sonic_reference.reference_tokens(self._reference_q, self._reference_qd, self._reference_ori6)
        self._body_extra = torch.cat((proprio, tokens), -1)
        self._body_extra_step = self._sim_step_counter
        return self._body_extra

    # ---------------------------------------------------------------- actions
    def _pre_physics_step(self, actions):
        if self._carry_ready:
            self._prev_actions.copy_(self._actions_now)
            self._actions_now.copy_(actions)
        super()._pre_physics_step(actions)

    # ------------------------------------------------------- dones / rewards
    def _carry_kinematics(self):
        palm_c, palm = palm_center_w(self)
        obj = self.object.data
        offset = obj.root_pos_w-palm_c
        rel_linear = obj.root_lin_vel_w-(palm[:, 7:10]+torch.cross(palm[:, 10:13], offset, dim=-1))
        rel_angular = obj.root_ang_vel_w-palm[:, 10:13]
        return offset.norm(dim=-1), rel_linear, rel_angular

    def _get_dones(self):
        terminated, truncated = super()._get_dones()
        if not self._carry_ready:
            return terminated, truncated
        distance, _, _ = self._carry_kinematics()
        far = distance > self._home_distance+.12
        self._drop_count = torch.where(far, self._drop_count+1, torch.zeros_like(self._drop_count))
        low = (self.object.data.root_pos_w[:, 2]-self.scene.env_origins[:, 2]) < .25
        self._carry_dropped = (self._drop_count >= 6) | low
        # Environments without any captured grasp end immediately and earn zero.
        terminated = terminated | self._carry_dropped | ~self.carry_valid
        return terminated, truncated

    def _get_rewards(self):
        super()._get_rewards()  # keep source bookkeeping/final observations; discard reposing reward
        if not self._carry_ready:
            return torch.zeros(self.num_envs, device=self.device)
        final = self.extras.get('final_observation')
        if isinstance(final, dict) and 'critic' in final:
            final['critic'][:, self._goal_critic] = 0
        distance, rel_linear, rel_angular = self._carry_kinematics()
        data = self.robot.data
        yaw = yaw_of(data.root_quat_w)
        c, s = yaw.cos(), yaw.sin()
        v = data.root_lin_vel_w
        body_velocity = torch.stack((c*v[:, 0]+s*v[:, 1], -s*v[:, 0]+c*v[:, 1], data.root_ang_vel_w[:, 2]), -1)
        delta = self._actions_now-self._prev_actions
        reward, terms = carry_reward(self.reward_cfg, distance=distance, home_distance=self._home_distance,
            normal_n=self.touch_raw[..., 0], rel_linear=rel_linear, rel_angular=rel_angular,
            body_velocity=body_velocity, planned_velocity=self._planned_velocity,
            latent_delta=delta[:, :64], finger_delta=delta[:, 64:], fell=self._body_fallen, dropped=self._carry_dropped)
        reward = torch.where(self.carry_valid, reward, torch.zeros_like(reward))
        cumulative = self.extras.setdefault('episode_cumulative', {})
        for name, value in terms.items():
            cumulative['carry_'+name] = value
        final_stats = self.extras.setdefault('episode_final', {})
        elapsed = self._elapsed()
        final_stats['carry_drop'] = self._carry_dropped.float()
        final_stats['carry_settle_drop'] = (self._carry_dropped & (elapsed < self.settle_s)).float()
        final_stats['carry_seconds'] = elapsed
        root_xy = data.root_pos_w[:, :2]-self.scene.env_origins[:, :2]
        final_stats['carry_walked_m'] = (root_xy-self._reset_root_xy).norm(dim=-1)
        self.extras['carry/planned_speed'] = self._planned_velocity[:, :2].norm(dim=-1).mean()
        self.extras['carry/body_speed'] = body_velocity[:, :2].norm(dim=-1).mean()
        self.extras['carry/palm_object_distance_excess'] = (distance-self._home_distance).mean()
        return reward

    # ------------------------------------------------------------------ reset
    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if not self._carry_ready:
            return
        ids = self.robot._ALL_INDICES if env_ids is None else env_ids
        ids = torch.as_tensor(ids, device=self.device, dtype=torch.long)
        valid = ids[self.carry_valid[ids]]
        if valid.numel():
            k = (torch.rand(valid.numel(), device=self.device)*self._bank_count[valid]).long()
            snap = self._bank[valid, k]
            sl, origin = self._slices, self.scene.env_origins[valid]
            root = snap[:, sl['root']].clone(); root[:, :3] += origin
            self.robot.write_root_state_to_sim(root, env_ids=valid)
            q, qd = snap[:, sl['joint_pos']], snap[:, sl['joint_vel']]
            self.robot.write_joint_state_to_sim(q, qd, env_ids=valid)
            targets = snap[:, sl['targets']]
            self.robot.set_joint_position_target(targets, env_ids=valid)
            self._cur_targets[valid] = targets
            self._prev_targets[valid] = targets
            self._last_body_action[valid] = (targets[:, self._body_ids]-self._body_nominal)/self._body_scales
            obj = snap[:, sl['object']].clone(); obj[:, :3] += origin
            self.object.write_root_state_to_sim(obj, env_ids=valid)
            self._home_distance[valid] = snap[:, sl['palm_object_distance']].squeeze(-1)
            self._reset_root_xy[valid] = snap[:, sl['root']][:, :2]
            self._clip_heading[valid] = yaw_of(snap[:, sl['root']][:, 3:7])
        pose = self.table.data.root_state_w[ids, :7].clone()
        pose[:, 2] = self.scene.env_origins[ids, 2]+TABLE_PARKING_Z
        self.table.write_root_pose_to_sim(pose, env_ids=ids)
        self._clip[ids] = self._idle_clip
        self._clip_t0[ids] = 0
        self._clip_started[ids] = False
        self._blend_q[ids] = self._standing_reference_q[ids]
        self._reference_q[ids] = self._standing_reference_q[ids]
        self._drop_count[ids] = 0
        self._carry_dropped[ids] = False
        self._actions_now[ids] = 0
        self._prev_actions[ids] = 0

    def carry_contract(self):
        return dict(type='g1_frozen_sonic_carry_expert_v1', grasp_bank=self._bank_meta,
            bank_coverage=float(self.carry_valid.float().mean()), clips=self.clips.names,
            settle_s=self.settle_s, idle_clip_probability=self.idle_clip_probability,
            table='teleported 2 m below floor after grasp restore; kinematic, no gravity',
            goal='none; keypoints_rel_goal zeroed in actor and critic observations',
            observation_frame='palm/object terms in yaw-only moving frame, identity at standing reset pose',
            arm_reference='trained standing arm reference held; legs/waist from walking clips',
            reward=self.reward_cfg.__dict__, drop='palm-object distance > grasp + 0.12 m for 6 steps, or object below 0.25 m')
