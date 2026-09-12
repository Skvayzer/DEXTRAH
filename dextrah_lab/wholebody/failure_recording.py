"""Opt-in host ring buffer for a failed training reproduction, not main logging.

Store every environment until the failing one is known. Read actual PhysX body
poses: the renderer must not invent a different finger coupling or interpolate
through a reset. No physics, policy action, random draw or reward is modified.
"""
import json
from pathlib import Path
import numpy as np
import torch


class FailureRecorder:
    def __init__(self, env, output, seconds=10., epoch=lambda: 0):
        self.env, self.output, self.epoch = env, Path(output), epoch
        self.capacity = int(np.ceil(seconds/env.cfg.sim.dt))+2
        self.count = 0
        self.data = None
        self.steps = np.zeros(self.capacity, dtype=np.int64)
        self.epochs = np.zeros(self.capacity, dtype=np.int64)
        self.reset_counts = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.saved = False
        apply_action, get_dones, reset = env._apply_action, env._get_dones, env._reset_idx

        def recorded_apply():
            apply_action()
            self.record(env._sim_step_counter-1)

        def recorded_dones():
            try:
                return get_dones()
            except BaseException:
                self.record(env._sim_step_counter)  # include the rejected post-step state
                raise

        def recorded_reset(ids):
            reset(ids)
            self.reset_counts[slice(None) if ids is None else ids] += 1

        env._apply_action, env._get_dones, env._reset_idx = recorded_apply, recorded_dones, recorded_reset

    @torch.no_grad()
    def record(self, step):
        e = self.env
        origins = e.scene.env_origins
        def pose(asset):
            return torch.cat((asset.data.root_pos_w-origins, asset.data.root_quat_w), -1)
        parts = dict(joint_pos=e.robot.data.joint_pos, joint_vel=e.robot.data.joint_vel,
            commanded_joint_targets=e._cur_targets, body_pos=e.robot.data.body_pos_w-origins[:, None],
            body_quat=e.robot.data.body_quat_w, robot=pose(e.robot), object=pose(e.object),
            table=pose(e.table), goal=pose(e.goal_viz), touch=e.touch_raw,
            resets=self.reset_counts[:, None], goal_hits=e._successes[:, None],
            episode_steps=e.episode_length_buf[:, None])
        packed = torch.cat([x.reshape(e.num_envs, -1).float() for x in parts.values()], -1).cpu().numpy()
        if self.data is None:
            self.shapes = {k: tuple(x.shape[1:]) for k, x in parts.items()}
            self.data = np.empty((self.capacity, *packed.shape), dtype=np.float32)
            print(f'FAILURE_CAPTURE_READY host_ring_gib={self.data.nbytes/2**30:.3f} '
                  f'environments={e.num_envs} physics_hz={1/e.cfg.sim.dt}', flush=True)
        slot = self.count % self.capacity
        self.data[slot] = packed
        self.steps[slot], self.epochs[slot] = step, self.epoch()
        self.count += 1

    def save(self, error=None):
        if self.saved or self.data is None:
            return None
        e = self.env
        velocity = e.robot.data.joint_vel.detach().abs().cpu().numpy()
        env_id, joint_id = np.unravel_index(np.nan_to_num(velocity, nan=np.inf).argmax(), velocity.shape)
        env_id, joint_id = int(env_id), int(joint_id)
        order = np.arange(max(0, self.count-self.capacity), self.count) % self.capacity
        # Export just the affected environment; release the multi-GB ring after
        # selecting it. Exported clips are small and independent of live physics.
        selected = self.data[order, env_id].copy()
        trace, column = {}, 0
        for key, shape in self.shapes.items():
            size = int(np.prod(shape))
            trace[key] = selected[:, column:column+size].reshape(len(order), *shape)
            column += size
        trace['physics_step'] = self.steps[order].copy()
        trace['time_s'] = trace['physics_step']*e.cfg.sim.dt
        trace['epoch'] = self.epochs[order].copy()
        self.output.mkdir(parents=True, exist_ok=False)
        np.savez_compressed(self.output/'trajectory.npz', **trace)
        from isaacsimenvs.tasks.play.pose_viewer import object_urdf_for_env, table_urdf_for_env
        from dextrah_lab.object_shape.evaluation import family_name
        import yourdfpy
        for name, resolver in (('object', object_urdf_for_env), ('table', table_urdf_for_env)):
            text, path = resolver(e, env_id)
            yourdfpy.URDF.load(str(path)).scene.export(self.output/f'{name}.glb')
            (self.output/f'{name}.urdf').write_text(text)
        finite = np.isfinite(velocity[env_id, joint_id])
        meta = dict(recording_type='new_reproduction_of_failed_training_not_original_job_506',
            original_job=506, frames=len(order), env_id=env_id, num_envs=e.num_envs,
            family=family_name(e._object_urdf_paths[int(e._object_asset_index_per_env[env_id])]),
            asset_index=int(e._object_asset_index_per_env[env_id]), robot_urdf=str(e.g1_urdf),
            joint_names=e.robot.joint_names, body_names=e.robot.body_names,
            physics_hz=1/e.cfg.sim.dt, policy_hz=1/e.step_dt, tactile_hz=e.cfg.touch.sensor_hz,
            first_time_s=float(trace['time_s'][0]), last_time_s=float(trace['time_s'][-1]),
            failed_joint=e.robot.joint_names[joint_id], failed_joint_index=joint_id,
            final_joint_speed_rad_s=float(velocity[env_id, joint_id]) if finite else None,
            failure_reproduced=bool(not finite or velocity[env_id, joint_id]>1000),
            stop_error=None if error is None else str(error), epoch=int(trace['epoch'][-1]),
            body_poses='measured PhysX link poses; no generated motion or smoothing',
            self_collision=False, native_finger_coupling=False, fabrics=False, pca=False,
            reset_semantics='original task resets plus robot-fall termination; reset jumps are marked',
            selection='environment and joint with the largest absolute velocity at the safety stop')
        (self.output/'metadata.json').write_text(json.dumps(meta, indent=2))
        self.saved = True
        self.data = None
        print('FAILURE_CAPTURE_SAVED '+json.dumps(meta), flush=True)
        return meta
