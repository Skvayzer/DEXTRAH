#!/usr/bin/env python3
"""Exercise the actual source BPS+touch task with floating SONIC; no RL yet."""
import argparse
import faulthandler
import json
import os
from pathlib import Path
import subprocess
import time
import traceback


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Use a Slurm GPU allocation')
    from isaaclab.app import AppLauncher
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--num-envs', type=int, default=12)
    p.add_argument('--seconds', type=float, default=30.)
    p.add_argument('--exercise-hand', action='store_true')
    AppLauncher.add_app_launcher_args(p)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    args.headless = True
    app = AppLauncher(args).app
    env = None
    stack_log = (args.output/'stacks.log').open('w')
    faulthandler.enable(file=stack_log)
    faulthandler.dump_traceback_later(45, repeat=True, file=stack_log)
    report = dict(passed=False, training_updates=0, commit=os.environ.get('FULLBODY_SOURCE_COMMIT'))
    try:
        import numpy as np
        import torch
        from isaaclab.utils.io import dump_yaml
        from pxr import Usd, UsdPhysics, PhysxSchema
        import isaaclab.sim as sim
        from dextrah_lab.wholebody.source_config import source_task_config
        from dextrah_lab.wholebody.task_contract import P2P_REVISION
        from dextrah_lab.wholebody.sonic import FrozenSonic
        from dextrah_lab.tasks.g1_revo2_adept.g1_sonic_touch_env import G1SonicTouchEnv
        torch.set_num_threads(4)
        workspace = Path('/data1/users/konstantin.smirnov')
        revision = subprocess.check_output(['git', '-C', str(workspace/'play2perfect'), 'rev-parse', 'HEAD'], text=True).strip()
        if revision != P2P_REVISION:
            raise ValueError('Original task code revision changed')
        sonic = FrozenSonic(workspace/'GRAIL', workspace/'G1-SONIC-models/checkpoint/SONIC/models/sonic_manipulation_base', args.device)
        cfg, contract = source_task_config(args.output, args.num_envs, args.device)
        report['contract'] = contract
        (args.output/'task_contract.json').write_text(json.dumps(contract, indent=2))
        env = G1SonicTouchEnv(cfg, sonic=sonic)
        dump_yaml(str(args.output/'env_resolved.yaml'), cfg)
        robot_root = sim.get_current_stage().GetPrimAtPath('/World/envs/env_0/Robot')
        rigid = [x for x in Usd.PrimRange(robot_root) if x.HasAPI(UsdPhysics.RigidBodyAPI)]
        disabled = [str(x.GetPath()) for x in rigid if PhysxSchema.PhysxRigidBodyAPI(x).GetDisableGravityAttr().Get()]
        if disabled or env.robot.is_fixed_base:
            raise RuntimeError(f'Floating-body gravity contract failed: {disabled}')
        if len(env._body_ids) != 29 or len(env._hand_joint_ids) != 6:
            raise RuntimeError('Wrong body/finger action ownership')
        report.update(robot_bodies=len(rigid), mass_kg=float(env.robot.root_physx_view.get_masses()[0].sum()),
                      is_fixed_base=env.robot.is_fixed_base, object_count=len(env._bps_bank),
                      action_joint_count=35, hand_physics='source_PD_followers', self_collision=False)
        obs, _ = env.reset()
        trace = {k: [] for k in ('time_s', 'height', 'upright', 'body_q', 'body_action', 'reward', 'touch', 'done')}
        done_count, falls, max_speed, min_height = 0, 0, 0., 1e9
        start = time.monotonic()
        for step in range(round(args.seconds/env.step_dt)):
            with torch.no_grad():
                body_obs = obs['policy'][:, 249:]
                decoder_input = torch.cat((body_obs[:, 930:], body_obs[:, :930]), -1)
                action_body = sonic.actor.actor_module.decoders['g1_dyn'].module(decoder_input)
                fingers = action_body.new_full((env.num_envs, 6), -1.)
                if args.exercise_hand:
                    fingers[:] = .8*math_sin(step*env.step_dt*2*np.pi*.3)
                action = torch.cat((env.sonic_to_policy_body(action_body), fingers), -1)
                obs, reward, terminated, truncated, extras = env.step(action)
                speed = float(env.robot.data.joint_vel.abs().max())
                if not torch.isfinite(obs['policy']).all() or not np.isfinite(speed) or speed > 1000:
                    raise RuntimeError(f'Numerically invalid physics: joint_speed={speed}')
                height = env.robot.data.root_pos_w[:, 2]-env.scene.env_origins[:, 2]
                upright = -env.robot.data.projected_gravity_b[:, 2]
                max_speed, min_height = max(max_speed, speed), min(min_height, float(height.min()))
                done_count += int((terminated | truncated).sum())
                falls += int(extras['episode_final']['robot_fall'].sum())
                values = dict(time_s=step*env.step_dt, height=height[:4], upright=upright[:4],
                    body_q=env.robot.data.joint_pos[:4, env._body_ids], body_action=action_body[:4],
                    reward=reward[:4], touch=obs['policy'][:4, 224:249], done=(terminated | truncated)[:4])
                for key, value in values.items():
                    trace[key].append(value.detach().cpu().numpy().copy() if isinstance(value, torch.Tensor) else value)
                if (step+1) % 120 == 0:
                    print('SOURCE_TASK_PROGRESS '+json.dumps(dict(sim_s=(step+1)*env.step_dt,
                        wall_s=time.monotonic()-start, falls=falls, episodes=done_count,
                        minimum_height=min_height, maximum_joint_speed=max_speed,
                        memory_gb=torch.cuda.max_memory_reserved()/2**30)), flush=True)
        np.savez_compressed(args.output/'trajectory.npz', **{k: np.asarray(v) for k, v in trace.items()})
        # Distinguish finite simulation plumbing from standing/task competence.
        report.update(passed=True, standing_no_fall=(falls == 0), task_success_validated=False,
            simulated_seconds=args.seconds, robot_falls=falls, completed_episodes=done_count,
            maximum_joint_speed=max_speed, minimum_height=min_height,
            tactile_acquisitions=env.touch_model.acquisition_count,
            tactile_publications=env.touch_model.publication_count,
            tactile_effective_hz=env.touch_model.acquisition_count/args.seconds,
            wall_seconds=time.monotonic()-start, gpu_reserved_gb=torch.cuda.max_memory_reserved()/2**30)
    except BaseException as error:
        report.update(error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
        raise
    finally:
        (args.output/'validation.json').write_text(json.dumps(report, indent=2))
        print('SOURCE_TASK_VALIDATION '+json.dumps(report), flush=True)
        faulthandler.cancel_dump_traceback_later()
        if env is not None:
            env.close()
        app.close()


def math_sin(value):
    import math
    return math.sin(value)


if __name__ == '__main__':
    main()
