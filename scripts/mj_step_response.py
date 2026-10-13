#!/usr/bin/env python3
"""Stage 2 (mjlab side): PD step responses with the Isaac protocol.

Same groups/offsets as `dump_isaac_reference.py --step-response`: reset, hold
every joint target at its reset value for 0.1 s, then offset one joint and
record 1.5 s of joint position for envs 0-7 (raw PD targets, no policy/SONIC).
"""
import argparse
import json
import os
from pathlib import Path


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Use an allocated Slurm step')
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--num-envs', type=int, default=1200)
    p.add_argument('--no-velocity-limits', action='store_true')
    args = p.parse_args()
    args.output.mkdir(parents=True)
    import numpy as np
    import torch
    from dextrah_lab.mj.factory import make_mj_env
    import importlib.util
    spec = importlib.util.spec_from_file_location('ref', Path(__file__).with_name('dump_isaac_reference.py'))
    ref = importlib.util.module_from_spec(spec); spec.loader.exec_module(ref)
    env = make_mj_env(args.num_envs, 'cuda:0', velocity_limits=not args.no_velocity_limits)
    names = list(env.robot.joint_names)
    results = {}
    with torch.inference_mode():
        for group, (joint, offset) in ref.STEP_GROUPS.items():
            env.reset()
            j = names.index(joint)
            base = env._cur_targets.clone()
            trace = []
            for step in range(round(1.5/env.cfg.sim.dt)):
                target = base.clone()
                if step >= 12:
                    target[:, j] += offset
                env._cur_targets[:] = target
                env._sim_step_counter += 1
                env._apply_action()
                env._write_data_to_sim()
                env.sim.step()
                env._after_physics_step()
                trace.append(env.robot.data.joint_pos[:8, j].cpu().numpy().copy())
            results[group] = dict(joint=joint, offset=offset, dt=env.cfg.sim.dt,
                                  trace=np.stack(trace).tolist(), initial_target=float(base[0, j]))
    (args.output/'step_response.json').write_text(json.dumps(results))
    print('MJ_STEP_RESPONSE_DONE', flush=True)
    os._exit(0)


if __name__ == '__main__':
    main()
