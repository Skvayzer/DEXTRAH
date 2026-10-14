#!/usr/bin/env python3
"""Self-test of the mjlab NaN capture + replay (diagnostic tooling only).

Passive robot (zero latent residual, open fingers) at the run-1157 buffer
sizes. After a few policy steps a NaN is injected into one world's actuator
command right before a physics step; the capture must fire on that step,
dump the world, and `mj_nan_replay.py` must (a) replay the earlier buffered
steps with small error and (b) place the first non-finite stage at
actuation.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Use an allocated Slurm step')
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--num-envs', type=int, default=64)
    p.add_argument('--inject-world', type=int, default=5)
    p.add_argument('--inject-policy-step', type=int, default=12)
    args = p.parse_args()
    import torch
    from dextrah_lab.mj.factory import make_mj_env
    env = make_mj_env(args.num_envs, 'cuda:0', nconmax=64, njmax=300, spawn_on_table=True)
    cap = env.enable_nan_capture(args.output/'nan_capture')
    cap.stats_interval = 5
    env.reset()
    action = torch.zeros(env.num_envs, 70, device='cuda:0')
    action[:, 64:] = -1.
    original = env._write_data_to_sim
    state = dict(policy_step=0)

    def write(advance=True):
        original(advance)
        if advance and state['policy_step'] == args.inject_policy_step:
            env.sim.data.ctrl[args.inject_world, 0] = float('nan')
    env._write_data_to_sim = write
    raised = None
    with torch.inference_mode():
        for i in range(args.inject_policy_step+3):
            state['policy_step'] = i
            try:
                env.step(action)
            except RuntimeError as e:
                raised = str(e)
                break
    result = dict(raised=raised, dumped=str(cap.dumped), stats=cap.stats(), info=cap.info)
    print('SELFTEST capture', json.dumps(result, indent=1), flush=True)
    assert raised and cap.dumped is not None, 'capture did not fire'
    summary = json.loads((cap.dumped/'summary.json').read_text())
    w = summary['per_world'][str(args.inject_world)]
    assert summary['worlds'] == [args.inject_world], summary['worlds']
    assert w['nonfinite_inputs_at_failing_step'] == ['ctrl'], w['nonfinite_inputs_at_failing_step']
    replay = subprocess.run([sys.executable, 'scripts/mj_nan_replay.py', str(cap.dumped)],
                            capture_output=True, text=True)
    print(replay.stdout[-6000:], replay.stderr[-6000:], flush=True)
    assert replay.returncode == 0
    out = json.loads((cap.dumped/f'replay_w{args.inject_world}.json').read_text())
    print('SELFTEST first stage', out['first_nonfinite_stage'], 'fidelity',
          [round(f['qpos_max_abs_diff'], 9) for f in out['fidelity']], flush=True)
    assert out['first_nonfinite_stage'] == 'actuation', out['first_nonfinite_stage']
    print('SELFTEST PASSED', flush=True)
    sys.stdout.flush()
    os._exit(0)


if __name__ == '__main__':
    main()
