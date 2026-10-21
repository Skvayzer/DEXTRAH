#!/usr/bin/env python3
"""Transplant a frozen-SONIC whole-body checkpoint onto the fixed-body (welded base,
no SONIC) touch task, as a resumable fixed-body checkpoint.

The whole-body actor holds the fixed-body actor unchanged in structure as
``a2c_network.source`` (recurrent task core, 13-action head, SAPG embeddings,
exploration sigma, value head) and adds a SONIC latent adapter. The adapter and its
latent exploration are dropped; everything else is copied. The 7 arm outputs come
from the source's arm rows, which the whole-body run never trained (they are the
teacher's). The critic's first layer drops its 994 SONIC body columns; the task
columns and the SAPG embedding columns are copied.

Optimizers start fresh (empty Adam state). Epoch/frame counters, run provenance and
the SGS state (configurations + outcome windows) come from the fixed-body template
run (the whole-body SGS state has floating-base held states, not usable here).
Writes OUTPUT/nn/transplant_<frame>.pth (+ .sgs.pt), OUTPUT/params, the template's
provenance files and OUTPUT/transplant_report.json. CPU only.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import torch


def sha256(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def load(path):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    return payload[0] if 0 in payload else payload


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--wholebody', type=Path, required=True, help='whole-body complete_*.pth (frozen SONIC)')
    p.add_argument('--template', type=Path, required=True, help='fixed-body complete_*.pth (+ .sgs.pt) of the same task')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    from dextrah_lab.wholebody.task_contract import BODY_EXTRA_DIM, TASK_CRITIC_DIM
    t, w = load(args.template), load(args.wholebody)
    # Actor: source subnetwork -> fixed-body network; task normaliser -> normaliser.
    actor, dropped = {}, []
    for key, value in w['model'].items():
        if key.startswith('a2c_network.latent_adapter') or key == 'a2c_network.logstd':
            dropped.append(key)
            continue
        actor[key.replace('a2c_network.source.', 'a2c_network.').replace('running_mean_std.task.', 'running_mean_std.')] = value
    # Critic: drop the SONIC body normaliser and body input columns.
    critic = {}
    for key, value in w['assymetric_vf_nets'].items():
        if '.running_mean_std.body.' in key:
            dropped.append('critic.'+key)
            continue
        key = key.replace('running_mean_std.task.', 'running_mean_std.')
        if key == 'model.a2c_network.actor_mlp.0.weight':
            if value.shape[1] != TASK_CRITIC_DIM+BODY_EXTRA_DIM+32:
                raise ValueError(f'Unexpected whole-body critic input width {value.shape[1]}')
            value = torch.cat((value[:, :TASK_CRITIC_DIM], value[:, -32:]), 1)
        critic[key] = value
    for name, new, ref in (('actor', actor, t['model']), ('critic', critic, t['assymetric_vf_nets'])):
        if set(new) != set(ref):
            raise ValueError(f'{name} keys differ: {sorted(set(new) ^ set(ref))[:6]}')
        bad = [k for k in ref if new[k].shape != ref[k].shape]
        if bad:
            raise ValueError(f'{name} shapes differ: {bad}')
    arm_rows_unchanged = bool(torch.equal(actor['a2c_network.mu.weight'][:7], t['model']['a2c_network.mu.weight'][:7]))
    state = dict(t)
    for key in ('obs', 'rnn_states', 'dones', 'current_rewards', 'current_shaped_rewards', 'current_lengths', 'env_state'):
        state.pop(key, None)
    state['model'], state['assymetric_vf_nets'] = actor, critic
    state['optimizer'] = dict(state=dict(), param_groups=t['optimizer']['param_groups'])
    state['continuation_critic_optimizer'] = dict(state=dict(), param_groups=t['continuation_critic_optimizer']['param_groups'])
    state['last_mean_rewards'] = -1e9
    template_run, wb_run = args.template.resolve().parent.parent, args.wholebody.resolve().parent.parent
    (args.output/'nn').mkdir(parents=True)
    name = f"transplant_{int(t['frame'])}"
    out = args.output/'nn'/f'{name}.pth'
    torch.save(state, out)
    shutil.copy2(args.template.with_suffix('.sgs.pt'), out.with_suffix('.sgs.pt'))
    shutil.copytree(template_run/'params', args.output/'params')
    for f in ('warmstart_validation.json', 'calibration.json'):
        shutil.copy2(template_run/f, args.output/f)
    report = dict(checkpoint=str(out), checkpoint_sha256=sha256(out),
                  wholebody=str(args.wholebody), wholebody_sha256=sha256(args.wholebody),
                  wholebody_epoch=int(w['epoch']), wholebody_frame=int(w['frame']),
                  template=str(args.template), template_sha256=sha256(args.template),
                  epoch=int(t['epoch']), frame=int(t['frame']), dropped=dropped,
                  arm_rows='source arm rows (never trained in the whole-body run) = template arm rows: '+str(arm_rows_unchanged),
                  critic_first_layer=f'task columns [0:{TASK_CRITIC_DIM}] + SAPG embedding columns kept, {BODY_EXTRA_DIM} body columns dropped',
                  optimizers='fresh (empty Adam state)', sgs_state='template run (fixed-body configurations and windows)',
                  counters='template run epoch/frame', wholebody_run=str(wb_run), template_run=str(template_run))
    (args.output/'transplant_report.json').write_text(json.dumps(report, indent=2))
    print('TRANSPLANT '+json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
