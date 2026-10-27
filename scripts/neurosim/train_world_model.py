#!/usr/bin/env python3
"""Train the hand-object world model on logged carry transitions (dextrah_lab/neurosim), then report the
go / no-go metrics (dextrah_lab/neurosim/evaluate.py) on objects never seen in training.

SuperMimic recipe (arXiv 2610.28480, App. A): k = 8 step autoregressive windows, Adam 2e-4, global-norm
clip 1.0, cosine decay; here a smaller MLP (the state is ~40-D, not a whole body).
"""
import argparse
import json
import math
import os
from pathlib import Path
import time
import traceback


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data', type=Path, nargs='+', required=True, help='collector output directories (combined along worlds)')
    p.add_argument('--members', type=int, default=0, help='>0: probabilistic ensemble with this many members (ensemble.py)')
    p.add_argument('--nll-weight', type=float, default=.1, help='ensemble: weight of the one-step Gaussian NLL')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--k', type=int, default=16, help='autoregressive window (v1: 8)')
    p.add_argument('--exo-mode', choices=('post', 'pre'), default='pre', help="'pre': only inputs known before the step (v1: post)")
    p.add_argument('--lowpass-tau', type=float, default=.3, help='low-pass relative twist state (s); 0 = off (v1)')
    p.add_argument('--labels', choices=('lost', 'drop'), default='drop', help="'drop': failed grasp checks censored (v1: lost)")
    p.add_argument('--support-in-secure', action='store_true', help='keep secure-phase support labels (v1 behaviour)')
    p.add_argument('--oversample', type=float, default=.1, help='share of each batch drawn from pre-drop windows, importance-weighted')
    p.add_argument('--iters', type=int, default=20000)
    p.add_argument('--batch', type=int, default=4096)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--hidden', default='1024,1024,1024')
    p.add_argument('--lost-pos-weight', type=float, default=1.)
    p.add_argument('--test-fraction', type=float, default=.1)
    p.add_argument('--eval-every', type=int, default=2000)
    p.add_argument('--max-chunks', type=int, default=None)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--wandb', choices=['online', 'disabled'], default='online')
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    report = dict(completed=False, args={k: str(v) for k, v in vars(args).items()})
    import wandb
    if args.wandb == 'online':
        job = os.environ.get('SLURM_JOB_ID', 'local')
        wandb.init(project='adept', entity='skvayzer', group='neurosim', name=f'neurosim_{args.output.parent.name}_{job}',
                   tags=['neurosim', 'world-model', 'hand-object'], dir=str(args.output),
                   config={k: str(v) for k, v in vars(args).items()})
        (args.output/'wandb.json').write_text(json.dumps(dict(id=wandb.run.id, url=wandb.run.url), indent=2))
        print(f'WANDB {wandb.run.url}', flush=True)
    try:
        report.update(train(args))
        report['completed'] = True
    except BaseException as error:
        report.update(error=str(error), traceback=traceback.format_exc())
        traceback.print_exc()
        if wandb.run:
            wandb.run.summary.update(dict(error=str(error)[:500]))
    finally:
        if wandb.run:
            wandb.finish(exit_code=0 if report['completed'] else 1)
        (args.output/'report.json').write_text(json.dumps(report, indent=2))
        print('NEUROSIM_TRAIN_DONE '+json.dumps(dict(completed=report['completed'], final=report.get('final'))), flush=True)


def train(args):
    import torch
    from dextrah_lab.neurosim.data import FIELDS, Layout, build_tensors, combine, gather, load_chunks, pre_event_mask, sample_windows, split_envs
    from dextrah_lab.neurosim.ensemble import EnsembleMean, EnsembleWorldModel, ensemble_window_loss, uncertainty_report
    from dextrah_lab.neurosim.evaluate import drop_prediction_report, loss_prediction, open_loop, open_loop_smoothed
    from dextrah_lab.neurosim.model import HandObjectWorldModel, window_loss
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    dev = args.device
    t0 = time.time()
    parts = []
    for directory in args.data:
        data, context = load_chunks(directory, fields=FIELDS, max_chunks=args.max_chunks)
        parts.append(build_tensors(data, context, exo_mode=args.exo_mode, lowpass_tau=args.lowpass_tau or None, labels=args.labels))
        del data
    t = combine(parts) if len(parts) > 1 else parts[0]
    del parts
    T, N = t['done'].shape
    asset = t.pop('asset')
    lowpass = 6 if args.lowpass_tau else 0
    layout = Layout(hand=int(t['state'].shape[-1]-15-lowpass)//2, actuated=int(t['action'].shape[-1]), bps=int(t['bps'].shape[-1]),
                    lowpass=lowpass, exo=int(t['exo'].shape[-1]))
    train_mask, test_mask = split_envs(asset, args.test_fraction, args.seed)
    train_envs, test_envs = torch.nonzero(train_mask).flatten(), torch.nonzero(test_mask).flatten()
    stats = dict(rows=T, envs=N, transitions=int(T*N), train_envs=len(train_envs), test_envs=len(test_envs),
                 test_assets=int(torch.unique(asset[test_mask]).numel()), episode_ends=int(t['done'].sum()),
                 losses=int(t['lost'].sum()), loss_rate_per_s=float(t['lost'].sum())/max(T*N*context['dt'], 1e-9),
                 contact_rate=[round(float(x), 4) for x in t['contact'].mean((0, 1))], load_s=round(time.time()-t0, 1))
    print('NEUROSIM_DATA '+json.dumps(stats), flush=True)
    import wandb
    if wandb.run:
        wandb.config.update(dict(data=stats), allow_val_change=True)
    t = {k: v.to(dev) for k, v in t.items()}
    g = torch.Generator().manual_seed(args.seed)
    hidden = tuple(int(x) for x in args.hidden.split(','))
    if args.members > 0:
        model = EnsembleWorldModel(layout, members=args.members, hidden=hidden, exo_mode=args.exo_mode).to(dev)
        loss_fn = lambda w, **kw: ensemble_window_loss(model, w, args.k, nll_weight=args.nll_weight, **kw)
    else:
        model = HandObjectWorldModel(layout, hidden=hidden, exo_mode=args.exo_mode).to(dev)
        loss_fn = lambda w, **kw: window_loss(model, w, args.k, **kw)
    model.dt = float(context['dt'])
    # normalisers from up to 2M training rows
    m = min(2_000_000, T*len(train_envs))
    rows = (torch.randint(1, T, (m,), generator=g).to(dev), train_envs.to(dev)[torch.randint(0, len(train_envs), (m,), generator=g).to(dev)])
    model.fit_normalizers(t, rows)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: .03+.97*.5*(1+math.cos(math.pi*min(i, args.iters)/args.iters)))
    history = []
    train_envs_d, test_envs_d = train_envs.to(dev), test_envs.to(dev)
    # importance-weighted oversampling of windows that reach a loss event (review Q2): a share `oversample`
    # of each batch comes from those starts; weight = uniform probability / mixture probability
    in_train = torch.zeros(N, dtype=torch.bool, device=dev)
    in_train[train_envs_d] = True
    pre = pre_event_mask(t, args.k) & in_train[None]
    pre[T-args.k:] = False
    pre_idx = torch.nonzero(pre)
    U = float((T-args.k)*len(train_envs))
    n_over = int(round(args.oversample*args.batch)) if args.oversample > 0 and len(pre_idx) else 0
    print(f'NEUROSIM_OVERSAMPLE pre-loss starts {len(pre_idx)} per batch {n_over}', flush=True)
    for it in range(1, args.iters+1):
        model.train()
        s, e = sample_windows(t, train_envs_d, args.k, args.batch-n_over, generator=g, device=dev)
        if n_over:
            pick = pre_idx[torch.randint(0, len(pre_idx), (n_over,), generator=g).to(dev)]
            s, e = torch.cat((s, pick[:, 0])), torch.cat((e, pick[:, 1]))
            q = (1-args.oversample)/U+args.oversample*pre[s, e].float()/len(pre_idx)
            weight = (1/U)/q
        else:
            weight = None
        loss, logs = loss_fn(gather(t, s, e, args.k), lost_pos_weight=args.lost_pos_weight, window_weight=weight,
                             support_in_secure=args.support_in_secure)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.))
        opt.step(); sched.step()
        if it % 200 == 0:
            print(f'NEUROSIM_ITER {it} loss {float(loss):.4f} grad {grad:.3f} '+' '.join(f'{k} {v:.4f}' for k, v in logs.items()), flush=True)
            if wandb.run:
                wandb.log(dict({'train/loss': float(loss), 'train/grad_norm': grad, 'train/lr': sched.get_last_lr()[0]},
                               **{f'train/{k}': v for k, v in logs.items()}), step=it)
        if it % args.eval_every == 0 or it == args.iters:
            model.eval()
            with torch.no_grad():
                s, e = sample_windows(t, test_envs_d, args.k, args.batch, generator=g, device=dev)
                test_loss, test_logs = loss_fn(gather(t, s, e, args.k), lost_pos_weight=args.lost_pos_weight,
                                               support_in_secure=args.support_in_secure)
            row = dict(iter=it, train_loss=round(float(loss), 5), test_loss=round(float(test_loss), 5),
                       test=dict((k, round(v, 5)) for k, v in test_logs.items()), wall_s=round(time.time()-t0, 1))
            history.append(row)
            print('NEUROSIM_EVAL '+json.dumps(row), flush=True)
            if wandb.run:
                wandb.log(dict({'test/loss': row['test_loss']}, **{f'test/{k}': v for k, v in row['test'].items()}), step=it)
    model.eval()
    evaluated = EnsembleMean(model) if args.members > 0 else model
    evaluated.dt = model.dt
    final = dict(open_loop=open_loop(evaluated, t, test_envs_d, horizons=(1, 8, 16, 32), generator=g),
                 open_loop_smoothed=open_loop_smoothed(evaluated, t, test_envs_d, generator=g),
                 loss_within_0p5s=loss_prediction(evaluated, t, test_envs_d, generator=g),
                 drop_prediction=drop_prediction_report(evaluated, t, train_envs_d, test_envs_d, 30, g),
                 open_loop_train_objects=open_loop(evaluated, t, train_envs_d, generator=g))
    if args.members > 0:
        final['uncertainty'] = uncertainty_report(model, t, test_envs_d, generator=g)
    print('NEUROSIM_FINAL '+json.dumps(final), flush=True)
    if wandb.run:
        flat = {}
        for split in ('open_loop', 'open_loop_train_objects'):
            for h, row in final[split].items():
                for name in ('pos_mm', 'rot_deg', 'hand_q_rad', 'force_log1p'):
                    for who in ('model', 'persistence'):
                        flat[f'{split}/h{h}/{name}/{who}_median'] = row[name][who]['median']
                for name in ('pos_mm', 'rot_deg'):
                    if 'moving' in row:
                        for who in ('model', 'persistence'):
                            flat[f'{split}/h{h}/moving/{name}/{who}_median'] = row['moving'][name][who]['median']
                flat[f'{split}/h{h}/contact_auc_mean'] = sum(row['contact_auc'])/len(row['contact_auc'])
                flat[f'{split}/h{h}/support_auc'] = row['support_auc']
        for h, row in final['open_loop_smoothed'].items():
            for name in ('pos_mm', 'rot_deg'):
                for who in ('model', 'persistence'):
                    flat[f'smoothed/h{h}/{name}/{who}_median'] = row[name][who]['median']
                    if f'drifting_{name}' in row:
                        flat[f'smoothed/h{h}/drifting/{name}/{who}_median'] = row[f'drifting_{name}'][who]['median']
        for neg in ('random_negatives', 'phase_matched_negatives'):
            for k, v in final['drop_prediction'][neg].items():
                flat[f'drops/{neg}/{k}/auc'] = v['auc']
        for k, v in final['drop_prediction']['calibration_random_windows'].items():
            flat[f'drops/calibration/{k}/brier'] = v['brier']
            flat[f'drops/calibration/{k}/brier_climatology'] = v['brier_climatology']
        lp = final['loss_within_0p5s']
        flat.update({f'loss_0p5s/{k}': lp[k] for k in ('auc', 'events', 'base_rate', 'mean_predicted', 'brier', 'brier_climatology')})
        wandb.run.summary.update(flat)
    torch.save(dict(state_dict=model.state_dict(), layout=layout.__dict__, hidden=hidden, dt=model.dt, exo_mode=args.exo_mode,
                    members=args.members,
                    lowpass_tau=args.lowpass_tau, labels=args.labels,
                    test_assets=torch.unique(asset[test_mask]).tolist(), args=vars(args)), args.output/'world_model.pt')
    return dict(data=stats, history=history, final=final, wall_s=round(time.time()-t0, 1))


if __name__ == '__main__':
    main()
