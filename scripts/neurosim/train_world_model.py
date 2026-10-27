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
    p.add_argument('--data', type=Path, required=True, help='collector output directory (chunk_*.pt, context.pt)')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--k', type=int, default=8)
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
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    report = dict(completed=False, args={k: str(v) for k, v in vars(args).items()})
    try:
        report.update(train(args))
        report['completed'] = True
    except BaseException as error:
        report.update(error=str(error), traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        (args.output/'report.json').write_text(json.dumps(report, indent=2))
        print('NEUROSIM_TRAIN_DONE '+json.dumps(dict(completed=report['completed'], final=report.get('final'))), flush=True)


def train(args):
    import torch
    from dextrah_lab.neurosim.data import Layout, build_tensors, gather, load_chunks, sample_windows, split_envs
    from dextrah_lab.neurosim.evaluate import loss_prediction, open_loop
    from dextrah_lab.neurosim.model import HandObjectWorldModel, window_loss
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    dev = args.device
    t0 = time.time()
    data, context = load_chunks(args.data, max_chunks=args.max_chunks)
    t = build_tensors(data, context)
    del data
    T, N = t['done'].shape
    asset = t.pop('asset')
    layout = Layout(hand=int(t['state'].shape[-1]-15)//2, actuated=int(t['action'].shape[-1]), bps=int(t['bps'].shape[-1]))
    train_mask, test_mask = split_envs(asset, args.test_fraction, args.seed)
    train_envs, test_envs = torch.nonzero(train_mask).flatten(), torch.nonzero(test_mask).flatten()
    stats = dict(rows=T, envs=N, transitions=int(T*N), train_envs=len(train_envs), test_envs=len(test_envs),
                 test_assets=int(torch.unique(asset[test_mask]).numel()), episode_ends=int(t['done'].sum()),
                 losses=int(t['lost'].sum()), loss_rate_per_s=float(t['lost'].sum())/max(T*N*context['dt'], 1e-9),
                 contact_rate=[round(float(x), 4) for x in t['contact'].mean((0, 1))], load_s=round(time.time()-t0, 1))
    print('NEUROSIM_DATA '+json.dumps(stats), flush=True)
    t = {k: v.to(dev) for k, v in t.items()}
    g = torch.Generator().manual_seed(args.seed)
    hidden = tuple(int(x) for x in args.hidden.split(','))
    model = HandObjectWorldModel(layout, hidden=hidden).to(dev)
    model.dt = float(context['dt'])
    # normalisers from up to 2M training rows
    m = min(2_000_000, T*len(train_envs))
    rows = (torch.randint(1, T, (m,), generator=g).to(dev), train_envs.to(dev)[torch.randint(0, len(train_envs), (m,), generator=g).to(dev)])
    model.fit_normalizers(t, rows)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: .03+.97*.5*(1+math.cos(math.pi*min(i, args.iters)/args.iters)))
    history = []
    train_envs_d, test_envs_d = train_envs.to(dev), test_envs.to(dev)
    for it in range(1, args.iters+1):
        model.train()
        s, e = sample_windows(t, train_envs_d, args.k, args.batch, generator=g, device=dev)
        loss, logs = window_loss(model, gather(t, s, e, args.k), args.k, lost_pos_weight=args.lost_pos_weight)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.))
        opt.step(); sched.step()
        if it % 200 == 0:
            print(f'NEUROSIM_ITER {it} loss {float(loss):.4f} grad {grad:.3f} '+' '.join(f'{k} {v:.4f}' for k, v in logs.items()), flush=True)
        if it % args.eval_every == 0 or it == args.iters:
            model.eval()
            with torch.no_grad():
                s, e = sample_windows(t, test_envs_d, args.k, args.batch, generator=g, device=dev)
                test_loss, test_logs = window_loss(model, gather(t, s, e, args.k), args.k, lost_pos_weight=args.lost_pos_weight)
            row = dict(iter=it, train_loss=round(float(loss), 5), test_loss=round(float(test_loss), 5),
                       test=dict((k, round(v, 5)) for k, v in test_logs.items()), wall_s=round(time.time()-t0, 1))
            history.append(row)
            print('NEUROSIM_EVAL '+json.dumps(row), flush=True)
    model.eval()
    final = dict(open_loop=open_loop(model, t, test_envs_d, generator=g),
                 loss_within_0p5s=loss_prediction(model, t, test_envs_d, generator=g),
                 open_loop_train_objects=open_loop(model, t, train_envs_d, generator=g))
    print('NEUROSIM_FINAL '+json.dumps(final), flush=True)
    torch.save(dict(state_dict=model.state_dict(), layout=layout.__dict__, hidden=hidden, dt=model.dt,
                    test_assets=torch.unique(asset[test_mask]).tolist(), args=vars(args)), args.output/'world_model.pt')
    return dict(data=stats, history=history, final=final, wall_s=round(time.time()-t0, 1))


if __name__ == '__main__':
    main()
