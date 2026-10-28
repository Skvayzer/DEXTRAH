#!/usr/bin/env python3
"""Why is the world model action-blind? One-step action sensitivity, model vs data (2026-10-28), CPU.

Gradient check v5 (randomisation off, perturbation right after the restore): the model's effects of a finger
command change are 4-6% of MuJoCo Warp's and uncorrelated with them. This separates "the data carry little
independent action variation" from "the model ignores the action it could use":

* model: autograd Jacobian of the one-step prediction w.r.t. the action (the finger targets applied in that
  step), for the next hand joint positions and the next pad / palm contact forces (log1p N), on held-out objects;
* data: ridge regression of the same one-step targets on the model's inputs (state, forces, action, exogenous
  inputs; context left out), its action coefficients in the same units, and the variance the action explains
  beyond the other inputs (R^2 with minus without the action);
* the per-step change of the action and how much of it the state predicts (the excitation left for learning).
"""
import argparse
import json
from pathlib import Path
import time


def ridge(X, Y, lam):
    """Standardised ridge: coefficients in original units (d_in, d_out), R^2 per output."""
    import torch
    mx, sx = X.mean(0), X.std(0).clamp_min(1e-6)
    my = Y.mean(0)
    Z = (X-mx)/sx
    A = Z.T@Z+lam*len(Z)*torch.eye(Z.shape[1], dtype=Z.dtype)
    W = torch.linalg.solve(A, Z.T@(Y-my))
    res = Y-my-Z@W
    r2 = 1-res.pow(2).mean(0)/(Y-my).pow(2).mean(0).clamp_min(1e-12)
    return W/sx[:, None], r2


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--max-chunks', type=int, default=2)
    p.add_argument('--samples', type=int, default=200000)
    p.add_argument('--lam', type=float, default=1e-3)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    import torch
    from dextrah_lab.neurosim.data import FIELDS, Layout, build_tensors, load_chunks
    from dextrah_lab.neurosim.model import HandObjectWorldModel
    t0 = time.time()
    torch.manual_seed(args.seed)
    ck = torch.load(args.model, map_location='cpu', weights_only=False)
    data, context = load_chunks(args.data, fields=FIELDS, max_chunks=args.max_chunks)
    t = build_tensors(data, context, exo_mode=ck.get('exo_mode', 'post'), lowpass_tau=ck.get('lowpass_tau') or None,
                      labels=ck.get('labels', 'lost'))
    del data
    layout = Layout(**{k: v for k, v in ck['layout'].items() if k in Layout.__dataclass_fields__})
    model = HandObjectWorldModel(layout, hidden=tuple(ck['hidden']), exo_mode=ck.get('exo_mode', 'post'))
    model.load_state_dict(ck['state_dict'])
    model.eval()
    T, N = t['done'].shape
    test = torch.isin(t['asset'], torch.tensor(ck['test_assets']))
    # transitions r -> r+1 inside an episode (row r+1 not a reset row), held-out objects
    valid = ~t['done'][1:]                                                    # (T-1, N)
    valid &= test[None]
    rows, envs = torch.nonzero(valid, as_tuple=True)
    pick = torch.randperm(len(rows))[:args.samples]
    r, e = rows[pick], envs[pick]
    x, R, f = t['state'][r, e], t['rot'][r, e], t['forces'][r, e]
    a, u = t['action'][r+1, e], t['exo'][r+1, e]
    ctx = torch.cat((t['bps'][e], t['masscom'][r, e]), -1)
    q0 = 3+6+3+3                                                              # obj pos, rot6, lin, ang -> hand q
    hand = slice(q0, q0+layout.hand)
    y_q = t['state'][r+1, e][:, hand]-x[:, hand]                              # one-step hand joint change
    y_f = t['forces'][r+1, e][:, :6]                                          # next pad + palm log1p force
    # ---- model Jacobian (chunks; samples are independent, so grad of the sum gives per-sample rows)
    Jq, Jf, pred_q, pred_f = [], [], [], []
    for b in range(0, len(r), 8192):
        sl = slice(b, b+8192)
        ab = a[sl].clone().requires_grad_(True)
        z = model.embed(ctx[sl])
        xn, _, fn, _ = model.step(x[sl], R[sl], f[sl], ab, u[sl], z)
        dq = xn[:, hand]-x[sl][:, hand]
        out = torch.cat((dq, fn[:, :6]), -1)
        rows_j = []
        for i in range(out.shape[1]):
            g, = torch.autograd.grad(out[:, i].sum(), ab, retain_graph=i < out.shape[1]-1)
            rows_j.append(g)
        J = torch.stack(rows_j, 1)                                            # (B, 12, actuated)
        Jq.append(J[:, :layout.hand].detach()); Jf.append(J[:, layout.hand:].detach())
        pred_q.append(dq.detach()); pred_f.append(fn[:, :6].detach())
    Jq, Jf = torch.cat(Jq), torch.cat(Jf)
    pred_q, pred_f = torch.cat(pred_q), torch.cat(pred_f)
    # ---- data: ridge on the model's inputs (64-bit)
    X_wo = torch.cat((x, f, u), -1).double()
    X_w = torch.cat((x, f, u, a), -1).double()
    na = a.shape[1]
    W_q, r2_q = ridge(X_w, y_q.double(), args.lam)
    W_f, r2_f = ridge(X_w, y_f.double(), args.lam)
    _, r2_q0 = ridge(X_wo, y_q.double(), args.lam)
    _, r2_f0 = ridge(X_wo, y_f.double(), args.lam)
    Bq, Bf = W_q[-na:].T, W_f[-na:].T                                         # (out, actuated) like the Jacobian
    # ---- excitation: action change per step, and how much of the action the state + exo predict
    da = (a-t['action'][r, e])
    _, r2_a = ridge(torch.cat((x, f, u), -1).double(), a.double(), args.lam)
    q = lambda v: [round(float(v.quantile(p)), 5) for p in (.1, .5, .9)]
    diag = lambda J: torch.diagonal(J, dim1=-2, dim2=-1)
    report = dict(
        model=str(args.model), data=str(args.data), samples=len(r), held_out_assets=len(ck['test_assets']),
        hand_q=dict(
            model_jacobian_diag_median=[round(float(v), 4) for v in diag(Jq).median(0).values],
            model_jacobian_offdiag_abs_median=round(float((Jq-torch.diag_embed(diag(Jq))).abs().median()), 5),
            ridge_action_coef_diag=[round(float(v), 4) for v in torch.diagonal(Bq)],
            r2_with_action=[round(float(v), 4) for v in r2_q], r2_without_action=[round(float(v), 4) for v in r2_q0],
            model_r2=[round(float(v), 4) for v in (1-(pred_q-y_q).pow(2).mean(0)/(y_q-y_q.mean(0)).pow(2).mean(0))]),
        pad_force=dict(
            model_jacobian_abs_median=round(float(Jf.abs().median()), 5),
            model_jacobian_frobenius_median=round(float(Jf.flatten(1).norm(dim=1).median()), 5),
            ridge_action_coef_frobenius=round(float(Bf.norm()), 5),
            ridge_action_coef=[[round(float(v), 4) for v in row] for row in Bf],
            model_jacobian_median=[[round(float(v), 4) for v in row] for row in Jf.median(0).values],
            r2_with_action=[round(float(v), 4) for v in r2_f], r2_without_action=[round(float(v), 4) for v in r2_f0]),
        action=dict(step_change_abs_q10_50_90=q(da.abs().flatten()), std_per_joint=[round(float(v), 4) for v in a.std(0)],
                    r2_from_inputs=[round(float(v), 4) for v in r2_a]),
        wall_s=round(time.time()-t0, 1))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print('ACTION_SENSITIVITY '+json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
