#!/usr/bin/env python3
"""What did the difference loss teach? Per-component agreement on held-out paired counterfactuals (CPU, 2026-10-28).

Fine-tuning on paired counterfactuals raised the held-out pair cosine (all state dimensions + forces, scaled) to
0.35, yet the held-out gradient check of object outcomes stayed at chance. This splits the comparison by state
component on the pair groups of the fine-tune's held-out pair objects (reconstructed as train_world_model.py drew
them): cosine and magnitude ratio of the predicted vs simulated copy - baseline difference over the window, and
the size of the effect against the model's own error on the baseline rollout (same window, same units).
"""
import argparse
import json
from pathlib import Path

COMPONENTS = dict(object_pos=(0, 3), object_rot6=(3, 9), object_linvel=(9, 12), object_angvel=(12, 15),
                  hand_q=(15, 21), hand_qd=(21, 27), lowpass_twist=(27, 33))
FORCES = dict(pad_forces=(0, 5), palm_force=(5, 6))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--pairs', type=Path, required=True)
    p.add_argument('--models', type=Path, nargs='+', required=True)
    p.add_argument('--finetune-report', type=Path, required=True, help='report.json of the pair fine-tune (its args)')
    p.add_argument('--groups', type=int, default=4096)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    import torch
    from dextrah_lab.neurosim.data import Layout, gather
    from dextrah_lab.neurosim.model import HandObjectWorldModel
    from dextrah_lab.neurosim.pairs import load_pairs
    ft_args = json.loads(args.finetune_report.read_text())['args']
    K = int(ft_args['pair_copies'])
    first = torch.load(args.models[0], map_location='cpu', weights_only=False)
    tp = load_pairs(args.pairs, exo_mode=first['exo_mode'], lowpass_tau=first['lowpass_tau'] or None, labels=first['labels'],
                    dt=first['dt'], copies=K)
    # held-out pair objects exactly as train_world_model.py drew them (first use of its generator under --init)
    g = torch.Generator().manual_seed(int(ft_args['seed']))
    base_asset = tp['asset'][::K]
    usable = ~tp['group_end'] & ~torch.isin(base_asset, torch.tensor(first['test_assets']))
    pair_assets = base_asset[usable].unique()
    held = pair_assets[torch.randperm(len(pair_assets), generator=g)[:int(round(float(ft_args['pair_test_fraction'])*len(pair_assets)))]]
    groups = torch.nonzero(usable & torch.isin(base_asset, held)).flatten()
    groups = groups[torch.randperm(len(groups), generator=torch.Generator().manual_seed(1))[:args.groups]]
    k = tp['state'].shape[0]-2
    c = torch.randint(1, K, (len(groups),), generator=torch.Generator().manual_seed(2))
    base, other = groups*K, groups*K+c
    s = torch.ones(len(groups), dtype=torch.long)
    wb, wc = gather(tp, s, base, k), gather(tp, s, other, k)
    sim_x = wc['state'][:, 1:]-wb['state'][:, 1:]
    sim_f = wc['forces'][:, 1:]-wb['forces'][:, 1:]
    report = dict(held_out_pair_objects=len(held), groups=len(groups), k=k, models={})
    for path in args.models:
        ck = torch.load(path, map_location='cpu', weights_only=False)
        layout = Layout(**{kk: v for kk, v in ck['layout'].items() if kk in Layout.__dataclass_fields__})
        model = HandObjectWorldModel(layout, hidden=tuple(ck['hidden']), exo_mode=ck.get('exo_mode', 'post'))
        model.load_state_dict(ck['state_dict'])
        model.eval()
        with torch.no_grad():
            xb, _, fb, _ = model.rollout(wb, k)
            xc, _, fc, _ = model.rollout(wc, k)
        xb, xc, fb, fc = torch.stack(xb, 1), torch.stack(xc, 1), torch.stack(fb, 1), torch.stack(fc, 1)
        pred_x, pred_f = xc-xb, fc-fb
        err_x, err_f = xb-wb['state'][:, 1:], fb-wb['forces'][:, 1:]
        rows = {}
        for name, (a, b), pred, sim, err in ([(n, sl, pred_x, sim_x, err_x) for n, sl in COMPONENTS.items()] +
                                            [(n, sl, pred_f, sim_f, err_f) for n, sl in FORCES.items()]):
            P, S, E = pred[..., a:b].flatten(1), sim[..., a:b].flatten(1), err[..., a:b].flatten(1)
            keep = S.norm(dim=1) > 0
            cos = (P*S).sum(1)/(P.norm(dim=1)*S.norm(dim=1)).clamp_min(1e-12)
            rows[name] = dict(cosine_median=round(float(cos[keep].median()), 4),
                              magnitude_ratio_median=round(float((P.norm(dim=1)/S.norm(dim=1).clamp_min(1e-12))[keep].median()), 4),
                              effect_norm_median=float(S.norm(dim=1).median()),
                              baseline_error_norm_median=float(E.norm(dim=1).median()),
                              error_to_effect_median=round(float((E.norm(dim=1)/S.norm(dim=1).clamp_min(1e-12))[keep].median()), 2))
        report['models'][str(path)] = rows
    args.output.write_text(json.dumps(report, indent=2))
    print('PAIR_DIAGNOSTIC '+json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
