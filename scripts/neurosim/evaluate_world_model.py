#!/usr/bin/env python3
"""Drop-prediction report of a trained hand-object world model (review, 2026-10-27), CPU or GPU.

On the model's held-out objects: drop-only labels (failed grasp checks and timeouts inside the horizon are
excluded), model AUC under the 'teacher' (logged inputs over the horizon) and 'frozen' (only what is known
at the window start) protocols, logistic-regression baselines on current features without / with the phase
bucket, random and phase-matched negatives, stratified bootstrap 95% intervals, calibration at the natural
rate (dextrah_lab/neurosim/evaluate.drop_prediction_report).
"""
import argparse
import json
from pathlib import Path
import time


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', type=Path, required=True, help='world_model.pt of train_world_model.py')
    p.add_argument('--data', type=Path, required=True, help='the collection the model was trained on')
    p.add_argument('--output', type=Path, required=True, help='report .json')
    p.add_argument('--horizon', type=int, default=30)
    p.add_argument('--boot', type=int, default=1000)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='cpu')
    args = p.parse_args()
    import torch
    from dextrah_lab.neurosim.data import FIELDS, Layout, build_tensors, load_chunks
    from dextrah_lab.neurosim.evaluate import drop_prediction_report
    from dextrah_lab.neurosim.model import HandObjectWorldModel
    t0 = time.time()
    data, context = load_chunks(args.data, fields=FIELDS)
    t = build_tensors(data, context)
    del data
    asset = t.pop('asset')
    ck = torch.load(args.model, map_location='cpu', weights_only=False)
    layout = Layout(**{k: v for k, v in ck['layout'].items() if k in Layout.__dataclass_fields__})
    model = HandObjectWorldModel(layout, hidden=tuple(ck['hidden']))
    model.load_state_dict(ck['state_dict'])
    model.dt = ck['dt']
    model.eval().to(args.device)
    t = {k: v.to(args.device) for k, v in t.items()}
    test = torch.isin(asset, torch.tensor(ck['test_assets']))
    train_envs, test_envs = torch.nonzero(~test).flatten(), torch.nonzero(test).flatten()
    g = torch.Generator().manual_seed(args.seed)
    report = dict(model=str(args.model), data=str(args.data), test_assets=len(ck['test_assets']), test_worlds=len(test_envs),
                  drops_total=int(t['drop'].sum()), censored_grasp_checks=int(t['censor'].sum()),
                  drop_prediction=drop_prediction_report(model, t, train_envs, test_envs, args.horizon, g, args.boot))
    report['wall_s'] = round(time.time()-t0, 1)
    args.output.write_text(json.dumps(report, indent=2))
    print('NEUROSIM_DROP_REPORT '+json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
