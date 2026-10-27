#!/usr/bin/env python3
"""Mix real M -> C hand-off states into M's SGS holding states (carry expert starts), CPU only.

The carry expert C trains from M's SGS holding states (stable_grasp / near_goal slots): filtered, stable
grasps. On the table-to-table route C starts from whatever M grasped (one finger on the object for 1 s
suffices), and C v2 loses 17% of those grasps while its arm moves to the carry posture, against ~5% in
training (jobs 1588 / 1589). This writes a copy of M's .sgs.pt in which a share (--mix) of the filled holding
slots is overwritten by hand-off states harvested with scripts/mj_table_to_table.py --save-grasp-states
--grasp-only (the state at the switch, same fields as MjSgsTask._store_held). Slot indices, families and the
file layout are unchanged, so the carry env loads it like the original.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

FAMILIES = ('reaching', 'stable_grasp', 'near_goal')     # dextrah_lab/sgs/mj_task.FAMILIES
FIELDS = ('asset', 'qpos', 'qvel', 'act', 'targets', 'setpoint', 'table_z', 'object_init_z', 'object_pose',
          'joint_pos', 'joint_vel', 'history', 'goal_pose')


def slot_families(settings):
    """Family index of every slot (MjSgsTask's balanced family blocks)."""
    n, fams = settings['num_configs'], list(settings['families'])
    sizes = [n//len(fams)+(i < n % len(fams)) for i in range(len(fams))]
    bounds = np.cumsum([0]+sizes)
    family = torch.empty(n, dtype=torch.long)
    for i, f in enumerate(fams):
        family[int(bounds[i]):int(bounds[i+1])] = FAMILIES.index(f)
    return family


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--sgs', type=Path, required=True, help="M's .sgs.pt (the carry expert's CARRY_HOLDING_STATE)")
    p.add_argument('--grasp-states', type=Path, nargs='+', required=True, help='grasp_states.pt files of harvest runs')
    p.add_argument('--mix', type=float, default=.5, help='share of holding slots replaced by hand-offs')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    state = torch.load(args.sgs, map_location='cpu', weights_only=False)
    cfg, filled = state['cfg'], state['filled']
    family = slot_families(state['settings'])
    holding = filled & ((family == FAMILIES.index('stable_grasp')) | (family == FAMILIES.index('near_goal')))
    slots = torch.nonzero(holding).flatten()
    parts = [torch.load(f, map_location='cpu', weights_only=False) for f in args.grasp_states]
    hand = {k: torch.cat([q[k] for q in parts]) for k in parts[0]}
    n_hand = len(hand['asset'])
    g = torch.Generator().manual_seed(args.seed)
    m = int(round(args.mix*len(slots)))
    target = slots[torch.randperm(len(slots), generator=g)[:m]]
    source = torch.randperm(n_hand, generator=g)[:m] if n_hand >= m else torch.randint(0, n_hand, (m,), generator=g)
    for k in FIELDS:
        if k not in cfg:
            continue
        if k not in hand:
            if k == 'act' and cfg[k].shape[1] == 1:          # scenes without actuator activations
                continue
            raise KeyError(f'hand-off states lack {k}')
        if hand[k].shape[1:] != cfg[k].shape[1:]:
            raise ValueError(f'{k}: hand-off {tuple(hand[k].shape[1:])} vs SGS {tuple(cfg[k].shape[1:])} (other scene?)')
        cfg[k][target] = hand[k][source].to(cfg[k].dtype)
    state['cfg'] = cfg
    state['handoff_mix'] = dict(source_sgs=str(args.sgs), grasp_states=[str(f) for f in args.grasp_states], mix=args.mix,
                                replaced_slots=target.tolist(), handoff_states=n_hand, holding_slots=len(slots),
                                handoff_index=source.tolist())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, args.output)
    touching = hand.get('touching')
    report = dict(holding_slots=len(slots), replaced=m, handoff_states=n_hand, sampled_with_replacement=n_hand < m,
                  handoff_assets=int(torch.unique(hand['asset']).numel()),
                  touching_fingers=({int(v): int((touching == v).sum()) for v in torch.unique(touching)} if touching is not None else None),
                  output=str(args.output))
    print('HANDOFF_HOLDING_STATES '+json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
