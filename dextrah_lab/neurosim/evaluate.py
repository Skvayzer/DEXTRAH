"""Go / no-go metrics of the hand-object world model on held-out envs (objects never seen in training).

* Open-loop prediction error at horizons 1 / 8 / 32 steps (seeded at a logged state, logged actions and
  exogenous inputs): object position (mm), object rotation (deg), hand joints (rad), contact forces
  (log1p N), against the persistence baseline (nothing changes from the seed state).
* Contact / support classification (AUC) at horizons 1 and 8.
* Loss of the object within the next 0.5 s: P = 1 - prod_j (1 - p_j) over the rolled-out lost head.
  AUC on all held-out loss events (starts 1..H steps before) plus random starts; calibration (Brier,
  binned reliability) on random starts only, at the natural rate. Windows where the episode ends for
  another reason (timeout, fall) inside the horizon are excluded.
"""
import math

import torch

from .data import EXO_INDEX, gather
from .geometry import rotation_angle


def auc(score, label):
    """Area under the ROC curve (ties averaged); nan if one class is missing."""
    score, label = score.double().flatten(), label.bool().flatten()
    pos, neg = int(label.sum()), int((~label).sum())
    if pos == 0 or neg == 0:
        return float('nan')
    order = torch.argsort(score)
    ranks = torch.empty_like(score)
    ranks[order] = torch.arange(1, len(score)+1, dtype=score.dtype, device=score.device)
    # average ranks over ties
    uniq, inv, counts = torch.unique(score, return_inverse=True, return_counts=True)
    sums = torch.zeros(len(uniq), dtype=score.dtype, device=score.device).index_add_(0, inv, ranks)
    ranks = (sums/counts)[inv]
    return float((ranks[label].sum()-pos*(pos+1)/2)/(pos*neg))


@torch.no_grad()
def open_loop(model, t, envs, horizons=(1, 8, 32), samples=8192, generator=None):
    K = max(horizons)
    dev = t['state'].device
    T = t['state'].shape[0]
    s = torch.randint(0, T-K, (samples,), generator=generator).to(dev)
    e = envs.to(dev)[torch.randint(0, len(envs), (samples,), generator=generator).to(dev)]
    w = gather(t, s, e, K)
    xs, Rs, fs, ls = model.rollout(w, K)
    H = model.L.hand
    out = {}
    for h in horizons:
        m = w['has_next'][:, h-1]
        if int(m.sum()) < 10:
            continue
        x, R, f = w['state'][m, h], w['rot'][m, h], w['forces'][m, h]
        x0, R0, f0 = w['state'][m, 0], w['rot'][m, 0], w['forces'][m, 0]
        xh, Rh, fh, lh = xs[h-1][m], Rs[h-1][m], fs[h-1][m], ls[h-1][m]
        stats = lambda v: dict(median=round(float(v.median()), 4), mean=round(float(v.mean()), 4), p90=round(float(v.quantile(.9)), 4))
        row = dict(n=int(m.sum()))
        for name, model_err, base_err in (
                ('pos_mm', 1e3*(xh[:, :3]-x[:, :3]).norm(dim=-1), 1e3*(x0[:, :3]-x[:, :3]).norm(dim=-1)),
                ('rot_deg', torch.rad2deg(rotation_angle(Rh, R)), torch.rad2deg(rotation_angle(R0, R))),
                ('hand_q_rad', (xh[:, 15:15+H]-x[:, 15:15+H]).abs().mean(-1), (x0[:, 15:15+H]-x[:, 15:15+H]).abs().mean(-1)),
                ('force_log1p', (fh-f).abs().mean(-1), (f0-f).abs().mean(-1))):
            row[name] = dict(model=stats(model_err), persistence=stats(base_err))
        # windows where the object really moves in the hand (>= 5 mm or >= 5 deg over the horizon): slips,
        # re-grips, drops in progress -- persistence is a strong baseline only when nothing happens
        moving = ((x0[:, :3]-x[:, :3]).norm(dim=-1) >= .005) | (torch.rad2deg(rotation_angle(R0, R)) >= 5.)
        if int(moving.sum()) >= 10:
            row['moving'] = dict(n=int(moving.sum()), fraction=round(float(moving.float().mean()), 4))
            for name, model_err, base_err in (
                    ('pos_mm', 1e3*(xh[moving, :3]-x[moving, :3]).norm(dim=-1), 1e3*(x0[moving, :3]-x[moving, :3]).norm(dim=-1)),
                    ('rot_deg', torch.rad2deg(rotation_angle(Rh[moving], R[moving])), torch.rad2deg(rotation_angle(R0[moving], R[moving])))):
                row['moving'][name] = dict(model=stats(model_err), persistence=stats(base_err))
        contact = w['contact'][m, h]
        row['contact_auc'] = [round(auc(lh[:, i], contact[:, i]), 4) for i in range(6)]
        row['support_auc'] = round(auc(lh[:, 8], w['supported'][m, h]), 4)
        row['contact_rate'] = [round(float(contact[:, i].mean()), 4) for i in range(6)]
        out[h] = row
    return out


@torch.no_grad()
def loss_prediction(model, t, envs, horizon=30, random_samples=20000, generator=None, bins=10):
    dev = t['state'].device
    T = t['state'].shape[0]
    envs = envs.to(dev)
    in_test = torch.zeros(t['lost'].shape[1], dtype=torch.bool, device=dev)
    in_test[envs] = True
    rows, cols = torch.nonzero((t['lost'] > .5) & in_test[None], as_tuple=True)
    # positives: starts 1..horizon steps before each loss event
    back = torch.randint(1, horizon+1, (len(rows),), generator=generator).to(dev)
    s_pos, e_pos = (rows-back).clamp(min=0), cols
    s_rand = torch.randint(0, T-horizon, (random_samples,), generator=generator).to(dev)
    e_rand = envs[torch.randint(0, len(envs), (random_samples,), generator=generator).to(dev)]
    s = torch.cat((s_pos, s_rand)).clamp(max=T-horizon-1)
    e = torch.cat((e_pos, e_rand))
    P, label, clean = [], [], []
    for a in range(0, len(s), 4096):
        w = gather(t, s[a:a+4096], e[a:a+4096], horizon)
        _, _, _, ls = model.rollout(w, horizon)
        hazard = torch.stack([torch.sigmoid(l[:, 9]) for l in ls], 1)*w['alive'].float()
        P.append(1-torch.prod(1-hazard, 1))
        lost_any = (w['lost'][:, 1:]*w['alive'].float()).amax(1)
        ended = (~w['has_next'] & w['alive']).any(1)        # an episode end inside the horizon
        label.append(lost_any)
        clean.append(~ended | (lost_any > .5))               # drop windows ending in another way
    P, label, clean = torch.cat(P), torch.cat(label), torch.cat(clean)
    n_pos = len(s_pos)
    rand = torch.zeros_like(clean); rand[n_pos:] = True
    keep = clean
    res = dict(horizon_s=round(horizon*float(model.dt if hasattr(model, 'dt') else 1/60), 3), events=int(n_pos),
               auc=round(auc(P[keep], label[keep]), 4))
    r = keep & rand
    p, y = P[r], label[r]
    res.update(random_windows=int(r.sum()), base_rate=round(float(y.mean()), 5), mean_predicted=round(float(p.mean()), 5),
               brier=round(float(((p-y)**2).mean()), 6),
               brier_climatology=round(float(((y.mean()-y)**2).mean()), 6))
    edges = torch.quantile(p, torch.linspace(0, 1, bins+1, device=dev))
    rel = []
    for i in range(bins):
        m = (p >= edges[i]) & (p <= edges[i+1]) if i == bins-1 else (p >= edges[i]) & (p < edges[i+1])
        if int(m.sum()):
            rel.append(dict(predicted=round(float(p[m].mean()), 5), observed=round(float(y[m].mean()), 5), n=int(m.sum())))
    res['reliability'] = rel
    return res


# ---------------------------------------------------------------- drop prediction v2 (review, 2026-10-27)
# Labels: physical drops only; a window whose episode ends any other way inside the horizon (timeout, failed
# grasp check, fall) is excluded. Protocols: 'teacher' = logged actions / exogenous inputs over the horizon;
# 'frozen' = only what is known at the window start, held (the first step's action, decided at the start, and
# its exogenous vector with the after-step palm velocity replaced by the before-step one), so no future
# information (the arm's or the policy's reaction to a slip) can enter. Negatives: random starts, and
# phase-matched starts (same world / object, same phase bucket) that did not drop.

def freeze_inputs(w, exo_mode='post'):
    a = w['action'][:, 1:2].expand_as(w['action'][:, 1:]).clone()
    u0 = w['exo'][:, 1].clone()
    idx = EXO_INDEX[exo_mode]
    if idx['after'] is not None:                       # 'post': the after-step palm velocity is not known in advance
        u0[:, idx['after']] = u0[:, idx['before']]
    u = u0[:, None].expand_as(w['exo'][:, 1:]).clone()
    w = dict(w)
    w['action'] = torch.cat((w['action'][:, :1], a), 1)
    w['exo'] = torch.cat((w['exo'][:, :1], u), 1)
    return w


def auc_ci(score, label, n_boot=1000, generator=None):
    """AUC and a stratified bootstrap 95% interval."""
    score, label = score.flatten().cpu(), label.bool().flatten().cpu()
    pos, neg = score[label], score[~label]
    if len(pos) == 0 or len(neg) == 0:
        return dict(auc=float('nan'), ci=[float('nan')]*2, positives=int(len(pos)), negatives=int(len(neg)))
    vals = []
    for _ in range(n_boot):
        p = pos[torch.randint(0, len(pos), (len(pos),), generator=generator)]
        q = neg[torch.randint(0, len(neg), (len(neg),), generator=generator)]
        vals.append(auc(torch.cat((p, q)), torch.cat((torch.ones(len(p)), torch.zeros(len(q))))))
    v = torch.tensor(vals)
    return dict(auc=round(auc(score, label), 4), ci=[round(float(v.quantile(.025)), 4), round(float(v.quantile(.975)), 4)],
                positives=int(len(pos)), negatives=int(len(neg)))


def drop_windows(t, envs, horizon, generator=None, random_n=20000, matched_per_positive=5, tries=64):
    """Window starts (s, e) and their kind: 0 positive candidates (1..horizon steps before a drop), 1 random,
    2 phase-matched candidates (same world, same phase bucket as a positive's start)."""
    T, N = t['drop'].shape
    dev = t['drop'].device
    envs = envs.to(dev)
    in_set = torch.zeros(N, dtype=torch.bool, device=dev)
    in_set[envs] = True
    rows, cols = torch.nonzero((t['drop'] > .5) & in_set[None], as_tuple=True)
    back = torch.randint(1, horizon+1, (len(rows),), generator=generator).to(dev)
    s_pos, e_pos = (rows-back).clamp(min=0), cols
    s_rand = torch.randint(0, T-horizon-1, (random_n,), generator=generator).to(dev)
    e_rand = envs[torch.randint(0, len(envs), (random_n,), generator=generator).to(dev)]
    s_m, e_m = [], []
    if 'bucket' in t and len(s_pos):
        b = t['bucket'][s_pos, e_pos].long()
        cand = torch.randint(0, T-horizon-1, (len(s_pos), tries), generator=generator).to(dev)
        ok = t['bucket'][cand, e_pos[:, None]].long() == b[:, None]
        rank = torch.cumsum(ok.long(), 1)
        keep = ok & (rank <= matched_per_positive)
        idx = torch.nonzero(keep, as_tuple=True)
        s_m, e_m = cand[idx], e_pos[idx[0]]
    s = torch.cat((s_pos, s_rand)+((s_m,) if len(s_m) else ())).clamp(max=T-horizon-1)
    e = torch.cat((e_pos, e_rand)+((e_m,) if len(e_m) else ()))
    kind = torch.cat((torch.zeros(len(s_pos)), torch.ones(random_n), torch.full((len(s_m),), 2.))).to(dev)
    return s, e, kind


@torch.no_grad()
def window_labels_and_scores(model, t, s, e, horizon, protocol='teacher', batch=4096, label='drop'):
    """Per window: drop within the horizon, clean (no other episode end inside), predicted P(drop within)."""
    P, label, clean = [], [], []
    for a in range(0, len(s), batch):
        w = gather_drop(t, s[a:a+batch], e[a:a+batch], horizon)
        if protocol == 'frozen':
            w = freeze_inputs(w, getattr(model, 'exo_mode', 'post'))
        _, _, _, ls = model.rollout(w, horizon)
        hazard = torch.stack([torch.sigmoid(l[:, 9]) for l in ls], 1)*w['alive'].float()
        P.append(1-torch.prod(1-hazard, 1))
        dropped = (w['drop'][:, 1:]*w['alive'].float()).amax(1) > .5
        ended = (~w['has_next'] & w['alive']).any(1)
        label.append(dropped)
        clean.append(~ended | dropped)
    return torch.cat(P), torch.cat(label), torch.cat(clean)


def gather_drop(t, s, e, k):
    from .data import gather
    w = gather(t, s, e, k)
    idx = s[:, None]+torch.arange(k+1, device=s.device)[None]
    w['drop'] = t['drop'][idx, e[:, None]]
    return w


def baseline_features(t, s, e, with_phase=False, exo_mode='post'):
    """Current-state features at the window start: 6 contact flags, 8 log forces, supported, table present
    (+ one-hot phase bucket)."""
    k = EXO_INDEX[exo_mode]['parked']
    feats = [t['contact'][s, e], t['forces'][s, e], t['supported'][s, e][:, None], 1-t['exo'][s, e][:, k:k+1]]
    if with_phase:
        feats.append(torch.nn.functional.one_hot(t['bucket'][s, e].long(), 5).float())
    return torch.cat(feats, -1)


def fit_logistic(x, y, l2=1e-3, steps=200):
    mean, std = x.mean(0), x.std(0).clamp_min(1e-6)
    z = (x-mean)/std
    w = torch.zeros(z.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=steps, line_search_fn='strong_wolfe')

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(z@w+b, y)+l2*(w*w).sum()
        loss.backward()
        return loss
    opt.step(closure)
    return lambda q: torch.sigmoid(((q-mean)/std)@w.detach()+b.detach())


def drop_prediction_report(model, t, train_envs, test_envs, horizon=30, generator=None, n_boot=1000):
    """The review's drop-prediction table: model (teacher / frozen protocols) and logistic baselines
    (without / with phase), on random and on phase-matched negatives, with bootstrap intervals."""
    s, e, kind = drop_windows(t, test_envs, horizon, generator)
    report = dict(horizon_steps=horizon)
    scores = {}
    for protocol in ('teacher', 'frozen'):
        # labels come from the log, so they are the same for both protocols
        scores['model_'+protocol], y, clean = window_labels_and_scores(model, t, s, e, horizon, protocol)
    # baselines fit on training worlds with the same window sampling
    s_tr, e_tr, _ = drop_windows(t, train_envs, horizon, generator)
    _, y_tr, c_tr = window_labels_and_scores(model, t, s_tr, e_tr, horizon, 'teacher')
    for name, with_phase in (('logistic', False), ('logistic_phase', True)):
        if with_phase and 'bucket' not in t:
            continue
        mode = getattr(model, 'exo_mode', 'post')
        f = fit_logistic(baseline_features(t, s_tr[c_tr], e_tr[c_tr], with_phase, mode).cpu().float(), y_tr[c_tr].float().cpu())
        scores[name] = f(baseline_features(t, s, e, with_phase, mode).cpu().float()).to(s.device)
    pos = clean & y
    for neg_name, neg in (('random_negatives', clean & ~y & (kind == 1)), ('phase_matched_negatives', clean & ~y & (kind == 2))):
        keep = pos | neg
        report[neg_name] = {k: auc_ci(v[keep], y[keep], n_boot, generator) for k, v in scores.items()}
    rand = clean & (kind == 1)
    report['calibration_random_windows'] = {
        k: dict(base_rate=round(float(y[rand].float().mean()), 5), mean_predicted=round(float(scores[k][rand].mean()), 5),
                brier=round(float(((scores[k][rand]-y[rand].float())**2).mean()), 6),
                brier_climatology=round(float(((y[rand].float().mean()-y[rand].float())**2).mean()), 6))
        for k in scores}
    if 'bucket' in t:
        b = t['bucket'][s, e].long()
        report['positives_by_phase'] = {name: int((pos & (b == i)).sum()) for i, name in enumerate(PHASE_NAMES)}
    return report


PHASE_NAMES = ('secure', 'walk<5s', 'walk5-10s', 'walk>10s', 'final')


def project_so3(M):
    """Nearest rotation (SVD) of each 3x3 matrix in M (..., 3, 3)."""
    U, _, Vh = torch.linalg.svd(M)
    d = torch.det(U@Vh)
    D = torch.diag_embed(torch.stack((torch.ones_like(d), torch.ones_like(d), d), -1))
    return U@D@Vh


@torch.no_grad()
def open_loop_smoothed(model, t, envs, horizons=(8, 16, 32), samples=8192, avg=5, generator=None):
    """Criterion 1 on smoothed pose (review, after the v1 result): positions and orientations averaged over
    `avg` rows at both ends (start rows s-avg+1..s, end rows s+h-avg+1..s+h; orientation = projected mean
    rotation), which cancels the ~30 Hz contact vibration. Model = the same average over its own predictions.
    Errors overall and on 'drifting' windows (smoothed change >= 5 mm or >= 5 deg)."""
    K = max(horizons)
    dev = t['state'].device
    T = t['state'].shape[0]
    s = torch.randint(avg-1, T-K-1, (samples,), generator=generator).to(dev)
    e = envs.to(dev)[torch.randint(0, len(envs), (samples,), generator=generator).to(dev)]
    back = s[:, None]-torch.arange(avg-1, -1, -1, device=dev)[None]            # rows s-avg+1 .. s
    same_start = ~t['done'][back[:, 1:], e[:, None]].any(1)                    # no episode start inside
    w = gather(t, s, e, K)
    xs, Rs, _, _ = model.rollout(w, K)
    p0 = t['state'][back, e[:, None]][..., :3].mean(1)
    R0 = project_so3(t['rot'][back, e[:, None]].mean(1))
    out = {}
    stats = lambda v: dict(median=round(float(v.median()), 4), mean=round(float(v.mean()), 4), p90=round(float(v.quantile(.9)), 4))
    for h in horizons:
        m = w['has_next'][:, h-1] & same_start
        if int(m.sum()) < 10:
            continue
        rows = slice(h-avg+1, h+1)
        p1 = w['state'][:, rows, :3].mean(1)
        R1 = project_so3(w['rot'][:, rows].mean(1))
        ph = torch.stack(xs[h-avg:h], 1)[..., :3].mean(1)
        Rh = project_so3(torch.stack(Rs[h-avg:h], 1).mean(1))
        pos_m, pos_p = 1e3*(ph-p1).norm(dim=-1), 1e3*(p0-p1).norm(dim=-1)
        rot_m, rot_p = torch.rad2deg(rotation_angle(Rh, R1)), torch.rad2deg(rotation_angle(R0, R1))
        drift = m & ((pos_p >= 5.) | (rot_p >= 5.))
        row = dict(n=int(m.sum()), drifting=int(drift.sum()), drifting_fraction=round(float(drift.sum())/max(int(m.sum()), 1), 4),
                   pos_mm=dict(model=stats(pos_m[m]), persistence=stats(pos_p[m])),
                   rot_deg=dict(model=stats(rot_m[m]), persistence=stats(rot_p[m])))
        if int(drift.sum()) >= 10:
            row['drifting_pos_mm'] = dict(model=stats(pos_m[drift]), persistence=stats(pos_p[drift]))
            row['drifting_rot_deg'] = dict(model=stats(rot_m[drift]), persistence=stats(rot_p[drift]))
        out[h] = row
    return out
