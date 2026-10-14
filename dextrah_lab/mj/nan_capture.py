"""Capture the first non-finite MuJoCo Warp world (diagnostic only, opt-in).

Run 1157 aborted after 6.5 h (about 1 in 9e8 transitions) with "Non-finite
simulation state" and nothing recorded about which world or why. This keeps,
for every world, the complete mj_step input (MuJoCo's integration state:
time, qpos, qvel, act, history, warmstart, ctrl, applied forces, equality
flags, mocap, userdata, sleep) of the last `depth` physics steps, and checks
every world for NaN/Inf right after each physics step. On the first bad world
it writes, for that world:

* `inputs.npz`: the buffered pre-step inputs (oldest -> the failing step);
* `data.npz`: the post-step Data slice. mj_step = forward at the pre-step
  state, then integration, so contacts, constraint rows, solver output and
  qacc are those of the failing step; `summary.json` lists which of them are
  non-finite, which localizes the pipeline stage (collision / constraint
  assembly / solver / integration);
* `model.npz` + `model.mjb` + `model_scalars.json`: every Model array (the
  per-world expanded ones sliced to this world) and the compiled MjModel, so
  `scripts/mj_nan_replay.py` can rebuild the world as a single-world model;
* `context.npz`: task-side state of the world (targets, wrench, episode step).

It never writes simulator state; the caller then aborts exactly as before.
"""
import dataclasses
import json
import time
from pathlib import Path

import numpy as np
import torch
import warp as wp

# mjSTATE_INTEGRATION in mujoco_warp Data names.
INPUT_FIELDS = ('time', 'qpos', 'qvel', 'act', 'history', 'qacc_warmstart', 'ctrl', 'qfrc_applied',
                'xfrc_applied', 'eq_active', 'mocap_pos', 'mocap_quat', 'userdata', 'tree_asleep')
CHECK_FIELDS = ('qpos', 'qvel', 'qacc')
FULL_ARRAY_LIMIT = 8_000_000   # elements; larger non-per-world arrays are listed, not saved


def _torch(arr):
    return wp.to_torch(arr)


def _finite_rows(t):
    """(nworld,) bool: row has only finite entries (non-float dtypes are finite)."""
    if not torch.is_floating_point(t):
        return torch.ones(t.shape[0], dtype=torch.bool, device=t.device)
    return torch.isfinite(t.reshape(t.shape[0], -1)).all(-1)


def _walk(obj, prefix=''):
    """Yield (name, value) for every field of a (nested) mujoco_warp dataclass."""
    for f in dataclasses.fields(obj):
        value = getattr(obj, f.name, None)
        name = prefix+f.name
        if dataclasses.is_dataclass(value):
            yield from _walk(value, name+'.')
        else:
            yield name, value


def _nonfinite(a):
    a = np.asarray(a)
    return bool(np.issubdtype(a.dtype, np.floating) and a.size and not np.isfinite(a).all())


class NanCapture:
    def __init__(self, env, out_dir, depth=4, stats_interval=2000):
        self.env = env
        self.sim = env.sim
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.depth = int(depth)
        self.stats_interval = int(stats_interval)
        d = self.sim.wp_data
        self.nworld = d.nworld
        self.naconmax = d.naconmax
        self.inputs = {f: _torch(getattr(d, f)) for f in INPUT_FIELDS
                       if getattr(d, f, None) is not None and getattr(d, f).size > 0}
        self.ring = {f: torch.empty((self.depth,)+tuple(v.shape), dtype=v.dtype, device=v.device)
                     for f, v in self.inputs.items()}
        self.ring_step = [-1]*self.depth
        self.check = {f: _torch(getattr(d, f)) for f in CHECK_FIELDS}
        dev = self.check['qpos'].device
        # Running per-world maxima and overflow bits (no host sync per step).
        self.max_nefc = torch.zeros(self.nworld, dtype=torch.int32, device=dev)
        self.max_ncon = torch.zeros(self.nworld, dtype=torch.int32, device=dev)
        self.max_nacon = torch.zeros((), dtype=torch.int32, device=dev)
        self.max_ncollision = torch.zeros((), dtype=torch.int32, device=dev)
        self.overflow = torch.zeros(self.nworld, dtype=torch.int32, device=dev)
        self._ncon = torch.zeros(self.nworld, dtype=torch.int32, device=dev)
        self._arange = torch.arange(self.naconmax, device=dev)
        self.steps = 0
        self.check_seconds = 0.
        self.dumped = None
        ring_bytes = sum(t.numel()*t.element_size() for t in self.ring.values())
        self.info = dict(depth=self.depth, nworld=self.nworld, naconmax=self.naconmax, njmax=d.njmax,
                         fields={f: list(v.shape[1:]) for f, v in self.inputs.items()},
                         ring_mib=ring_bytes/2**20)
        (self.out_dir/'capture_info.json').write_text(json.dumps(self.info, indent=1))
        print(f'MJ_NAN_CAPTURE ready {json.dumps(self.info)}', flush=True)

    # ------------------------------------------------------------------
    def before_step(self, step):
        slot = step % self.depth
        for f, v in self.inputs.items():
            self.ring[f][slot].copy_(v)
        self.ring_step[slot] = step

    def after_step(self, step):
        t0 = time.perf_counter()
        d = self.sim.wp_data
        ok = torch.ones(self.nworld, dtype=torch.bool, device=self.max_nefc.device)
        for t in self.check.values():
            ok &= _finite_rows(t)
        self._stats(d)
        bad = ~ok
        if bool(bad.any()):
            path = self.dump(step, bad)
            raise RuntimeError(f'Non-finite simulation state; refusing to feed it to SAPG (captured in {path})')
        self.steps += 1
        self.check_seconds += time.perf_counter()-t0
        if self.steps % self.stats_interval == 0:
            self.write_stats(step)

    def _stats(self, d):
        nefc = _torch(d.nefc)
        torch.maximum(self.max_nefc, nefc.to(torch.int32), out=self.max_nefc)
        nacon = _torch(d.nacon)
        active = (self._arange < nacon).to(torch.int32)
        world = _torch(d.contact.worldid).long().clamp_(0, self.nworld-1)
        self._ncon.zero_().scatter_add_(0, world, active)
        torch.maximum(self.max_ncon, self._ncon, out=self.max_ncon)
        torch.maximum(self.max_nacon, nacon[0].to(torch.int32), out=self.max_nacon)
        torch.maximum(self.max_ncollision, _torch(d.ncollision)[0].to(torch.int32), out=self.max_ncollision)
        self.overflow |= _torch(d.overflow).to(torch.int32)

    def stats(self):
        over = self.overflow
        return dict(physics_steps=self.steps, check_ms_per_step=1e3*self.check_seconds/max(self.steps, 1),
                    max_nefc=int(self.max_nefc.max()), njmax=self.sim.wp_data.njmax,
                    max_contacts_per_world=int(self.max_ncon.max()),
                    max_nacon=int(self.max_nacon), max_ncollision=int(self.max_ncollision), naconmax=self.naconmax,
                    overflow_worlds=int((over != 0).sum()),
                    overflow_bits=sorted(set(over[over != 0].tolist())))

    def write_stats(self, step):
        s = dict(step=step, wall=time.time(), **self.stats())
        with (self.out_dir/'stats.jsonl').open('a') as f:
            f.write(json.dumps(s)+'\n')

    def dump_env_state(self, step, tensors):
        """The task's own finiteness check failed although qpos/qvel/qacc
        passed after every physics step (non-finite derived state): dump the
        worlds whose task tensors are non-finite."""
        bad = torch.zeros(self.nworld, dtype=torch.bool, device=self.max_nefc.device)
        for t in tensors:
            bad |= ~_finite_rows(t)
        if not bool(bad.any()):
            return None
        return self.dump(step, bad)

    # ------------------------------------------------------------------
    def dump(self, step, bad, max_worlds=4):
        worlds = bad.nonzero().flatten().tolist()
        out = self.out_dir/f'nan_step{step}'
        out.mkdir(parents=True, exist_ok=True)
        env, d, m = self.env, self.sim.wp_data, self.sim.wp_model
        order = sorted(range(self.depth), key=lambda s: self.ring_step[s])
        order = [s for s in order if self.ring_step[s] >= 0]
        summary = dict(step=step, worlds=worlds, ring_steps=[self.ring_step[s] for s in order],
                       stats=self.stats(), env_step=int(env.common_step_counter), per_world={})
        # Inputs: oldest -> failing step, every bad world (cheap).
        inputs = {f'{f}': self.ring[f][order][:, worlds].cpu().numpy() for f in self.ring}
        np.savez_compressed(out/'inputs.npz', worlds=np.array(worlds), steps=np.array(summary['ring_steps']), **inputs)
        nacon = int(_torch(d.nacon)[0])
        cworld = _torch(d.contact.worldid)[:min(nacon, self.naconmax)]
        for w in worlds[:max_worlds]:
            arrays, skipped = {}, []
            cidx = (cworld == w).nonzero().flatten()
            for name, value in _walk(d):
                if not isinstance(value, wp.array) or value.size == 0:
                    continue
                try:
                    t = _torch(value)
                except Exception as e:
                    skipped.append([name, repr(e)])
                    continue
                if name.startswith('contact.'):
                    arrays[name] = t[cidx].cpu().numpy()
                elif t.ndim >= 1 and t.shape[0] == self.nworld:
                    arrays[name] = t[w].cpu().numpy()
                elif t.numel() <= FULL_ARRAY_LIMIT:
                    arrays[name] = t.cpu().numpy()
                else:
                    skipped.append([name, list(t.shape)])
            arrays['contact.index'] = cidx.cpu().numpy()
            np.savez_compressed(out/f'data_w{w}.npz', **arrays)
            pre = {f: _nonfinite(inputs[f][-1, worlds.index(w)]) for f in inputs}
            post = sorted(k for k, a in arrays.items() if _nonfinite(a))
            mjm = self.sim.mj_model
            contacts = []
            for i in range(len(cidx)):
                g = arrays['contact.geom'][i]
                contacts.append(dict(geom=[int(x) for x in g],
                                     names=[mjm.geom(int(x)).name or f'geom{int(x)}' for x in g],
                                     types=[int(mjm.geom_type[int(x)]) for x in g],
                                     bodies=[mjm.body(int(mjm.geom_bodyid[int(x)])).name for x in g],
                                     dist=float(arrays['contact.dist'][i]), dim=int(arrays['contact.dim'][i]),
                                     efc_address=[int(x) for x in np.ravel(arrays['contact.efc_address'][i])[:6]],
                                     nonfinite=any(_nonfinite(arrays[k][i]) for k in
                                                   ('contact.dist', 'contact.pos', 'contact.frame'))))
            summary['per_world'][w] = dict(
                nonfinite_inputs_at_failing_step=sorted(k for k, v in pre.items() if v),
                nonfinite_post_step=post, nefc=int(arrays['nefc']), ncon=len(cidx),
                overflow=int(arrays['overflow']), solver_niter=int(arrays['solver_niter']),
                max_nefc_seen=int(self.max_nefc[w]), max_ncon_seen=int(self.max_ncon[w]),
                contacts=contacts, skipped_large=skipped)
        # Model: per-world (expanded) arrays sliced; shared arrays whole.
        model_arrays, scalars, expanded = {}, {}, []
        for name, value in _walk(m):
            if isinstance(value, wp.array):
                if value.size == 0:
                    continue
                try:
                    t = _torch(value)
                except Exception:
                    continue
                if t.ndim >= 1 and t.shape[0] == self.nworld and self.nworld > 1:
                    # Per-world (expanded) field, or a model dimension that
                    # happens to equal nworld: keep the slice, and the whole
                    # array when small; the replay tells them apart by shape.
                    expanded.append(name)
                    model_arrays[name] = t[worlds[:max_worlds]].cpu().numpy()
                    if t.numel() <= 2_000_000:
                        model_arrays[name+'#full'] = t.cpu().numpy()
                elif t.numel() <= FULL_ARRAY_LIMIT:
                    model_arrays[name] = t.cpu().numpy()
            elif isinstance(value, (bool, int, float, str)) or value is None:
                scalars[name] = value
            elif hasattr(value, 'value'):   # enums
                scalars[name] = int(value.value)
            elif isinstance(value, (tuple, list)) and all(isinstance(x, (bool, int, float)) for x in value):
                scalars[name] = list(value)
        np.savez_compressed(out/'model.npz', **model_arrays)
        (out/'model_scalars.json').write_text(json.dumps(dict(scalars=scalars, expanded=expanded,
                                                              worlds=worlds[:max_worlds]), indent=1, default=str))
        import mujoco
        mujoco.mj_saveModel(self.sim.mj_model, str(out/'model.mjb'), None)
        # Task-side context.
        ctx = {}
        for name in ('episode_length_buf', '_cur_targets', '_object_asset_index_per_env', '_last_latent_action',
                     '_wholebody_action_queue', '_object_init_z', '_table_z_per_env', 'touch_force_w',
                     '_body_table_force_step'):
            v = getattr(env, name, None)
            if torch.is_tensor(v) and v.ndim >= 1 and v.shape[0] == self.nworld:
                ctx[name] = v[worlds].cpu().numpy()
        for owner, attrs in ((env.robot, ('_targets', '_setpoint')),):
            for a in attrs:
                v = getattr(owner, a, None)
                if torch.is_tensor(v):
                    ctx['robot'+a] = v[worlds].cpu().numpy()
        wrench = getattr(env.object, '_wrench', None)
        if wrench is not None:
            ctx['object_force'], ctx['object_torque'] = (x[worlds].cpu().numpy() for x in wrench)
        np.savez_compressed(out/'context.npz', **ctx)
        paths = getattr(env, '_object_urdf_paths', None)
        if paths is not None:
            summary['objects'] = {w: paths[int(env._object_asset_index_per_env[w])] for w in worlds[:max_worlds]}
        (out/'summary.json').write_text(json.dumps(summary, indent=1, default=str))
        self.dumped = out
        print(f'MJ_NAN_CAPTURE dumped {out}: {json.dumps(summary, default=str)[:4000]}', flush=True)
        return out
