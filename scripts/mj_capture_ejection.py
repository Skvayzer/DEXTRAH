"""Capture the pre-step state of worlds whose object is launched by one physics step (CPU ok)."""
import sys, os, json, torch, numpy as np
import warp as wp
from dextrah_lab.mj.factory import make_mj_env
n, dev, seconds, out = int(sys.argv[1]), sys.argv[2], float(sys.argv[3]), sys.argv[4]
env = make_mj_env(n, dev, spawn_on_table=True)
m = env.sim.mj_model; d = env.sim.data; md = env.sim.model
obj = env.object.entity.indexing.root_body_id
dof = env.object._free_dof
env.reset()
a = torch.zeros(n, 70, device=dev); a[:, 64:] = -1
FIELDS = ['qpos', 'qvel', 'ctrl', 'act', 'mocap_pos', 'mocap_quat', 'xfrc_applied', 'qacc_warmstart']
MODEL = ['geom_size', 'geom_pos', 'geom_quat', 'geom_rbound', 'body_mass', 'body_ipos', 'body_iquat', 'body_inertia']
prev = None
saved = []
orig_write = env._write_data_to_sim
def write(advance=True):
    global prev
    orig_write(advance)
    if advance:   # state right before mj_step (ctrl written)
        prev = {f: getattr(d, f).clone() for f in FIELDS}
env._write_data_to_sim = write
orig = env._after_physics_step
def after():
    if prev is not None:
        v0 = prev['qvel'][:, dof:dof+3].norm(dim=-1)
        v1 = d.qvel[:, dof:dof+3].norm(dim=-1)
        for w in ((v1-v0) > 8.).nonzero().flatten().tolist():
            if len(saved) >= 12: break
            rec = {f: prev[f][w].cpu().numpy() for f in FIELDS}
            rec.update({'model_'+k: getattr(md, k)[w].cpu().numpy() if getattr(md, k).dim() > 1 and getattr(md, k).shape[0] == n else getattr(md, k).cpu().numpy() for k in MODEL})
            rec['post_qvel'] = d.qvel[w].cpu().numpy(); rec['post_qpos'] = d.qpos[w].cpu().numpy()
            rec['world'] = w; rec['step'] = env._sim_step_counter
            np.savez(f'{out}/eject_{len(saved):02d}.npz', **rec)
            saved.append((w, env._sim_step_counter, float(v0[w]), float(v1[w])))
            print('CAPTURED', saved[-1], flush=True)
    orig()
env._after_physics_step = after
with torch.inference_mode():
    for i in range(int(seconds/env.step_dt)):
        env.step(a)
        if len(saved) >= 12: break
import mujoco
mujoco.mj_saveModel(m, f'{out}/model.mjb', None)
print('DONE', len(saved), flush=True)
sys.stdout.flush(); os._exit(0)
