import sys, torch, numpy as np
sys.path.insert(0, '/data1/users/konstantin.smirnov/DEXTRAH-G1-WholeBody')
import warp as wp
from dextrah_lab.mj.factory import make_mj_env
dev = sys.argv[1]; W = int(sys.argv[2]); n = int(sys.argv[3])
env = make_mj_env(n, dev)
m = env.sim.mj_model
env.reset()
a = torch.zeros(n, 70, device=dev); a[:, 64:] = -1
d = env.sim.wp_data
gname = lambda g: m.geom(int(g)).name or f'geom{int(g)}({m.body(int(m.geom_bodyid[g])).name})'
orig = env._after_physics_step
def after():
    o = env.object.data
    nacon = int(wp.to_torch(d.nacon)[0])
    world = wp.to_torch(d.contact.worldid)[:nacon]
    idx = (world == W).nonzero().flatten()
    geom = wp.to_torch(d.contact.geom)[idx].cpu().numpy()
    dist = wp.to_torch(d.contact.dist)[idx].cpu().numpy()
    pairs = [f'{gname(g0)}|{gname(g1)}:{dd*1000:.1f}mm' for (g0, g1), dd in zip(geom, dist)]
    print(f'step {env._sim_step_counter} obj pos {o.root_pos_w[W].cpu().numpy().round(3)} linvel {o.root_lin_vel_w[W].norm().item():.2f} angvel {o.root_ang_vel_w[W].norm().item():.1f} '
          f'palm {env.robot.data.body_pos_w[W, env._palm_body_id].cpu().numpy().round(3)} contacts {pairs}', flush=True)
    orig()
env._after_physics_step = after
print('table z', env._table_z_per_env[W].item(), 'object init z', env._object_init_z[W].item())
for i in range(6):
    try: env.step(a)
    except RuntimeError as e: print('stopped', e); break
