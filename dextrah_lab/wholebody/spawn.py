"""Object spawn on the table: no table intersection, whole footprint on the table.

Play2Perfect spawns the object origin 0.25 m above the table centre (10 cm
above the top) with a uniformly random 3D orientation, in an XY box only
4-14 cm from the robot-side edge. Measured on Isaac (spawn checks 1111/1112):
22% of spawns start intersecting the table (median 3 cm, up to 16 cm deep)
and 31% fall off with no fingertip contact (72% over the robot-side edge).

Applied after the source reset, keeping its random orientation:
* XY re-sampled uniformly from the source spawn box intersected with the
  centre positions whose oriented footprint lies `margin` inside the table;
* height set so the object's lowest point is `clearance` above the top.
Simulator independent (torch); used identically by the Isaac and mjlab envs.
"""
import torch

TABLE_HALF = (0.475/2, 0.4/2, 0.3/2)   # play2perfect/assets/urdf/table_narrow.urdf


class TableSpawn:
    def __init__(self, urdf_paths, asset_per_env, device, *, clearance=0.01, margin=0.01):
        from dextrah_lab.mj.objects import build_object_bank
        bank = build_object_bank(urdf_paths)
        points, radius = [], []
        for i in range(len(urdf_paths)):
            p, r = [], []
            for s in range(4):
                if not bank.used[i, s]:
                    continue
                q = torch.tensor(bank.quat[i, s], dtype=torch.float32)
                if s % 2 == 0:   # box: 8 corners
                    h = bank.size[i, s]
                    local = torch.tensor([[x, y, z] for x in (-h[0], h[0]) for y in (-h[1], h[1]) for z in (-h[2], h[2])],
                                         dtype=torch.float32)
                    rad = 0.
                else:            # capsule: 2 segment end points, swept by radius
                    local = torch.tensor([[0., 0., -bank.size[i, s, 1]], [0., 0., bank.size[i, s, 1]]], dtype=torch.float32)
                    rad = float(bank.size[i, s, 0])
                world = _rotate(q.expand(len(local), 4), local)+torch.tensor(bank.pos[i, s], dtype=torch.float32)
                p.append(world)
                r += [rad]*len(local)
            points.append(torch.cat(p))
            radius.append(torch.tensor(r))
        size = max(len(p) for p in points)
        pad = lambda t, v: torch.cat((t, t[:1].expand(size-len(t), *t.shape[1:]) if v is None else torch.full((size-len(t),), v)))
        self.points = torch.stack([pad(p, None) for p in points]).to(device)        # (assets, P, 3), body frame
        self.radius = torch.stack([pad(r, 0.) for r in radius]).to(device)         # (assets, P)
        self.asset = asset_per_env.to(device)
        self.clearance, self.margin = clearance, margin

    def place(self, quat, table_z, spawn_min, spawn_max, env_ids, assets=None):
        """Return env-local object positions (n, 3) for the given orientations
        (objects: the envs' assigned assets, or ``assets`` when given)."""
        assets = self.asset[env_ids] if assets is None else assets
        pts = self.points[assets]
        n, k, _ = pts.shape
        world = _rotate(quat[:, None].expand(n, k, 4).reshape(-1, 4), pts.reshape(-1, 3)).reshape(n, k, 3)
        r = self.radius[assets][..., None]
        low, high = (world-r).amin(1), (world+r).amax(1)                          # extents about the origin
        hx, hy, hz = TABLE_HALF
        lo = torch.stack((-hx+self.margin-low[:, 0], -hy+self.margin-low[:, 1]), -1)
        hi = torch.stack((hx-self.margin-high[:, 0], hy-self.margin-high[:, 1]), -1)
        smin = torch.as_tensor(spawn_min, device=quat.device, dtype=quat.dtype)
        smax = torch.as_tensor(spawn_max, device=quat.device, dtype=quat.dtype)
        a, b = torch.maximum(lo, smin), torch.minimum(hi, smax)
        # Spawn box and on-table set disjoint (long object near the robot edge):
        # sample the on-table set's part closest to the spawn box instead.
        empty = a > b
        a = torch.where(empty, torch.minimum(torch.maximum(smin, lo), hi), a)
        b = torch.where(empty, torch.maximum(torch.minimum(smax, hi), lo), b)
        b = torch.maximum(a, b)
        xy = a+(b-a)*torch.rand_like(a)
        z = table_z+hz+self.clearance-low[:, 2]
        return torch.cat((xy, z[:, None]), -1)


def _rotate(q, v):
    """Rotate vectors v by wxyz quaternions q (same leading shape)."""
    xyz = q[..., 1:]
    t = 2*torch.linalg.cross(xyz, v)
    return v+q[..., :1]*t+torch.linalg.cross(xyz, t)


def apply_table_spawn(env, env_ids):
    """Re-place the objects just reset by the source task (orientation kept)."""
    if env_ids.numel() == 0:
        return
    origins = env.scene.env_origins[env_ids]
    quat = env.object.data.root_quat_w[env_ids].clone()
    reset = env.cfg.reset
    pos = env._table_spawn.place(quat, env._table_z_per_env[env_ids],
                                 reset.object_spawn_xy_mins, reset.object_spawn_xy_maxs, env_ids)
    env.object.write_root_pose_to_sim(torch.cat((pos+origins, quat), -1), env_ids=env_ids)
    env.object.write_root_velocity_to_sim(torch.zeros(len(env_ids), 6, device=pos.device), env_ids=env_ids)
    env._object_init_z[env_ids] = pos[:, 2]
