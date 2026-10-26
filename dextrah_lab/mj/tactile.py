"""Fingertip pad forces and body-table forces from the MuJoCo Warp contact buffer.

Equivalent of the PhysX path (`g1_revo2_touch_env.capture_touch` and the
body-table ContactSensors), computed in one pass over all active contacts:

* fingertip channel: contacts between a right distal-link geom and the
  object/table/ground (the Isaac filter partners). The force ON the distal
  link is split into normal and friction parts; each contact is kept only if
  its point lies inside that finger's CAD pad window (same `pad_tensors`).
* body-table channel: the normal force of the table on each monitored link,
  summed as vectors per link, then its norm (as `force_matrix_w.norm()`).

MuJoCo convention: contact frame row 0 is the normal from geom[0] to geom[1];
the solver force in that frame acts on geom[1] (minus on geom[0]).
No host synchronisation: inactive slots are masked, never compacted.
"""
import torch

OBJECT, TABLE, GROUND, ROBOT, OTHER = 0, 1, 2, 3, 4


def _quat_rotate_inverse(q, v):
    xyz = -q[..., 1:]
    t = 2*torch.linalg.cross(xyz, v)
    return v+q[..., :1]*t+torch.linalg.cross(xyz, t)


class ContactReader:
    def __init__(self, sim, mj_model, *, finger_bodies, monitored_bodies, object_body,
                 table_body, ground_geom, robot_bodies, pad_tensors, palm_body=None):
        import mujoco_warp as mjwarp
        import warp as wp
        self.sim, self.n = sim, sim.num_envs
        device = sim.device
        self.device = device
        ngeom = mj_model.ngeom
        body = torch.as_tensor(mj_model.geom_bodyid, device=device, dtype=torch.long)
        self.partner = torch.full((ngeom,), OTHER, dtype=torch.long, device=device)
        self.partner[body == object_body] = OBJECT
        self.partner[body == table_body] = TABLE
        self.partner[ground_geom] = GROUND
        self.partner[torch.isin(body, torch.as_tensor(robot_bodies, device=device))] = ROBOT
        self.finger = torch.full((ngeom,), -1, dtype=torch.long, device=device)
        for i, b in enumerate(finger_bodies):
            self.finger[body == b] = i
        self.link = torch.full((ngeom,), -1, dtype=torch.long, device=device)
        for i, b in enumerate(monitored_bodies):
            self.link[body == b] = i
        self.finger_bodies = torch.as_tensor(finger_bodies, device=device)
        # Opt-in (grasp reward): object-only contact magnitudes of the distal links and palm.
        self.palm = None if palm_body is None else (body == palm_body)
        self.num_links = len(monitored_bodies)
        self.pad_origin = torch.stack([p[0] for p in pad_tensors])      # (5, 3)
        self.pad_rotation = torch.stack([p[1] for p in pad_tensors])    # (5, 3, 3)
        self.pad_bounds = torch.stack([p[2] for p in pad_tensors])      # (5, 2, 3)
        d = sim.wp_data
        self.naconmax = d.naconmax
        self._ids = wp.array(list(range(self.naconmax)), dtype=int, device=str(device))
        self._force = wp.zeros(self.naconmax, dtype=wp.spatial_vector, device=str(device))
        self._contact_force = lambda: mjwarp.contact_force(sim.wp_model, sim.wp_data, self._ids, True, self._force)
        self._mjwarp = mjwarp
        self._wp = wp

    active_only = False   # opt-in (env.enable_fast_step): process the first nacon contacts only
    object_table = False  # opt-in (evaluation): per-env object-table normal force, out['object_table_force']

    def read(self):
        """Return per-env fingertip and body-table force tensors for the last step."""
        wp = self._wp
        nacon = wp.to_torch(self.sim.wp_data.nacon)
        # Active-only: same forces; inactive slots (all routed to world 0 slot 0 below)
        # dominate the atomics otherwise. Costs one host read of nacon.
        k = min(int(nacon[0]), self.naconmax) if self.active_only else self.naconmax
        self._force.zero_()
        with wp.ScopedDevice(str(self.device)):
            if self.active_only:
                self._mjwarp.contact_force(self.sim.wp_model, self.sim.wp_data, self._ids[:k], True, self._force[:k])
            else:
                self._contact_force()
        c = self.sim.wp_data.contact
        force = wp.to_torch(self._force)[:k, :3]                    # world, acts on geom[1]
        frame = wp.to_torch(c.frame)[:k]                              # (C, 3, 3)
        pos = wp.to_torch(c.pos)[:k]
        geom = wp.to_torch(c.geom)[:k].long()
        world = wp.to_torch(c.worldid)[:k].long()
        active = torch.arange(k, device=self.device) < nacon
        active &= (wp.to_torch(c.type)[:k] & 1) > 0
        # Inactive slots hold stale or uninitialised ids: never index with them.
        ngeom = self.partner.numel()
        active &= (geom >= 0).all(-1) & (geom < ngeom).all(-1) & (world >= 0) & (world < self.n)
        world = torch.where(active, world, 0)
        g0 = torch.where(active, geom[:, 0], 0)
        g1 = torch.where(active, geom[:, 1], 0)
        normal = frame[:, 0]
        fn = (force*normal).sum(-1, keepdim=True)*normal             # normal part on geom[1]
        ff = force-fn                                                 # friction part on geom[1]

        out = {}
        # ---- fingertips -----------------------------------------------------
        sensor_partner = lambda g: (self.partner[g] == OBJECT) | (self.partner[g] == TABLE) | (self.partner[g] == GROUND)
        on1 = active & (self.finger[g1] >= 0) & sensor_partner(g0)
        on0 = active & (self.finger[g0] >= 0) & sensor_partner(g1)
        sign = on1.float()-on0.float()                                # force ON the finger
        finger = torch.where(on1, self.finger[g1], self.finger[g0]).clamp_min(0)
        hit = on1 | on0
        body_pos = self.sim.data.xpos[world, self.finger_bodies[finger]]
        body_quat = self.sim.data.xquat[world, self.finger_bodies[finger]]
        local = _quat_rotate_inverse(body_quat, pos-body_pos)
        in_pad = torch.einsum('ci,cij->cj', local-self.pad_origin[finger], self.pad_rotation[finger])
        bounds = self.pad_bounds[finger]
        accepted = hit & ((in_pad >= bounds[:, 0]) & (in_pad <= bounds[:, 1])).all(-1)
        slot = world*5+finger
        def add(values, mask):
            acc = torch.zeros(self.n*5, values.shape[-1], device=self.device)
            acc.index_add_(0, slot, values*(mask.float()*sign)[:, None])
            return acc.view(self.n, 5, -1)
        out['normal_masked_w'] = add(fn, accepted)
        out['friction_masked_w'] = add(ff, accepted)
        out['normal_all_w'] = add(fn, hit)
        out['friction_all_w'] = add(ff, hit)
        load = (fn.norm(dim=-1)*accepted.float())
        weight = torch.zeros(self.n*5, device=self.device).index_add_(0, slot, load)
        centroid = torch.zeros(self.n*5, 3, device=self.device).index_add_(0, slot, pos*load[:, None])
        out['centroid_w'] = (centroid/weight.clamp_min(1e-12)[:, None]).view(self.n, 5, 3)
        out['finger_contacts'] = torch.zeros(self.n*5, device=self.device).index_add_(0, slot, hit.float()).view(self.n, 5)
        # ---- object-only contact magnitudes (grasp reward) -------------------
        if self.palm is not None:
            obj0, obj1 = self.partner[g0] == OBJECT, self.partner[g1] == OBJECT
            f1 = active & (self.finger[g1] >= 0) & obj0
            f0 = active & (self.finger[g0] >= 0) & obj1
            fing = torch.where(f1, self.finger[g1], self.finger[g0]).clamp_min(0)
            mag = force.abs().sum(-1)
            out['finger_object_force_l1'] = torch.zeros(self.n*5, device=self.device).index_add_(
                0, world*5+fing, mag*(f1 | f0).float()).view(self.n, 5)
            palm = active & ((self.palm[g1] & obj0) | (self.palm[g0] & obj1))
            out['palm_object_force_l1'] = torch.zeros(self.n, device=self.device).index_add_(0, world, mag*palm.float())
        # ---- body-table -----------------------------------------------------
        if self.num_links:
            l1 = active & (self.link[g1] >= 0) & (self.partner[g0] == TABLE)
            l0 = active & (self.link[g0] >= 0) & (self.partner[g1] == TABLE)
            link = torch.where(l1, self.link[g1], self.link[g0]).clamp_min(0)
            s = l1.float()-l0.float()
            acc = torch.zeros(self.n*self.num_links, 3, device=self.device)
            acc.index_add_(0, world*self.num_links+link, fn*s[:, None])
            out['body_table_force'] = acc.view(self.n, self.num_links, 3).norm(dim=-1)
        # ---- object-table (opt-in: placement support) ------------------------
        if self.object_table:
            ot = active & (((self.partner[g0] == OBJECT) & (self.partner[g1] == TABLE))
                           | ((self.partner[g0] == TABLE) & (self.partner[g1] == OBJECT)))
            out['object_table_force'] = torch.zeros(self.n, device=self.device).index_add_(
                0, world, fn.norm(dim=-1)*ot.float())
        out['active_contacts'] = nacon
        return out
