"""Hand-to-surface observation (opt-in: train_g1_sonic_sapg.py --hand-surface-obs).

For 17 points of the right Revo2 hand (palm centre, the five fingertip pads, the
five distal and six proximal/thumb-base link origins), the vector from the point
to the nearest point on the object's surface. Robot geometry comes from the
simulated link poses (forward kinematics of the URDF; on hardware: encoders +
URDF), object geometry from the per-world primitives (boxes / capsules of the
object bank), so the policy reads "this fingertip is 2 cm from the handle"
directly instead of composing it from joint angles, BPS and the object pose
(cf. RobustDexGrasp's finger-to-surface vectors, InterMimicGen's link-to-surface
vectors). The actor uses the same delayed/noised object pose as its other object
observations; the critic uses the true pose.

Encoding: world-axis vectors (as the other relative observations), in decimetres,
norm clipped at 3 (30 cm): 17 x 3 = 51 values, normalised by their own running
mean/std. Network columns for them start at zero (sapg_network/frozen_sapg), so
a policy expanded with them initially acts exactly as before.
"""
import torch

HAND_SURFACE_LINKS = ('right_index_proximal_link', 'right_middle_proximal_link', 'right_ring_proximal_link',
                      'right_pinky_proximal_link', 'right_thumb_metacarpal_link', 'right_thumb_proximal_link',
                      'right_index_distal_link', 'right_middle_distal_link', 'right_ring_distal_link',
                      'right_pinky_distal_link', 'right_thumb_distal_link')
HAND_SURFACE_POINTS = 1+5+len(HAND_SURFACE_LINKS)       # palm centre + fingertip pads + links
HAND_SURFACE_DIM = 3*HAND_SURFACE_POINTS
SCALE = 10.                                             # metres -> decimetres
CLIP = 3.                                               # 30 cm

MJ_GEOM_CAPSULE, MJ_GEOM_BOX = 3, 6


def quat_to_mat(q):
    """wxyz quaternions (..., 4) -> rotation matrices (..., 3, 3) (local -> world)."""
    q = q/q.norm(dim=-1, keepdim=True)
    w, x, y, z = q.unbind(-1)
    return torch.stack((1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
                        2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
                        2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)), -1).reshape(*q.shape[:-1], 3, 3)


def nearest_surface_vectors(points, centre, rot, size, is_box, used):
    """Vectors from points to the nearest surface point of a union of boxes/capsules.

    points (n, P, 3); centre (n, S, 3); rot (n, S, 3, 3) local->world; size (n, S, 3)
    (box: half extents; capsule: radius, half length along local z); is_box (S,) bool;
    used (n, S) bool. Points inside a primitive get the vector to its nearest face.
    Returns (n, P, 3).
    """
    local = torch.einsum('nsji,npsj->npsi', rot, points[:, :, None]-centre[:, None])     # R^T (p - c)
    h = size[:, None].expand_as(local)
    # Box: clamp outside; inside, push to the nearest face.
    q = torch.maximum(torch.minimum(local, h), -h)
    gap = h-local.abs()
    inside = (gap > 0).all(-1, keepdim=True)
    axis = torch.nn.functional.one_hot(gap.argmin(-1), 3).bool()
    sign = torch.where(local >= 0, 1., -1.)
    q = torch.where(inside & axis, sign*h, q)
    box = q-local
    # Capsule: nearest point on the axis segment, then out by the radius.
    r, half = size[:, None, :, 0], size[:, None, :, 1]
    a = torch.zeros_like(local)
    a[..., 2] = torch.maximum(torch.minimum(local[..., 2], half), -half)
    d = local-a
    n = d.norm(dim=-1, keepdim=True)
    fallback = torch.zeros_like(d)
    fallback[..., 0] = 1.
    unit = torch.where(n > 1e-9, d/n.clamp_min(1e-9), fallback)
    capsule = a+r[..., None]*unit-local
    v_local = torch.where(is_box[None, None, :, None], box, capsule)
    v = torch.einsum('nsij,npsj->npsi', rot, v_local)
    dist = v.norm(dim=-1).masked_fill(~used[:, None, :], float('inf'))
    best = dist.argmin(-1)
    return torch.gather(v, 2, best[..., None, None].expand(-1, -1, 1, 3)).squeeze(2)


def encode(vectors):
    """(n, P, 3) metres -> (n, 3P): decimetres, norm clipped at CLIP."""
    v = vectors*SCALE
    n = v.norm(dim=-1, keepdim=True)
    return (v*torch.clamp(CLIP/n.clamp_min(1e-9), max=1.)).flatten(1)


class HandSurface:
    """Per-env hand-to-surface observation of the mjlab whole-body env."""

    def __init__(self, env):
        from dextrah_lab.mj.compat import p2p
        from dextrah_lab.mj.scene import object_indices
        self.env, device = env, env.device
        self._offset = p2p('obs_utils')._apply_local_offset
        names = env.robot.body_names
        missing = [n for n in HAND_SURFACE_LINKS if n not in names]
        if missing:
            raise ValueError(f'Hand-surface links missing from the robot: {missing}')
        self.link_ids = torch.tensor([names.index(n) for n in HAND_SURFACE_LINKS], device=device)
        bank = env.object_bank
        t = lambda x: torch.as_tensor(x, device=device, dtype=torch.float32)
        self.pos, self.size = t(bank.pos), t(bank.size)                # link frame, per asset and slot
        self.rot = quat_to_mat(t(bank.quat))
        self.used = torch.as_tensor(bank.used, device=device, dtype=torch.bool)
        _, slots = object_indices(env.sim.mj_model)
        types = [int(env.sim.mj_model.geom_type[g]) for g in slots]
        if any(x not in (MJ_GEOM_BOX, MJ_GEOM_CAPSULE) for x in types):
            raise ValueError(f'Object slots must be boxes or capsules, got types {types}')
        self.is_box = torch.tensor([x == MJ_GEOM_BOX for x in types], device=device)

    def points(self):
        env = self.env
        origin = env.scene.env_origins[:, None]
        state = env.robot.data.body_state_w
        palm = state[:, env._palm_body_id]
        palm = self._offset(palm[:, None, :3], palm[:, None, 3:7], env._palm_center_offset, (env.num_envs, 1))
        tips = state[:, env._fingertip_body_ids]
        tips = self._offset(tips[:, :, :3], tips[:, :, 3:7], env._fingertip_offset, (env.num_envs, env._num_fingertips))
        links = state[:, self.link_ids, :3]
        return torch.cat((palm, tips, links), 1)-origin

    def vectors(self, obj_pos, obj_quat):
        """Nearest-surface vectors for an env-local object link pose (pos, wxyz quat)."""
        a = self.env._object_asset_index_per_env
        R = quat_to_mat(obj_quat)
        centre = obj_pos[:, None]+torch.einsum('nij,nsj->nsi', R, self.pos[a])
        rot = R[:, None] @ self.rot[a]
        return nearest_surface_vectors(self.points(), centre, rot, self.size[a], self.is_box, self.used[a])

    def observation(self, obj_pos, obj_quat):
        return encode(self.vectors(obj_pos, obj_quat))


def install_actor_pose_capture(obs_utils):
    """Keep the delayed/noised object pose the actor observes (computed inside
    obs_utils.build_observations) on the env, stamped with the physics step."""
    original = obs_utils._apply_object_state_dr
    if getattr(original, 'captures_actor_pose', False):
        return

    def capture(env, *args, **kwargs):
        out = original(env, *args, **kwargs)
        env._actor_object_pose = (out[0], out[1], env._sim_step_counter)
        return out
    capture.captures_actor_pose = True
    obs_utils._apply_object_state_dr = capture


# ---------------------------------------------------------------- weight surgery
def insert_zero_columns(weight, at, count):
    """(out, in) -> (out, in+count) with zero columns inserted before column ``at``."""
    return torch.cat((weight[:, :at], weight.new_zeros(weight.shape[0], count), weight[:, at:]), 1)


def expand_lstm_input(wrapper, at, count):
    """Replace wrapper.rnn (a 1-layer nn.LSTM) by one with ``count`` more inputs whose
    input weights are zero, inserted before input ``at``."""
    old = wrapper.rnn
    if not isinstance(old, torch.nn.LSTM) or old.num_layers != 1 or old.bidirectional:
        raise ValueError('Expected a single-layer unidirectional LSTM')
    new = torch.nn.LSTM(old.input_size+count, old.hidden_size, num_layers=1, batch_first=old.batch_first,
                        bias=old.bias).to(old.weight_ih_l0)
    with torch.no_grad():
        new.weight_ih_l0.copy_(insert_zero_columns(old.weight_ih_l0, at, count))
        for name in ('weight_hh_l0', 'bias_ih_l0', 'bias_hh_l0'):
            if hasattr(old, name):
                getattr(new, name).copy_(getattr(old, name))
    wrapper.rnn = new


def expand_state_dict(state, model_state, columns):
    """Load an older checkpoint into an expanded model: for each key in ``columns``
    (key -> (at, count)) insert zero input columns; keys only the expanded model has
    (the new normaliser) keep the expanded model's values. Returns (state, added keys)."""
    out, added = {}, []
    for key, value in model_state.items():
        if key not in state:
            out[key] = value
            added.append(key)
            continue
        old = state[key]
        if key in columns:
            old = insert_zero_columns(old, *columns[key])
        if old.shape != value.shape:
            raise ValueError(f'{key}: checkpoint {tuple(old.shape)} vs model {tuple(value.shape)}')
        out[key] = old
    unknown = sorted(set(state)-set(model_state))
    if unknown:
        raise ValueError(f'Checkpoint keys unknown to the expanded model: {unknown[:5]}')
    return out, added


def load_expanded_initial_weights(algo, checkpoint, hand_dim):
    """--init-weights from a checkpoint without hand-surface inputs into an expanded
    model: every trained tensor copied, zero input columns for the new inputs (actor
    task LSTM, critic first layer), fresh normalizer for them; fresh optimizers and
    counters (as carry_training.load_initial_weights)."""
    import hashlib
    from pathlib import Path
    from .task_contract import BODY_EXTRA_DIM, TASK_ACTOR_DIM, TASK_CRITIC_DIM
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    state = payload[0] if 0 in payload else payload
    actor, added_actor = expand_state_dict(
        state['model'], algo.model.state_dict(), {'a2c_network.source.rnn.rnn.weight_ih_l0': (TASK_ACTOR_DIM, hand_dim)})
    algo.model.load_state_dict(actor, strict=True)
    critic, added_critic = expand_state_dict(
        state['assymetric_vf_nets'], algo.central_value_net.state_dict(),
        {'model.a2c_network.actor_mlp.0.weight': (TASK_CRITIC_DIM+BODY_EXTRA_DIM, hand_dim)})
    algo.central_value_net.load_state_dict(critic, strict=True)
    with Path(checkpoint).open('rb') as f:
        digest = hashlib.file_digest(f, 'sha256').hexdigest()
    return dict(checkpoint=str(Path(checkpoint).resolve()), checkpoint_sha256=digest,
                source_epoch=int(state.get('epoch', -1)), source_frame=int(state.get('frame', -1)),
                optimizers='fresh', epoch_and_frame='start at zero', hand_surface_dim=hand_dim,
                zero_columns=dict(actor_lstm_input=[TASK_ACTOR_DIM, TASK_ACTOR_DIM+hand_dim],
                                  critic_first_layer=[TASK_CRITIC_DIM+BODY_EXTRA_DIM, TASK_CRITIC_DIM+BODY_EXTRA_DIM+hand_dim]),
                new_tensors=added_actor+['critic.'+k for k in added_critic])


@torch.no_grad()
def check_zero_hand_columns(model, critic, obs_dim, state_dim, hand_dim, device, n=64):
    """The expanded actor and critic ignore the hand inputs at initialization: same
    outputs for zero and random hand-surface values."""
    model.eval()
    critic.eval()
    try:
        obs = torch.randn(n, obs_dim+1, device=device)
        obs[:, -1] = 0.                                   # SAPG leader coefficient
        alt = obs.clone()
        alt[:, obs_dim-hand_dim:obs_dim] = torch.randn(n, hand_dim, device=device)*3
        states = tuple(s[:, :n].to(device) for s in model.get_default_rnn_state())
        a = model(dict(obs=obs, is_train=False, prev_actions=None, rnn_states=states))
        b = model(dict(obs=alt, is_train=False, prev_actions=None, rnn_states=states))
        st = torch.randn(n, state_dim+1, device=device)
        st[:, -1] = 0.
        st_alt = st.clone()
        st_alt[:, state_dim-hand_dim:state_dim] = torch.randn(n, hand_dim, device=device)*3
        va, vb = critic(dict(obs=st, is_train=False)), critic(dict(obs=st_alt, is_train=False))
        va = va['values'] if isinstance(va, dict) else va
        vb = vb['values'] if isinstance(vb, dict) else vb
        report = dict(actor_mus_max_abs_diff=float((a['mus']-b['mus']).abs().max()),
                      actor_values_max_abs_diff=float((a['values']-b['values']).abs().max()),
                      critic_values_max_abs_diff=float((va-vb).abs().max()))
    finally:
        model.train()
        critic.train()
    if max(report.values()) > 1e-5:
        raise RuntimeError(f'Hand-surface inputs are not inert at initialization: {report}')
    return report
