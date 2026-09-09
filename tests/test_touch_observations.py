import pytest
import torch
from dextrah_lab.g1_adept.touch_observations import TouchObservationConfig, TouchObservationModel


def test_rate_hold_release_and_partial_reset():
    m = TouchObservationModel(2, 1/120, 'cpu', TouchObservationConfig(publish_hz=10))
    f = torch.ones(2, 5, 3)
    for _ in range(11):
        m.advance(f)
        assert not m.valid.any()
    m.advance(f)
    assert m.valid.all()
    before = m.force.clone()
    for _ in range(11):
        m.advance(f*2)
        torch.testing.assert_close(m.force, before)
    m.advance(f*2)
    assert m.force[0, 0, 0] == 2
    m.reset(torch.tensor([0]))
    assert not m.valid[0].any() and m.valid[1].all()
    assert not m.observation()[0].count_nonzero()
    for _ in range(24):
        m.advance(torch.zeros_like(f))
    assert not m.force.count_nonzero()
    assert m.observation().shape == (2, 25)


def test_70hz_acquisition_delivery_with_120hz_physics_and_60hz_policy():
    model = TouchObservationModel(2, 1/120, 'cpu')
    packets, policy_samples, max_age = [], [], 0.
    previous = 0
    for tick in range(1, 1201):
        model.advance(torch.full((2, 5, 3), tick / 100.))
        if model.publication_count != previous:
            packets.append(tick)
        previous = model.publication_count
        if tick % 2 == 0:
            assert model.valid.all()
            policy_samples.append(float(model.force[0, 0, 0]))
            max_age = max(max_age, float(model.age_s.max()))
    assert model.acquisition_count == model.publication_count == 700
    assert len(set(policy_samples)) == 600  # fresh value at every policy tick
    assert set(b-a for a,b in zip(packets, packets[1:])) == {1, 2}
    assert max_age <= 1/120 + 1e-6
    before = (model.steps, model.acquisition_count, model.publication_count)
    model.reset(torch.tensor([0]))
    assert not model.valid[0].any() and model.valid[1].all()
    assert (model.steps, model.acquisition_count, model.publication_count) == before


def test_delay_invalid_direction_quantization():
    cfg = TouchObservationConfig(sensor_hz=120, publish_hz=120, latency_s=2/120)
    m = TouchObservationModel(1, 1/120, 'cpu', cfg)
    f = torch.tensor([1.234, 2., -.01]).expand(1, 5, 3)
    m.advance(f)
    m.advance(f)
    assert not m.valid.any()
    m.advance(f)
    assert m.valid.all()
    torch.testing.assert_close(m.force[..., 0], torch.full((1, 5), 1.23))
    torch.testing.assert_close(m.age_s, torch.full((1, 5), 2/120))
    for _ in range(3):
        m.advance(torch.full_like(f, float('nan')))
    assert not m.valid.any()
    assert not m.observation()[:, :3].count_nonzero()
    assert torch.isfinite(m.observation()).all()


@pytest.mark.parametrize('touch,torque,dim', [(False,False,0),(True,False,25),(False,True,7),(True,True,32)])
def test_independent_options(touch, torque, dim):
    m = TouchObservationModel(2, 1/120, 'cpu', TouchObservationConfig(enabled=touch, arm_torques=torque))
    obs = m.observation(torch.ones(2, 7)*40)
    assert obs.shape == (2, dim)
    if torque:
        torch.testing.assert_close(obs[:, -7:], torch.ones(2, 7))
        with pytest.raises(ValueError):
            m.observation()


def test_bad_rates_fail_and_overrange_not_silently_clipped():
    with pytest.raises(ValueError):
        TouchObservationModel(1, 1/60, 'cpu')
    m = TouchObservationModel(1, 1/120, 'cpu')
    for _ in range(12):
        m.advance(torch.tensor([30., -30., 0.]).expand(1, 5, 3))
    assert m.force[..., 0].min() > 25
    assert m.force[..., 1].max() < -25


def test_identity_model_does_not_consume_random_numbers():
    before = torch.random.get_rng_state().clone()
    model = TouchObservationModel(2, 1/120, 'cpu')
    for _ in range(24):
        model.advance(torch.zeros(2,5,3))
    torch.testing.assert_close(torch.random.get_rng_state(), before)


def test_noise_dropout_and_reset_never_leak_other_environment():
    cfg = TouchObservationConfig(noise_std_n=.03, gain_range=(.9,1.1),
                                 bias_std_n=.01, dropout_probability=1.)
    model = TouchObservationModel(2,1/120,'cpu',cfg)
    for _ in range(24):
        model.advance(torch.ones(2,5,3))
    assert not model.valid.any()
    other_gain = model.gain[1].clone()
    model.reset(torch.tensor([0]))
    torch.testing.assert_close(model.gain[1],other_gain)
    assert torch.isfinite(model.observation()).all()


def test_center_ray_and_single_display_frame():
    import numpy as np
    import trimesh
    from dextrah_lab.g1_adept.touch_frames import ray_surface_point, pose_matrix, display_pose
    mesh = trimesh.creation.box(extents=(.02,.04,.06))
    point,normal = ray_surface_point(mesh,np.array([.1,0.,0.]),np.array([-1.,0.,0.]))
    np.testing.assert_allclose(point,[.01,0.,0.],atol=1e-7)
    np.testing.assert_allclose(normal,[1.,0.,0.])
    with pytest.raises(ValueError,match='missed'):
        ray_surface_point(mesh,np.array([.1,.2,0.]),np.array([-1.,0.,0.]))
    frame=pose_matrix([1.,2.,3.],[1.,0.,0.,0.])
    p,q=display_pose(np.linalg.inv(frame),[1.,2.,3.],[1.,0.,0.,0.])
    np.testing.assert_allclose(p,0.)
    np.testing.assert_allclose(q,[1.,0.,0.,0.])


@pytest.mark.parametrize('base', [224, 246])
def test_tactile_checkpoint_columns_preserve_sapg_embedding(base):
    from dextrah_lab.object_shape.warmstart import expand_state_dict
    prefix = '' if base == 224 else 'model.'
    weight = prefix + ('a2c_network.rnn.rnn.weight_ih_l0' if base == 224 else 'a2c_network.actor_mlp.0.weight')
    mean, var = prefix+'running_mean_std.running_mean', prefix+'running_mean_std.running_var'
    old = {mean: torch.zeros(base), var: torch.ones(base), weight: torch.randn(8, base+32)}
    target = {mean: torch.zeros(base+32), var: torch.ones(base+32), weight: torch.zeros(8, base+64)}
    expanded = expand_state_dict(old, target, base, torch.zeros(32), torch.ones(32))
    x = torch.randn(4, base+32)
    xx = torch.cat((x[:, :base], torch.randn(4, 32), x[:, base:]), -1)
    torch.testing.assert_close(x@old[weight].T, xx@expanded[weight].T, atol=2e-5, rtol=1e-5)
