import numpy as np
import pytest
import trimesh
from dextrah_lab.object_shape.bank import build_bank, load_primitive_urdf, map_features, mesh_features
from dextrah_lab.object_shape.grail.compute_bps import compute_bps, fibonacci_sphere, normalize_points


def urdf(path, geometry='<box size=".2 .04 .06"/>', origin='xyz=".03 -.02 .08"'):
    path.write_text(f'<robot name="test"><link name="root"><collision><origin {origin}/>'
                    f'<geometry>{geometry}</geometry></collision></link></robot>')
    return path


def test_primitive_frame_and_cylinder_rotation(tmp_path):
    box = load_primitive_urdf(urdf(tmp_path / "box.urdf"))
    np.testing.assert_allclose(box.bounds, [[-.07, -.04, .05], [.13, 0., .11]], atol=1e-8)
    cylinder = load_primitive_urdf(urdf(tmp_path / "cyl.urdf",
        '<cylinder radius=".02" length=".2"/>', 'rpy="0 -1.5707963267948966 0"'))
    np.testing.assert_allclose(cylinder.extents, [.2, .04, .04], atol=1e-8)


def test_fast_bank_matches_grail_and_keeps_rng():
    mesh = trimesh.creation.box([.2, .04, .06])
    before = np.random.get_state()
    features = mesh_features(mesh)
    after = np.random.get_state()
    for a, b in zip(before, after):
        np.testing.assert_array_equal(a, b)
    cloud = trimesh.sample.sample_surface(mesh, 16384, seed=42)[0].astype(np.float32)
    normalized, center, radius = normalize_points(cloud)
    expected = compute_bps(normalized, fibonacci_sphere(128).astype(np.float32))
    np.testing.assert_allclose(features[:128], expected, atol=1e-7)
    np.testing.assert_array_equal(features[128:131], center)
    assert features[-1] == radius
    scaled = mesh.copy()
    scaled.apply_scale(2)
    doubled = mesh_features(scaled)
    np.testing.assert_allclose(doubled[:128], features[:128], atol=1e-6)
    np.testing.assert_allclose(doubled[128:], features[128:]*2, atol=1e-6)


def test_cache_content_identity_and_shuffled_mapping(tmp_path):
    paths = [urdf(tmp_path / "a.urdf"), urdf(tmp_path / "b.urdf", '<box size=".05 .1 .3"/>')]
    bank, meta = build_bank(paths, tmp_path / "cache")
    cached, _ = build_bank(paths, tmp_path / "cache")
    np.testing.assert_array_equal(bank, cached)
    np.testing.assert_array_equal(map_features(bank, [1, 0, 1, 1]), bank[[1, 0, 1, 1]])
    assert not np.allclose(bank[0], bank[1])
    urdf(paths[0], '<box size=".1 .1 .1"/>')
    changed, new_meta = build_bank(paths, tmp_path / "cache")
    assert new_meta["entries"][0]["cache_key"] != meta["entries"][0]["cache_key"]
    assert not np.allclose(changed[0], bank[0])


@pytest.mark.parametrize("indices", [[-1], [2], [0.0], [[0]]])
def test_bad_indices_rejected(indices):
    with pytest.raises(ValueError):
        map_features(np.zeros((2, 132)), indices)


def test_unsupported_geometry_rejected(tmp_path):
    with pytest.raises(ValueError):
        load_primitive_urdf(urdf(tmp_path / "bad.urdf", '<sphere radius=".02"/>'))
