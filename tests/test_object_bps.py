"""CPU geometry/encoding tests; no Isaac Lab or robot dependencies."""
import numpy as np
import pytest
import trimesh

from dextrah_lab.object_shape.bps import (
    encode_mesh, encode_points, load_object_mesh, validate_representation,
)
from dextrah_lab.object_shape.grail import compute_bps as upstream


def test_exact_grail_functions_and_correspondences():
    cloud = np.random.default_rng(5).normal(size=(500, 3)).astype(np.float32) * .08
    actual = encode_points(cloud)
    normalized, centroid, radius = upstream.normalize_points(cloud)
    basis = upstream.fibonacci_sphere(128).astype(np.float32)
    np.testing.assert_array_equal(actual.basis, basis)
    np.testing.assert_array_equal(actual.distances, upstream.compute_bps(normalized, basis))
    np.testing.assert_array_equal(actual.centroid_m, centroid)
    assert actual.scale_m == radius
    np.testing.assert_allclose(np.linalg.norm(actual.basis, axis=1), 1., atol=1e-7)
    assert actual.distances.shape == (128,)
    np.testing.assert_allclose(np.linalg.norm(actual.basis-actual.nearest_normalized, axis=1),
                               actual.distances, atol=1e-7)
    np.testing.assert_allclose(np.linalg.norm(actual.basis_m-actual.nearest_m, axis=1),
                               actual.distances*actual.scale_m, atol=1e-7)


def test_translation_scale_and_sample_order_invariance():
    cloud = np.random.default_rng(6).normal(size=(300, 3)).astype(np.float32) * .08
    baseline = encode_points(cloud)
    transformed = encode_points(cloud*3. + np.array([.3, -.2, .9], np.float32))
    shuffled = encode_points(cloud[::-1])
    np.testing.assert_allclose(transformed.distances, baseline.distances, atol=2e-6)
    np.testing.assert_allclose(shuffled.distances, baseline.distances, atol=2e-6)
    assert transformed.scale_m == pytest.approx(3*baseline.scale_m, abs=2e-6)


def test_basis_does_not_depend_on_object_and_rotation_is_not_removed():
    mesh = trimesh.creation.box([.19, .08, .03])
    one = encode_mesh(mesh, surface_count=2000)
    two = encode_points(one.surface_m[:, [1, 2, 0]])
    np.testing.assert_array_equal(one.basis, two.basis)
    assert np.max(np.abs(one.distances-two.distances)) > .1


def test_sampling_uses_faces_is_repeatable_and_does_not_change_global_rng():
    mesh = trimesh.creation.box([.19, .08, .03])
    state = np.random.get_state()
    one = encode_mesh(mesh, surface_count=2000, seed=42)
    two = encode_mesh(mesh, surface_count=2000, seed=42)
    after = np.random.get_state()
    for before_value, after_value in zip(state, after):
        np.testing.assert_array_equal(before_value, after_value)
    np.testing.assert_array_equal(one.surface_m, two.surface_m)
    np.testing.assert_array_equal(one.distances, two.distances)
    assert len(one.surface_m) == 2000  # NOT eight corner vertices
    assert np.min(np.linalg.norm(one.surface_m[:, None]-mesh.vertices[None], axis=-1)) > 1e-6
    report = validate_representation(mesh, one)
    assert report['max_endpoint_off_mesh_m'] < 1e-7
    assert report['max_sampling_distance_error_m'] < .01


def test_scene_flattening_preserves_instances_and_nested_transforms(tmp_path):
    scene = trimesh.Scene()
    scene.graph.update(frame_to="parent", matrix=trimesh.transformations.translation_matrix([.4, .2, -.1]))
    transform = trimesh.transformations.rotation_matrix(.45, [0, 0, 1])
    transform[:3, 3] = [-.2, .1, .08]
    scene.add_geometry(trimesh.creation.box([.10, .02, .04]), node_name="one",
                       geom_name="shared", parent_node_name="parent", transform=transform)
    scene.graph.update(frame_to="two", matrix=np.eye(4), geometry="shared")
    path = tmp_path / "transformed.glb"
    scene.export(str(path))
    flattened = load_object_mesh(path)
    np.testing.assert_allclose(flattened.bounds, scene.bounds, atol=2e-8)
    assert len(flattened.faces) == 24


@pytest.mark.parametrize("cloud", [np.zeros((3, 3)), np.ones((1, 3)),
                                     np.zeros((3, 2)), np.array([[0, 0, 0], [1, np.nan, 0]])])
def test_invalid_clouds_are_rejected(cloud):
    with pytest.raises(ValueError):
        encode_points(cloud)


def test_invalid_counts_are_rejected():
    with pytest.raises(ValueError):
        encode_points(np.eye(3), num_basis=0)
    with pytest.raises(ValueError):
        encode_mesh(trimesh.creation.box(), surface_count=1)
