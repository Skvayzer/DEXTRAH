from dataclasses import asdict

import numpy as np
import pytest
import torch

from dextrah_lab.g1_adept.collision_geometry import (
    G1_REVO2_DYNAMIC_SPHERES, G1_REVO2_FIXED_SPHERES,
    G1_REVO2_SELF_COLLISION_PAIRS, build_g1_collision_batch,
)
from scripts.fabric_recording_geometry import rotation, sphere_clearances, verify_sphere_centers


def geometry(include_self=True):
    return dict(dynamic=[asdict(s) for s in G1_REVO2_DYNAMIC_SPHERES],
                fixed=[asdict(s) for s in G1_REVO2_FIXED_SPHERES],
                self_pairs=G1_REVO2_SELF_COLLISION_PAIRS,
                include_self_collision=include_self)


@pytest.mark.parametrize("include_self", [True, False])
def test_renderer_clearance_order_matches_training(include_self):
    rng = np.random.default_rng(12)
    moving, fixed = rng.normal(size=(14, 3)), rng.normal(size=(7, 3))
    gaps, moving_min, fixed_min = sphere_clearances(geometry(include_self), moving, fixed, .8)
    runtime = build_g1_collision_batch(torch.tensor(moving)[None],
        torch.zeros(1, 14, 3, 13, dtype=torch.float64), torch.tensor(fixed)[None],
        table_height=.8, include_self_collision=include_self)
    np.testing.assert_allclose(gaps, runtime.clearance.numpy()[0], atol=1.e-12)
    assert gaps.size == (125 if include_self else 97)
    assert min(moving_min) == min(gaps)
    assert np.isfinite(fixed_min).all()


def test_disabled_body_and_table_pairs_are_not_highlighted():
    moving = np.zeros((14, 3))
    moving[:, 0] = np.arange(14)
    moving[0, 2] = -1.
    fixed = np.ones((7, 3)) * 10.
    fixed[0] = moving[0]
    _, moving_min, _ = sphere_clearances(geometry(False), moving, fixed, 0.)
    assert moving_min[0] == np.inf  # adjacent upper arm excluded by training
    assert moving_min[1] < 0.  # upper-arm second sample does avoid the table


def test_sphere_centers_preserve_link_and_root_transforms():
    spec = geometry()
    names = list(dict.fromkeys(s["link_name"] for s in spec["dynamic"]))
    body_pos = np.random.default_rng(7).normal(size=(len(names), 3))
    quat = np.tile([.5, .5, .5, .5], (len(names), 1))
    root = np.array([.3, -.2, 1., .5, .5, .5, .5])
    moving = np.stack([body_pos[names.index(s["link_name"])] + rotation(quat[0]) @ s["offset"]
                       for s in spec["dynamic"]])
    fixed = np.array([s["center"] for s in spec["fixed"]]) @ rotation(root[3:]).T + root[:3]
    assert verify_sphere_centers(spec, names, body_pos, quat, root, moving, fixed) < 1.e-12
    moving[4, 1] += .005
    with pytest.raises(RuntimeError, match="do not match"):
        verify_sphere_centers(spec, names, body_pos, quat, root, moving, fixed)
