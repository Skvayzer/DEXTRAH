import numpy as np
import pytest

from scripts.prepare_pca_presentation import feasible_interval


def test_sweep_interval_respects_both_joint_limits():
    mean=np.array([.3,.7,.5]); direction=np.array([1.,-2.,0.])
    low,high=feasible_interval(mean,direction,np.zeros(3),np.ones(3))
    assert low == pytest.approx(-.15)
    assert high == pytest.approx(.35)
    for scalar in np.linspace(low,high,101):
        q=mean+scalar*direction
        assert q.min() >= -1.e-12 and q.max() <= 1+1.e-12


def test_basis_projection_blend_keeps_residual_direction():
    rng=np.random.default_rng(42)
    orthogonal,_=np.linalg.qr(rng.normal(size=(6,6)))
    a=orthogonal[:,:5].T
    residual=orthogonal[:,5]
    q=orthogonal[:,0]+.3*residual
    projected=q @ a.T @ a
    soft=q+.05*(projected-q)
    np.testing.assert_allclose(soft @ a.T,q @ a.T,atol=1.e-12)
    assert soft @ residual == pytest.approx(.95*(q @ residual))
