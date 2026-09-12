import numpy as np
import torch
from scipy.spatial.transform import Rotation
from dextrah_lab.wholebody.clip_task import compose,qapply,retimed_smoothing
from dextrah_lab.wholebody.kinematics import pose_matrix


def test_scene_pose_composition_preserves_quaternion_convention():
    a=torch.tensor([.2,.3,.5,.7071067812,0,0,.7071067812],dtype=torch.float64)
    b=torch.tensor([[.1,0,.2,1,0,0,0],[-.1,.1,0,1,0,0,0]],dtype=torch.float64)
    result=compose(a,b)
    np.testing.assert_allclose(pose_matrix(result.numpy()),pose_matrix(a.numpy())@pose_matrix(b.numpy()),atol=1e-9)
    vectors=torch.randn(5,3,dtype=torch.float64)
    q=a[3:].expand(5,-1)
    np.testing.assert_allclose(qapply(q,vectors).numpy(),Rotation.from_quat(a[[4,5,6,3]].numpy()).apply(vectors.numpy()),atol=1e-9)


def test_finger_smoothing_preserves_one_second_response():
    alpha=retimed_smoothing(.1,1/60,1/50)
    np.testing.assert_allclose((1-alpha)**50,.9**60,atol=1e-12)
