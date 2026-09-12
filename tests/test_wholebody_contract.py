import ast
from pathlib import Path
import numpy as np
import pytest

from dextrah_lab.wholebody.contract import (
    BODY_JOINTS, hand_joints, joint_indices, nominal_body_pose, checkpoint_reference_command,
)


def test_command_ownership_and_permutation():
    commands = (*BODY_JOINTS,*hand_joints('left'),*hand_joints('right'))
    assert len(commands) == len(set(commands)) == 41
    assert sum('right_' in n and any(x in n for x in ('shoulder','elbow','wrist')) for n in BODY_JOINTS) == 7
    actual = tuple(reversed(commands))
    assert tuple(actual[i] for i in joint_indices(actual,BODY_JOINTS)) == BODY_JOINTS
    with pytest.raises(ValueError,match='Missing'):
        joint_indices(actual[1:],commands)
    with pytest.raises(ValueError,match='Duplicate'):
        joint_indices((*actual,actual[0]),commands)
    with pytest.raises(ValueError):
        hand_joints('mirrored')


def test_nominal_pose_and_released_packing():
    nominal=nominal_body_pose()
    assert nominal.shape == (29,)
    assert nominal[BODY_JOINTS.index('left_shoulder_roll_joint')] == pytest.approx(.2)
    assert nominal[BODY_JOINTS.index('right_shoulder_roll_joint')] == pytest.approx(-.2)
    q=np.arange(2*10*29).reshape(2,10,29)
    qd=q+10000
    result=checkpoint_reference_command(q,qd)
    expected=np.concatenate((q.reshape(2,-1),qd.reshape(2,-1)),axis=1).reshape(2,10,58)
    np.testing.assert_array_equal(result,expected)
    assert not np.array_equal(result,np.concatenate((q,qd),axis=-1))
    with pytest.raises(ValueError):
        checkpoint_reference_command(q[:,:9],qd[:,:9])


def test_joint_order_matches_pinned_grail_when_available():
    upstream=Path(__file__).parents[2]/'GRAIL/imports/SONIC/gear_sonic/envs/env_utils/joint_utils.py'
    if not upstream.exists():
        pytest.skip('Pinned GRAIL checkout not beside this worktree')
    tree=ast.parse(upstream.read_text())
    assignment=next(n for n in tree.body if isinstance(n,ast.Assign)
                    and any(isinstance(t,ast.Name) and t.id=='G1_ISAACLab_ORDER' for t in n.targets))
    assert tuple(ast.literal_eval(assignment.value)) == BODY_JOINTS
