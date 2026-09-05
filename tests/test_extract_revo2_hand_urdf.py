import importlib.util
from pathlib import Path

import torch

from dextrah_lab.retargeting import Revo2Kinematics


ROOT = Path(__file__).parents[1]
URDF = ROOT / "dextrah_lab/assets/revo2_description/urdf/revo2_right_hand.urdf"
PACKAGE_ROOT = ROOT / "dextrah_lab/assets/revo2_description"
SCRIPT = ROOT / "scripts/extract_revo2_hand_urdf.py"


def _script_module():
    spec = importlib.util.spec_from_file_location("extract_revo2_hand_urdf", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_extracted_subtree_preserves_revo2_forward_kinematics(tmp_path) -> None:
    output = tmp_path / "right_hand.urdf"
    _script_module().extract_hand_subtree(URDF, output, PACKAGE_ROOT)
    original = Revo2Kinematics(URDF)
    extracted = Revo2Kinematics(output)
    q = original.lower + 0.37 * (original.upper - original.lower)

    torch.testing.assert_close(
        extracted.fingertip_positions(q), original.fingertip_positions(q)
    )
