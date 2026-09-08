"""Static guardrails; the allocated-GPU validator checks actual runtime behavior."""
import ast
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "dextrah_lab/tasks/g1_revo2_adept"


def test_direct_task_does_not_override_control_or_observations():
    tree = ast.parse((TASK / "g1_revo2_direct_env.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "G1Revo2DirectEnv")
    assert [b.id for b in cls.bases] == ["PlayEnv"]
    methods = {n.name for n in cls.body if isinstance(n, ast.FunctionDef)}
    assert not methods & {"_pre_physics_step", "_apply_action", "_get_observations", "_get_dones"}


def test_sapg_algorithm_parameters_match_existing_run():
    baseline = yaml.safe_load((TASK / "agents/g1_revo2_adept_sapg.yaml").read_text())["params"]
    direct = yaml.safe_load((TASK / "agents/g1_revo2_direct_sapg.yaml").read_text())["params"]
    baseline["config"].pop("name")
    direct["config"].pop("name")
    assert direct == baseline


def test_direct_launcher_has_no_old_checkpoint_or_engine_fabric_toggle():
    script = (ROOT / "scripts/slurm/train_g1_direct_sapg_single.sbatch").read_text()
    assert "--checkpoint" not in script
    assert "--disable_fabric" not in script  # Isaac Sim I/O != geometric controller
    assert "G1-Revo2-SimToolReal-Repose-Direct" in script
