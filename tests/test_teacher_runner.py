import importlib.util
from pathlib import Path
import json

import pytest


spec=importlib.util.spec_from_file_location('teacher_runner',Path(__file__).parents[1]/'scripts/run_g1_teacher_suite.py')
runner=importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_commands_keep_protocol_and_separate_capture():
    candidate=dict(run='/run',checkpoint='/weights.pth')
    for case in runner.cases():
        command=runner.rollout_command(case,candidate,Path('/output'),Path('/p2p'))
        assert command[command.index('--seconds')+1]=='120'
        assert command[command.index('--num-envs')+1]=='1200'
        assert command[command.index('--seed')+1]==str(case['seed'])
        assert ('--capture-reference' in command) == (case['condition']=='training' and case['seed']==42)
        assert ('--metrics-only' in command) != ('--capture-reference' in command)


def test_changed_metadata_cannot_resume(tmp_path):
    (tmp_path/'metadata.json').write_text(json.dumps(dict(seed=999)))
    (tmp_path/'evaluation.json').write_text('{}')
    with pytest.raises(ValueError,match='protocol mismatch'):
        runner.validate_result(tmp_path,runner.cases()[0],dict(checkpoint_sha256='abc'))


def test_atomic_marker(tmp_path):
    runner.atomic_json(tmp_path/'done.json',dict(complete=False))
    runner.atomic_json(tmp_path/'done.json',dict(complete=True))
    assert json.loads((tmp_path/'done.json').read_text())==dict(complete=True)
    assert not (tmp_path/'done.tmp').exists()
