from copy import deepcopy
from pathlib import Path
import pytest
import yaml
from dextrah_lab.wholebody.task_contract import TOUCH_RUN, verify_task_contract


@pytest.fixture
def source():
    path = Path(TOUCH_RUN)/'params/env_resolved.yaml'
    if not path.exists():
        pytest.skip('Authoritative source config is on the workstation')
    class Loader(yaml.SafeLoader):
        pass
    Loader.add_constructor('tag:yaml.org,2002:python/tuple', lambda l, n: l.construct_sequence(n))
    return yaml.load(path.read_text(), Loader=Loader)


def body_cfg(source):
    cfg = deepcopy(source)
    cfg['assets'].update(robot_profile='g1_brainco', robot_fix_base=False,
                        robot_actuate_left_hand=True, robot_self_collision=False,
                        robot_body_collision_enabled=False)
    cfg['scene']['num_envs'] = 12
    return cfg


def test_source_contract(source):
    report = verify_task_contract(source, body_cfg(source))
    assert report['policy_hz'] == 60 and report['tactile_hz'] == 70


@pytest.mark.parametrize('section', ['action', 'reward', 'termination', 'reset', 'obs', 'touch', 'domain_randomization'])
def test_reject_task_change(source, section):
    cfg = body_cfg(source)
    cfg[section]['accidental_change'] = 1
    with pytest.raises(ValueError, match=section):
        verify_task_contract(source, cfg)


def test_reject_retiming(source):
    cfg = body_cfg(source)
    cfg['sim']['dt'], cfg['decimation'] = .005, 4
    with pytest.raises(ValueError, match='60 Hz'):
        verify_task_contract(source, cfg)
