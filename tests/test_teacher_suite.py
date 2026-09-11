from types import SimpleNamespace
import hashlib

import pytest

from dextrah_lab.object_shape.teacher_suite import (
    cases, geometry_hash, object_seed_override, summarize,
)


def urdf(size=1, mass=1):
    return f'<robot><link name="obj"><inertial><mass value="{mass}"/></inertial><collision><geometry><box size="{size} 1 1"/></geometry></collision></link></robot>'


def test_geometry_identity_ignores_density(tmp_path):
    a, b = tmp_path/'a.urdf', tmp_path/'b.urdf'
    a.write_text(urdf(mass=1)); b.write_text(urdf(mass=4))
    assert geometry_hash(a) == geometry_hash(b)
    b.write_text(urdf(size=2))
    assert geometry_hash(a) != geometry_hash(b)


def test_seed_override_restores_and_audits_geometry(tmp_path):
    def original(*, out_dir, seed=42):
        from pathlib import Path
        path = Path(out_dir)/'object.urdf'
        path.write_text(urdf(size=seed))
        return [path], [(1, 1, 1)]
    module = SimpleNamespace(generate_handle_head_urdfs=original)
    source = {'entries': [{'urdf_sha256': hashlib.sha256(urdf(size=42).encode()).hexdigest()}]}
    with object_seed_override(module, 914271, source) as audit:
        paths, _ = module.generate_handle_head_urdfs(out_dir=tmp_path)
        assert geometry_hash(paths[0])
        assert audit['training_geometry_overlap'] == 0
    assert module.generate_handle_head_urdfs is original
    with pytest.raises(RuntimeError):
        with object_seed_override(module, 42, source):
            raise RuntimeError('scene failed')
    assert module.generate_handle_head_urdfs is original


def test_density_only_change_is_not_heldout(tmp_path):
    def original(*, out_dir, seed=42):
        from pathlib import Path
        path = Path(out_dir)/'object.urdf'
        path.write_text(urdf(mass=seed))
        return [path], [(1, 1, 1)]
    module = SimpleNamespace(generate_handle_head_urdfs=original)
    source = {'entries': [{'urdf_sha256': hashlib.sha256(urdf(mass=42).encode()).hexdigest()}]}
    with pytest.raises(ValueError, match='overlaps'):
        with object_seed_override(module, 99, source):
            module.generate_handle_head_urdfs(out_dir=tmp_path)
    assert module.generate_handle_head_urdfs is original


def records():
    return [dict(c, report=dict(episode_any_goal_success_rate=.7,
        episode_ever_lifted_rate=.9,goals_per_simulated_minute=16.)) for c in cases()]


def test_selection_requires_complete_unique_suite():
    data = records()
    assert len(data) == 27
    with pytest.raises(ValueError, match='every unique'):
        summarize(data[:-1])
    with pytest.raises(ValueError, match='every unique'):
        summarize(data+[data[0]])


def test_heldout_cannot_select_teacher_and_regression_guard_works():
    data = records()
    for row in data:
        if row['candidate'] == 'touch_best':
            row['report']['goals_per_simulated_minute'] = 18.
        if row['candidate'] == 'touch_final':
            row['report']['goals_per_simulated_minute'] = 100.
            row['report']['episode_any_goal_success_rate'] = .60
        if row['candidate'] == 'bps_baseline' and row['condition'] == 'heldout':
            row['report']['goals_per_simulated_minute'] = 1000.
    result = summarize(data)
    assert result['selected_candidate'] == 'touch_best'
    assert not result['heldout_used_for_selection']
    assert not result['causal_tactile_benefit_established']
    assert 'touch_final' not in result['eligible_candidates']


def test_invalid_metric_rejected():
    data = records()
    data[0]['report']['goals_per_simulated_minute'] = float('nan')
    with pytest.raises(ValueError, match='Invalid'):
        summarize(data)
