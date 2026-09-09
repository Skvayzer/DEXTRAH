"""Explicit rollout denominators and faithful BPS display geometry (no Isaac)."""
import numpy as np


def family_name(path):
    from pathlib import Path
    parts = Path(path).name.split('_', 3)
    if len(parts) != 4 or not parts[0].isdigit() or parts[2] != 'handle':
        raise ValueError(f'Unknown procedural object name: {path}')
    return parts[1]


def select_family_envs(paths, asset_indices, families):
    """Preselect by asset name only, never by rollout results."""
    selected = {}
    for env, asset in enumerate(asset_indices):
        family = family_name(paths[int(asset)])
        if family in families and family not in selected:
            selected[family] = env
    if set(selected) != set(families):
        raise ValueError(f'Missing requested families: {set(families)-set(selected)}')
    return {family: selected[family] for family in families}


class ReposeStats:
    """Count before auto-reset; ongoing episodes/attempts remain censored.

    A hit ending an episode resolves ONE successful goal attempt, not a success
    plus an imaginary failure on the next goal. Throughput includes all elapsed
    simulated time, including grasping, resets and unfinished attempts.
    """
    def __init__(self, n_envs, dt):
        if n_envs < 1 or not np.isfinite(dt) or dt <= 0:
            raise ValueError('Invalid environment count or timestep')
        self.n, self.dt, self.steps = n_envs, dt, 0
        for name in ('previous', 'hits', 'failed_attempts', 'episodes', 'any_goal',
                     'lifted', 'all_goals', 'completed_goals', 'fall', 'timeout', 'hand_far'):
            setattr(self, name, np.zeros(n_envs, dtype=np.int64))
        self.reward = np.zeros(n_envs, dtype=np.float64)

    def update(self, final, done, reward):
        done = np.asarray(done, dtype=bool)
        count = np.asarray(final['successes'])
        reward = np.asarray(reward)
        if any(x.shape != (self.n,) for x in (done, count, reward)):
            raise ValueError('Expected one value per environment')
        if not np.isfinite(count).all() or not np.isfinite(reward).all():
            raise ValueError('Nonfinite rollout metric')
        if np.any(count != np.floor(count)):
            raise ValueError('Goal count is not integer')
        count = count.astype(np.int64)
        delta = count - self.previous
        if np.any((delta < 0) | (delta > 1)):
            raise ValueError('Lost or double-counted pre-reset goal counter')
        self.steps += 1
        self.hits += delta
        self.failed_attempts += done & (delta == 0)
        self.episodes += done
        self.any_goal += done & (count > 0)
        self.completed_goals += np.where(done, count, 0)
        self.lifted += done & np.asarray(final['ever_lifted_rate'], dtype=bool)
        self.all_goals += done & np.asarray(final['all_goals_hit'], dtype=bool)
        for name in ('fall', 'timeout', 'hand_far'):
            getattr(self, name)[:] += done & np.asarray(final['done_'+name], dtype=bool)
        self.reward += reward
        self.previous[:] = np.where(done, 0, count)
        return delta

    def report(self, ids=None):
        ids = np.arange(self.n) if ids is None else np.asarray(ids, dtype=int)
        n, seconds = len(ids), self.steps*self.dt
        sums = {name: int(getattr(self, name)[ids].sum()) for name in
                ('hits', 'failed_attempts', 'episodes', 'any_goal', 'lifted',
                 'all_goals', 'completed_goals', 'fall', 'timeout', 'hand_far')}
        ep, attempts = sums['episodes'], sums['hits']+sums['failed_attempts']
        return dict(environments=n, seconds_per_environment=seconds,
            total_environment_seconds=n*seconds, **sums,
            goal_success_fraction_resolved=(sums['hits']/attempts if attempts else None),
            goals_per_simulated_minute=(60*sums['hits']/(n*seconds) if seconds else None),
            episode_any_goal_success_rate=(sums['any_goal']/ep if ep else None),
            episode_ever_lifted_rate=(sums['lifted']/ep if ep else None),
            episode_all_50_goals_rate=(sums['all_goals']/ep if ep else None),
            goals_per_completed_episode=(sums['completed_goals']/ep if ep else None),
            objects_reaching_any_goal=int((self.hits[ids]>0).sum()),
            object_any_goal_rate=float((self.hits[ids]>0).mean()) if n else None,
            ongoing_episode_goals=int(self.previous[ids].sum()),
            censoring='Ongoing episodes and unresolved attempts are excluded from success fractions, not counted as failures; throughput includes all time.')


def bps_display_geometry(mesh, expected):
    """Reconstruct the EXACT sampled surface and descriptor used by training."""
    import trimesh
    from scipy.spatial import cKDTree
    from .grail.compute_bps import fibonacci_sphere, normalize_points
    samples, _ = trimesh.sample.sample_surface(mesh, 16384, seed=42)
    points, centroid, radius = normalize_points(samples.astype(np.float32))
    basis = fibonacci_sphere(128).astype(np.float32)
    _, ids = cKDTree(points).query(basis, k=1)
    distances = np.linalg.norm(basis-points[ids], axis=1)
    descriptor = np.r_[distances, centroid, radius].astype(np.float32)
    np.testing.assert_allclose(descriptor, expected, rtol=1e-6, atol=1e-7)
    return dict(basis=basis, nearest=points[ids], distances=distances,
                centroid=centroid, radius=np.asarray(radius),
                vertices=(mesh.vertices-centroid)/radius, faces=mesh.faces)
