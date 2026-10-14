"""Per-env object physics for the Play2Perfect G1 scenes (Isaac).

The factorized replicated scene simulated env 0's object in every env while
BPS, observations and rendering used each env's own URDF (2026-10-08, job
1095: 1 unique PhysX mass in 120 envs). Upstream Play.yaml: the replication
flags must be false for MultiUsdFileCfg to spawn per-env distinct handle_head
USDs. The same settings as G1SonicTouchEnv's `per_env_object_physics`, for
the fixed-body touch task, plus a startup check that every env really
simulates its assigned object.
"""
import re
from pathlib import Path


def configure(cfg):
    cfg.assets.replicate_factorized_object_templates = False
    cfg.assets.replicate_grouped_physics_templates = False
    cfg.assets.replicate_single_physics_template = False
    cfg.scene.replicate_physics = False
    cfg.scene.clone_in_fabric = False


def filter_inter_env_collisions(env):
    """The direct-env scene path never filters inter-env collisions (job 1101:
    a neighbour's object touched this robot); filter every env against all others."""
    if env.num_envs > 1:
        env.scene.filter_collisions(global_prim_paths=['/World/ground'])


def with_per_env_objects(env_class):
    """Subclass of a Play2Perfect env class that filters inter-env collisions after scene setup."""
    class PerEnvObjects(env_class):
        def _setup_scene(self):
            super()._setup_scene()
            filter_inter_env_collisions(self)
            print(f'PER_ENV_OBJECT_PHYSICS inter-env collisions filtered envs={self.num_envs}', flush=True)
    PerEnvObjects.__name__ = PerEnvObjects.__qualname__ = env_class.__name__+'PerEnvObjects'
    return PerEnvObjects


def verify(env, sample=2048, tolerance=1e-3):
    """Every sampled env's PhysX object mass equals its assigned URDF's mass, and
    masses differ across envs. Raises if the scene still shares one object."""
    import numpy as np
    masses = env.object.root_physx_view.get_masses().cpu().numpy().reshape(env.num_envs, -1)[:, 0]
    assignment = env._object_asset_index_per_env.cpu().numpy()
    cache = {}

    def urdf_mass(index):
        if index not in cache:
            text = Path(env._object_urdf_paths[index]).read_text()
            cache[index] = float(re.search(r'<mass value="([^"]+)"', text).group(1))
        return cache[index]
    step = max(1, env.num_envs//sample)
    checked = list(range(0, env.num_envs, step))
    errors = [abs(masses[i]-urdf_mass(int(assignment[i]))) for i in checked]
    report = dict(envs=int(env.num_envs), unique_physx_masses=int(len(np.unique(np.round(masses, 6)))),
                  assigned_assets=int(len(np.unique(assignment))), checked_envs=len(checked),
                  max_abs_mass_error_kg=float(max(errors)), mismatched=int(sum(e > tolerance for e in errors)))
    print('PER_ENV_OBJECT_PHYSICS_VERIFY '+str(report), flush=True)
    if report['mismatched'] or (report['assigned_assets'] > 1 and report['unique_physx_masses'] == 1):
        raise RuntimeError(f'Per-env object physics not in effect: {report}')
    return report
