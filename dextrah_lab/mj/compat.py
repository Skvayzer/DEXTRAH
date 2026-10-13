"""Run the pinned Play2Perfect task utilities without Isaac Sim.

The reward/observation/reset/termination/goal modules of Play2Perfect are
pure torch; they import Isaac only for (a) `isaaclab.utils.math` (pure torch,
which itself imports `omni.log`) and (b) robot-profile tables in
`scene_utils`, whose module also imports USD/PhysX spawners at top level.

`install()` registers placeholder modules for the simulator-only imports, so
these files execute verbatim from the pinned revision. Any attempt to *use*
a placeholder (call/instantiate/inherit) raises, so nothing silently falls
back to a fake Isaac implementation.
"""
import importlib.util
import logging
import sys
import types
from pathlib import Path

WORKSPACE = Path('/data1/users/konstantin.smirnov')
ISAACLAB_SOURCE = WORKSPACE/'IsaacLab/source/isaaclab'
P2P_ROOT = WORKSPACE/'play2perfect'

_SIM_ONLY = (
    'omni', 'omni.kit', 'omni.usd', 'omni.physx', 'carb', 'pxr',
    'isaacsim', 'isaacsim.core', 'isaacsim.core.utils', 'isaacsim.core.utils.prims',
    'isaacsim.core.utils.stage',
    'isaaclab.sim', 'isaaclab.sim.converters', 'isaaclab.sim.spawners',
    'isaaclab.sim.spawners.from_files', 'isaaclab.sim.spawners.wrappers', 'isaaclab.sim.utils',
    'isaaclab.actuators', 'isaaclab.assets', 'isaaclab.sensors', 'isaaclab.envs',
    'isaaclab.scene', 'isaaclab.markers', 'isaaclab.managers',
)


class _Unavailable:
    """Stands in for an Isaac class or function that must never run here."""

    def __init__(self, name):
        self._name = name

    def __call__(self, *args, **kwargs):
        raise RuntimeError(f'{self._name} is Isaac-only and unavailable on the mjlab backend')

    def __mro_entries__(self, bases):
        raise RuntimeError(f'Cannot subclass Isaac-only {self._name} on the mjlab backend')

    def __getattr__(self, item):
        if item.startswith('__'):
            raise AttributeError(item)
        return _Unavailable(f'{self._name}.{item}')


class _SimOnlyModule(types.ModuleType):
    def __getattr__(self, item):
        if item.startswith('__'):
            raise AttributeError(item)
        return _Unavailable(f'{self.__name__}.{item}')


def _package(name, path):
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    module.__file__ = str(path/'__init__.py')
    module.__package__ = name
    return module


def install():
    if getattr(install, 'done', False):
        return
    for name in _SIM_ONLY:
        if name not in sys.modules:
            module = _SimOnlyModule(name)
            module.__path__ = []
            sys.modules[name] = module
    log = sys.modules['omni'].__dict__.setdefault('log', types.ModuleType('omni.log'))
    logger = logging.getLogger('omni')
    for level in ('info', 'warn', 'warning', 'error', 'verbose'):
        setattr(log, level, getattr(logger, 'warning' if level == 'warn' else
                                    ('debug' if level == 'verbose' else level)))
    sys.modules['omni.log'] = log
    if str(ISAACLAB_SOURCE) not in sys.path:
        sys.path.append(str(ISAACLAB_SOURCE))
    # Import the task utility subpackage without isaacsimenvs/__init__ (which
    # registers every Isaac task) or tasks/play/__init__ (which imports the env).
    for name, path in (('isaacsimenvs', P2P_ROOT/'isaacsimenvs'),
                       ('isaacsimenvs.tasks', P2P_ROOT/'isaacsimenvs/tasks'),
                       ('isaacsimenvs.tasks.play', P2P_ROOT/'isaacsimenvs/tasks/play')):
        if name not in sys.modules:
            sys.modules[name] = _package(name, path)
    install.done = True


def p2p(module):
    """Import ``isaacsimenvs.tasks.play.utils.<module>`` from the pinned checkout."""
    install()
    return importlib.import_module(f'isaacsimenvs.tasks.play.utils.{module}')
