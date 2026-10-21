"""Opt-in fast env step (G1SonicTouchMjEnv.enable_fast_step), CPU parts.

The GPU check against the default path on real states is scripts/check_fast_env_step.py.
"""
from pathlib import Path
import re
import unittest

import torch

REPO = Path(__file__).resolve().parents[1]
WORKSPACE = Path('/data1/users/konstantin.smirnov')
BUNDLE = WORKSPACE/'G1-SONIC-models/checkpoint/SONIC/models/sonic_manipulation_base'


@unittest.skipUnless((BUNDLE/'last.pt').exists(), 'SONIC bundle not on this machine')
class SonicFastPathTests(unittest.TestCase):
    def test_fast_path_equals_full_controller(self):
        from dextrah_lab.mj.compat import install
        install()
        from dextrah_lab.wholebody.sonic import FrozenSonic
        sonic = FrozenSonic(WORKSPACE/'GRAIL', BUNDLE, 'cpu')
        sonic.check_fast_path(n=128)
        g = torch.Generator().manual_seed(3)
        n = 300
        proprio, rq, o = torch.randn(n, 930, generator=g), torch.randn(n, 10, 29, generator=g)*.3, torch.randn(n, 10, 6, generator=g)
        rqd = torch.zeros(n, 10, 29)
        for scale in (0., .3, 3.):
            residual = torch.randn(n, 64, generator=g)*scale
            tokens, latent = sonic.reference_tokens_and_latent(rq, rqd, o)
            self.assertTrue(torch.equal(tokens, sonic.reference_tokens(rq, rqd, o)))
            self.assertTrue(torch.equal(sonic.decode_latent(proprio, latent, residual), sonic(proprio, rq, rqd, o, residual)))
        with self.assertRaises(ValueError):
            sonic.decode_latent(proprio, latent, torch.full((n, 64), float('nan')))


class RefreshReadSetTests(unittest.TestCase):
    def test_task_does_not_read_full_forward_only_quantities(self):
        """The kinematics refresh leaves qacc / actuator forces of the last physics
        substep; the mjlab task path must not read them between steps."""
        paths = [REPO/'dextrah_lab/mj', REPO/'dextrah_lab/wholebody', REPO/'dextrah_lab/sgs',
                 WORKSPACE/'play2perfect/isaacsimenvs/tasks/play/utils']      # p2p obs/reward/reset/termination utils
        pattern = re.compile(r'\.(joint_acc|actuator_force|applied_torque|qacc|qfrc_\w+|cfrc_\w+|sensordata|body_acc_w|computed_torque)\b')
        allowed = {'dextrah_lab/mj/assets.py', 'dextrah_lab/mj/nan_capture.py'}   # definition; diagnostic after sim.step
        hits = []
        for root in paths:
            for f in root.rglob('*.py'):
                rel = str(f.relative_to(REPO)) if f.is_relative_to(REPO) else str(f)
                if rel in allowed:
                    continue
                for i, line in enumerate(f.read_text().splitlines(), 1):
                    if pattern.search(line) and not line.lstrip().startswith('#'):
                        hits.append(f'{rel}:{i}: {line.strip()}')
        self.assertEqual(hits, [], 'task code reads full-forward-only quantities:\n'+'\n'.join(hits))


if __name__ == '__main__':
    unittest.main()
