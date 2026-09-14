import tempfile
from pathlib import Path
import unittest

from dextrah_lab.wholebody.recording_contract import action_layout, execution_means, snapshot_checkpoint


class RecordingContractTests(unittest.TestCase):
    def frozen(self):
        return dict(controller_mode='frozen_pretrained_latent', action_dim=70,
                    architecture='frozen_pretrained_sonic_sapg_latent64_fingers6_v1',
                    latent_residual_scale=.1, latent_clipping=False)

    def test_architecture(self):
        self.assertEqual(action_layout(self.frozen()), (True, 70))
        self.assertEqual(action_layout({}), (False, 35))

    def test_rejects_mismatched_actions(self):
        value = self.frozen()
        value['action_dim'] = 35
        with self.assertRaises(ValueError):
            action_layout(value)

    def test_rejects_clipped_latents(self):
        value = self.frozen()
        value['latent_clipping'] = True
        with self.assertRaises(ValueError):
            action_layout(value)

    def test_latents_are_never_clamped(self):
        class Means:
            shape = (120, 70)
            def clamp(self, *args):
                raise AssertionError('Do not clip latent means')
        means = Means()
        self.assertIs(execution_means(means, frozen=True), means)
        with self.assertRaises(ValueError):
            execution_means(means, frozen=False)

    def test_checkpoint_snapshot_does_not_change_source_or_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            source, target = Path(folder)/'source', Path(folder)/'target'
            source.write_bytes(b'checkpoint data')
            before = source.stat()
            report = snapshot_checkpoint(source, target)
            self.assertEqual(target.read_bytes(), source.read_bytes())
            self.assertEqual(report['bytes'], len(b'checkpoint data'))
            self.assertEqual(source.stat().st_mtime_ns, before.st_mtime_ns)
            self.assertEqual(len(report['sha256']), 64)
            with self.assertRaises(FileExistsError):
                snapshot_checkpoint(source, target)
