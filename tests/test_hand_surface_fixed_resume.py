"""Fixed-body resume expansion on the real run-1369 checkpoint (CPU; skipped if absent).

The expanded actor/critic must reproduce the original outputs exactly for any
hand-surface input, and every Adam moment must match its parameter's shape.
"""
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

from dextrah_lab.mj.compat import install

install()
from rl_games.algos_torch import model_builder  # noqa: E402
from dextrah_lab.g1_adept import touch_policy  # noqa: E402
from dextrah_lab.g1_adept.touch_continuation import load_yaml  # noqa: E402
from dextrah_lab.wholebody import hand_surface as hs  # noqa: E402

RUN = Path('/data1/users/konstantin.smirnov/DEXTRAH-G1-WholeBody/outputs/0_bps128_touch70_fixed_mjlab_sgs_1369')
CHECKPOINT = RUN/'nn/complete_55651467264.pth'
COEF = torch.tensor([50., 40., 30., 20., 10., 0.])


def build(hand_dim):
    touch_policy.register_models(hand_dim)
    params = load_yaml(RUN/'params/agent.yaml')['params']
    actor = model_builder.ModelBuilder().load(params).build(dict(
        actions_num=13, input_shape=(249+hand_dim+32,), num_seqs=32, value_size=1, normalize_value=True,
        normalize_input=True, type='extra_param', coef_ids=COEF, coef_id_idx=249+hand_dim))
    cv = params['config']['central_value_config']
    critic = model_builder.ModelBuilder().load(dict(model=cv['model'], network=cv['network'])).build(dict(
        actions_num=13, input_shape=(271+hand_dim+32,), num_seqs=1, value_size=1, normalize_value=True,
        normalize_input=True, type='extra_param', coef_ids=COEF, coef_id_idx=271+hand_dim))
    return actor.eval(), critic.eval()


@unittest.skipUnless(CHECKPOINT.exists(), 'run-1369 checkpoint not on this machine')
class FixedResumeExpansionTests(unittest.TestCase):
    def test_expanded_resume_is_exact(self):
        torch.manual_seed(0)
        payload = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
        state = payload[0] if 0 in payload else payload
        actor0, critic0 = build(0)
        actor0.load_state_dict(state['model'])
        critic0.load_state_dict({k.removeprefix('model.'): v for k, v in state['assymetric_vf_nets'].items()})
        H = hs.HAND_SURFACE_DIM
        actor1, critic1 = build(H)
        algo = SimpleNamespace(model=actor1, central_value_net=SimpleNamespace(
            model=critic1, state_dict=lambda: {'model.'+k: v for k, v in critic1.state_dict().items()}))
        new, report = hs.expand_fixed_resume_state(state, algo, H)
        self.assertTrue(report['expanded'])
        actor1.load_state_dict(new['model'], strict=True)
        critic1.load_state_dict({k.removeprefix('model.'): v for k, v in new['assymetric_vf_nets'].items()}, strict=True)
        # Adam moments match parameter shapes (actor and critic optimizers).
        for opt, net in ((new['optimizer'], actor1), (new['continuation_critic_optimizer'], critic1)):
            params = list(net.parameters())
            ids = opt['param_groups'][0]['params']
            self.assertEqual(len(ids), len(params))
            for i, p in zip(ids, params):
                if i in opt['state']:
                    self.assertEqual(opt['state'][i]['exp_avg'].shape, p.shape)
                    self.assertEqual(opt['state'][i]['exp_avg_sq'].shape, p.shape)
        # Same outputs as the original for any hand input.
        n = 32
        base = torch.randn(n, 249)
        base[:, 224:249] = torch.rand(n, 25)             # touch channels incl. validity bits
        hand = torch.randn(n, H)*3
        coef = torch.zeros(n, 1)
        h0 = tuple(s[:, :n] for s in actor0.get_default_rnn_state())
        with torch.no_grad():
            a0 = actor0(dict(obs=torch.cat((base, coef), -1), is_train=False, prev_actions=None, rnn_states=h0))
            a1 = actor1(dict(obs=torch.cat((base, hand, coef), -1), is_train=False, prev_actions=None, rnn_states=h0))
            torch.testing.assert_close(a1['mus'], a0['mus'], rtol=0, atol=1e-5)
            torch.testing.assert_close(a1['values'], a0['values'], rtol=0, atol=1e-4)
            cb = torch.randn(n, 271)
            cb[:, 246:271] = torch.rand(n, 25)
            v0 = critic0(dict(obs=torch.cat((cb, coef), -1), is_train=False))['values']
            v1 = critic1(dict(obs=torch.cat((cb, hand, coef), -1), is_train=False))['values']
            torch.testing.assert_close(v1, v0, rtol=0, atol=1e-4)
        # Already-expanded checkpoints pass through.
        again, rep = hs.expand_fixed_resume_state(new, algo, H)
        self.assertFalse(rep['expanded'])
        touch_policy.register_models(0)


if __name__ == '__main__':
    unittest.main()
