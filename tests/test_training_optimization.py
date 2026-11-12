import copy
import unittest

from training.optimization import selection_update, set_learning_rate, split_lora_optimizer

try:
    import torch
except ImportError:
    torch = None


class SelectionTests(unittest.TestCase):
    def test_small_gains_are_saved_but_patience_uses_cumulative_gain(self):
        best, anchor, bad, improved = selection_update(3.0, float('inf'), float('inf'), 0, .002)
        self.assertTrue(improved)
        best, anchor, bad, improved = selection_update(2.999, best, anchor, bad, .002)
        self.assertEqual((best, anchor, bad, improved), (2.999, 3.0, 1, True))
        best, anchor, bad, improved = selection_update(2.997, best, anchor, bad, .002)
        self.assertEqual((best, anchor, bad, improved), (2.997, 2.997, 0, True))
        self.assertEqual(selection_update(2.998, best, anchor, bad, .002), (best, anchor, 1, False))


@unittest.skipIf(torch is None, 'optional training dependencies not installed')
class OptimizerTests(unittest.TestCase):
    def test_small_matrix_spectrum_matches_dense_update(self):
        from scripts.analyze_lora_spectrum import spectrum
        torch.manual_seed(7)
        a, b = torch.randn(3, 8), torch.randn(7, 3)
        result = spectrum(a, b)
        values = torch.linalg.svdvals(b.double() @ a.double())
        self.assertAlmostEqual(result['frobenius_norm'], values.square().sum().sqrt().item())
        self.assertAlmostEqual(result['stable_rank'], (values.square().sum()/values[0].square()).item())
        self.assertEqual(spectrum(a, torch.zeros_like(b))['rank90'], 0)

    def test_split_preserves_historical_moments_and_resume_trajectory(self):
        a = torch.nn.Parameter(torch.ones(3, 2))
        b = torch.nn.Parameter(torch.ones(4, 3))
        named = [('layer.lora_A.default.weight', a), ('layer.lora_B.default.weight', b)]
        optimizer = torch.optim.AdamW([a, b], lr=1e-5)
        for p in (a, b):
            p.grad = torch.ones_like(p)
        optimizer.step()
        before = copy.deepcopy(optimizer.state_dict())
        split_lora_optimizer(optimizer, named, 4)
        for p, state in zip((a, b), before['state'].values()):
            torch.testing.assert_close(optimizer.state[p]['exp_avg'], state['exp_avg'])
            torch.testing.assert_close(optimizer.state[p]['exp_avg_sq'], state['exp_avg_sq'])
        set_learning_rate(optimizer, 3e-6)
        self.assertEqual([g['lr'] for g in optimizer.param_groups], [7.5e-7, 3e-6])
        a2, b2 = [torch.nn.Parameter(p.detach().clone()) for p in (a, b)]
        resumed = torch.optim.AdamW([a2, b2], lr=1e-5)
        split_lora_optimizer(resumed, [(named[0][0], a2), (named[1][0], b2)], 4)
        resumed.load_state_dict(copy.deepcopy(optimizer.state_dict()))
        for opt, params in ((optimizer, (a, b)), (resumed, (a2, b2))):
            set_learning_rate(opt, 2e-6)
            for p in params:
                p.grad = torch.full_like(p, .3)
            opt.step()
        torch.testing.assert_close(a, a2, rtol=0, atol=0)
        torch.testing.assert_close(b, b2, rtol=0, atol=0)

    def test_unknown_trainable_parameter_rejected(self):
        p = torch.nn.Parameter(torch.ones(2))
        optimizer = torch.optim.AdamW([p])
        with self.assertRaisesRegex(ValueError, 'Unsupported'):
            split_lora_optimizer(optimizer, [('unrelated.weight', p)], 4)


if __name__ == '__main__':
    unittest.main()
