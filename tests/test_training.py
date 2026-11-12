import copy
from pathlib import Path
import unittest

from training.policy import (external, learning_rate, microbatches,
                             update_groups, validate_cohorts)

try:
    import torch
except ImportError:
    torch = None


class PolicyTests(unittest.TestCase):
    def test_tail_is_not_a_single_clip_update(self):
        groups = list(update_groups(list(range(28001)), 16))
        self.assertEqual(len(groups), 1750)
        self.assertEqual(len(groups[-1]), 17)
        self.assertEqual([i for g in groups for i in g], list(range(28001)))
        for n in range(1, 50):
            self.assertEqual([i for g in update_groups(list(range(n)), 16) for i in g], list(range(n)))

    def test_long_reference_limits_batch(self):
        items = [dict(frames=10, length=500), dict(frames=10, length=510)]
        self.assertEqual([len(b) for b in microbatches(items, 195, 1000)], [1, 1])

    def test_schedule_endpoints(self):
        self.assertAlmostEqual(learning_rate(1, 176, 1e-5, 3e-6, 0), 1e-5)
        self.assertAlmostEqual(learning_rate(176, 176, 1e-5, 3e-6, 0), 3e-6)
        self.assertAlmostEqual(learning_rate(1, 100, 5e-5, 1e-5, 10), 5e-6)

    def test_repository_artifacts_rejected(self):
        root = Path(__file__).resolve().parents[1]
        if not (root / '.git').exists():
            self.skipTest('Requires a Git checkout')
        with self.assertRaises(ValueError):
            external(root / 'output')

    def test_reference_and_text_leakage_rejected(self):
        manifest = [dict(id=str(i), text=text, audio=f'{i}.wav', speaker_id='s',
                         split='train' if i < 2 else 'validation', pcm_sha256=str(i))
                    for i, text in enumerate(('train one', 'train two', 'val one', 'val two'))]
        def pair(i, j):
            return dict(manifest[i], ref_id=str(j), ref_text=manifest[j]['text'], ref_audio=f'{j}.wav')
        train, val = [pair(0, 1), pair(1, 0)], [pair(2, 3)]
        validate_cohorts(train, val, manifest)
        leaky = copy.deepcopy(manifest)
        leaky[3]['text'] = 'TRAIN, ONE!'
        val[0]['ref_text'] = leaky[3]['text']
        with self.assertRaisesRegex(ValueError, 'leakage'):
            validate_cohorts(train, val, leaky)


@unittest.skipIf(torch is None, 'optional training dependencies not installed')
class NumericalTests(unittest.TestCase):
    def test_masked_projection_preserves_loss_and_gradients(self):
        import torch.nn.functional as F
        torch.manual_seed(3)
        hidden = torch.randn(2, 7, 4, requires_grad=True)
        weight = torch.randn(13, 4, requires_grad=True)
        labels = torch.tensor([[-100, -100, 1, 2, 3, 4, 0], [-100, 5, 6, 0, -100, -100, -100]])
        old = F.cross_entropy(F.linear(hidden, weight).reshape(-1, 13), labels.reshape(-1),
                              reduction='sum', ignore_index=-100)
        expected = torch.autograd.grad(old, (hidden, weight))
        valid = labels != -100
        new = F.cross_entropy(F.linear(hidden[valid], weight), labels[valid], reduction='sum')
        actual = torch.autograd.grad(new, (hidden, weight))
        torch.testing.assert_close(old, new)
        for a, b in zip(expected, actual):
            torch.testing.assert_close(a, b)

    def test_fp32_retains_small_adam_update_and_moments(self):
        from training.engine import check_optimizer
        old = torch.nn.Parameter(torch.ones(8, dtype=torch.bfloat16))
        previous = torch.optim.AdamW([old], lr=1e-5)
        old.grad = torch.ones_like(old)
        previous.step()
        self.assertTrue(torch.equal(old, torch.ones_like(old)))
        new = torch.nn.Parameter(old.detach().float())
        optimizer = torch.optim.AdamW([new], lr=1e-5)
        optimizer.load_state_dict(previous.state_dict())
        check_optimizer(optimizer)
        new.grad = torch.ones_like(new)
        optimizer.step()
        self.assertTrue((new < 1).all())


if __name__ == '__main__':
    unittest.main()
