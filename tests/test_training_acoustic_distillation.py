import unittest
from test_training_sequence import AVAILABLE, tiny


@unittest.skipUnless(AVAILABLE, 'optional Torch/Qwen dependencies required')
class DistillationTests(unittest.TestCase):
    def setUp(self):
        import torch
        from training.targeted_voice import text_adapter
        from training.reference_protected import protect_reference
        torch.set_num_threads(1)
        self.model = tiny()
        text_adapter(self.model, rank=2, dropout=0)
        protect_reference(self.model)
        self.model.eval()
        self.ids = torch.tensor([[1, 2, 3, 10, 11, 12, 4, 5, 6, 7, 8]])
        self.ref = torch.tensor([[1, 2, 3, 20, 21, 4, 5]])
        self.codes = torch.randint(0, 64, (3, 16))
        self.speaker = torch.randn(32)

    def sequences(self):
        from training.sequence import build
        return [build(self.model, self.ids, self.codes, self.speaker, self.ref, self.codes, True)]

    def test_matches_original_ce_and_zero_initial_kl(self):
        import torch
        from training.sequence import loss_sums
        from training.acoustic_distillation import distillation_terms
        expected = loss_sums(self.model, self.sequences())
        a, b, kl, n, m = distillation_terms(self.model, self.sequences)
        torch.testing.assert_close(a, expected[0], rtol=0, atol=0)
        torch.testing.assert_close(b, expected[1], rtol=0, atol=0)
        self.assertEqual((n, m), expected[2:])
        self.assertLess(abs(kl.item()), 1e-6)

    def test_kl_changes_and_only_student_adapter_receives_gradients(self):
        import torch
        from training.acoustic_distillation import distillation_terms
        with torch.no_grad():
            for name, parameter in self.model.named_parameters():
                if '.lora_B.' in name:
                    parameter.normal_(std=3.0)
        a, b, kl, n, m = distillation_terms(self.model, self.sequences)
        # A random tiny model has near-uniform distributions; require an actual
        # change and nonzero KL gradients, not an arbitrary loss magnitude.
        self.assertGreater(kl.item(), 0)
        # KL itself must influence all four adapters, independently of CE.
        (kl/m).backward()
        self.assertTrue(all(p.grad is None for p in self.model.parameters() if not p.requires_grad))
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.count_nonzero()
                            for p in self.model.parameters() if p.requires_grad))

    def test_teacher_restores_scaling_modes_and_rng_even_after_error(self):
        import torch
        from training.acoustic_distillation import halfway_teacher
        self.model.talker.train()
        self.model.talker.text_projection.linear_fc2.eval()
        modes = [m.training for m in self.model.talker.modules()]
        rng = torch.get_rng_state().clone()
        with self.assertRaisesRegex(RuntimeError, 'test'):
            with halfway_teacher(self.model):
                self.assertFalse(torch.is_grad_enabled())
                self.assertEqual(self.model.talker.text_projection.linear_fc1.scaling['default'], 0)
                raise RuntimeError('test')
        self.assertEqual([m.training for m in self.model.talker.modules()], modes)
        self.assertEqual(self.model.talker.text_projection.linear_fc1.scaling['default'], 1)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))


if __name__ == '__main__':
    unittest.main()
