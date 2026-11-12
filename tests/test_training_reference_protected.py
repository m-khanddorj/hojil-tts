"""Routing invariants using the installed Qwen implementation, not a mock MLP."""
import unittest
from unittest.mock import patch

from test_training_sequence import AVAILABLE, tiny


@unittest.skipUnless(AVAILABLE, 'optional Torch/Qwen dependencies required')
class ProtectedReferenceTests(unittest.TestCase):
    def setUp(self):
        import torch
        from training.reference_protected import protect_reference
        from training.targeted_voice import text_adapter
        torch.set_num_threads(1)
        self.model = tiny()
        text_adapter(self.model, rank=2, dropout=0)
        self.projection = self.model.talker.text_projection
        self.original = self.projection.forward
        protect_reference(self.model)
        self.model.eval()
        self.ids = torch.tensor([[1, 2, 3, 10, 11, 12, 4, 5, 6, 7, 8]])
        self.ref = torch.tensor([[1, 2, 3, 20, 21, 4, 5]])
        self.codes = torch.randint(0, 64, (3, 16))
        self.speaker = torch.randn(32)

    def modify_adapter(self):
        import torch
        with torch.no_grad():
            for name, parameter in self.model.named_parameters():
                if '.lora_B.' in name:
                    parameter.normal_(std=0.2)

    def sequence(self):
        from training.sequence import build
        return build(self.model, self.ids, self.codes, self.speaker, self.ref, self.codes, True)

    def test_zero_adapter_exact_projection_parity(self):
        import torch
        x = torch.randn(1, 7, 32)
        expected = self.original(x)
        self.assertTrue(torch.equal(expected, self.projection(x)))
        self.projection._target_route['span'] = (3, 7)
        self.assertTrue(torch.equal(expected, self.projection(x)))
        self.projection._target_route['span'] = None

    def test_loaded_checkpoint_audit_preserves_rng_and_mode(self):
        import json
        from pathlib import Path
        import tempfile
        import torch
        from peft import get_peft_model_state_dict
        from training.reference_protected import audited_load_adapter
        self.modify_adapter()
        self.projection._protected_control_ids = [101, 102, 100]
        self.projection.train()
        rng = torch.get_rng_state().clone()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'audit.json'
            audited_load_adapter(self.model.talker,
                get_peft_model_state_dict(self.model.talker), path)
            audit = json.loads(path.read_text())
        self.assertEqual(audit['reference_projection_max_abs_change'], 0)
        self.assertEqual(audit['control_projection_max_abs_change'], 0)
        self.assertGreater(audit['target_projection_rms_change'], 0)
        self.assertTrue(self.projection.training)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertIsNone(self.projection._target_route['span'])

    def test_nonzero_adapter_changes_only_target_positions_and_gradients(self):
        import torch
        x = torch.randn(1, 7, 32)
        baseline = self.projection(x).detach().clone()
        self.modify_adapter()
        self.assertTrue(torch.equal(baseline, self.projection(x)))
        self.projection._target_route['span'] = (3, 7)
        changed = self.projection(x)
        self.assertTrue(torch.equal(baseline[:, :3], changed[:, :3]))
        self.assertFalse(torch.equal(baseline[:, 3:], changed[:, 3:]))
        changed[:, :3].sum().backward()
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.assertTrue(all(p.grad is not None and p.grad.count_nonzero() == 0 for p in params))
        self.model.zero_grad(set_to_none=True)
        self.projection(x)[:, 3:].sum().backward()
        self.assertTrue(all(p.grad is not None and p.grad.count_nonzero() > 0 for p in params))
        self.projection._target_route['span'] = None

    def test_training_and_generation_prefixes_match_after_update(self):
        import torch
        from training.sequence import loss_sums
        before = self.sequence()
        self.modify_adapter()
        seq = self.sequence()
        # Only the three target transcript vectors can differ in teacher input:
        # reference text, role, EOS, reference audio and target audio inputs match.
        differing = (seq.embeddings != before.embeddings).any(-1).nonzero().flatten()
        self.assertEqual(len(differing), 3)
        captured = {}
        class Captured(Exception):
            pass
        def intercept(**kwargs):
            captured.update(kwargs)
            raise Captured()
        prompt = dict(ref_spk_embedding=[self.speaker], ref_code=[self.codes],
                      icl_mode=[True], x_vector_only_mode=[False])
        with patch.object(self.model.talker, 'generate', side_effect=intercept):
            with self.assertRaises(Captured):
                self.model.generate(input_ids=[self.ids], ref_ids=[self.ref],
                    voice_clone_prompt=prompt, languages=['Auto'],
                    non_streaming_mode=True, max_new_tokens=2)
        self.assertTrue(torch.equal(seq.embeddings[:seq.prefix_length], captured['inputs_embeds'][0]))
        a, b, n, m = loss_sums(self.model, [seq])
        (a / n + 0.3 * b / m).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in self.model.parameters() if p.requires_grad))
        self.assertTrue(all(p.grad is None for p in self.model.parameters() if not p.requires_grad))
        self.assertIsNone(self.projection._target_route['span'])

    def test_route_clears_after_exception_and_rejects_unsupported_modes(self):
        import torch
        with patch.object(self.projection.linear_fc1, 'forward', side_effect=RuntimeError('test')):
            with self.assertRaisesRegex(RuntimeError, 'test'):
                self.sequence()
        self.assertIsNone(self.projection._target_route['span'])
        with self.assertRaisesRegex(ValueError, 'full-reference'):
            self.model.generate(input_ids=[self.ids], non_streaming_mode=True)
        with self.assertRaisesRegex(ValueError, 'non-streaming'):
            self.model.generate_icl_prompt(self.ids[:, 3:-5], self.ref[:, 3:-2],
                self.codes, torch.zeros(1, 1, 32), torch.zeros(1, 1, 32), False)


if __name__ == '__main__':
    unittest.main()
