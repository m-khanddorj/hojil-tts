import copy
import unittest
import torch
from test_training_sequence import AVAILABLE, tiny
from training import middle_attention_adapter as api
from training.speaker_preference_adapter import install as install_speaker
from training.preference_objective import dpo_loss
from training.quality_preference_objective import backward_pair, quality_pairs


@unittest.skipUnless(AVAILABLE, 'Qwen required')
class CalibratedGradientTests(unittest.TestCase):
    def test_replay_equals_full_graph_for_unequal_lengths_and_optional_anchor(self):
        torch.set_num_threads(1)
        torch.manual_seed(721)
        model = tiny().requires_grad_(False)
        install_speaker(model, rank=2)
        api.install(model, [0, 1], 2)
        with torch.no_grad():
            for module in api.modules(model).values():
                module.lora_B.normal_(std=.001)
        frozen, speaker = api.frozen_digest(model), api.speaker_digest(model)
        a = dict(input_ids=torch.tensor([[1,2,3,10,11,12,4,5,6,7,8]]),
                 ref_ids=torch.tensor([[1,2,3,20,21,4,5]]), codes=torch.randint(0,64,(3,16)),
                 ref_codes=torch.randint(0,64,(2,16)), ref_xvec=torch.randn(32), terminated=True)
        b = dict(a, codes=torch.randint(0,64,(5,16)))
        references = (-200., -320.)
        for anchor in (5.,):
            model.zero_grad(set_to_none=True)
            chosen, tokens = api.log_probability(model, a)
            rejected, _ = api.log_probability(model, b)
            dpo, _ = dpo_loss(chosen, rejected, *references, beta=.1)
            expected_loss = dpo - anchor*chosen/tokens
            (expected_loss*.25).backward()
            expected = {n:p.grad.clone() for n,p in model.named_parameters() if p.requires_grad}
            self.assertTrue(all(float(g.norm()) > 0 for g in expected.values()))
            model.zero_grad(set_to_none=True)
            event = backward_pair(model, a, b, references, beta=.1, weight=.25, sft_weight=anchor)
            self.assertAlmostEqual(event['total_loss'], float(expected_loss.detach()), places=7)
            for name, parameter in model.named_parameters():
                if parameter.requires_grad:
                    torch.testing.assert_close(parameter.grad, expected[name], atol=2e-5, rtol=2e-4)
                else:
                    self.assertIsNone(parameter.grad)
            self.assertEqual(api.frozen_digest(model), frozen)
            self.assertEqual(api.speaker_digest(model), speaker)
        for bad_weight in (-1., float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                backward_pair(model, a, b, references, sft_weight=bad_weight)


if __name__ == '__main__':
    unittest.main()
