import math
import unittest

from test_training_sequence import AVAILABLE, tiny


@unittest.skipUnless(AVAILABLE, 'optional Torch/Qwen dependencies required')
class PreferenceObjectiveTests(unittest.TestCase):
    def setUp(self):
        import torch
        from training.targeted_voice import text_adapter
        from training.reference_protected import protect_reference
        torch.set_num_threads(1)
        self.model = tiny()
        text_adapter(self.model, rank=2, dropout=0.2)
        protect_reference(self.model)
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if '.lora_B.' in name:
                    param.normal_(std=0.1)
        self.a = dict(input_ids=torch.tensor([[1,2,3,10,11,12,4,5,6,7,8]]),
            ref_ids=torch.tensor([[1,2,3,20,21,4,5]]), codes=torch.randint(0,64,(3,16)),
            ref_codes=torch.randint(0,64,(2,16)), ref_xvec=torch.randn(32), terminated=True)
        self.b = dict(self.a, codes=torch.randint(0,64,(4,16)))

    def test_true_joint_logp_includes_eos_and_unweighted_residuals(self):
        import torch
        from training.sequence import build, loss_sums
        from training.preference_objective import log_probability
        self.model.eval()
        a = self.a
        sequence = build(self.model, a['input_ids'], a['codes'], a['ref_xvec'], a['ref_ids'], a['ref_codes'], True)
        first, residual, n, m = loss_sums(self.model, [sequence])
        actual, count = log_probability(self.model, a)
        torch.testing.assert_close(actual, -(first.double()+residual.double()), rtol=0, atol=0)
        self.assertEqual(count, 3*16+1)
        self.assertGreater(abs(actual.item()+(first+0.3*residual).item()), 1)
        with self.assertRaisesRegex(ValueError, 'Truncated'):
            log_probability(self.model, dict(a, terminated=False))

    def test_replayed_backward_matches_full_graph_dpo(self):
        import torch
        from training.preference_objective import (log_probability, dpo_loss,
            backward_pair, deterministic_training)
        deterministic_training(self.model)
        params = [p for p in self.model.parameters() if p.requires_grad]
        a, _ = log_probability(self.model, self.a)
        b, _ = log_probability(self.model, self.b)
        references = (a.detach().item()-1.5,b.detach().item()+0.3)
        loss, _ = dpo_loss(a,b,*references)
        (loss/3).backward()
        expected = [p.grad.clone() for p in params]
        self.model.zero_grad(set_to_none=True)
        # Exercise the actual checkpointed path used for the full-size model.
        for decoder in (self.model.talker.model,self.model.talker.code_predictor.model):
            decoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        event = backward_pair(self.model,self.a,self.b,references,weight=1/3)
        self.assertAlmostEqual(event['loss'], loss.item(), places=7)
        for param, grad in zip(params, expected):
            self.assertGreater(param.grad.abs().sum(), 0)
            torch.testing.assert_close(param.grad, grad, rtol=2e-4, atol=2e-7)
        self.assertTrue(all(p.grad is None for p in self.model.parameters() if not p.requires_grad))

    def test_loss_direction_and_reference_cancellation(self):
        import torch
        from training.preference_objective import dpo_loss
        chosen = torch.tensor(-20., requires_grad=True)
        rejected = torch.tensor(-10., requires_grad=True)
        loss, margin = dpo_loss(chosen,rejected,-20.,-10.)
        self.assertAlmostEqual(loss.item(), math.log(2), places=6)
        self.assertEqual(margin.item(), 0)
        loss.backward()
        self.assertLess(chosen.grad.item(), 0)
        self.assertGreater(rejected.grad.item(), 0)
        for beta in (0,-1,float('nan')):
            with self.assertRaises(ValueError):
                dpo_loss(chosen,rejected,-20.,-10.,beta)


if __name__ == '__main__':
    unittest.main()
