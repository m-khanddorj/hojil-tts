import unittest
from test_training_sequence import AVAILABLE,tiny


@unittest.skipUnless(AVAILABLE,'optional Torch/Qwen dependencies required')
class TargetWeightTransferTests(unittest.TestCase):
    def setUp(self):
        import torch
        from training.targeted_voice import text_adapter
        from training.reference_protected import protect_reference
        torch.set_num_threads(1)
        self.model=tiny()
        text_adapter(self.model,rank=2,dropout=0)
        protect_reference(self.model)
        self.model.eval()
        self.projection=self.model.talker.text_projection
        self.weights={f'linear_fc{i}.weight':getattr(self.projection,f'linear_fc{i}').base_layer.weight.detach().clone()+0.02
            for i in (1,2)}

    def test_only_target_positions_use_transferred_weights(self):
        import torch
        import torch.nn.functional as F
        from training.target_weight_transfer import install
        old={n:p.detach().clone() for n,p in self.model.named_parameters()}
        x=torch.randn(1,9,32)
        baseline=self.projection(x).detach()
        install(self.model,self.weights)
        self.assertTrue(torch.equal(self.projection(x),baseline))
        self.projection._target_route['span']=(4,9)
        actual=self.projection(x)
        a,b=self.projection.linear_fc1.base_layer,self.projection.linear_fc2.base_layer
        full=F.linear(self.projection.act_fn(F.linear(x,self.weights['linear_fc1.weight'],a.bias)),
            self.weights['linear_fc2.weight'],b.bias)
        self.assertTrue(torch.equal(actual[:,:4],baseline[:,:4]))
        self.assertTrue(torch.equal(actual[:,4:],full[:,4:]))
        self.assertFalse(torch.equal(actual[:,4:],baseline[:,4:]))
        self.assertTrue(all(torch.equal(dict(self.model.named_parameters())[n],v) for n,v in old.items()))
        self.assertTrue(all(not p.requires_grad for p in self.model.parameters()))
        with self.assertRaisesRegex(ValueError,'fresh'):
            install(self.model,self.weights)

    def test_noop_is_exact_even_with_different_source_weights(self):
        import torch
        from training.target_weight_transfer import install
        x=torch.randn(1,9,32)
        baseline=self.projection(x).detach()
        install(self.model,self.weights,0)
        self.projection._target_route['span']=(4,9)
        self.assertTrue(torch.equal(self.projection(x),baseline))


if __name__=='__main__':unittest.main()
