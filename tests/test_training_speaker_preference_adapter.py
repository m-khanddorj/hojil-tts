import unittest
from unittest.mock import patch
from test_training_sequence import AVAILABLE,tiny


@unittest.skipUnless(AVAILABLE,'Torch/Qwen required')
class SpeakerPreferenceTests(unittest.TestCase):
    def setUp(self):
        import torch
        from training.speaker_preference_adapter import install
        torch.set_num_threads(1)
        self.model=tiny().requires_grad_(False)
        self.adapter=install(self.model,rank=2)

    def trajectory(self,frames=3):
        import torch
        return dict(input_ids=torch.tensor([[1,2,3,10,11,12,4,5,6,7,8]]),
            ref_ids=torch.tensor([[1,2,3,20,21,4,5]]),codes=torch.randint(0,64,(frames,16)),
            ref_codes=torch.randint(0,64,(2,16)),ref_xvec=torch.randn(32),terminated=True)

    def test_zero_output_and_finite_update_always_obey_bound(self):
        import torch
        x=torch.randn(7,32)
        self.assertTrue(torch.equal(self.adapter(x),x))
        with torch.no_grad():self.adapter.up.normal_(std=200)
        ratio=(self.adapter(x)-x).norm(dim=-1)/x.norm(dim=-1)
        self.assertTrue((ratio<=0.050001).all())
        self.assertTrue((ratio>0).all())
        self.assertTrue(all(not p.requires_grad for n,p in self.model.named_parameters() if not n.startswith('speaker_preference_adapter.')))

    def test_full_icl_training_prefix_matches_native_conditioning_hook(self):
        import torch
        from training.speaker_preference_adapter import preference_sequence
        t=self.trajectory()
        with torch.no_grad():self.adapter.up.normal_(std=.02)
        seq=preference_sequence(self.model,t);captured={}
        class Stop(Exception):pass
        def intercept(**kwargs):captured.update(kwargs);raise Stop()
        prompt=dict(ref_spk_embedding=[t['ref_xvec']],ref_code=[t['ref_codes']],icl_mode=[True],x_vector_only_mode=[False])
        with patch.object(self.model.talker,'generate',side_effect=intercept):
            with self.assertRaises(Stop):
                self.model.generate(input_ids=[t['input_ids']],ref_ids=[t['ref_ids']],
                    voice_clone_prompt=prompt,languages=['Auto'],non_streaming_mode=True,max_new_tokens=2)
        torch.testing.assert_close(seq.embeddings[:seq.prefix_length],captured['inputs_embeds'][0],atol=0,rtol=0)

    def test_replayed_dpo_gradient_matches_full_graph_and_keeps_backbone_frozen(self):
        import torch
        from training.speaker_preference_adapter import backward_pair,log_probability
        from training.preference_objective import dpo_loss
        a,b=self.trajectory(),self.trajectory(4)
        # Pair candidates share exactly the same conditioning, as real pool pairs do.
        for key in ('input_ids','ref_ids','ref_codes','ref_xvec'):b[key]=a[key]
        refs=[float(log_probability(self.model,t)[0].detach()) for t in (a,b)]
        la,_=log_probability(self.model,a);lb,_=log_probability(self.model,b)
        loss,_=dpo_loss(la,lb,*refs);loss.backward()
        expected={n:p.grad.clone() for n,p in self.adapter.named_parameters()}
        self.assertGreater(float(expected['up'].norm()),0)
        self.assertTrue(torch.equal(expected['down'],torch.zeros_like(expected['down'])))
        self.adapter.zero_grad(set_to_none=True)
        result=backward_pair(self.model,a,b,refs)
        self.assertAlmostEqual(result['loss'],.6931471805599453,places=10)
        for n,p in self.adapter.named_parameters():torch.testing.assert_close(p.grad,expected[n],atol=2e-5,rtol=2e-4)
        self.assertTrue(all(p.grad is None for n,p in self.model.named_parameters() if not n.startswith('speaker_preference_adapter.')))
        b['terminated']=False
        with self.assertRaisesRegex(ValueError,'truncated'):log_probability(self.model,b)


if __name__=='__main__':unittest.main()
