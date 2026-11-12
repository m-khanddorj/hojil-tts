import unittest
import torch
from test_training_sequence import AVAILABLE,tiny
from training import middle_attention_adapter as api
from training.speaker_preference_adapter import install as install_speaker


@unittest.skipUnless(AVAILABLE,'Qwen required')
class MiddleAttentionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.model=tiny().requires_grad_(False)
        speaker=install_speaker(self.model,rank=2)
        with torch.no_grad():speaker.up.normal_(std=.01)
        self.t=dict(input_ids=torch.tensor([[1,2,3,10,11,12,4,5,6,7,8]]),
            ref_ids=torch.tensor([[1,2,3,20,21,4,5]]),codes=torch.randint(0,64,(3,16)),
            ref_codes=torch.randint(0,64,(2,16)),ref_xvec=torch.randn(32),terminated=True)

    def test_zero_adapter_native_parity_and_canonical_frozen_identity(self):
        frozen=api.frozen_digest(self.model);speaker=api.speaker_digest(self.model)
        t=self.t
        kwargs=dict(input_ids=[t['input_ids']],ref_ids=[t['ref_ids']],
            voice_clone_prompt=dict(ref_spk_embedding=[t['ref_xvec']],ref_code=[t['ref_codes']],icl_mode=[True],x_vector_only_mode=[False]),
            languages=['Auto'],non_streaming_mode=True,max_new_tokens=4,do_sample=False,subtalker_dosample=False)
        with torch.inference_mode():expected=self.model.generate(**kwargs);logp=api.log_probability(self.model,t)[0]
        api.install(self.model,[0,1],2)
        with torch.inference_mode():actual=self.model.generate(**kwargs);newlogp=api.log_probability(self.model,t)[0]
        for a,b in zip(expected[0],actual[0]):self.assertTrue(torch.equal(a,b))
        self.assertTrue(torch.equal(logp,newlogp))
        self.assertEqual(api.frozen_digest(self.model),frozen)
        self.assertEqual(api.speaker_digest(self.model),speaker)
        self.assertEqual(len(api.parameters(self.model)),8)
        self.assertEqual(api.movement(self.model)['maximum'],0)
        self.assertTrue(all(not p.requires_grad for p in self.model.speaker_preference_adapter.parameters()))

    def test_exact_replayed_dpo_gradients_only_reach_lora(self):
        api.install(self.model,[0,1],2)
        with torch.no_grad():
            for m in api.modules(self.model).values():m.lora_B.normal_(std=.001)
        b=dict(self.t,codes=torch.randint(0,64,(4,16)))
        from training.preference_objective import dpo_loss
        refs=[float(api.log_probability(self.model,t)[0].detach()) for t in (self.t,b)]
        a0,_=api.log_probability(self.model,self.t);b0,_=api.log_probability(self.model,b)
        loss,_=dpo_loss(a0,b0,*refs);loss.backward()
        expected={n:p.grad.clone() for n,p in self.model.named_parameters() if p.requires_grad}
        self.assertTrue(all(float(g.norm())>0 for g in expected.values()))
        self.model.zero_grad(set_to_none=True)
        api.backward_pair(self.model,self.t,b,refs)
        for n,p in self.model.named_parameters():
            if p.requires_grad:torch.testing.assert_close(p.grad,expected[n],atol=2e-5,rtol=2e-4)
            else:self.assertIsNone(p.grad)

    def test_projection_serialization_and_frozen_parameter_tamper_detection(self):
        api.install(self.model,[0,1],2)
        frozen=api.frozen_digest(self.model);speaker=api.speaker_digest(self.model)
        with torch.no_grad():
            for m in api.modules(self.model).values():m.lora_B.normal_(std=10)
        self.assertGreater(api.movement(self.model)['maximum'],.01)
        api.project(self.model,.01)
        self.assertLessEqual(api.movement(self.model)['maximum'],.01000001)
        saved=api.snapshot(self.model)
        api.restore(self.model,saved)
        for k,v in api.snapshot(self.model).items():self.assertTrue(torch.equal(v,saved[k]))
        self.assertEqual(api.frozen_digest(self.model),frozen)
        self.assertEqual(api.speaker_digest(self.model),speaker)
        self.model.speaker_preference_adapter.up.requires_grad_(True)
        with self.assertRaisesRegex(ValueError,'trainable'):api.parameters(self.model)
        self.model.speaker_preference_adapter.up.requires_grad_(False)
        with torch.no_grad():next(iter(api.modules(self.model).values())).base_layer.weight[0,0].add_(1)
        self.assertNotEqual(api.frozen_digest(self.model),frozen)


if __name__=='__main__':unittest.main()
