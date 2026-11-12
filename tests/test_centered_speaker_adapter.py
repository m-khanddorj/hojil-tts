import unittest
import torch
from training.speaker_preference_adapter import SpeakerAdapter
from training.centered_speaker_adapter import CenteredSpeakerAdapter,fit_basis


class CenteredSpeakerAdapterTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)
        self.references = {str(i):[torch.randn(20),torch.randn(20)] for i in range(12)}
        self.basis = fit_basis(self.references,4)
        self.initial = SpeakerAdapter(20,4,.05)
        with torch.no_grad():
            self.initial.up.normal_(0,.1)

    def test_exact_nonzero_initial_policy_and_only_residual_gradients(self):
        adapter = CenteredSpeakerAdapter(self.initial,self.basis)
        x = torch.randn(3,20)
        self.assertTrue(torch.equal(adapter(x),self.initial(x)))
        expected = adapter.frozen_state()
        loss = (adapter(x)*torch.randn_like(x)).sum()
        loss.backward()
        self.assertGreater(float(adapter.residual.grad.norm()),0)
        self.assertTrue(all(p.grad is None for p in adapter.initial.parameters()))
        optimizer = torch.optim.AdamW([adapter.residual],lr=.01,weight_decay=0)
        optimizer.step()
        adapter.verify_frozen(expected)
        self.assertFalse(torch.equal(adapter(x),self.initial(x)))

    def test_centering_whitening_equal_speaker_weight_and_finite_bound(self):
        # Basis fitting averages within speaker: repeating every reference of
        # one speaker does not give that speaker extra covariance weight.
        repeated = dict(self.references)
        repeated['0'] = self.references['0']*5
        second = fit_basis(repeated,4)
        for key in ('center','basis','scale'):
            torch.testing.assert_close(second[key],self.basis[key])
        means = torch.stack([torch.nn.functional.normalize(torch.stack(v),dim=-1).mean(0)
                             for _,v in sorted(self.references.items())])
        z = (means-self.basis['center'])@self.basis['basis'].T/self.basis['scale']
        torch.testing.assert_close(z.mean(0),torch.zeros(4),atol=1e-6,rtol=0)
        torch.testing.assert_close(z.T@z/11,torch.eye(4),atol=1e-6,rtol=0)
        adapter = CenteredSpeakerAdapter(self.initial,self.basis)
        with torch.no_grad():
            adapter.residual.normal_(0,100)
            x = torch.randn(25,20)
            y = adapter(x)
        self.assertTrue(torch.isfinite(y).all())
        self.assertTrue((((y-x).norm(dim=-1)/x.norm(dim=-1)) <= .050001).all())

    def test_frozen_coordinate_tampering_detected(self):
        adapter = CenteredSpeakerAdapter(self.initial,self.basis)
        frozen = adapter.frozen_state()
        adapter.center[0] += .001
        with self.assertRaisesRegex(ValueError,'coordinates changed'):
            adapter.verify_frozen(frozen)

    def test_native_prefix_and_replayed_dpo_gradient_on_tiny_qwen(self):
        from unittest.mock import patch
        from test_training_sequence import AVAILABLE,tiny
        if not AVAILABLE:
            self.skipTest('Qwen required')
        from training.speaker_preference_adapter import install,preference_sequence,log_probability,backward_pair
        from training.preference_objective import dpo_loss
        torch.set_num_threads(1)
        model = tiny().requires_grad_(False)
        original = install(model,rank=2)
        with torch.no_grad():
            original.up.normal_(0,.01)
        basis = fit_basis({str(i):[torch.randn(32)] for i in range(6)},2)
        model.speaker_preference_adapter = CenteredSpeakerAdapter(original,basis)
        adapter = model.speaker_preference_adapter
        frozen = adapter.frozen_state()
        a = dict(input_ids=torch.tensor([[1,2,3,10,11,12,4,5,6,7,8]]),
                 ref_ids=torch.tensor([[1,2,3,20,21,4,5]]),codes=torch.randint(0,64,(3,16)),
                 ref_codes=torch.randint(0,64,(2,16)),ref_xvec=torch.randn(32),terminated=True)
        b = dict(a,codes=torch.randint(0,64,(4,16)))
        sequence = preference_sequence(model,a)
        captured = {}
        class Stop(Exception):
            pass
        def intercept(**kwargs):
            captured.update(kwargs)
            raise Stop()
        prompt = dict(ref_spk_embedding=[a['ref_xvec']],ref_code=[a['ref_codes']],
                      icl_mode=[True],x_vector_only_mode=[False])
        with patch.object(model.talker,'generate',side_effect=intercept):
            with self.assertRaises(Stop):
                model.generate(input_ids=[a['input_ids']],ref_ids=[a['ref_ids']],
                               voice_clone_prompt=prompt,languages=['Auto'],
                               non_streaming_mode=True,max_new_tokens=2)
        torch.testing.assert_close(sequence.embeddings[:sequence.prefix_length],
                                   captured['inputs_embeds'][0],atol=0,rtol=0)
        refs = [float(log_probability(model,t)[0].detach()) for t in (a,b)]
        la,_ = log_probability(model,a)
        lb,_ = log_probability(model,b)
        loss,_ = dpo_loss(la,lb,*refs)
        loss.backward()
        expected = adapter.residual.grad.clone()
        self.assertGreater(float(expected.norm()),0)
        adapter.zero_grad(set_to_none=True)
        backward_pair(model,a,b,refs)
        torch.testing.assert_close(adapter.residual.grad,expected,atol=2e-5,rtol=2e-4)
        self.assertTrue(all(p.grad is None for n,p in model.named_parameters()
                            if n != 'speaker_preference_adapter.residual'))
        optimizer = torch.optim.AdamW([adapter.residual],lr=.001,weight_decay=0)
        optimizer.step()
        adapter.verify_frozen(frozen)


if __name__ == '__main__':
    unittest.main()
