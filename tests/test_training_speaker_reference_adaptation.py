"""Input-gradient and trust-region tests, including the actual Qwen prefix."""
import unittest
from unittest.mock import patch
from test_training_sequence import AVAILABLE, tiny


@unittest.skipUnless(AVAILABLE,'Torch/Qwen required')
class SpeakerReferenceTests(unittest.TestCase):
    def test_reference_prefix_and_gradient_match_generation(self):
        import torch
        from training.speaker_reference_adaptation import speaker_sequence
        from training.sequence import build,loss_sums
        torch.set_num_threads(1)
        model=tiny().requires_grad_(False)
        ids=torch.tensor([[1,2,3,10,11,12,4,5,6,7,8]])
        codes=torch.randint(0,64,(3,16))
        speaker=torch.randn(32,requires_grad=True)
        seq=speaker_sequence(model,ids,codes,speaker)
        baseline=build(model,ids,codes,speaker,non_streaming=True)
        self.assertTrue(torch.equal(seq.embeddings,baseline.embeddings))
        captured={}
        class Stop(Exception):pass
        def intercept(**kwargs):
            captured.update(kwargs)
            raise Stop()
        prompt={'ref_spk_embedding':[speaker],'ref_code':[None],
            'icl_mode':[False],'x_vector_only_mode':[True]}
        with patch.object(model.talker,'generate',side_effect=intercept):
            with self.assertRaises(Stop):
                model.generate(input_ids=[ids],voice_clone_prompt=prompt,languages=['Auto'],
                    non_streaming_mode=True,max_new_tokens=2)
        actual=captured['inputs_embeds'][0]
        torch.testing.assert_close(seq.embeddings[:seq.prefix_length],actual,rtol=0,atol=0)
        # Equality alone would miss a detached or mislocated speaker gradient.
        weights=torch.randn_like(actual)
        # Public generation is inference_mode. A finite input perturbation
        # verifies which slot changes; the expected Jacobian is identity there.
        delta=torch.randn_like(speaker)*0.01
        prompt['ref_spk_embedding']=[speaker.detach()+delta]
        with patch.object(model.talker,'generate',side_effect=intercept):
            with self.assertRaises(Stop):
                model.generate(input_ids=[ids],voice_clone_prompt=prompt,languages=['Auto'],
                    non_streaming_mode=True,max_new_tokens=2)
        changed=captured['inputs_embeds'][0]-actual
        expected_change=torch.zeros_like(changed);expected_change[6]=delta
        torch.testing.assert_close(changed,expected_change,atol=3e-7,rtol=1e-4)
        expected=weights[6]
        observed,=torch.autograd.grad((seq.embeddings[:seq.prefix_length]*weights).sum(),speaker,retain_graph=True)
        torch.testing.assert_close(expected,observed,rtol=0,atol=0)
        a,b,n,m=loss_sums(model,[seq]);loss=a/n+0.3*b/m
        grad,=torch.autograd.grad(loss,speaker)
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(float(grad.norm()),0)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        # A finite difference guards the restored input gradient through CE.
        direction=grad/grad.norm();eps=0.01
        values=[]
        with torch.no_grad():
            for sign in (-1,1):
                s=speaker_sequence(model,ids,codes,speaker.detach()+sign*eps*direction)
                a,b,n,m=loss_sums(model,[s]);values.append(float(a/n+0.3*b/m))
        self.assertAlmostEqual((values[1]-values[0])/(2*eps),float(grad.norm()),delta=2e-4)

    def test_projection_bounds_every_update_and_preserves_origin(self):
        import torch
        from training.speaker_reference_adaptation import bounded_step
        origin=torch.tensor([3.,4.]);copy=origin.clone();vector=origin.clone()
        for _ in range(100):
            vector=bounded_step(vector,origin,torch.tensor([-1.,2.]))
            self.assertLessEqual(float((vector-origin).norm()),0.250001)
        self.assertTrue(torch.equal(origin,copy))
        self.assertFalse(vector.requires_grad)
        self.assertTrue(torch.equal(bounded_step(origin,origin,torch.zeros(2)),origin))
        with self.assertRaises(ValueError):bounded_step(vector,origin,torch.tensor([float('nan'),0.]))


if __name__=='__main__':unittest.main()
