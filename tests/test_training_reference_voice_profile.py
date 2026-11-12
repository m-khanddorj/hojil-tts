import unittest
import torch
from tests.test_training_sequence import tiny
from training.speaker_preference_adapter import install as install_speaker
from training import middle_attention_adapter as attention
from training import reference_voice_profile as profile_api
from training.codec_policy_replay import policy_outputs,backward_probability_gradient


class ReferenceVoiceProfileTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(51817);torch.set_num_threads(1)
        self.model=tiny().requires_grad_(False);install_speaker(self.model,rank=2)
        attention.install(self.model,[0,1],rank=2)
        with torch.no_grad():
            self.model.speaker_preference_adapter.up.normal_(std=.02)
            for m in attention.modules(self.model).values():m.lora_B.normal_(std=.002)
        self.original_speaker=attention.speaker_digest(self.model)
        self.original_weights={n:p.detach().clone() for n,p in self.model.named_parameters()}
        self.model.requires_grad_(False)
        self.profile=profile_api.install(self.model,'a'*64)
        self.trajectory=dict(input_ids=torch.tensor([[1,2,3,10,11,12,4,5,6,7,8]]),
            ref_ids=torch.tensor([[1,2,3,20,21,4,5]]),codes=torch.randint(0,64,(3,16)),
            ref_codes=torch.randint(0,64,(2,16)),ref_xvec=torch.randn(32),terminated=True)

    def test_reference_isolation_zero_parity_and_exception_restoration(self):
        vector=self.trajectory['ref_xvec'];expected=self.profile.base(vector)
        with self.profile.use_reference('a'*64):self.assertTrue(torch.equal(self.profile(vector),expected))
        with torch.no_grad():self.profile.raw.normal_(std=100)
        with self.profile.use_reference('b'*64):self.assertTrue(torch.equal(self.profile(vector),expected))
        with self.assertRaisesRegex(RuntimeError,'fixture'):
            with self.profile.use_reference('a'*64):
                delta=self.profile(vector)-expected
                self.assertLessEqual(float(delta.norm()/vector.norm()),.050001)
                with self.profile.use_reference('b'*64):self.assertTrue(torch.equal(self.profile(vector),expected))
                self.assertFalse(torch.equal(self.profile(vector),expected))
                raise RuntimeError('fixture')
        self.assertFalse(self.profile._active)
        self.assertTrue(torch.equal(self.profile(vector),expected))

    def test_replay_and_finite_difference_profile_gradient_with_frozen_originals(self):
        with self.profile.use_reference('a'*64):
            probabilities=policy_outputs(self.model,self.trajectory)['probabilities']
            gradient=torch.randn_like(probabilities)
            (probabilities*gradient).sum().backward();direct=self.profile.raw.grad.clone()
            self.model.zero_grad(set_to_none=True)
            with torch.no_grad():cached=policy_outputs(self.model,self.trajectory)['probabilities']
            backward_probability_gradient(self.model,self.trajectory,cached,gradient)
            torch.testing.assert_close(self.profile.raw.grad,direct,atol=1e-7,rtol=2e-4)
            self.assertGreater(float(direct.norm()),0)
            direction=direct/direct.norm();epsilon=.01;values=[]
            with torch.no_grad():
                for sign in (1,-1):
                    self.profile.raw.copy_(sign*epsilon*direction)
                    values.append(float((policy_outputs(self.model,self.trajectory)['probabilities']*gradient).sum()))
                self.profile.raw.zero_()
            finite=(values[0]-values[1])/(2*epsilon)
            self.assertAlmostEqual(finite,float(direct.dot(direction)),delta=max(2e-5,abs(finite)*.05))
        self.assertEqual(profile_api.shared_speaker_digest(self.profile),self.original_speaker)
        for name,p in self.model.named_parameters():
            if name=='speaker_preference_adapter.raw':continue
            canonical=name.replace('speaker_preference_adapter.base.','speaker_preference_adapter.')
            self.assertTrue(torch.equal(p,self.original_weights[canonical]));self.assertIsNone(p.grad)

    def test_profile_artifact_rejects_wrong_reference(self):
        state=profile_api.snapshot(self.profile)
        with torch.no_grad():self.profile.raw.fill_(.1)
        profile_api.restore(self.profile,state);self.assertEqual(float(self.profile.raw.norm()),0)
        with self.assertRaises(ValueError):profile_api.restore(self.profile,dict(state,reference_sha256='b'*64))


if __name__=='__main__':unittest.main()
