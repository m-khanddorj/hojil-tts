import copy
import unittest
import torch
from tests.test_training_sequence import tiny
from training.speaker_preference_adapter import install as install_speaker
from training import reference_voice_profile as profiles
from training.reference_profile_preference_objective import log_probability,objective,backward_batch,project_input_radius


class ProfilePreferenceObjectiveTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(61571);torch.set_num_threads(1)
        self.model=tiny().requires_grad_(False);install_speaker(self.model,rank=2)
        self.model.requires_grad_(False);self.profile=profiles.install(self.model,'a'*64)
        self.trajectory=dict(input_ids=torch.tensor([[1,2,3,10,11,12,4,5,6,7,8]]),
            ref_ids=torch.tensor([[1,2,3,20,21,4,5]]),codes=torch.randint(0,64,(3,16)),
            ref_codes=torch.randint(0,64,(2,16)),ref_xvec=torch.randn(32),terminated=True,reference_sha256='a'*64)

    def test_unique_replay_matches_direct_joint_gradient_with_frozen_originals(self):
        trajectories={str(i):dict(self.trajectory,codes=torch.randint(0,64,(3+i,16))) for i in range(3)}
        pairs=[dict(case='one',chosen='0',rejected='1'),dict(case='one',chosen='0',rejected='2')]
        originals={n:p.detach().clone() for n,p in self.model.named_parameters() if p is not self.profile.raw}
        with torch.no_grad():refs={k:float(log_probability(self.model,self.profile,t)[0]) for k,t in trajectories.items()}
        values={k:log_probability(self.model,self.profile,t)[0] for k,t in trajectories.items()}
        loss,_,_=objective(values,refs,pairs);self.assertAlmostEqual(float(loss.detach()),.69314718056,places=8)
        loss.backward();expected=self.profile.raw.grad.clone();self.assertGreater(float(expected.norm()),0)
        self.model.zero_grad(set_to_none=True)
        result=backward_batch(self.model,self.profile,trajectories,refs,pairs)
        torch.testing.assert_close(self.profile.raw.grad,expected,atol=2e-7,rtol=3e-4)
        self.assertEqual(result['unique_trajectories'],3);self.assertFalse(self.profile._active)
        self.assertEqual(result['weighted_chosen_logratio'],0)
        self.assertEqual(result['weighted_rejected_logratio'],0)
        self.assertEqual(result['weighted_both_likelihoods_decreased_fraction'],0)
        for name,p in self.model.named_parameters():
            if p is self.profile.raw:continue
            self.assertTrue(torch.equal(p,originals[name]));self.assertIsNone(p.grad)

    def test_equal_text_weight_and_reference_isolation(self):
        values={str(i):torch.tensor(float(i),dtype=torch.float64,requires_grad=True) for i in range(5)}
        refs={k:float(v.detach()) for k,v in values.items()}
        pairs=[dict(case='a',chosen='0',rejected='1'),dict(case='a',chosen='0',rejected='2'),dict(case='b',chosen='3',rejected='4')]
        loss,coefficients,margins=objective(values,refs,pairs)
        self.assertEqual([r['weight'] for r in margins],[.25,.25,.5]);loss.backward()
        for key in values:torch.testing.assert_close(values[key].grad,coefficients[key])
        with self.assertRaises(ValueError):log_probability(self.model,self.profile,dict(self.trajectory,reference_sha256='b'*64))
        with self.assertRaises(ValueError):log_probability(self.model,self.profile,dict(self.trajectory,terminated=False))
        self.assertFalse(self.profile._active)

    def test_projection_enforces_actual_input_displacement(self):
        vector=self.trajectory['ref_xvec'];before=self.profile.base(vector)
        with torch.no_grad():self.profile.raw.fill_(100)
        result=project_input_radius(self.profile,.01);self.assertTrue(result['projected'])
        with self.profile.use_reference('a'*64),torch.no_grad():actual=float((self.profile(vector)-before).norm()/vector.norm())
        self.assertAlmostEqual(actual,.01,places=6)
        with self.profile.use_reference('b'*64):self.assertTrue(torch.equal(self.profile(vector),before))
        with self.assertRaises(ValueError):project_input_radius(self.profile,.05)
        with torch.no_grad():self.profile.raw[0]=float('nan')
        with self.assertRaisesRegex(ValueError,'Nonfinite'):project_input_radius(self.profile,.01)


if __name__=='__main__':unittest.main()
