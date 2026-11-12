import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from training.refreshed_speaker_rollouts import refresh_cases, load_policy, adapter_digest


class RefreshedSpeakerRolloutTests(unittest.TestCase):
    def test_refresh_keeps_cases_and_speaker_split_but_uses_new_rng_streams(self):
        cases=[dict(case='a',speaker_id='a',preference_split='train',text='one',reference='a.wav',batch_seed=7301),
               dict(case='b',speaker_id='b',preference_split='validation',text='two',reference='b.wav',batch_seed=7302)]
        original=copy.deepcopy(cases)
        fresh=refresh_cases(cases)
        self.assertEqual(cases,original)
        self.assertEqual([c['batch_seed'] for c in fresh],[17301,17302])
        for a,b in zip(original,fresh):
            self.assertEqual({k:v for k,v in a.items() if k!='batch_seed'},
                             {k:v for k,v in b.items() if k!='batch_seed'})
        with self.assertRaisesRegex(ValueError,'Duplicate'):refresh_cases(cases+[cases[0]])
        cases[0]['batch_seed']=17301
        with self.assertRaisesRegex(ValueError,'overlap'):refresh_cases(cases)

    def test_initial_policy_loads_nonzero_adapter_instead_of_resetting_it(self):
        import torch
        from training.speaker_preference_adapter import SpeakerAdapter
        model=torch.nn.Module()
        model.add_module('speaker_preference_adapter',SpeakerAdapter(4,2))
        expected={k:torch.full_like(v,.2) for k,v in model.speaker_preference_adapter.state_dict().items()}
        wrapper=SimpleNamespace(model=model)
        before=adapter_digest(model)
        with patch('training.refreshed_speaker_rollouts.verify_initial',return_value=({},dict(adapter=expected))), \
                patch('training.refreshed_speaker_rollouts.previous_fit.load',return_value=wrapper):
            loaded=load_policy(dict(initial_policy={}))
        self.assertIs(loaded,wrapper)
        self.assertFalse(model.training)
        self.assertNotEqual(before,adapter_digest(model))
        for k,v in model.speaker_preference_adapter.state_dict().items():self.assertTrue(torch.equal(v,expected[k]))
        x=torch.ones(4)
        self.assertFalse(torch.equal(model.speaker_preference_adapter(x),x))


if __name__=='__main__':unittest.main()
