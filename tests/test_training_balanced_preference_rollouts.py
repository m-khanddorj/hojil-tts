import unittest
from types import SimpleNamespace
from training.balanced_preference_rollouts import capture_batch,separate_transcripts


class BalancedRolloutTests(unittest.TestCase):
    def test_reference_and_target_overlap_excluded_without_changing_validation(self):
        cases=[dict(id='v',speaker_id='v',preference_split='validation',text='same words',ref_text='other reference'),
            dict(id='a',speaker_id='a',preference_split='train',text='new text',ref_text='same words'),
            dict(id='b',speaker_id='b',preference_split='train',text='other reference',ref_text='another'),
            dict(id='c',speaker_id='c',preference_split='train',text='distinct text',ref_text='distinct reference')]
        kept,excluded=separate_transcripts(cases)
        self.assertEqual([c['id'] for c in kept],['v','c'])
        self.assertEqual(excluded,['a','b'])
        cases[-1]['speaker_id']='v'
        with self.assertRaisesRegex(ValueError,'Speaker overlap'):separate_transcripts(cases)

    def model(self):
        import torch
        codes=[]
        for first in ([8,9],[10,11,12],[4,5,6]):
            c=torch.zeros(len(first),16,dtype=torch.long);c[:,0]=torch.tensor(first);codes.append(c)
        talker=SimpleNamespace(generate=lambda:SimpleNamespace(sequences=torch.tensor([[8,9,99,99],[10,11,12,99],[4,5,6,7]])))
        def generate():talker.generate();return codes,None
        return SimpleNamespace(generate=generate,talker=talker,config=SimpleNamespace(talker_config=SimpleNamespace(codec_eos_token_id=99)))

    def test_batch_eos_padding_and_truncation_are_distinguished(self):
        import torch
        model=self.model();original,talker=model.generate,model.talker.generate
        with capture_batch(model) as captured:result=model.generate()
        self.assertEqual(captured['terminated'],[True,True,False])
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(result[0],captured['codes'])))
        self.assertIs(model.generate,original);self.assertIs(model.talker.generate,talker)
        with self.assertRaisesRegex(RuntimeError,'deliberate'):
            with capture_batch(model):raise RuntimeError('deliberate')
        self.assertIs(model.generate,original);self.assertIs(model.talker.generate,talker)

    def test_rejects_trajectory_that_disagrees_with_sampled_tokens(self):
        import torch
        model=self.model()
        model.talker.generate=lambda:SimpleNamespace(sequences=torch.tensor([[8,3,99,99],[10,11,12,99],[4,5,6,7]]))
        with self.assertRaisesRegex(ValueError,'disagrees'):
            with capture_batch(model):model.generate()


if __name__=='__main__':unittest.main()
