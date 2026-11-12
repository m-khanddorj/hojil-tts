import unittest
from collections import Counter
from training.speaker_preference_training import sampled_pair,paired_screen


class SpeakerPreferenceTrainingTests(unittest.TestCase):
    def test_sampling_weights_speakers_equally_despite_case_and_pair_imbalance(self):
        pairs=[dict(speaker_id='a',case='a1',chosen=f'a{i}',rejected=f'ar{i}',preference_split='train') for i in range(20)]
        pairs += [dict(speaker_id='b',case=c,chosen=c+'c',rejected=c+'r',preference_split='train') for c in ('b1','b2')]
        pairs += [dict(speaker_id='validation',case='v',chosen='vc',rejected='vr',preference_split='validation')]
        sampled=[sampled_pair(pairs,i,4811) for i in range(12)]
        self.assertEqual(Counter(p['speaker_id'] for p in sampled),{'a':6,'b':6})
        self.assertEqual(Counter(p['case'] for p in sampled),{'a1':6,'b1':3,'b2':3})
        self.assertEqual(len({p['chosen'] for p in sampled if p['speaker_id']=='a'}),6)
        self.assertEqual(sampled,[sampled_pair(list(reversed(pairs)),i,4811) for i in range(12)])
        self.assertEqual(sampled[8:],[sampled_pair(pairs,i,4811) for i in range(8,12)])

    def report(self,cer=.05,cosine=.98):
        records=[dict(case=str(i),seed=seed,text=f'text {i}',reference_sha256=f'ref {i}',speaker_id=str(i),cosine=cosine)
            for i in range(13) for seed in (2026,2027)]
        return dict(records=records,summary={g:dict(pooled_cer=cer,speaker_macro_cosine=cosine) for g in ('development','custom')},truncated=0)

    def test_absolute_voice_failure_does_not_stop_an_improving_full_model(self):
        baseline=self.report();candidate=self.report(.04,.981)
        result=paired_screen(baseline,candidate)
        self.assertFalse(result['goal_screen']);self.assertFalse(result['harmful'])
        self.assertTrue(paired_screen(baseline,self.report(.06,.979))['harmful'])
        self.assertTrue(paired_screen(baseline,self.report(.04,.974))['harmful'])

    def test_goal_requires_both_strict_thresholds_complete_pairs_and_no_truncation(self):
        baseline=self.report();candidate=self.report(.029,.985)
        self.assertTrue(paired_screen(baseline,candidate)['goal_screen'])
        for cer,cos in ((.03,.985),(.029,.984)):
            self.assertFalse(paired_screen(baseline,self.report(cer,cos))['goal_screen'])
        candidate['summary']['custom']['pooled_cer']=.031
        self.assertFalse(paired_screen(baseline,candidate)['goal_screen'])
        candidate=self.report(.029,.985);candidate['truncated']=1
        self.assertFalse(paired_screen(baseline,candidate)['goal_screen'])
        candidate['records'].pop()
        with self.assertRaisesRegex(ValueError,'Incomplete'):paired_screen(baseline,candidate)
        candidate=self.report();candidate['records'][0]['reference_sha256']='changed'
        with self.assertRaisesRegex(ValueError,'Changed'):paired_screen(baseline,candidate)

    def test_optimizer_resume_matches_uninterrupted_tiny_qwen_training(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from types import SimpleNamespace
        import torch
        from test_training_sequence import AVAILABLE,tiny
        if not AVAILABLE or not torch.cuda.is_available():self.skipTest('Qwen and CUDA required')
        from training import speaker_preference_training as trainer
        from training.speaker_preference_adapter import install
        from training.engine import digest,save,write
        from training.targeted_voice import frozen_digest
        torch.set_num_threads(1)
        def model(_):
            value=tiny().requires_grad_(False)
            torch.manual_seed(4811);install(value,rank=2)
            return SimpleNamespace(model=value)
        frozen=frozen_digest(model(None).model)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);pool=root/'pool';pool.mkdir();records=[]
            torch.manual_seed(92)
            conditioning=dict(input_ids=torch.tensor([[1,2,3,10,11,12,4,5,6,7,8]]),
                ref_ids=torch.tensor([[1,2,3,20,21,4,5]]),ref_codes=torch.randint(0,64,(2,16)),ref_xvec=torch.randn(32),terminated=True)
            for i in range(4):
                path=pool/f'{i}.pt';save(path,dict(conditioning,codes=torch.randint(0,64,(2+i%2,16))))
                records.append(dict(sample_id=str(i),trajectory=path.name,trajectory_sha256=digest(path)))
            write(pool/'results.json',dict(records=records))
            pairs=[dict(speaker_id='a',case='a',chosen='0',rejected='1',preference_split='train'),
                   dict(speaker_id='b',case='b',chosen='2',rejected='3',preference_split='validation')]
            p=dict(pool=str(pool),pairs=pairs,training=dict(trainer.DEFAULTS,rank=2,updates=2,
                checkpoints=[1,2],warmup_updates=0,pairs_per_update=1))
            outputs=[]
            for mode in ('resumed','uninterrupted'):
                run=root/mode;run.mkdir();write(run/'protocol.json',p)
                write(run/'pool-verification.json',dict(frozen_backbone_sha256=frozen))
                with patch.object(trainer,'verify',return_value=p),patch.object(trainer,'load',side_effect=model):
                    trainer.train(run,0)
                    zero=trainer.read_checkpoint(run,0,p)
                    self.assertEqual(zero['adapter_l2_change'],0)
                    self.assertAlmostEqual(zero['losses']['train']['loss'],.6931471805599453,places=10)
                    if mode=='resumed':trainer.train(run,1)
                    trainer.train(run,2)
                    outputs.append(trainer.read_checkpoint(run,2,p))
            for key,value in outputs[0]['adapter'].items():
                torch.testing.assert_close(value,outputs[1]['adapter'][key],atol=1e-6,rtol=1e-5)
            self.assertGreater(outputs[0]['adapter_l2_change'],0)
            self.assertEqual(outputs[0]['frozen_backbone_sha256'],frozen)
            for state in outputs[0]['optimizer']['state'].values():
                self.assertEqual(state['exp_avg'].dtype,torch.float32)
                self.assertEqual(state['exp_avg_sq'].dtype,torch.float32)


if __name__=='__main__':unittest.main()
