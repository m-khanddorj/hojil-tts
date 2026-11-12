import unittest


class CenteredTrainingTests(unittest.TestCase):
    def test_improvement_in_one_group_does_not_hide_regression_in_other(self):
        from training.centered_speaker_training import paired_screen
        from copy import deepcopy
        records=[dict(case=str(i),seed=seed,text=f'text {i}',reference_sha256=f'ref {i}',
                      speaker_id=str(i),cosine=.98) for i in range(13) for seed in (2026,2027)]
        baseline=dict(records=records,summary={g:dict(pooled_cer=.05,speaker_macro_cosine=.98)
                      for g in ('development','custom')},truncated=0)
        candidate=deepcopy(baseline)
        candidate['summary']['development']['speaker_macro_cosine']=.981
        candidate['summary']['custom']['pooled_cer']=.06
        candidate['summary']['custom']['speaker_macro_cosine']=.979
        measured=paired_screen(baseline,candidate)
        self.assertTrue(measured['harmful'])
        self.assertEqual(measured['regressed_groups'],['custom'])
        candidate['summary']['custom']['pooled_cer']=.04
        self.assertFalse(paired_screen(baseline,candidate)['harmful'])
        for group in candidate['summary']:
            candidate['summary'][group]=dict(pooled_cer=.029,speaker_macro_cosine=.985)
        self.assertTrue(paired_screen(baseline,candidate)['goal_screen'])
        candidate['summary']['custom']['speaker_macro_cosine']=.984
        self.assertFalse(paired_screen(baseline,candidate)['goal_screen'])

    def test_optimizer_resume_and_frozen_initial_policy(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from types import SimpleNamespace
        import torch
        from test_training_sequence import AVAILABLE,tiny
        if not AVAILABLE or not torch.cuda.is_available():
            self.skipTest('Qwen and CUDA required')
        from training import centered_speaker_training as trainer
        from training.speaker_preference_adapter import install
        from training.centered_speaker_adapter import CenteredSpeakerAdapter,fit_basis
        from training.engine import digest,save,write
        from training.centered_speaker_training import frozen_digest
        torch.set_num_threads(1)
        def model(_):
            value=tiny().requires_grad_(False)
            torch.manual_seed(4811)
            old=install(value,rank=2)
            with torch.no_grad():old.up.normal_(std=.01)
            basis=fit_basis({str(i):[torch.randn(32)] for i in range(6)},2)
            value.speaker_preference_adapter=CenteredSpeakerAdapter(old,basis)
            return SimpleNamespace(model=value)
        initial=model(None).model
        frozen=frozen_digest(initial)
        fixed=initial.speaker_preference_adapter.frozen_state()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);pool=root/'pool';pool.mkdir();records=[]
            torch.manual_seed(92)
            conditioning=dict(input_ids=torch.tensor([[1,2,3,10,11,12,4,5,6,7,8]]),
                ref_ids=torch.tensor([[1,2,3,20,21,4,5]]),ref_codes=torch.randint(0,64,(2,16)),
                ref_xvec=torch.randn(32),terminated=True)
            for i in range(4):
                path=pool/f'{i}.pt'
                save(path,dict(conditioning,codes=torch.randint(0,64,(2+i%2,16))))
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
                    self.assertEqual(float(zero['adapter']['residual'].abs().sum()),0)
                    self.assertGreater(float(zero['adapter']['initial.up'].abs().sum()),0)
                    self.assertAlmostEqual(zero['losses']['train']['loss'],.6931471805599453,places=10)
                    if mode=='resumed':trainer.train(run,1)
                    trainer.train(run,2)
                    outputs.append(trainer.read_checkpoint(run,2,p))
            for key,value in outputs[0]['adapter'].items():
                torch.testing.assert_close(value,outputs[1]['adapter'][key],atol=1e-6,rtol=1e-5)
            for key,value in fixed.items():
                self.assertTrue(torch.equal(value,outputs[0]['adapter'][key]))
            self.assertGreater(outputs[0]['adapter_l2_change'],0)
            self.assertEqual(outputs[0]['frozen_backbone_sha256'],frozen)
            self.assertEqual(len(outputs[0]['optimizer']['state']),1)
            for state in outputs[0]['optimizer']['state'].values():
                self.assertEqual(state['exp_avg'].dtype,torch.float32)
                self.assertEqual(state['exp_avg_sq'].dtype,torch.float32)


if __name__=='__main__':unittest.main()
