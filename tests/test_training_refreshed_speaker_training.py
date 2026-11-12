import unittest


class RefreshedTrainingTests(unittest.TestCase):
    def test_optimizer_resume_matches_uninterrupted_tiny_qwen_training(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from types import SimpleNamespace
        import torch
        from test_training_sequence import AVAILABLE,tiny
        if not AVAILABLE or not torch.cuda.is_available():self.skipTest('Qwen and CUDA required')
        from training import refreshed_speaker_training as trainer
        from training.speaker_preference_adapter import install
        from training.engine import digest,save,write
        from training.targeted_voice import frozen_digest
        torch.set_num_threads(1)
        def model(_):
            value=tiny().requires_grad_(False)
            torch.manual_seed(4811);install(value,rank=2)
            with torch.no_grad():value.speaker_preference_adapter.up.normal_(std=.01)
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
                    self.assertGreater(float(zero['adapter']['up'].abs().sum()),0)
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
