from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import torch
from tests.test_training_sequence import tiny
from training.speaker_preference_adapter import install as install_speaker
from training import middle_attention_adapter as adapters
from training import direct_voice_training as trainer
from training.codec_policy_replay import policy_outputs
from training.engine import write,save,digest,read


class DirectVoiceTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.run=Path(self.tmp.name)/'run';self.run.mkdir();self.pool=Path(self.tmp.name)/'pool';self.pool.mkdir()
        self.model=tiny().requires_grad_(False);install_speaker(self.model,rank=2);adapters.install(self.model,[0,1],rank=2)
        with torch.no_grad():
            for m in adapters.modules(self.model).values():m.lora_B.normal_(std=.002)
        self.state=dict(step=0,adapter=adapters.snapshot(self.model),optimizer=None,
            frozen_backbone_sha256=adapters.frozen_digest(self.model),frozen_speaker_sha256=adapters.speaker_digest(self.model))
        write(self.run/'protocol.json',dict(test_fixture=True));save(self.run/'checkpoint-000000.pt',self.state)
        out=self.run/'rollout-000001';out.mkdir();records=[];posts=[];grads=[];anchors=[]
        for i in range(2):
            t=dict(input_ids=torch.tensor([[1,2,3,10,11,12,4,5,6,7,8]]),ref_ids=torch.tensor([[1,2,3,20,21,4,5]]),
                codes=torch.randint(0,64,(3+i,16)),ref_codes=torch.randint(0,64,(2,16)),ref_xvec=torch.randn(32),terminated=True)
            save(out/f'{i}.pt',t);save(self.pool/f'{i}.pt',t)
            r=dict(case=str(i),seed=i,trajectory=f'{i}.pt',trajectory_sha256=digest(out/f'{i}.pt'))
            records.append(r);anchors.append(dict(trajectory=f'{i}.pt',trajectory_sha256=digest(self.pool/f'{i}.pt')))
            with torch.no_grad():probs=policy_outputs(self.model,t)['probabilities']
            selected=torch.nn.functional.one_hot(t['codes'].T,64).float()[None]
            gradient=-selected/probs.clamp_min(1e-9)/t['codes'].numel()
            save(out/f'p{i}.pt',probs);save(out/f'g{i}.pt',gradient)
            posts.append(dict(case=str(i),seed=i,path=f'p{i}.pt',sha256=digest(out/f'p{i}.pt')))
            grads.append(dict(case=str(i),seed=i,path=f'g{i}.pt',sha256=digest(out/f'g{i}.pt'),loss=.01))
        write(out/'results.json',dict(records=records,summary={'fitting':{'fixture':True}}))
        write(out/'posteriors.json',dict(records=posts,checkpoint_sha256=digest(self.run/'checkpoint-000000.pt')))
        write(out/'gradients.json',dict(records=grads,frozen_weights_unchanged=True,posteriors_sha256=digest(out/'posteriors.json')))
        self.p=dict(pool=str(self.pool),anchors=anchors,validation=anchors)
        self.settings=dict(trainer.SETTINGS,recordings_per_update=2,weight_radius=.1)

    def test_real_optimizer_update_is_constrained_and_keeps_original_weights(self):
        with patch.object(trainer,'SETTINGS',self.settings),patch.object(trainer,'load',return_value=(self.p,SimpleNamespace(model=self.model),self.state)):
            trainer.update(self.run,1)
        result=torch.load(self.run/'checkpoint-000001.pt',map_location='cpu',weights_only=True)
        self.assertEqual(result['step'],1)
        self.assertTrue(any(not torch.equal(v,self.state['adapter'][k]) for k,v in result['adapter'].items()))
        self.assertLessEqual(result['metrics']['anchor_nll_after'],result['metrics']['anchor_nll_before']+self.settings['anchor_tolerance'])
        self.assertLess(result['metrics']['backtracking'][-1]['predicted_voice_change'],0)
        self.assertEqual(adapters.frozen_digest(self.model),self.state['frozen_backbone_sha256'])
        self.assertEqual(adapters.speaker_digest(self.model),self.state['frozen_speaker_sha256'])
        self.assertEqual(read(self.run/'checkpoint-000001.json')['sha256'],digest(self.run/'checkpoint-000001.pt'))
        for item in result['optimizer']['state'].values():
            self.assertEqual(item['exp_avg'].dtype,torch.float32)
            self.assertEqual(item['exp_avg_sq'].dtype,torch.float32)

    def test_failed_anchor_backtracking_restores_adapter_and_saves_no_checkpoint(self):
        original=trainer.mean_nll
        def forced_failure(model,p,records,backward=False):
            return original(model,p,records,backward=True) if backward else 1e9
        with patch.object(trainer,'SETTINGS',self.settings),patch.object(trainer,'load',return_value=(self.p,SimpleNamespace(model=self.model),self.state)),patch.object(trainer,'mean_nll',side_effect=forced_failure):
            with self.assertRaisesRegex(ValueError,'anchor constraint'):trainer.update(self.run,1)
        self.assertFalse((self.run/'checkpoint-000001.pt').exists())
        self.assertTrue((self.run/'rollout-000001/rejected-update.json').exists())
        self.assertTrue(all(torch.equal(v,self.state['adapter'][k]) for k,v in adapters.snapshot(self.model).items()))


if __name__=='__main__':unittest.main()
