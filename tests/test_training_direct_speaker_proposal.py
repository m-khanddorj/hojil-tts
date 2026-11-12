import tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from tests.test_training_sequence import tiny
from training.speaker_preference_adapter import install as install_speaker
from training import middle_attention_adapter as attention
from training import direct_speaker_proposal as proposal
from training import direct_voice_training as direct
from training.engine import save,write,digest,read


class DirectSpeakerProposalTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4811);torch.set_num_threads(1)
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.out=self.root/'out';self.out.mkdir()
        self.pool=self.root/'pool';self.pool.mkdir();write(self.out/'protocol.json',{'fixture':True})
        self.model=tiny().requires_grad_(False)
        install_speaker(self.model,rank=2);attention.install(self.model,[0,1],rank=2)
        with torch.no_grad():self.model.speaker_preference_adapter.up.normal_(std=.02)
        self.initial=dict(adapter=attention.snapshot(self.model),frozen_backbone_sha256=attention.frozen_digest(self.model),
                          frozen_speaker_sha256=attention.speaker_digest(self.model))
        self.p=dict(pool=str(self.pool),anchors=[])
        for i in range(2):
            t=dict(input_ids=torch.tensor([[1,2,3,10,11,12,4,5,6,7,8]]),ref_ids=torch.tensor([[1,2,3,20,21,4,5]]),
                   codes=torch.randint(0,64,(3+i,16)),ref_codes=torch.randint(0,64,(2,16)),ref_xvec=torch.randn(32),terminated=True)
            path=self.pool/f'{i}.pt';save(path,t)
            self.p['anchors'].append(dict(trajectory=path.name,trajectory_sha256=digest(path)))
        self.model.requires_grad_(False);self.model.speaker_preference_adapter.requires_grad_(True)
        direct.mean_nll(self.model,self.p,self.p['anchors'],backward=True)
        gradient={n:p.grad.detach().cpu().clone() for n,p in self.model.named_parameters() if p.requires_grad}
        self.gradient_path=self.root/'gradients.pt';save(self.gradient_path,dict(voice=gradient,anchor=gradient))
        self.model.zero_grad(set_to_none=True);self.model.speaker_preference_adapter.requires_grad_(False)
        for m in attention.modules(self.model).values():m.lora_A.requires_grad_(True);m.lora_B.requires_grad_(True)
        self.registered=dict(initial_run=str(self.root),gradient_path=str(self.gradient_path),parameters=128)

    def run_proposals(self):
        original_to=torch.Tensor.to
        def cpu_to(t,*args,**kwargs):
            if args and args[0]=='cuda':args=('cpu',)+args[1:]
            return original_to(t,*args,**kwargs)
        with patch.object(proposal,'verify',return_value=self.registered), \
             patch.object(direct,'load',return_value=(self.p,SimpleNamespace(model=self.model),self.initial)), \
             patch.object(torch.Tensor,'to',cpu_to):
            proposal.proposals(self.out)

    def test_proposals_preserve_attention_and_restore_initial_speaker(self):
        initial_speaker=proposal.speaker_state(self.model)
        self.run_proposals();result=read(self.out/'results.json')
        feasible=[r for r in result['proposals'] if r['feasible']]
        self.assertTrue(feasible)
        for row in feasible:
            saved=torch.load(self.out/row['path'],map_location='cpu',weights_only=True)
            self.assertTrue(any(not torch.equal(v,initial_speaker[n]) for n,v in saved['speaker'].items()))
            self.assertLess(row['predicted_voice_change'],0)
            self.assertLessEqual(row['anchor_nll_after'],row['anchor_nll_before']+proposal.SETTINGS['anchor_tolerance'])
        direct.unchanged(self.model,self.initial)
        self.assertTrue(result['initial_model_restored'])
        self.assertTrue(result['training_speech_gate_pending'])

    def test_wrong_cached_anchor_gradient_prevents_candidate_creation(self):
        cached=torch.load(self.gradient_path,map_location='cpu',weights_only=True)
        for v in cached['anchor'].values():v.add_(1)
        save(self.gradient_path,cached)
        with self.assertRaises(AssertionError):self.run_proposals()
        self.assertFalse(list(self.out.glob('candidate-*.pt')))
        self.assertFalse((self.out/'results.json').exists())


if __name__=='__main__':unittest.main()
