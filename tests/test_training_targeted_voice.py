import copy
import unittest
import torch
from training.targeted_voice import text_adapter, frozen_digest, joint_gate


class TextProjection(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_fc1=torch.nn.Linear(4,8)
        self.linear_fc2=torch.nn.Linear(8,4)
    def forward(self,x):
        return self.linear_fc2(torch.relu(self.linear_fc1(x)))


class TargetedVoiceTests(unittest.TestCase):
    def test_only_text_adapter_updates_and_zero_init_preserves_output(self):
        model=torch.nn.Module();model.talker=torch.nn.Module()
        model.talker.text_projection=TextProjection()
        model.talker.decoder=torch.nn.Linear(4,4)
        inputs=torch.randn(6,4)
        def forward():return model.talker.decoder(model.talker.text_projection(inputs))
        initial=forward().detach().clone()
        named=text_adapter(model,rank=2,dropout=0)
        torch.testing.assert_close(initial,forward(),rtol=0,atol=0)
        frozen=frozen_digest(model)
        optimizer=torch.optim.AdamW([p for _,p in named],lr=0.01)
        for _ in range(2):
            optimizer.zero_grad();forward().square().mean().backward();optimizer.step()
        self.assertEqual(frozen,frozen_digest(model))
        self.assertFalse(torch.equal(initial,forward()))
        self.assertEqual(len(named),4)
        self.assertTrue(all(p.dtype==torch.float32 for _,p in named))

    def records(self):
        return [dict(group=g,seed=s,case='001',text='a b',reference_sha256='ref',speaker_id=g,
                     cosine=.98,score=dict(cer=.1,wer=.5,word_errors=1,words=2,character_errors=1,characters=10))
                for g in ('development','custom') for s in (2026,2027)]

    def improved(self):
        new=copy.deepcopy(self.records())
        for r in new:r['score'].update(cer=0,wer=0,word_errors=0,character_errors=0)
        return new

    def test_joint_improvement_is_eligible_but_identical_is_not(self):
        self.assertTrue(joint_gate(self.records(),self.improved())['eligible'])
        self.assertFalse(joint_gate(self.records(),self.records())['eligible'])

    def test_better_pronunciation_cannot_override_voice_regression(self):
        new=self.improved();new[0]['cosine']=.97
        self.assertFalse(joint_gate(self.records(),new)['eligible'])
        self.assertFalse(joint_gate(self.records(),new)['voice_pass'])

    def test_bad_case_cannot_be_hidden_by_a_better_average(self):
        new=self.improved();new[0]['cosine']=.97;new[1]['cosine']=.999
        self.assertFalse(joint_gate(self.records(),new)['voice_pass'])

    def test_seed_regression_cannot_be_hidden_by_pooled_improvement(self):
        new=self.improved();new[0]['score'].update(wer=1,word_errors=2)
        self.assertFalse(joint_gate(self.records(),new)['pronunciation_pass'])

    def test_incomplete_duplicate_and_changed_references_fail(self):
        a=self.records()
        with self.assertRaises(ValueError):joint_gate(a,a[:-1])
        with self.assertRaises(ValueError):joint_gate(a,a+[a[0]])
        b=copy.deepcopy(a);b[0]['reference_sha256']='changed'
        with self.assertRaises(ValueError):joint_gate(a,b)
