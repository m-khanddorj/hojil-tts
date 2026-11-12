import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from training.engine import write,digest,save
from training import reference_profile_preference_training as trainer
from training import reference_profile_preference_collection as collection
from training.targeted_voice import aggregate
from training.voice_preferences import joint_pairs


class ProfilePreferenceTrainingTests(unittest.TestCase):
    def fixture(self):
        cases=[dict(case=f'c{i}',group='profile-preference-train',text='a'*100,speaker_id='custom',reference='r',
            reference_text='reference',reference_sha256='a'*64) for i in range(16)]
        p=dict(cases=cases,settings=collection.SETTINGS)
        rows=[dict(c,seed=seed,terminated=True,cosine=cosine,seconds=4.,
            score=dict(cer=errors/100,character_errors=errors,characters=100,wer=.1,word_errors=1,words=10)) for c in cases
            for seed,cosine,errors in zip(collection.SETTINGS['seeds'],[.98,.985,.99,.981],[3,0,4,0])]
        report=dict(records=rows,summary=aggregate(rows),truncated=0);records=collection.preference_records(p,report)
        pairs=[]
        for case in cases:pairs.extend(joint_pairs([r for r in records if r['case']==case['case']],.001))
        audit=dict(status='verified',recordings=64,truncated=0,all_weights_unchanged=True,
            maximum_cosine_recomputation_error=0,summary=report['summary'])
        data=dict(records=records,pairs=pairs,pair_count=len(pairs),cases_with_pairs=16,total_cases=16)
        return p,report,audit,data

    def test_complete_generated_training_pool_and_exact_pair_rule(self):
        p,report,audit,data=self.fixture();records,pairs=trainer.pool_payload(p,report,audit,data)
        self.assertEqual(len(records),64);self.assertTrue(pairs)
        statistics=trainer.pool_statistics(records,pairs);self.assertEqual(statistics['cases_with_pairs'],16)
        self.assertAlmostEqual(statistics['pronunciation_improving_case_balanced_mass']+statistics['voice_only_case_balanced_mass'],1.)
        changed=copy.deepcopy(data);changed['pairs']=changed['pairs'][:-1]
        with self.assertRaises(ValueError):trainer.pool_payload(p,report,audit,changed)
        changed=copy.deepcopy(report);changed['records'][0]['group']='profile-validation'
        with self.assertRaises(ValueError):trainer.pool_payload(p,changed,audit,data)
        with self.assertRaises(ValueError):trainer.pool_payload(p,report,dict(audit,truncated=1),data)
        with tempfile.TemporaryDirectory() as tmp:
            pool=Path(tmp)/'pool';write(Path(tmp)/'reference-profile-preferences-status.json',dict(status='running'))
            with self.assertRaises(ValueError):trainer.pool_evidence(pool)

    def test_optimizer_checkpoint_restoration_and_tamper_checks(self):
        value=torch.nn.Parameter(torch.zeros(1024));optimizer=torch.optim.AdamW([value],lr=.0005,weight_decay=0.)
        for _ in range(4):
            value.grad=torch.ones_like(value);optimizer.step();optimizer.zero_grad(set_to_none=True)
        state=optimizer.state_dict();trainer.validate_optimizer(state,4)
        other=torch.nn.Parameter(value.detach().clone());restored=torch.optim.AdamW([other],lr=.0005,weight_decay=0.)
        restored.load_state_dict(copy.deepcopy(state));value.grad=torch.ones_like(value);other.grad=torch.ones_like(other)
        optimizer.step();restored.step();torch.testing.assert_close(value,other)
        bad=copy.deepcopy(state);bad['state'][0]['exp_avg_sq'][0]=-1
        with self.assertRaises(ValueError):trainer.validate_optimizer(bad,5)
        # Capture the fifth update as a declared test checkpoint to isolate artifact integrity.
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp);p=dict(checkpoints=[0,5],profile_reference=dict(reference_sha256='a'*64),settings=trainer.SETTINGS)
            write(run/'protocol.json',p);write(run/'references.json',dict(fixed=True))
            checkpoint=dict(step=5,profile=dict(raw=value.detach().clone(),reference_sha256='a'*64,radius_fraction=.05),
                optimizer=optimizer.state_dict(),protocol_sha256=digest(run/'protocol.json'),reference_cache_sha256=digest(run/'references.json'))
            trainer.save_checkpoint(run,checkpoint);read_back=trainer.read_checkpoint(run,5,p)
            torch.testing.assert_close(read_back['profile']['raw'],checkpoint['profile']['raw'])
            with self.assertRaises(ValueError):trainer.save_checkpoint(run,checkpoint)
            changed=copy.deepcopy(checkpoint);changed['profile']['reference_sha256']='b'*64
            path=trainer.checkpoint_path(run,5);save(path,changed)
            with self.assertRaises(ValueError):trainer.read_checkpoint(run,5,p)
            write(path.with_suffix('.json'),dict(step=5,sha256=digest(path),protocol_sha256=digest(run/'protocol.json')))
            with self.assertRaises(ValueError):trainer.read_checkpoint(run,5,p)

    def test_native_trajectory_conditioning_rejects_text_or_reference_changes(self):
        ref_vector=torch.randn(32);ref_codes=torch.ones((2,16),dtype=torch.long)
        prompt=SimpleNamespace(ref_spk_embedding=ref_vector,ref_code=ref_codes)
        wrapper=SimpleNamespace(create_voice_clone_prompt=lambda **kwargs:[prompt],
            model=SimpleNamespace(config=SimpleNamespace(talker_config=SimpleNamespace(codec_eos_token_id=63))))
        records=[dict(sample_id=f's{i}',case='same',text='target',reference_sha256='a'*64,terminated=True) for i in range(2)]
        codes=torch.full((3,16),2,dtype=torch.long)
        base=dict(codes=codes,first_sequence=torch.tensor([[2,2,2,63]]),terminated=True,reference_sha256='a'*64,
            ref_codes=ref_codes,ref_xvec=ref_vector,input_ids=torch.tensor([[1]]),ref_ids=torch.tensor([[3]]))
        trajectories={r['sample_id']:copy.deepcopy(base) for r in records}
        p=dict(pool='unused',profile_reference=dict(reference='r',reference_text='reference'))
        with patch.object(trainer,'ids',side_effect=lambda wrapper,text,is_ref=False:torch.tensor([[3 if is_ref else 1]])), \
             patch.object(trainer.collection.direct,'trajectory',side_effect=lambda pool,r:trajectories[r['sample_id']]):
            self.assertEqual(len(trainer.load_trajectories(p,wrapper,records)),2)
            trajectories['s1']['input_ids']=torch.tensor([[9]])
            with self.assertRaises(ValueError):trainer.load_trajectories(p,wrapper,records)
            trajectories['s1']=copy.deepcopy(base);trajectories['s1']['ref_xvec'][0]+=1
            with self.assertRaises(ValueError):trainer.load_trajectories(p,wrapper,records)


if __name__=='__main__':unittest.main()
