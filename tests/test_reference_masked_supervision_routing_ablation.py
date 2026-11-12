"""Pure synthetic metadata/artifact tests; no tensors, models or audio fixtures."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.score_latest_audio import score
from training.evaluate import ASR_REV
from training import reference_masked_supervision_routing_ablation as m


def cases():
    rows = []
    for seed in (2026,2027):
        for group,count in (('development',32),('custom',8)):
            for i in range(count):
                rows.append(dict(case=f'{group}-{i}',group=group,speaker_id=f'speaker-{i//2}' if group=='development' else 'custom',
                    text='a'*100,reference='reference.wav',reference_sha256='a'*64,reference_text='reference',
                    feature_id=f'feature-{i}',source_pair_id=f'pair-{i}',half_reference=group=='custom',seed=seed))
    return rows


def scored(rows):
    return [dict(r,cosine=.985,terminated=True,hypothesis=r['text'],score=score(r['text'],r['text'],True),
                 asr_revision=ASR_REV,decoding={'do_sample':False}) for r in rows]


def dump(path,value):
    Path(path).write_text(json.dumps(value),encoding='utf-8')


class RoutingAblationTests(unittest.TestCase):
    def test_panel_changes_only_development_route(self):
        original=cases(); before=copy.deepcopy(original); routed=m.panel(original)
        self.assertEqual(original,before); self.assertEqual(len(routed),64)
        self.assertEqual(routed,[dict(r,half_reference=True) for r in original if r['group']=='development'])
        self.assertTrue(all(r['half_reference'] for r in routed))

    def test_scope_rejects_missing_duplicate_wrong_seed_and_route(self):
        for mutation in ('missing','duplicate','seed','route','metadata'):
            rows=cases()
            if mutation=='missing': rows.pop()
            if mutation=='duplicate': rows[-1]=rows[0]
            if mutation=='seed': rows[0]['seed']=999
            if mutation=='route': rows[0]['half_reference']=True
            if mutation=='metadata': rows[0]['text']='changed'
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):m.panel(rows)

    def test_development_scores_require_exact_metadata_finite_cosine_and_coverage(self):
        cs=m.panel(cases()); rows=scored(cs)
        self.assertEqual(m.validate_development(cs,rows)['development']['recordings'],64)
        for field,value in [('cosine',float('nan')),('cosine',1.1),('half_reference',False),('text','different'),('terminated',1)]:
            changed=copy.deepcopy(rows); changed[0][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):m.validate_development(cs,changed)
        changed=copy.deepcopy(rows);changed[0]['score']['character_errors']=1
        with self.assertRaises(ValueError):m.validate_development(cs,changed)
        with self.assertRaises(ValueError):m.validate_development(cs,rows[:-1])

    def test_shared_custom_controls_strict_thresholds_and_no_promotion(self):
        cs=cases(); old=scored(cs); new=scored(m.panel(cs)); before=copy.deepcopy(old)
        good=m.combined(cs,old,new)
        self.assertTrue(good['calibration_thresholds_met']);self.assertEqual(old,before)
        self.assertFalse(good['goal_achieved']);self.assertFalse(good['original114_qualification'])
        for custom_bad,dev_bad,truncated in ((True,False,False),(False,True,False),(False,False,True)):
            a,b=copy.deepcopy(old),copy.deepcopy(new)
            if custom_bad:
                for r in a:
                    if r['group']=='custom':r['cosine']=.984
            if dev_bad:
                for r in b:r.update(hypothesis='a'*97,score=score(r['text'],'a'*97,True))
            if truncated:b[0]['terminated']=False
            with self.subTest(custom=custom_bad,dev=dev_bad,truncated=truncated):
                self.assertFalse(m.combined(cs,a,b)['calibration_thresholds_met'])

    def fixture(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);out=Path(temp.name)/'out';out.mkdir()
        train=out.parent/'train';train.mkdir();dest=m.directory(out,0);dest.mkdir()
        (out/'protocol.json').write_bytes(b'protocol');(train/'checkpoint-000000.pt').write_bytes(b'checkpoint')
        (train/'native-verification.json').write_bytes(b'native audit');ref=out/'reference.wav';ref.write_bytes(b'reference')
        cs=m.panel(cases())
        for c in cs:c.update(reference=str(ref),reference_sha256=m.digest(ref))
        e=dict(cases=cs,train=str(train),features_audit_sha256='f'*64,decoding={'do_sample':False})
        rows=[]
        for i,c in enumerate(cs):
            wav=dest/f'{i}.wav';wav.write_bytes(f'wave{i}'.encode());traj=dest/f'{i}.pt';traj.write_bytes(f'codes{i}'.encode())
            rows.append(dict(c,wav=wav.name,audio_sha256=m.digest(wav),trajectory=traj.name,trajectory_sha256=m.digest(traj),
                cosine=.985,terminated=True,step=0,checkpoint_sha256=m.digest(train/'checkpoint-000000.pt'),
                ablation_protocol_sha256=m.digest(out/'protocol.json')))
        dump(dest/'generated.json',dict(records=rows,all_weights_unchanged=True,
            checkpoint_sha256=m.digest(train/'checkpoint-000000.pt'),ablation_protocol_sha256=m.digest(out/'protocol.json')))
        records=scored(rows);summary=m.validate_development(cs,records)
        dump(dest/'results.json',dict(records=records,summary=summary,truncated=0))
        identity={'original':'x','half':'y'};p={'frozen_identity':identity};movement={'maximum':0.}
        dump(train/'checkpoint-000000.json',dict(sha256=m.digest(train/'checkpoint-000000.pt'),step=0,
            protocol_sha256=m.PROTOCOL_SHA,frozen_identity=identity,adapter_movement=movement))
        audit=dict(status='verified',step=0,recordings=64,native_inputs_verified=64,truncated=0,
            frozen_weights_verified=True,native_conditioning_verified=True,all_artifact_hashes_verified=True,
            all_recomputed_cosines_finite=True,max_cosine_recomputation_error=0.,summary=summary,
            protocol_sha256=m.digest(out/'protocol.json'),ablation_protocol_sha256=m.digest(out/'protocol.json'),
            results_sha256=m.digest(dest/'results.json'),generated_sha256=m.digest(dest/'generated.json'),
            native_verification_sha256=m.digest(train/'native-verification.json'),features_audit_sha256='f'*64,
            checkpoint_sha256=m.digest(train/'checkpoint-000000.pt'),frozen_identity=identity,adapter_movement=movement,**m.FLAGS)
        dump(out/'verification-000000.json',audit)
        return out,p,e,records,audit

    def test_artifacts_and_scored_field_tamper_rejected(self):
        out,p,e,rows,audit=self.fixture();m.bound_generation(out,e,0,rows)
        changed=copy.deepcopy(rows);changed[0]['cosine']=.999
        with self.assertRaises(ValueError):m.bound_generation(out,e,0,changed)
        path=m.directory(out,0)/rows[0]['wav'];path.write_bytes(b'changed')
        with self.assertRaises(ValueError):m.bound_generation(out,e,0,rows)

    def test_native_arm_requires_complete_finite_bound_audit(self):
        out,p,e,rows,audit=self.fixture();self.assertEqual(m.audited_arm(out,p,e,0)[1],rows)
        for field,value in [('recordings',63),('native_inputs_verified',63),('truncated',1),
                ('max_cosine_recomputation_error',float('nan')),('results_sha256','0'*64),('frozen_weights_verified',False)]:
            changed=dict(audit);changed[field]=value;dump(out/'verification-000000.json',changed)
            with self.subTest(field=field),self.assertRaises(ValueError):m.audited_arm(out,p,e,0)

    def test_verify_binds_inherited_features_runtime_and_protocol_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);train=out/'train';train.mkdir();(train/'speech-protocol.json').write_bytes(b'speech')
            p=dict(config={},features='correct',numeric_config={},runtime={})
            original=dict(cases=cases(),features_audit_sha256='a',generation={},decoding={},asr_manifest='asr',asr_manifest_sha256='b')
            e=dict(train=str(train),helper='helper',settings=m.SETTINGS,evidence={},parent={'step':32},
                training_protocol_sha256=m.PROTOCOL_SHA,speech_protocol_sha256=m.digest(train/'speech-protocol.json'),
                original_cases=original['cases'],cases=m.panel(original['cases']),**p,**m.FLAGS)
            e.update({k:v for k,v in original.items() if k!='cases'})
            with patch.object(m,'parent_evidence',return_value=(p,original,{},e['parent'])):
                dump(out/'protocol.json',e);m.verify(out,m.digest(out/'protocol.json'))
                with self.assertRaises(ValueError):m.verify(out,'wrong')
                for field in ('features','runtime','cases'):
                    bad=copy.deepcopy(e);bad[field]='changed';dump(out/'protocol.json',bad)
                    with self.subTest(field=field),self.assertRaises(ValueError):m.verify(out,m.digest(out/'protocol.json'))

    def test_parent_requires_failed_terminal32_and_evidence_gate(self):
        m.validate_test_log('Ran 8 tests in 0.1s\n\nOK\n')
        for invalid in ('Ran 7 tests in 0.1s\nOK', 'skipped one\nRan 8 tests in 0.1s\nOK',
                        'Ran 8 tests in 0.1s\nOK (skipped=1)', 'Ran 8 tests in 0.1s\nFAILED'):
            with self.assertRaises(ValueError):m.validate_test_log(invalid)
        with tempfile.TemporaryDirectory() as tmp:
            train=Path(tmp);(train/'protocol.json').write_bytes(b'parent')
            api=SimpleNamespace(TRAIN=train,metadata=lambda:({}, {}, []),parent_evidence=lambda p:({},dict(step=32,reason='maximum_updates_reached')))
            with patch.object(m,'helper',return_value=api),patch.object(m,'PROTOCOL_SHA',m.digest(train/'protocol.json')):
                self.assertEqual(m.parent_evidence(train,'helper')[3]['step'],32)
                for step,reason in ((8,'maximum_updates_reached'),(32,'calibration_qualified_needs_original114')):
                    api.parent_evidence=lambda p:({},dict(step=step,reason=reason))
                    with self.assertRaises(ValueError):m.parent_evidence(train,'helper')
                def pending(p):raise ValueError('Authoritative closure missing')
                api.parent_evidence=pending
                with self.assertRaisesRegex(ValueError,'closure'):m.parent_evidence(train,'helper')


if __name__=='__main__':unittest.main()
