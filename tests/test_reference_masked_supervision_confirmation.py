"""Pure scope, closure and artifact tests; no model/audio inference."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from scripts.score_latest_audio import score
from training import reference_masked_supervision_confirmation as confirmation
from training.reference_masked_supervision_evaluation import progression


class ConfirmationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.reference=self.root/'reference.wav';self.reference.write_bytes(b'custom original reference')
        self.development=self.root/'development.wav';self.development.write_bytes(b'development original reference')
        self.custom=dict(reference=str(self.reference),reference_sha256=confirmation.digest(self.reference),
            reference_text='original custom reference text',speaker_id='custom')
        cases=[]
        for group,count in (('development',44),('custom',13)):
            for i in range(count):
                ref=self.custom if group=='custom' else dict(reference=str(self.development),
                    reference_sha256=confirmation.digest(self.development),reference_text='original development text',speaker_id='dev'+str(i%10))
                cases.append(dict(case=group+str(i),group=group,text='a'*100,**ref))
        self.panel=dict(cases=cases,seeds=[2026,2027],development_recordings=88,custom_recordings=26,
            criteria=dict(pooled_cer_strictly_below=.03,speaker_macro_cosine_strictly_above=.984,
                groups=['development','custom'],require_complete_panel=True))
        self.cases=confirmation.expanded_panel(self.panel,self.custom)
        self.records=[dict(c,terminated=True,cosine=.985,hypothesis=c['text'],score=score(c['text'],c['text'],True)) for c in self.cases]

    def audited(self):
        return dict(status='verified',recordings=114,native_inputs_verified=114,**confirmation.measurement(self.panel,self.records),
            frozen_weights_verified=True,native_conditioning_verified=True,all_artifact_hashes_verified=True,
            all_recomputed_cosines_finite=True,max_cosine_recomputation_error=0.)

    @staticmethod
    def calibration(step):
        return dict(step=step,status='verified',recordings=80,truncated=0,
            frozen_weights_verified=True,native_conditioning_verified=True,all_artifact_hashes_verified=True,
            all_recomputed_cosines_finite=True,max_cosine_recomputation_error=0.,adapter_movement=dict(maximum=.001 if step else 0.),
            summary={g:dict(recordings=n,speakers=s,pooled_cer=.02 if step else .04,pooled_wer=.05,speaker_macro_cosine=.985)
                for g,n,s in (('development',64,16),('custom',16,1))})

    def test_exact_original_scope_and_metadata_preserved(self):
        measured=confirmation.measurement(self.panel,self.records)
        self.assertTrue(measured['confirmation_pass']);self.assertEqual(len(self.cases),114)
        self.assertEqual(measured['summary']['development']['speakers'],10)
        changed=copy.deepcopy(self.records);changed[0]['reference_text']='changed'
        with self.assertRaisesRegex(ValueError,'case changed'):confirmation.measurement(self.panel,changed)
        changed=copy.deepcopy(self.records);changed[0]['seed']=2028
        with self.assertRaisesRegex(ValueError,'Incomplete'):confirmation.measurement(self.panel,changed)
        changed=copy.deepcopy(self.custom);changed['reference_text']='changed'
        with self.assertRaisesRegex(ValueError,'metadata changed'):confirmation.expanded_panel(self.panel,changed)

    def test_incomplete_duplicate_or_changed_edit_counts_rejected(self):
        for records in (self.records[:-1],self.records[:-1]+[self.records[0]]):
            with self.assertRaisesRegex(ValueError,'Incomplete'):confirmation.measurement(self.panel,records)
        changed=copy.deepcopy(self.records);changed[0]['score']['character_errors']+=1
        with self.assertRaisesRegex(ValueError,'edit counts'):confirmation.measurement(self.panel,changed)
        audit=self.audited();audit['native_inputs_verified']=113
        with self.assertRaisesRegex(ValueError,'Complete original114'):confirmation.complete_audit(audit)

    def test_strict_two_group_boundaries_and_no_truncation(self):
        self.assertTrue(confirmation.complete_audit(self.audited()))
        for group,metric,value in (('development','pooled_cer',.03),('custom','speaker_macro_cosine',.984),
                ('custom','pooled_cer',.04),('development','speaker_macro_cosine',.98)):
            audit=self.audited();audit['summary'][group][metric]=value;audit['confirmation_pass']=False
            self.assertFalse(confirmation.complete_audit(audit))
            audit['confirmation_pass']=True
            with self.assertRaisesRegex(ValueError,'completion flag'):confirmation.complete_audit(audit)
        audit=self.audited();audit['truncated']=1;audit['confirmation_pass']=False
        self.assertFalse(confirmation.complete_audit(audit))

    def test_nan_cosine_summary_and_incomplete_speaker_coverage_rejected(self):
        changed=copy.deepcopy(self.records);changed[0]['cosine']=float('nan')
        with self.assertRaisesRegex(ValueError,'finite'):confirmation.measurement(self.panel,changed)
        for field in ('pooled_cer','pooled_wer','speaker_macro_cosine'):
            audit=self.audited();audit['summary']['custom'][field]=float('nan')
            with self.assertRaisesRegex(ValueError,'Invalid complete'):confirmation.complete_audit(audit)
        audit=self.audited();audit['summary']['development']['speakers']=16
        with self.assertRaisesRegex(ValueError,'Invalid complete'):confirmation.complete_audit(audit)

    def test_qualified_positive_checkpoint_and_authoritative_process_closure_required(self):
        history=[self.calibration(0),self.calibration(8)]
        bindings=dict(driver_sha256='d',terminal_status_sha256='s',driver_inputs_sha256='i',verification_sha256='v')
        status=dict(status='complete',step=8,reason=confirmation.QUALIFICATION,progression=progression(history),
            summary=history[-1]['summary'],original114_confirmation_required=True,goal_achieved=False,
            global_model_selection_changed=False,driver_inputs_sha256='i',verification_sha256='v')
        closed=dict(status='closed',exit_code=0,original_session_id=33720,step=8,
            driver_sha256='d',terminal_status_sha256='s',driver_inputs_sha256='i')
        self.assertEqual(confirmation.validate_parent(status,closed,history,8,bindings),progression(history))
        for changes in ({'status':'running'},{'exit_code':1},{'step':16},{'terminal_status_sha256':'changed'}):
            with self.assertRaisesRegex(ValueError,'process-closed'):
                confirmation.validate_parent(status,dict(closed,**changes),history,8,bindings)
        with self.assertRaisesRegex(ValueError,'Positive registered'):confirmation.validate_parent(status,closed,history,0,bindings)
        bad=copy.deepcopy(history);bad[-1]['summary']['custom']['pooled_cer']=.03
        with self.assertRaisesRegex(ValueError,'qualification'):confirmation.validate_parent(status,closed,bad,8,bindings)

    def test_generated_artifacts_and_scored_metadata_bound_to_all114(self):
        dest=self.root/'speech';dest.mkdir();(self.root/'protocol.json').write_bytes(b'protocol')
        p=dict(cases=self.cases,step=8,checkpoint_sha256='checkpoint',reference_sha256=self.custom['reference_sha256'])
        rows=[]
        for index,c in enumerate(self.cases):
            wav=dest/f'{index}.wav';wav.write_bytes(str(index).encode());trajectory=dest/f'{index}.pt';trajectory.write_bytes(b'codes')
            rows.append(dict(c,wav=wav.name,audio_sha256=confirmation.digest(wav),trajectory=trajectory.name,
                trajectory_sha256=confirmation.digest(trajectory),step=8,checkpoint_sha256='checkpoint',
                protocol_sha256=confirmation.digest(self.root/'protocol.json'),half_reference=c['group']=='custom'))
        generated=dict(records=rows,all_weights_unchanged=True,checkpoint_sha256='checkpoint',
            protocol_sha256=confirmation.digest(self.root/'protocol.json'))
        (dest/'generated.json').write_text(json.dumps(generated))
        self.assertEqual(len(confirmation.bound_generation(self.root,p)['records']),114)
        scored=[dict(r,hypothesis=r['text'],score=score(r['text'],r['text'],True),asr_revision='fixed',decoding={}) for r in rows]
        self.assertEqual(len(confirmation.bound_generation(self.root,p,scored)['records']),114)
        scored[0]['reference_text']='changed'
        with self.assertRaisesRegex(ValueError,'scoring altered'):confirmation.bound_generation(self.root,p,scored)
        (dest/rows[0]['wav']).write_bytes(b'changed waveform')
        with self.assertRaisesRegex(ValueError,'artifact or metadata'):confirmation.bound_generation(self.root,p)


if __name__=='__main__':unittest.main()
