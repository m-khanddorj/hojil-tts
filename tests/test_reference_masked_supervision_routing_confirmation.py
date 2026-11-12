"""Pure conditional-selection, closure and original114 artifact tests."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.score_latest_audio import score
from training import reference_masked_supervision_routing_confirmation as m


def arms():
    return {str(step):dict(truncated=0,calibration_thresholds_met=True,
        half_development_shared_custom_summary={g:dict(recordings=n,speakers=s,pooled_cer=.01 if step==0 else .02,
            pooled_wer=.03,speaker_macro_cosine=.986) for g,n,s in (('development',64,16),('custom',16,1))})
        for step in (0,32)}


def dump(path,value):Path(path).write_text(json.dumps(value),encoding='utf-8')


class RoutingConfirmationTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);self.root=Path(temp.name)
        self.ref=self.root/'custom.wav';self.ref.write_bytes(b'custom');self.dev=self.root/'dev.wav';self.dev.write_bytes(b'dev')
        self.custom=dict(reference=str(self.ref),reference_sha256=m.digest(self.ref),reference_text='custom ref',speaker_id='custom')
        cases=[]
        for group,count in (('development',44),('custom',13)):
            for i in range(count):
                ref=self.custom if group=='custom' else dict(reference=str(self.dev),reference_sha256=m.digest(self.dev),
                    reference_text='development ref',speaker_id=f'dev{i%10}')
                cases.append(dict(case=f'{group}{i}',group=group,text='a'*100,**ref))
        self.panel=dict(cases=cases,seeds=[2026,2027],development_recordings=88,custom_recordings=26,
            criteria=dict(pooled_cer_strictly_below=.03,speaker_macro_cosine_strictly_above=.984,
                groups=['development','custom'],require_complete_panel=True))
        self.cases=[dict(c,half_reference=True) for c in m.expanded_panel(self.panel,self.custom)]
        self.rows=[dict(c,terminated=True,cosine=.985,hypothesis=c['text'],score=score(c['text'],c['text'],True)) for c in self.cases]

    def audited(self):
        return dict(status='verified',recordings=114,native_inputs_verified=114,**m.measurement(self.panel,self.rows),
            frozen_weights_verified=True,native_conditioning_verified=True,all_artifact_hashes_verified=True,
            all_recomputed_cosines_finite=True,max_cosine_recomputation_error=0.)

    def test_zero_control_never_selected_even_if_its_calibration_is_better(self):
        original=arms();before=copy.deepcopy(original);result=m.selection(original)
        self.assertEqual(result['step'],32);self.assertEqual(result['eligible_steps'],[32]);self.assertEqual(original,before)
        self.assertFalse(result['goal_achieved']);self.assertEqual(result['rule']['eligible_checkpoints'],[32])

    def test_no_zero_fallback_at_strict_group_boundary_or_truncation(self):
        for group,field,value in [('development','pooled_cer',.03),('custom','speaker_macro_cosine',.984),
                ('custom','pooled_cer',.04),('development','speaker_macro_cosine',.98)]:
            changed=arms();changed['32']['half_development_shared_custom_summary'][group][field]=value
            changed['32']['calibration_thresholds_met']=False
            with self.subTest(group=group,field=field),self.assertRaises(ValueError):m.selection(changed)
        changed=arms();changed['32'].update(truncated=1,calibration_thresholds_met=False)
        with self.assertRaises(ValueError):m.selection(changed)

    def test_both_arms_coverage_finite_metrics_and_honest_pass_flag_required(self):
        for mutation in ('missing','nan','incomplete','falseflag'):
            changed=arms()
            if mutation=='missing':del changed['0']
            if mutation=='nan':changed['0']['half_development_shared_custom_summary']['custom']['pooled_cer']=float('nan')
            if mutation=='incomplete':changed['0']['half_development_shared_custom_summary']['development']['recordings']=63
            if mutation=='falseflag':changed['32']['calibration_thresholds_met']=False
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):m.selection(changed)

    def test_authoritative_original23429_closed_success_required(self):
        status=dict(status='complete',**m.FLAGS);bindings=dict(protocol_sha256='p',results_sha256='r',terminal_status_sha256='s',driver_sha256='d')
        closed=dict(status='closed',exit_code=0,original_session_id=23429,**bindings)
        m.validate_closure(status,closed,bindings,23429)
        for changes in ({'status':'running'},{'exit_code':1},{'exit_code':False},{'original_session_id':33720},{'results_sha256':'x'}):
            with self.subTest(changes=changes),self.assertRaises(ValueError):m.validate_closure(status,dict(closed,**changes),bindings,23429)
        with self.assertRaises(ValueError):m.validate_closure(dict(status,status='running'),closed,bindings,23429)

    def test_exact114_metadata_and_original_speaker_scope(self):
        self.assertEqual(len(self.cases),114);self.assertTrue(all(c['half_reference'] for c in self.cases))
        self.assertEqual(m.measurement(self.panel,self.rows)['summary']['development']['speakers'],10)
        for change in ('text','reference_text','seed','missing'):
            rows=copy.deepcopy(self.rows)
            if change=='missing':rows.pop()
            elif change=='seed':rows[0]['seed']=9
            else:rows[0][change]='changed'
            with self.subTest(change=change),self.assertRaises(ValueError):m.measurement(self.panel,rows)

    def test_full_native_completion_rejects_boundary_nan_or_missing_coverage(self):
        self.assertTrue(m.complete_audit(self.audited()))
        for field in ('pooled_cer','pooled_wer','speaker_macro_cosine'):
            audit=self.audited();audit['summary']['custom'][field]=float('nan')
            with self.assertRaises(ValueError):m.complete_audit(audit)
        audit=self.audited();audit['summary']['custom']['speaker_macro_cosine']=.984;audit['confirmation_pass']=False
        self.assertFalse(m.complete_audit(audit))
        for key,value in [('recordings',113),('native_inputs_verified',113),('native_conditioning_verified',False)]:
            audit=self.audited();audit[key]=value
            with self.assertRaises(ValueError):m.complete_audit(audit)

    def test_generated_all_half_metadata_scores_and_artifacts_are_bound(self):
        dest=self.root/'speech';dest.mkdir();(self.root/'protocol.json').write_bytes(b'protocol')
        p=dict(cases=self.cases,step=32,checkpoint_sha256='checkpoint');rows=[]
        for i,c in enumerate(self.cases):
            wav=dest/f'{i}.wav';wav.write_bytes(f'audio{i}'.encode());traj=dest/f'{i}.pt';traj.write_bytes(b'codes')
            rows.append(dict(c,wav=wav.name,audio_sha256=m.digest(wav),trajectory=traj.name,trajectory_sha256=m.digest(traj),
                step=32,checkpoint_sha256='checkpoint',protocol_sha256=m.digest(self.root/'protocol.json')))
        payload=dict(records=rows,all_weights_unchanged=True,checkpoint_sha256='checkpoint',protocol_sha256=m.digest(self.root/'protocol.json'))
        dump(dest/'generated.json',payload);self.assertEqual(len(m.bound_generation(self.root,p)['records']),114)
        changed=copy.deepcopy(payload);changed['records'][0]['half_reference']=False;dump(dest/'generated.json',changed)
        with self.assertRaises(ValueError):m.bound_generation(self.root,p)
        dump(dest/'generated.json',payload);(dest/rows[0]['wav']).write_bytes(b'changed')
        with self.assertRaises(ValueError):m.bound_generation(self.root,p)

    def test_completion_written_only_after_bound_confirmation_exit_zero(self):
        out=self.root/'confirmation';out.mkdir();(out/'speech').mkdir()
        for path in (out/'protocol.json',out/'verification.json',out/'speech/results.json'):path.write_bytes(b'bound')
        p=dict(driver_sha256='driver',checkpoint_sha256='checkpoint');audit=self.audited()
        status=dict(status='complete',step=32,confirmation_pass=True,protocol_sha256=m.digest(out/'protocol.json'),
            verification_sha256=m.digest(out/'verification.json'),summary=audit['summary'],truncated=0,logs={},**m.FLAGS)
        for phase in ('generate','score','audit','verify'):
            log=self.root/f'{phase}.log';log.write_bytes(b'phase');status['logs'][phase]=dict(path=str(log),sha256=m.digest(log),exit_code=0)
        status_path=self.root/'confirmation-status.json';dump(status_path,status)
        closed=dict(status='closed',exit_code=1,original_session_id=999,
            protocol_sha256=m.digest(out/'protocol.json'),driver_sha256='driver',terminal_status_sha256=m.digest(status_path),
            verification_sha256=m.digest(out/'verification.json'),results_sha256=m.digest(out/'speech/results.json'))
        path=self.root/'confirmation-process-closed.json';dump(path,closed)
        with patch.object(m,'verify',return_value=(p,{},{})),patch.object(m,'verified_result',return_value=audit):
            with self.assertRaises(ValueError):m.complete(out,m.digest(out/'protocol.json'),path)
            self.assertFalse((out/'completion.json').exists())
            closed['exit_code']=0;dump(path,closed);m.complete(out,m.digest(out/'protocol.json'),path)
            receipt=m.read(out/'completion.json');self.assertEqual(receipt['status'],'verified_complete');self.assertFalse(receipt['goal_achieved'])


if __name__=='__main__':unittest.main()
