import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from training.engine import write,digest
from training.reference_profile_preference_confirmation import validate_entry,confirmation_measurement,cosine_error,prepare,verify
from training.targeted_voice import aggregate


class PreferenceConfirmationTests(unittest.TestCase):
    def fixture(self):
        cases=[dict(case=f'dev-{i}',group='development',speaker_id=str(i%10),text='a'*100,
            reference='r',reference_sha256=f'dev-ref-{i%10}',reference_text='ref',row_id=f'row-{i}') for i in range(44)]
        cases += [dict(case=f'custom-{i}',group='custom',speaker_id='custom',text='a'*100,
            reference='c',reference_sha256='custom-ref',reference_text='ref',text_source_row_id=str(i)) for i in range(13)]
        panel=dict(cases=cases,seeds=[2026,2027],development_recordings=88,custom_recordings=26,
            criteria=dict(pooled_cer_strictly_below=.03,speaker_macro_cosine_strictly_above=.984,
                groups=['development','custom'],require_complete_panel=True))
        def scored(c,seed):
            return dict(c,seed=seed,terminated=True,cosine=.986,audio_sha256=f"{c['case']}-{seed}",
                score=dict(cer=.01,wer=.05,character_errors=1,characters=100,word_errors=1,words=20))
        full=[scored(c,s) for c in cases for s in panel['seeds']]
        selected={c['case'] for c in cases[:10]+cases[44:47]};screen=[r for r in full if r['case'] in selected]
        probes=[scored(dict(cases[44],case=f'fit-{i}',group='profile-preference-train'),41301) for i in range(16)]
        probes += [scored(dict(cases[44],case=f'val-{i}',group='profile-validation'),s) for i in range(4) for s in (41201,41202)]
        return panel,full,screen,probes

    def entry(self,screen,probes):
        fields=('case','speaker_id','group','text','reference','reference_sha256','reference_text','seed')
        cases={kind:[{k:r[k] for k in fields} for r in rows] for kind,rows in [('probes',probes),('screen',screen)]}
        reports={kind:dict(records=rows,summary=aggregate(rows),truncated=sum(not r['terminated'] for r in rows))
            for kind,rows in [('probes',probes),('screen',screen)]}
        audits={kind:dict(status='verified',step=4,kind=kind,recordings=len(report['records']),all_weights_unchanged=True,
            summary=report['summary'],truncated=report['truncated'],maximum_cosine_recomputation_error=0,
            baseline_waveforms_checked=20 if kind=='screen' else 0) for kind,report in reports.items()}
        receipt=dict(status='verified',step=4,recordings=50,all_weights_unchanged=True,inactive_development_exact_parity=True,
            goal_screen=True,all_baseline_waveforms_exact=False,summary=aggregate(screen+probes),
            screen_summary=reports['screen']['summary'],screen_truncated=reports['screen']['truncated'],
            truncated=sum(not r['terminated'] for r in screen+probes))
        return cases,reports,audits,receipt

    def test_complete50_entry_and_independent_audit_coverage(self):
        _,_,screen,probes=self.fixture();args=self.entry(screen,probes);validate_entry(*args,4)
        changed=copy.deepcopy(args);changed[1]['probes']['records'].pop()
        with self.assertRaises(ValueError):validate_entry(*changed,4)
        changed=copy.deepcopy(args);changed[1]['screen']['records'][-1]=changed[1]['screen']['records'][0]
        with self.assertRaises(ValueError):validate_entry(*changed,4)
        changed=copy.deepcopy(args);changed[2]['screen']['maximum_cosine_recomputation_error']=float('nan')
        with self.assertRaises(ValueError):validate_entry(*changed,4)
        changed=copy.deepcopy(args);changed[3]['inactive_development_exact_parity']=False
        with self.assertRaises(ValueError):validate_entry(*changed,4)
        self.assertAlmostEqual(cosine_error(.981,.98),.001)
        for first,second in ((float('nan'),.98),(float('inf'),.98),(.98,float('nan'))):
            with self.assertRaises(ValueError):cosine_error(first,second)

    def test_actual_strict_two_group_screen_thresholds(self):
        _,_,screen,probes=self.fixture()
        for group in ('development','custom'):
            for metric in ('cer','cosine'):
                changed=copy.deepcopy(screen)
                for r in changed:
                    if r['group']==group:
                        if metric=='cer':r['score'].update(cer=.03,character_errors=3)
                        else:r['cosine']=.984
                with self.assertRaises(ValueError):validate_entry(*self.entry(changed,probes),4)
        changed=copy.deepcopy(screen);changed[-1]['terminated']=False
        with self.assertRaises(ValueError):validate_entry(*self.entry(changed,probes),4)

    def test_full114_scope_metadata_and_screen_waveform_parity(self):
        panel,full,screen,_=self.fixture();result=confirmation_measurement(panel,full,screen)
        self.assertTrue(result['confirmation_pass']);self.assertEqual(result['screening_recordings_reproduced_exactly'],26)
        self.assertEqual(result['summary']['development']['recordings'],88);self.assertEqual(result['summary']['custom']['recordings'],26)
        for changed in (full[:-1],full[:-1]+[full[0]]):
            with self.assertRaises(ValueError):confirmation_measurement(panel,changed,screen)
        changed=copy.deepcopy(full);changed[0]['audio_sha256']='changed'
        with self.assertRaisesRegex(ValueError,'parity'):confirmation_measurement(panel,changed,screen)
        for index,key in ((0,'row_id'),(-1,'text_source_row_id')):
            changed=copy.deepcopy(full);changed[index][key]='changed'
            with self.assertRaisesRegex(ValueError,'case changed'):confirmation_measurement(panel,changed,screen)
        changed=copy.deepcopy(full);changed[-1]['terminated']=False
        self.assertFalse(confirmation_measurement(panel,changed,screen)['confirmation_pass'])
        weaker=copy.deepcopy(panel);weaker['criteria']['pooled_cer_strictly_below']=.04
        with self.assertRaises(ValueError):confirmation_measurement(weaker,full,screen)

    def test_no_output_before_terminal_passing_dpo_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);run=root/'training';out=root/'confirmation';run.mkdir()
            with patch('training.reference_profile_preference_confirmation.evaluation.verify',return_value=({'checkpoints':[0,4,8,16]}, {}, {})):
                for state in (dict(status='running'),dict(status='complete',reason='maximum_updates_reached')):
                    write(root/'reference-profile-preference-training-status.json',state)
                    with self.assertRaises(ValueError):prepare(run,out,root/'unused-driver.py')
                    self.assertFalse(out.exists())
            # A same-revision replacement manifest must fail before any model or
            # expensive checkpoint verification, even if it declares new hashes.
            asr=root/'asr';asr.mkdir();manifest=asr/'download-manifest.json'
            content=dict(repo='BuzzASR/mongolian',revision='unchanged-revision',files={'weights':{'sha256':'original'}})
            write(manifest,content);out.mkdir()
            write(out/'protocol.json',dict(training_run=str(run),config=dict(asr=str(asr)),asr_manifest_sha256=digest(manifest)))
            with patch('training.reference_profile_preference_confirmation.evidence',side_effect=ValueError('passed manifest gate')) as gate:
                with self.assertRaisesRegex(ValueError,'passed manifest gate'):verify(out)
                gate.assert_called_once()
                content['files']['weights']['sha256']='replacement';write(manifest,content);gate.reset_mock()
                with self.assertRaisesRegex(ValueError,'ASR manifest changed'):verify(out)
                gate.assert_not_called()


if __name__=='__main__':unittest.main()
