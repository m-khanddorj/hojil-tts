import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from training.engine import write,digest
from training.reference_prefix_generation_pad_confirmation import (
    validate_parent,validate_scope,validate_panel_registration,measurement,cosine_error,prepare,verify)
from training.targeted_voice import aggregate


class PadConfirmationTests(unittest.TestCase):
    def fixture(self):
        cases=[dict(case=f'dev-{i}',group='development',speaker_id=str(i%10),text='a'*100,reference='r',
            reference_sha256='b'*64,reference_text='ref',row_id=str(i)) for i in range(44)]
        cases += [dict(case=f'custom-{i}',group='custom',speaker_id='custom',text='a'*100,reference='custom',
            reference_sha256='a'*64,reference_text='ref',text_source_row_id=str(i)) for i in range(13)]
        panel=dict(cases=cases,seeds=[2026,2027],development_recordings=88,custom_recordings=26,
            criteria=dict(pooled_cer_strictly_below=.03,speaker_macro_cosine_strictly_above=.984,
                groups=['development','custom'],require_complete_panel=True))
        rows=[dict(c,seed=seed,terminated=True,cosine=.986,audio_sha256=f"{c['case']}-{seed}",
            score=dict(cer=.01,wer=.05,character_errors=1,characters=100,word_errors=1,words=20)) for seed in panel['seeds'] for c in cases]
        selected={c['case'] for c in cases[:10]+cases[44:47]};screen=[r for r in rows if r['case'] in selected]
        keys=('case','group','speaker_id','text','reference','reference_sha256','reference_text','seed')
        screen_cases=[{k:r[k] for k in keys} for r in screen]
        return panel,rows,screen,screen_cases

    def parent_fixture(self):
        panel,rows,screen,cases=self.fixture();control=copy.deepcopy(screen)
        for r in control:
            if r['group']=='custom':r['score'].update(cer=.04,character_errors=4);r['cosine']=.98
        reports={stage:dict(records=data,summary=aggregate(data),truncated=0) for stage,data in [('control',control),('full-pad',screen)]}
        audits={stage:dict(status='verified',stage=stage,recordings=26,all_weights_unchanged=True,
            all_recomputed_cosines_finite=True,maximum_cosine_recomputation_error=0.,optimizer_updates=0,
            summary=report['summary'],truncated=0,baseline_waveforms_checked=26 if stage=='control' else 20,
            paired_prefill_and_first_frame_verified=0 if stage=='control' else 26,goal_screen=stage=='full-pad',model_identity={'same':'identity'})
            for stage,report in reports.items()}
        return dict(status='complete',reason='needs_broader_confirmation'),cases,reports,audits

    def test_terminal_complete_audits_and_strict_both_groups(self):
        args=self.parent_fixture();validate_parent(*args)
        changed=copy.deepcopy(args);changed[0]['status']='running'
        with self.assertRaises(ValueError):validate_parent(*changed)
        for key,value in [('recordings',25),('paired_prefill_and_first_frame_verified',25),('all_recomputed_cosines_finite',False),
                          ('maximum_cosine_recomputation_error',float('nan'))]:
            changed=copy.deepcopy(args);changed[3]['full-pad'][key]=value
            with self.assertRaises(ValueError):validate_parent(*changed)
        changed=copy.deepcopy(args);changed[2]['control']['records'].pop()
        with self.assertRaises(ValueError):validate_parent(*changed)
        for group in ('custom','development'):
            for metric in ('cer','cosine','termination'):
                changed=copy.deepcopy(args);records=changed[2]['full-pad']['records']
                for r in records:
                    if r['group']==group:
                        if metric=='cer':r['score'].update(cer=.03,character_errors=3)
                        elif metric=='cosine':r['cosine']=.984
                        else:r['terminated']=False
                summary=aggregate(records);truncated=sum(not r['terminated'] for r in records)
                changed[2]['full-pad'].update(summary=summary,truncated=truncated)
                changed[3]['full-pad'].update(summary=summary,truncated=truncated,goal_screen=False)
                with self.assertRaises(ValueError):validate_parent(*changed)

    def test_full_original_scope_metadata_and_reference_dispatch(self):
        panel,_,_,cases=self.fixture();validate_scope(panel,cases,'a'*64)
        for field,value in [('seeds',[2026]),('development_recordings',86)]:
            changed=copy.deepcopy(panel);changed[field]=value
            with self.assertRaises(ValueError):validate_scope(changed,cases,'a'*64)
        for index in (0,56):
            changed=copy.deepcopy(panel);changed['cases'][index]['reference_sha256']='a'*64 if index==0 else 'b'*64
            with self.assertRaises(ValueError):validate_scope(changed,cases,'a'*64)
        changed=copy.deepcopy(cases);changed[0]['text']='changed'
        with self.assertRaises(ValueError):validate_scope(panel,changed,'a'*64)
        changed=copy.deepcopy(panel);changed['criteria']['pooled_cer_strictly_below']=.04
        with self.assertRaises(ValueError):validate_scope(changed,cases,'a'*64)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'original-panel.json';write(path,panel)
            registration=dict(panel=str(path),panel_sha256=digest(path));validate_panel_registration(path,registration)
            changed=copy.deepcopy(panel);changed['cases'][20]['text']='changed unscreened text'
            validate_scope(changed,cases,'a'*64)  # Counts and screening subset alone cannot detect this.
            write(path,changed)
            with self.assertRaisesRegex(ValueError,'previously registered'):validate_panel_registration(path,registration)

    def test_complete114_parity_and_finite_original_voice(self):
        panel,rows,screen,_=self.fixture();result=measurement(panel,rows,screen)
        self.assertTrue(result['confirmation_pass']);self.assertEqual(result['screening_scores_reproduced'],26)
        for changed in (rows[:-1],rows[:-1]+[rows[0]]):
            with self.assertRaises(ValueError):measurement(panel,changed,screen)
        for key,value in [('audio_sha256','changed'),('cosine',float('nan')),('row_id','changed')]:
            changed=copy.deepcopy(rows);changed[0][key]=value
            with self.assertRaises(ValueError):measurement(panel,changed,screen)
        changed=copy.deepcopy(rows);changed[0]['score']['word_errors']=9
        with self.assertRaises(ValueError):measurement(panel,changed,screen)
        changed=copy.deepcopy(rows);changed[56]['text_source_row_id']='changed'
        with self.assertRaises(ValueError):measurement(panel,changed,screen)
        changed=copy.deepcopy(rows);changed[-1]['terminated']=False
        self.assertFalse(measurement(panel,changed,screen)['confirmation_pass'])
        self.assertAlmostEqual(cosine_error(.986,.985),.001)
        for a,b in [(float('nan'),.98),(.98,float('inf'))]:
            with self.assertRaises(ValueError):cosine_error(a,b)

    def test_entry_and_manifest_drift_fail_before_model_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);run=root/'parent';out=root/'confirmation'
            write(root/'reference-prefix-generation-pad-status.json',dict(status='running'))
            with self.assertRaises(ValueError):prepare(run,out,root/'driver.py')
            self.assertFalse(out.exists())
            asr=root/'asr';asr.mkdir();manifest=asr/'download-manifest.json';content=dict(revision='same',files={'weights':'old'})
            write(manifest,content);out.mkdir()
            write(out/'protocol.json',dict(parent_run=str(run),config=dict(asr=str(asr)),asr_manifest_sha256=digest(manifest)))
            with patch('training.reference_prefix_generation_pad_confirmation.evidence',side_effect=ValueError('passed hash gate')) as gate:
                with self.assertRaisesRegex(ValueError,'passed hash gate'):verify(out)
                gate.assert_called_once();gate.reset_mock();content['files']['weights']='new';write(manifest,content)
                with self.assertRaisesRegex(ValueError,'ASR manifest changed'):verify(out)
                gate.assert_not_called()


if __name__=='__main__':unittest.main()
