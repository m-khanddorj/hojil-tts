import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from training.engine import write,digest
from training.reference_prefix_lora_pad_evaluation import (
    DISABLED,COUNTS,cases_for,validate_scope,validate_records,score_parity,cosine_error,completed,prepare)
from training.targeted_voice import aggregate


class LoRAPadEvaluationTests(unittest.TestCase):
    def fixture(self):
        cases=[dict(case=f'dev-{i}',group='development',speaker_id=str(i%10),text='reference text',reference='r',
            reference_sha256='b'*64,reference_text='ref',row_id=str(i)) for i in range(44)]
        cases += [dict(case=f'custom-{i}',group='custom',speaker_id='custom',text='reference text',reference='custom',
            reference_sha256='a'*64,reference_text='ref',text_source_row_id=str(i)) for i in range(13)]
        panel=dict(cases=cases,seeds=[2026,2027],development_recordings=88,custom_recordings=26,
            criteria=dict(pooled_cer_strictly_below=.03,speaker_macro_cosine_strictly_above=.984,
                groups=['development','custom'],require_complete_panel=True))
        screen=[dict(c,seed=s) for s in panel['seeds'] for c in cases[:10]+cases[44:47]]
        allcases=validate_scope(panel,screen,'a'*64)
        records=[dict(c,cosine=.986,terminated=True,wav=f"{c['case']}-{c['seed']}.wav",trajectory=f"{c['case']}-{c['seed']}.pt",
            audio_sha256='same',score=dict(cer=.01,wer=.1,character_errors=1,characters=100,word_errors=1,words=10)) for c in allcases]
        return panel,screen,dict(cases=allcases),records

    def test_full_scope_and_custom_only_control(self):
        panel,screen,p,_=self.fixture()
        self.assertEqual([len(cases_for(p,s)) for s in COUNTS],[114,26,114])
        self.assertTrue(all(c['group']=='custom' for c in cases_for(p,'control')))
        self.assertEqual(len({c['case'] for c in cases_for(p,'control')}),13)
        self.assertEqual(sorted({c['seed'] for c in cases_for(p,'control')}),[2026,2027])
        for index,newsha in ((0,'a'*64),(-1,'b'*64),(0,DISABLED)):
            changed=copy.deepcopy(panel);changed['cases'][index]['reference_sha256']=newsha
            with self.assertRaises(ValueError):validate_scope(changed,screen,'a'*64)
        changed=copy.deepcopy(screen);changed[0]['row_id']='changed'
        with self.assertRaises(ValueError):validate_scope(panel,changed,'a'*64)
        with self.assertRaises(ValueError):cases_for(p,'unknown')

    def test_complete_metadata_finite_voice_and_no_seed_selection(self):
        _,_,p,rows=self.fixture();validate_records(p['cases'],rows)
        for bad in (rows[:-1],rows[:-1]+[rows[0]]):
            with self.assertRaises(ValueError):validate_records(p['cases'],bad)
        for index,key,value in ((0,'row_id','changed'),(-1,'text_source_row_id','changed'),
                                (0,'cosine',float('nan')),(0,'terminated',1)):
            changed=copy.deepcopy(rows);changed[index][key]=value
            with self.assertRaises(ValueError):validate_records(p['cases'],changed)

    def test_exact_score_parity_and_finite_independent_cosine(self):
        _,_,_,rows=self.fixture();score_parity(rows[0],copy.deepcopy(rows[0]))
        for key,value in [('audio_sha256','different'),('cosine',float('nan')),('score',{})]:
            changed=copy.deepcopy(rows[0]);changed[key]=value
            with self.assertRaises(ValueError):score_parity(changed,rows[0])
        self.assertAlmostEqual(cosine_error(.986,.985),.001)
        for a,b in [(float('nan'),.98),(.98,float('inf'))]:
            with self.assertRaises(ValueError):cosine_error(a,b)

    def test_complete_prior_stage_required_before_advancing(self):
        _,_,p,rows=self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory);dest=out/'baseline';dest.mkdir();write(out/'protocol.json',p)
            for row in rows:
                (dest/row['wav']).write_bytes(b'wave');(dest/row['trajectory']).write_bytes(b'codes')
                row['audio_sha256']=digest(dest/row['wav']);row['trajectory_sha256']=digest(dest/row['trajectory'])
            report=dict(records=rows,summary=aggregate(rows),truncated=0);write(dest/'results.json',report)
            audit=dict(status='verified',stage='baseline',recordings=114,all_weights_unchanged=True,
                all_recomputed_cosines_finite=True,native_conditioning_verified=True,pad_traces_verified=114,
                historical_screen_reproduced=26,inactive_development_exact=0,custom_prefill_and_first_frame_verified=0,
                optimizer_updates=0,goal_achieved=False,global_model_selection_changed=False,
                maximum_cosine_recomputation_error=0.,summary=report['summary'],truncated=0,
                protocol_sha256=digest(out/'protocol.json'),results_sha256=digest(dest/'results.json'),prior_audit_sha256={})
            write(dest/'verification.json',audit);completed(out,p,'baseline')
            for key,value in [('recordings',113),('historical_screen_reproduced',25),('pad_traces_verified',113),
                              ('all_recomputed_cosines_finite',False),('maximum_cosine_recomputation_error',float('nan'))]:
                changed=dict(audit);changed[key]=value
                (dest/'verification.json').write_text(json.dumps(changed),encoding='utf-8')
                with self.assertRaises(ValueError):completed(out,p,'baseline')
            write(dest/'verification.json',audit);(dest/rows[0]['wav']).write_bytes(b'changed')
            with self.assertRaises(ValueError):completed(out,p,'baseline')

    def test_no_output_without_completed_failure_trigger(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)/'new'
            with patch('training.reference_prefix_lora_pad_evaluation.trigger',side_effect=ValueError('not completed')):
                with self.assertRaises(ValueError):prepare(Path('initial'),Path('previous'),out,Path('driver'))
            self.assertFalse(out.exists())


if __name__=='__main__':unittest.main()
