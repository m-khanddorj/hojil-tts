import copy
import unittest
from training.reference_profile_confirmation import screen_entry,confirmation_measurement
from training.targeted_voice import aggregate


class ProfileConfirmationTests(unittest.TestCase):
    def fixture(self):
        cases=[dict(case=f'dev-{i}',group='development',speaker_id=str(i%10),text='a'*100,
            reference='r',reference_sha256=f'dev-ref-{i%10}',reference_text='ref') for i in range(44)]
        cases += [dict(case=f'custom-{i}',group='custom',speaker_id='custom',text='a'*100,
            reference='c',reference_sha256='custom-ref',reference_text='ref',text_source_row_id=str(i)) for i in range(13)]
        panel=dict(cases=cases,seeds=[2026,2027],development_recordings=88,custom_recordings=26,
            criteria=dict(pooled_cer_strictly_below=.03,speaker_macro_cosine_strictly_above=.984,
                groups=['development','custom'],require_complete_panel=True))
        records=[dict(c,seed=s,terminated=True,cosine=.986,audio_sha256=f"{c['case']}-{s}",
            score=dict(cer=.01,wer=.05,character_errors=1,characters=100,word_errors=1,words=20)) for c in cases for s in panel['seeds']]
        selected={c['case'] for c in cases[:10]+cases[44:47]}
        screen=[r for r in records if r['case'] in selected]
        return panel,records,screen

    def test_screen_requires_actual_strict_both_group_pass(self):
        panel,records,screen=self.fixture();cases=[{k:v for k,v in r.items() if k not in ('score','cosine','audio_sha256','terminated')} for r in screen]
        def evidence(rows):
            report=dict(records=rows,summary=aggregate(rows),truncated=sum(not r['terminated'] for r in rows))
            audit=dict(status='verified',recordings=26,goal_screen=True,all_weights_unchanged=True,
                inactive_development_exact_parity=True,maximum_cosine_recomputation_error=0,
                summary=report['summary'],truncated=report['truncated'])
            return report,audit
        screen_entry(*evidence(screen),cases)
        with self.assertRaises(ValueError):screen_entry(*evidence(screen[:-1]),cases)
        for group in ('development','custom'):
            for metric in ('cer','cosine'):
                changed=copy.deepcopy(screen)
                for r in changed:
                    if r['group']==group:
                        if metric=='cer':r['score'].update(cer=.03,character_errors=3)
                        else:r['cosine']=.984
                with self.assertRaises(ValueError):screen_entry(*evidence(changed),cases)

    def test_complete_scope_and_termination(self):
        panel,records,screen=self.fixture();result=confirmation_measurement(panel,records,screen)
        self.assertTrue(result['confirmation_pass']);self.assertEqual(result['screening_recordings_reproduced_exactly'],26)
        self.assertEqual(result['summary']['development']['recordings'],88)
        self.assertEqual(result['summary']['custom']['recordings'],26)
        for changed in (records[:-1],records[:-1]+[records[0]]):
            with self.assertRaises(ValueError):confirmation_measurement(panel,changed,screen)
        changed=copy.deepcopy(records);changed[-1]['terminated']=False
        self.assertFalse(confirmation_measurement(panel,changed,screen)['confirmation_pass'])
        weaker=copy.deepcopy(panel);weaker['criteria']['pooled_cer_strictly_below']=.04
        with self.assertRaises(ValueError):confirmation_measurement(weaker,records,screen)

    def test_screen_waveform_and_full_case_metadata_are_preserved(self):
        panel,records,screen=self.fixture();changed=copy.deepcopy(records);changed[0]['audio_sha256']='changed'
        with self.assertRaisesRegex(ValueError,'parity'):confirmation_measurement(panel,changed,screen)
        changed=copy.deepcopy(records);changed[-1]['text_source_row_id']='wrong'
        with self.assertRaisesRegex(ValueError,'case changed'):confirmation_measurement(panel,changed,screen)
        with self.assertRaises(ValueError):confirmation_measurement(panel,records,screen[:-1])


if __name__=='__main__':unittest.main()
