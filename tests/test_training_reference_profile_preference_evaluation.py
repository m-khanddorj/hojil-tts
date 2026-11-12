import copy
import unittest
from training.reference_profile_preference_evaluation import validate_records,paired_changes,progression


class ProfilePreferenceEvaluationTests(unittest.TestCase):
    def records(self):
        cases=[dict(case=f'c{i}',seed=2026,group='custom',speaker_id='voice',text='a'*100,
            reference='r',reference_sha256='a'*64,reference_text='ref') for i in range(3)]
        rows=[dict(c,cosine=.98,terminated=True,score=dict(cer=.01,wer=.05,character_errors=1,characters=100,
            word_errors=1,words=20)) for c in cases]
        return cases,rows

    def checkpoint(self,step,cer,cosine,development_cer=.02,development_cosine=.985):
        screen=dict(custom=dict(pooled_cer=cer,speaker_macro_cosine=cosine),
            development=dict(pooled_cer=development_cer,speaker_macro_cosine=development_cosine))
        passed=cer<.03 and cosine>.984 and development_cer<.03 and development_cosine>.984
        return dict(status='verified',step=step,recordings=50,inactive_development_exact_parity=True,
            screen_summary=screen,truncated=0,screen_truncated=0,goal_screen=passed)

    def test_complete_matched_conditions_and_paired_diagnostics(self):
        cases,rows=self.records();validate_records(cases,rows)
        for changed in (rows[:-1],rows[:-1]+[rows[0]]):
            with self.assertRaises(ValueError):validate_records(cases,changed)
        changed=copy.deepcopy(rows);changed[0]['reference_sha256']='b'*64
        with self.assertRaises(ValueError):validate_records(cases,changed)
        changed=copy.deepcopy(rows);changed[0]['cosine']=float('nan')
        with self.assertRaises(ValueError):validate_records(cases,changed)
        changed=copy.deepcopy(rows);changed[0]['cosine']=.981;changed[0]['score']['character_errors']=2
        paired=paired_changes(rows,changed)
        self.assertAlmostEqual(paired[0]['cosine_delta'],.001);self.assertEqual(paired[0]['character_errors_after'],2)

    def test_two_consecutive_pareto_regressions_stop_with_comparator(self):
        history=[self.checkpoint(0,.02,.98),self.checkpoint(4,.03,.979),self.checkpoint(8,.025,.9795)]
        result=progression(history);self.assertTrue(result['stop']);self.assertEqual(result['reason'],'consecutive_pareto_regressions')
        self.assertEqual(result['decisions'][-1]['consecutive_regressions'],2)
        self.assertEqual(result['decisions'][-1]['dominating_earlier_checkpoints'][0]['step'],0)

    def test_voice_improvement_does_not_trigger_cer_only_stop(self):
        history=[self.checkpoint(0,.02,.98),self.checkpoint(4,.04,.981),self.checkpoint(8,.05,.982)]
        self.assertFalse(progression(history)['stop'])
        history=[self.checkpoint(0,.02,.98),self.checkpoint(4,.03,.979),self.checkpoint(8,.025,.981),self.checkpoint(16,.03,.9805)]
        result=progression(history);self.assertFalse(result['stop']);self.assertEqual(result['decisions'][-1]['consecutive_regressions'],1)

    def test_strict_both_group_pass_and_complete_audits(self):
        baseline=self.checkpoint(0,.02,.98)
        self.assertEqual(progression([baseline,self.checkpoint(4,.029,.985)])['reason'],'needs_broader_confirmation')
        for row in (self.checkpoint(4,.03,.985),self.checkpoint(4,.02,.984),self.checkpoint(4,.02,.985,development_cer=.03)):
            self.assertNotEqual(progression([baseline,row])['reason'],'needs_broader_confirmation')
        incomplete=self.checkpoint(4,.02,.985);incomplete['recordings']=49
        with self.assertRaises(ValueError):progression([baseline,incomplete])
        changed=self.checkpoint(4,.02,.985);changed['screen_truncated']=1
        with self.assertRaises(ValueError):progression([baseline,changed])
        with self.assertRaises(ValueError):progression([self.checkpoint(4,.02,.98)])
        with self.assertRaises(ValueError):progression([baseline,self.checkpoint(8,.02,.98)])


if __name__=='__main__':unittest.main()
