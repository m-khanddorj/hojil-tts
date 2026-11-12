import copy
import unittest
from training.reference_masked_supervision_evaluation import calibration_cases, progression, validate_records
from training.reference_masked_supervision_plan import build_plan
from scripts.score_latest_audio import score
from tests.test_reference_masked_supervision_plan import rows


def audit(step, cer=.04, cosine=.985):
    return dict(status='verified', step=step, recordings=80, truncated=0,
        frozen_weights_verified=True, native_conditioning_verified=True,
        all_artifact_hashes_verified=True, all_recomputed_cosines_finite=True,
        max_cosine_recomputation_error=0., adapter_movement=dict(maximum=0. if step == 0 else .0001),
        summary={g:dict(recordings=n, speakers=s, pooled_cer=cer, pooled_wer=.1, speaker_macro_cosine=cosine)
            for g,n,s in [('development',64,16),('custom',16,1)]})


class EvaluationTests(unittest.TestCase):
    def test_complete_registered_scope_and_edit_counts(self):
        fit, cal = rows(47,'fit'), rows(16,'cal')
        for r in cal:
            r.update(ref_audio='reference.wav', ref_audio_sha256='a'*64, ref_text='reference words')
        plan = build_plan(fit, cal)
        reference = dict(reference='custom.wav', reference_sha256='b'*64,
            reference_text='custom words', speaker_id='custom')
        cases = calibration_cases(plan, cal, reference)
        records = [dict(c, terminated=True, cosine=.986, hypothesis=c['text'],
            score=score(c['text'], c['text'], True)) for c in cases]
        summary = validate_records(cases, records)
        self.assertEqual(summary['development']['recordings'], 64)
        self.assertEqual(summary['custom']['recordings'], 16)
        for altered in (records[:-1], records[:-1]+[records[0]]):
            with self.assertRaises(ValueError): validate_records(cases, altered)
        for field, value in [('half_reference', True), ('cosine', float('nan')), ('terminated',1), ('hypothesis','wrong')]:
            altered = copy.deepcopy(records); altered[0][field] = value
            with self.assertRaises(ValueError): validate_records(cases, altered)

    def test_baseline_and_boundaries_never_qualify(self):
        self.assertFalse(progression([audit(0,.02,.99)])['stop'])
        for cer, cosine in [(.03,.99),(.02,.984)]:
            self.assertFalse(progression([audit(0), audit(8,cer,cosine)])['stop'])
        current = audit(8,.02,.99); current['adapter_movement']['maximum'] = 0.
        self.assertFalse(progression([audit(0),current])['stop'])
        current = audit(8,.02,.99); current['truncated'] = 1
        self.assertFalse(progression([audit(0),current])['stop'])
        result = progression([audit(0),audit(8,.02,.99)])
        self.assertTrue(result['stop']); self.assertFalse(result['goal_achieved'])
        self.assertEqual(result['reason'],'calibration_qualified_needs_original114')

    def test_two_consecutive_dominated_checkpoints_stop(self):
        history = [audit(0),audit(8,.05,.9845),audit(16,.06,.9844)]
        self.assertEqual(progression(history)['reason'],'consecutive_pareto_regressions')
        history[2] = audit(16,.06,.986)
        self.assertFalse(progression(history)['stop'])
        self.assertEqual(progression([audit(0),audit(8),audit(16),audit(32)])['reason'],'maximum_updates_reached')

    def test_incomplete_nonfinite_and_out_of_order_audits_reject(self):
        for field,value in [('recordings',79),('all_recomputed_cosines_finite',False),
                            ('max_cosine_recomputation_error',float('nan'))]:
            current=audit(0); current[field]=value
            with self.assertRaises(ValueError): progression([current])
        current=audit(0); current['summary']['custom']['speaker_macro_cosine']=float('nan')
        with self.assertRaises(ValueError): progression([current])
        with self.assertRaises(ValueError): progression([audit(0),audit(16)])
        with self.assertRaises(ValueError): progression([audit(0),audit(8,.02,.99),audit(16)])


if __name__ == '__main__': unittest.main()
