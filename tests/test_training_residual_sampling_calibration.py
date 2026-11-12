import copy
from types import SimpleNamespace
import unittest

from scripts.score_latest_audio import score
from training.residual_sampling_calibration import select_cases, generation, observe_residual, gate


class ResidualSamplingTests(unittest.TestCase):
    def test_split_selection_is_order_invariant_and_excludes_fitting(self):
        cases = [dict(case=f'{s}-{i}', speaker_id=str(s), preference_split='validation' if s < 6 else 'train')
                 for s in range(10) for i in range(2)]
        result = select_cases(cases)
        self.assertEqual(result, select_cases(list(reversed(cases))))
        self.assertEqual(len(result), 8)
        self.assertEqual(len({c['speaker_id'] for c in result}), 4)
        self.assertTrue(all(c['preference_split'] == 'validation' for c in result))
        with self.assertRaises(ValueError): select_cases(cases[:6])

    def test_residual_settings_are_forwarded_without_touching_main_policy(self):
        parent = dict(do_sample=True, temperature=.9, top_k=50, top_p=1., max_new_tokens=320)
        saved = dict(parent)
        control, candidate = generation(parent,'control'), generation(parent,'residual_06')
        self.assertEqual(parent, saved)
        self.assertEqual({k for k in control if control[k] != candidate[k]}, {'subtalker_temperature'})
        self.assertTrue(all(candidate[k] == v for k,v in parent.items()))
        self.assertEqual(generation(dict(parent,subtalker_dosample=True),'residual_06'),candidate)
        with self.assertRaises(ValueError): generation(dict(parent,subtalker_dosample=False),'control')
        calls = []
        def generate(**kwargs):
            calls.append(kwargs)
            return 'unchanged-output'
        predictor = SimpleNamespace(generate=generate)
        model = SimpleNamespace(talker=SimpleNamespace(code_predictor=predictor))
        expected = dict(do_sample=True,temperature=.6,top_k=50,top_p=1.,max_new_tokens=15)
        with observe_residual(model,expected) as receipt:
            self.assertEqual(predictor.generate(**expected),'unchanged-output')
        self.assertEqual(calls,[expected]); self.assertEqual(receipt['calls'],1)
        self.assertIs(predictor.generate,generate)
        with self.assertRaises(ValueError):
            with observe_residual(model,expected): predictor.generate(**dict(expected,temperature=.9))
        self.assertIs(predictor.generate,generate)
        with self.assertRaises(ValueError):
            with observe_residual(model,expected): pass

    def test_gate_requires_complete_pairs_and_checks_speech_harm(self):
        cases = [dict(case=str(i),speaker_id=str(i),group='calibration',text='one two') for i in range(4)]
        baseline = [dict(c,seed=s,cosine=.98,terminated=True,score=score(c['text'],'one two',True))
                    for c in cases for s in (2026,2027)]
        candidate = [dict(r,cosine=.982) for r in baseline]
        self.assertTrue(gate(cases,[2026,2027],baseline,candidate)['calibration_pass'])
        self.assertFalse(gate(cases,[2026,2027],baseline,baseline)['calibration_pass'])
        for altered in (candidate[:-1],candidate+[candidate[0]]):
            with self.assertRaises(ValueError): gate(cases,[2026,2027],baseline,altered)
        worse = copy.deepcopy(candidate); worse[0]['cosine']=.974
        self.assertFalse(gate(cases,[2026,2027],baseline,worse)['calibration_pass'])
        worse = copy.deepcopy(candidate); worse[0]['score']=score('one two','one',True)
        self.assertFalse(gate(cases,[2026,2027],baseline,worse)['calibration_pass'])
        worse = copy.deepcopy(candidate); worse[0]['terminated']=False
        self.assertFalse(gate(cases,[2026,2027],baseline,worse)['calibration_pass'])
        worse = copy.deepcopy(candidate); worse[0]['text']='different'
        with self.assertRaises(ValueError): gate(cases,[2026,2027],baseline,worse)


if __name__ == '__main__': unittest.main()
