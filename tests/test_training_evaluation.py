import unittest

from scripts.score_latest_audio import score
from training.evaluate import compare, pooled
from training.compare_experiments import combine


class EvaluationTests(unittest.TestCase):
    def test_speech_diagnostics_preserve_a_large_failure_in_totals(self):
        from training.speech_diagnostics import diagnose
        plan = dict(seed=1,cases=[dict(case='1',speaker_id='s'),dict(case='2',speaker_id='s')])
        def row(case, text):
            return dict(case=case,seconds=1,without_punctuation=score('one two',text,True))
        report = diagnose([(plan,[row('1','one two'),row('2','one')],
                                  [row('1','wrong wrong wrong'),row('2','one two')])])
        self.assertEqual(report['summary']['delta_word_errors'],2)
        self.assertEqual(report['summary']['improved_recordings'],1)
        self.assertEqual(report['summary']['worsened_recordings'],1)
        self.assertEqual(report['largest_regressions'][0]['case'],'1')
        self.assertEqual(len(report['recordings']),2)

    def test_pooled_error_is_not_mean_of_clip_rates(self):
        values = [score('a', '', True), score('abcdefghij', 'abcdefghij', True)]
        result = pooled(values)
        self.assertAlmostEqual(result['mean_cer'], 0.5)
        self.assertAlmostEqual(result['pooled_cer'], 1/11)

    def test_paired_comparison_and_duplicate_rejection(self):
        cases = [{'case': '1', 'speaker_id': 'a'}, {'case': '2', 'speaker_id': 'b'}]
        old = [{'case': '1', 'without_punctuation': score('one two', 'one', True)},
               {'case': '2', 'without_punctuation': score('three', 'three', True)}]
        new = [{'case': '1', 'without_punctuation': score('one two', 'one two', True)}, old[1]]
        result = compare(old, new, cases)
        self.assertAlmostEqual(result['wer_delta'], -1/3)
        self.assertEqual(result['speakers'], 2)
        with self.assertRaises(ValueError):
            compare(old, new + [new[0]], cases)

    def test_changed_reference_denominators_fail(self):
        case = [{'case': '1', 'speaker_id': 'a'}]
        a = [{'case': '1', 'without_punctuation': score('one', 'one', True)}]
        b = [{'case': '1', 'without_punctuation': score('one two', 'one', True)}]
        with self.assertRaisesRegex(ValueError, 'denominators'):
            compare(a, b, case)

    def test_seeds_keep_the_same_speaker_cluster(self):
        cases = [{'case': '1', 'speaker_id': 'a'}]
        a = [{'case': '1', 'without_punctuation': score('one two', 'one', True)}]
        b = [{'case': '1', 'without_punctuation': score('one two', 'one two', True)}]
        pairs = [(dict(seed=1, cases=cases), a, b), (dict(seed=2, cases=cases), b, a)]
        result = combine(pairs)
        self.assertEqual(result['speakers'], 1)
        self.assertEqual(result['clips'], 2)
        self.assertEqual(result['wer_delta'], 0)
        self.assertEqual(result['wer_delta_speaker_bootstrap_95'], [0, 0])
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            combine(pairs + [pairs[0]])
        with self.assertRaisesRegex(ValueError, 'settings'):
            combine([pairs[0], (dict(seed=2, cases=cases, temperature=.8), b, a)])


if __name__ == '__main__':
    unittest.main()
