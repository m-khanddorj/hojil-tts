import unittest
from collections import Counter
from training.reference_masked_supervision_plan import (
    build_plan, custom_calibration_texts, fitting_schedule, transcript_key)


def rows(speakers, prefix):
    result = []
    for s in range(speakers):
        for i in range(2):
            result.append(dict(id=f'{prefix}-{s}-{i}', ref_id=f'{prefix}-{s}-{1-i}',
                speaker_id=f'{prefix}-speaker-{s}', case=f'{prefix}-case-{s}-{i}',
                text=f'{prefix} sentence {s} variant {i}'))
    return result


class PlanTests(unittest.TestCase):
    def test_schedule_is_order_invariant_balanced_and_paired(self):
        data = rows(47, 'fit')
        plan = fitting_schedule(data)
        self.assertEqual(plan, fitting_schedule(list(reversed(data))))
        index = {r['id']: r for r in data}
        visits = Counter()
        for step in plan:
            speakers = [index[i]['speaker_id'] for i in step['pair_ids']]
            self.assertEqual(len(set(speakers)), 8)
            visits.update(speakers)
            self.assertEqual(set(step['modes']), {'full', 'half'})
            self.assertEqual(step['pair_weight'] * step['mode_weight'], 1 / 16)
        self.assertEqual(sum(visits.values()), 256)
        self.assertLessEqual(max(visits.values()) - min(visits.values()), 1)
        self.assertNotEqual(plan[0]['modes'], plan[1]['modes'])

    def test_reference_closure_and_identity_fail_closed(self):
        data = rows(8, 'fit')
        for changed in (data[:-1], data + [data[0]],
                        [dict(data[0], ref_id=data[0]['id'])] + data[1:],
                        [dict(data[0], ref_id=data[2]['id'])] + data[1:]):
            with self.assertRaises(ValueError):
                fitting_schedule(changed)

    def test_custom_texts_deduplicate_without_quality_selection(self):
        data = rows(16, 'cal')
        first = custom_calibration_texts(data)
        self.assertEqual(first, custom_calibration_texts(list(reversed(data))))
        self.assertEqual(len({r['text_key'] for r in first}), 8)
        duplicate = dict(data[0], case='zz-duplicate', text=data[0]['text'].upper() + '!!!')
        self.assertEqual(first, custom_calibration_texts(data + [duplicate]))
        self.assertEqual(transcript_key('Ａ a, Б!'), 'ａaб')

    def test_feature_plan_covers_every_consumed_pair_and_excludes_unused(self):
        fit = rows(47, 'fit')
        cal = rows(16, 'cal')
        plan = build_plan(fit, cal)
        index = {r['id']: r for r in fit}
        selected = set(plan['fitting_probe_pair_ids']) | {
            i for step in plan['updates'] for i in step['pair_ids']}
        expected = {r[k] for r in [index[i] for i in selected] + cal for k in ('id', 'ref_id')}
        self.assertEqual(set(plan['real_feature_ids']), expected)
        self.assertEqual(len(plan['fitting_probe_pair_ids']), 16)
        self.assertEqual(plan['calibration_recordings'], 80)
        self.assertFalse(plan['baseline_is_selection_candidate'])
        with self.assertRaises(ValueError):
            build_plan(fit, cal[:-1])
        with self.assertRaises(ValueError):
            build_plan(fit, [dict(cal[0], speaker_id=fit[0]['speaker_id'])] + cal[1:])


if __name__ == '__main__':
    unittest.main()
