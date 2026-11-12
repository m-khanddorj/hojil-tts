import unittest

from training.speaker_diagnostics import summarize


class SpeakerDiagnosticsTests(unittest.TestCase):
    def test_speaker_macro_does_not_overweight_frequent_speakers(self):
        rows = [dict(seed=2026, case=str(i), speaker_id='a', cosine=0.9) for i in range(3)]
        rows.append(dict(seed=2027, case='3', speaker_id='b', cosine=0.5))
        result = summarize(rows)
        self.assertAlmostEqual(result['mean_cosine'], 0.8)
        self.assertAlmostEqual(result['speaker_macro_cosine'], 0.7)
        self.assertEqual(result['per_seed'], {'2026': 0.9, '2027': 0.5})

    def test_duplicate_recordings_cannot_inflate_a_score(self):
        row = dict(seed=2026, case='1', speaker_id='a', cosine=0.8)
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            summarize([row, row])
        self.assertEqual(summarize([row, dict(row, seed=2027)])['recordings'], 2)

    def test_missing_or_invalid_measurements_fail(self):
        with self.assertRaises(ValueError):
            summarize([])
        for value in (float('nan'), float('inf'), 1.1, -1.1):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'Invalid'):
                summarize([dict(seed=2026, case='1', speaker_id='a', cosine=value)])
