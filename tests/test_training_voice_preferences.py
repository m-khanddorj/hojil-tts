import copy
import unittest
from training.voice_preferences import joint_pairs


def candidate(name, errors, cosine):
    return dict(sample_id=name, case='train-1', speaker_id='speaker-1', text='fixed transcript',
        reference_sha256='fixed-reference', partition='train', origin='model_generated',
        score=dict(character_errors=errors, characters=100, cer=errors/100), cosine=cosine)


class PreferenceTests(unittest.TestCase):
    def test_accepts_joint_dominance_and_rejects_tradeoffs(self):
        a, b = candidate('a', 2, .986), candidate('b', 5, .985)
        self.assertEqual([(p['chosen'],p['rejected']) for p in joint_pairs([a,b])], [('a','b')])
        b['cosine'] = .987
        self.assertEqual(joint_pairs([a,b]), [])

    def test_no_self_pairs_equal_pairs_or_tiny_voice_only_preferences(self):
        a, b = candidate('a', 2, .986), candidate('b', 2, .986)
        self.assertEqual(joint_pairs([a,b]), [])
        b['cosine'] -= .00001
        self.assertEqual(joint_pairs([a,b]), [])
        b['cosine'] -= .002
        self.assertEqual(len(joint_pairs([a,b])), 1)

    def test_rejects_eval_data_real_audio_and_changed_conditions(self):
        a, b = candidate('a', 2, .986), candidate('b', 5, .985)
        for key, value in [('partition','development'), ('origin','human_recording'),
                           ('reference_sha256','different'), ('text','different'), ('sample_id','a')]:
            changed = copy.deepcopy(b)
            changed[key] = value
            with self.assertRaises(ValueError):
                joint_pairs([a,changed])

    def test_rejects_nonfinite_and_inconsistent_metrics(self):
        a, b = candidate('a', 2, .986), candidate('b', 5, .985)
        b['cosine'] = float('nan')
        with self.assertRaises(ValueError):
            joint_pairs([a,b])
        b = candidate('b', 5, .985)
        b['score']['cer'] = .99
        with self.assertRaises(ValueError):
            joint_pairs([a,b])


if __name__ == '__main__':
    unittest.main()
