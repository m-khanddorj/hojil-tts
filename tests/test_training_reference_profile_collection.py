import unittest
from data.pipeline import text_key
from training.reference_profile_collection import select_texts


class ReferenceProfileCollectionTests(unittest.TestCase):
    def setUp(self):
        self.records=[dict(sample_id=str(i),case=str(i),speaker_id=str(i),preference_split='train',
            text=f'Unique sentence number {i} with enough characters for selection.') for i in range(16)]

    def test_deterministic_disjoint_unique_training_texts(self):
        self.records[0]['preference_split']='validation'
        excluded=[self.records[1]['text'],self.records[2]['text']]
        duplicate=dict(self.records[3],sample_id='duplicate')
        a=select_texts(self.records+[duplicate],excluded)
        b=select_texts(list(reversed(self.records+[duplicate])),excluded)
        self.assertEqual(a,b);self.assertEqual(len(a),12)
        keys={text_key(r['text']) for r in a};self.assertEqual(len(keys),12)
        self.assertFalse(keys & {text_key(t) for t in excluded+[self.records[0]['text']]})

    def test_insufficient_eligible_texts_fail_without_relaxing_exclusions(self):
        with self.assertRaises(ValueError):select_texts(self.records,[r['text'] for r in self.records[:6]])


if __name__=='__main__':unittest.main()
