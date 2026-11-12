import copy
import unittest
from training.reference_profile_preference_collection import SETTINGS,fresh_texts,preference_records
from training.voice_preferences import joint_pairs


class PreferenceCollectionTests(unittest.TestCase):
    def test_fresh_texts_exclude_evaluation_and_all_old_profile_texts(self):
        records=[dict(text='a'*40+chr(97+i%26)+chr(97+i//26),case=f'c{i}',sample_id=f's{i}',
            speaker_id=f'v{i}',preference_split='train') for i in range(50)]
        panel=dict(cases=[dict(text=records[0]['text'],reference_text=records[1]['text'])])
        old=[dict(text=r['text']) for r in records[2:14]];forbidden={r['text'] for r in records[:14]}
        records[-1]['preference_split']='validation'
        chosen=fresh_texts(records,panel,old)
        self.assertEqual(len(chosen),16);self.assertEqual(len({r['text'] for r in chosen}),16)
        self.assertFalse({r['text'] for r in chosen}&forbidden)
        self.assertNotIn(records[-1]['text'],{r['text'] for r in chosen})
        self.assertEqual(chosen,fresh_texts(list(reversed(records)),panel,old))

    def test_complete_train_only_coverage_and_joint_pair_dominance(self):
        cases=[dict(case=f'c{i}',group='profile-preference-train',text='a'*100,speaker_id='voice',
            reference='r',reference_text='reference',reference_sha256='a'*64) for i in range(16)]
        p=dict(cases=cases,settings=SETTINGS)
        records=[dict(c,seed=s,terminated=True,cosine=cosine,score=dict(cer=errors/100,character_errors=errors,
            characters=100,wer=.1,word_errors=1,words=10)) for c in cases
            for s,cosine,errors in zip(SETTINGS['seeds'],[.98,.985,.99,.981],[3,0,4,0])]
        rows=preference_records(p,dict(records=records));self.assertEqual(len({r['sample_id'] for r in rows}),64)
        pairs=joint_pairs(rows[:4],.001);by_id={r['sample_id']:r for r in rows}
        self.assertTrue(pairs)
        for pair in pairs:
            a,b=by_id[pair['chosen']],by_id[pair['rejected']]
            self.assertLessEqual(a['score']['character_errors'],b['score']['character_errors'])
            self.assertGreaterEqual(a['cosine'],b['cosine'])
        for changed in (records[:-1],records[:-1]+[records[0]]):
            with self.assertRaises(ValueError):preference_records(p,dict(records=changed))
        changed=copy.deepcopy(records);changed[0]['group']='profile-validation'
        with self.assertRaises(ValueError):preference_records(p,dict(records=changed))
        changed=copy.deepcopy(records);changed[0]['terminated']=False
        with self.assertRaises(ValueError):preference_records(p,dict(records=changed))


if __name__=='__main__':unittest.main()
