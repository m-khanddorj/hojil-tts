import unittest

from training.preference_pilot import fitting_pairs, voice_guard


class PreferencePilotTests(unittest.TestCase):
    def test_preference_validation_excludes_shared_text_and_speaker(self):
        cases = [dict(case='v', speaker_id='v',preference_split='validation',text='One.',reference_text='Two.'),
            dict(case='t1',speaker_id='a',preference_split='train',text='One.',reference_text='New.'),
            dict(case='t2',speaker_id='b',preference_split='train',text='Different.',reference_text='Other.')]
        pairs = [dict(case=c['case']) for c in cases]
        selected, excluded = fitting_pairs(dict(cases=cases), pairs)
        self.assertEqual(excluded, ['t1'])
        self.assertEqual(selected, [pairs[0],pairs[2]])
        cases[2]['speaker_id']='v'
        with self.assertRaisesRegex(ValueError,'shares training speakers'):
            fitting_pairs(dict(cases=cases),pairs)

    def reports(self):
        records = [dict(case=str(i),seed=s,cosine=0.99) for i in range(13) for s in (2026,2027)]
        baseline = dict(records=records)
        candidate = dict(records=[dict(r) for r in records],
            summary={k:dict(speaker_macro_cosine=0.985) for k in ('development','custom')})
        return baseline,candidate

    def test_voice_guard_requires_absolute_means_and_paired_clip_protection(self):
        a,b=self.reports()
        self.assertTrue(voice_guard(a,b)['passed'])
        b['summary']['custom']['speaker_macro_cosine']=0.984
        self.assertFalse(voice_guard(a,b)['passed'])
        a,b=self.reports()
        b['records'][0]['cosine']=0.98
        self.assertFalse(voice_guard(a,b)['passed'])
        b['records'].pop()
        with self.assertRaisesRegex(ValueError,'Incomplete'):
            voice_guard(a,b)


if __name__=='__main__':
    unittest.main()
