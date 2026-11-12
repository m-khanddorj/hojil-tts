import unittest
from training.reference_profile_gate import compare


class ProfileGateTests(unittest.TestCase):
    settings=dict(maximum_clip_cosine_drop=.005,maximum_added_character_errors_per_clip=1)

    def rows(self,group,count):
        return [dict(case=str(i),seed=1,speaker_id='voice',group=group,text='hello',reference='r',
            reference_sha256='a'*64,reference_text='reference',terminated=True,cosine=.98,
            score=dict(cer=.02,wer=.1,characters=50,character_errors=1,words=10,word_errors=1)) for i in range(count)]

    def test_fit_isolation_and_clip_damage(self):
        before=self.rows('profile-fit',16);after=[dict(r,cosine=.981) for r in before]
        self.assertTrue(compare(before,after,self.settings,'profile-fit')['passed'])
        after[0]=dict(after[0],group='profile-validation')
        with self.assertRaises(ValueError):compare(before,after,self.settings,'profile-fit')
        after[0]=dict(before[0],cosine=.974)
        self.assertFalse(compare(before,after,self.settings,'profile-fit')['passed'])

    def test_reserved_exact_coverage_and_averages(self):
        before=self.rows('profile-validation',8);after=[dict(r,cosine=.981) for r in before]
        result=compare(before,after,self.settings,'profile-validation')
        self.assertTrue(result['passed']);self.assertEqual(result['after']['recordings'],8)
        self.assertAlmostEqual(result['after']['speaker_macro_cosine'],.981)
        self.assertAlmostEqual(result['after']['pooled_cer'],.02)
        with self.assertRaises(ValueError):compare(before,after[:-1],self.settings,'profile-validation')


if __name__=='__main__':unittest.main()
