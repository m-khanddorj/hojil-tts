import unittest
from training.reference_profile_mean_selection import select
from training.targeted_voice import aggregate


class MeanSelectionTests(unittest.TestCase):
    settings=dict(maximum_clip_cosine_drop=.005,maximum_added_character_errors_per_clip=1)

    def rows(self,errors=4,cosine=.98):
        return [dict(case=str(i),seed=1,speaker_id='voice',group='profile-fit',text='x',reference='r',
            reference_sha256='a'*64,reference_text='ref',terminated=True,cosine=cosine,
            score=dict(cer=errors/100,wer=.1,characters=100,character_errors=errors,words=20,word_errors=2)) for i in range(16)]

    def trial(self,index,rows):
        return dict(index=index,records=rows,audit=dict(status='verified',recordings=16,candidate_index=index,
            all_weights_unchanged=True,maximum_cosine_recomputation_error=0,summary=aggregate(rows)))

    def test_joint_mean_priority_without_hiding_clip_damage(self):
        baseline=self.rows();a=self.rows(2,.985);a[0]['cosine']=.97
        b=self.rows(1,.982);c=self.rows(3,.99)
        result=select(baseline,[self.trial(0,a),self.trial(1,b),self.trial(2,c)],self.settings,[0,1,2])
        self.assertEqual(result['selected'],0)
        self.assertFalse(result['trials'][0]['gate']['checks']['worst_clip_voice_preserved'])
        self.assertFalse(result['validation_used'])

    def test_no_subset_reserved_or_unverified_selection(self):
        baseline=self.rows();trial=self.trial(0,self.rows(2,.985))
        with self.assertRaises(ValueError):select(baseline,[trial],self.settings,[0,1])
        trial['records'][0]['group']='profile-validation'
        with self.assertRaises(ValueError):select(baseline,[trial],self.settings,[0])
        trial=self.trial(0,self.rows(2,.985));trial['audit']['status']='pending'
        with self.assertRaises(ValueError):select(baseline,[trial],self.settings,[0])


if __name__=='__main__':unittest.main()
