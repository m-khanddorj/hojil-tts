import copy,unittest
from training.direct_speaker_gate import fitting_gate


class DirectSpeakerGateTests(unittest.TestCase):
    def setUp(self):
        self.settings=dict(maximum_clip_cosine_drop=.005,maximum_added_character_errors_per_clip=1)
        self.before=[dict(case=str(i),seed=i,speaker_id=str(i),group='fitting',text='a',reference=str(i),
            reference_sha256=str(i),reference_text='b',terminated=True,cosine=.98,
            score=dict(character_errors=2,characters=100,word_errors=1,words=20,cer=.02,wer=.05)) for i in range(16)]
        self.after=copy.deepcopy(self.before)
        for r in self.after:r['cosine']+=.001

    def test_joint_improvement_and_strict_voice_requirement(self):
        self.assertTrue(fitting_gate(self.before,self.after,self.settings)['passed'])
        self.assertFalse(fitting_gate(self.before,self.before,self.settings)['passed'])

    def test_global_improvement_does_not_hide_one_damaged_clip(self):
        for r in self.after:r['score'].update(character_errors=0,cer=0)
        self.after[0]['score'].update(character_errors=4,cer=.04)
        result=fitting_gate(self.before,self.after,self.settings)
        self.assertTrue(result['checks']['pooled_cer_preserved'])
        self.assertFalse(result['passed'])
        self.after[0]['score'].update(character_errors=0,cer=0)
        self.after[0]['cosine']=.974
        self.assertFalse(fitting_gate(self.before,self.after,self.settings)['passed'])

    def test_missing_duplicate_or_changed_condition_rejected(self):
        for invalid in (self.after[:-1],self.after[:-1]+[self.after[0]]):
            with self.assertRaises(ValueError):fitting_gate(self.before,invalid,self.settings)
        self.after[0]['seed']=999
        with self.assertRaises(ValueError):fitting_gate(self.before,self.after,self.settings)


if __name__=='__main__':unittest.main()
