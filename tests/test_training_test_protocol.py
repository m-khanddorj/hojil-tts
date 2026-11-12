import copy
import unittest

from training.policy import validate_cohorts
from training.test_protocol import assert_development_disjoint, evaluation_folder
from training.compare_experiments import combine
from scripts.score_latest_audio import score


class TestProtocolTests(unittest.TestCase):
    def setUp(self):
        self.manifest = [dict(id=str(i),text=f'utterance {i}',audio=f'{i}.wav',
            pcm_sha256=str(i),speaker_id=str(i//2),split=('train','test','validation')[i//2]) for i in range(6)]
        def pair(i,j):
            return dict(self.manifest[i],ref_id=str(j),ref_text=self.manifest[j]['text'],ref_audio=f'{j}.wav')
        self.train=[pair(0,1),pair(1,0)]
        self.test=[pair(2,3)]
        self.dev=[pair(4,5)]

    def test_reserved_split_requires_explicit_test_mode(self):
        validate_cohorts(self.train,self.test,self.manifest,heldout_split='test')
        with self.assertRaises(ValueError):
            validate_cohorts(self.train,self.test,self.manifest)

    def test_test_references_cannot_leak_training_text(self):
        manifest=copy.deepcopy(self.manifest)
        manifest[3]['text']='Utterance, 0!'
        heldout=copy.deepcopy(self.test)
        heldout[0]['ref_text']=manifest[3]['text']
        with self.assertRaisesRegex(ValueError,'leakage'):
            validate_cohorts(self.train,heldout,manifest,heldout_split='test')

    def test_development_reference_overlap_is_rejected(self):
        assert_development_disjoint(self.test,self.dev,self.manifest)
        for field in ('text','pcm_sha256','audio','speaker_id'):
            manifest=copy.deepcopy(self.manifest)
            manifest[3][field]=manifest[5][field]
            with self.assertRaises(ValueError):
                assert_development_disjoint(self.test,self.dev,manifest)

    def test_test_caches_are_separate_for_both_seeds(self):
        for seed in (2026,2027):
            self.assertNotEqual(evaluation_folder('unseen-test',seed),evaluation_folder('unseen-validation',seed))
        with self.assertRaises(ValueError):
            evaluation_folder('../unseen-test',2026)

    def test_test_comparison_is_labeled_and_cannot_pool_development(self):
        cases=[dict(case='001',speaker_id='s')]
        scores=[dict(case='001',without_punctuation=score('one two','one',True))]
        plan=dict(seed=2026,cohort='unseen-test',cases=cases)
        result=combine([(plan,scores,scores)])
        self.assertIn('test ASR proxy',result['interpretation'])
        self.assertIn('test ASR proxy',result['per_seed'][0]['interpretation'])
        with self.assertRaises(ValueError):
            combine([(plan,scores,scores),(dict(plan,seed=2027,cohort='unseen-validation'),scores,scores)])


if __name__=='__main__':
    unittest.main()
