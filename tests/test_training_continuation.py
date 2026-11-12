import copy
import unittest

from training.continuation import validate_parent


class ContinuationTests(unittest.TestCase):
    def setUp(self):
        self.provenance = dict(config=dict(method='lora',rank=16,peak_lr=1e-5),features='f',
                               shards='s',base_config='c',base_weights='w',validation='v',versions='p')
        self.state = dict(provenance=copy.deepcopy(self.provenance),adapter={},optimizer={},
                          rng_cpu='cpu',rng_cuda='gpu',step=1750)

    def test_new_schedule_is_allowed_without_replacing_parent_state(self):
        self.state['provenance']['config']['peak_lr'] = 5e-5
        before = copy.deepcopy(self.state)
        self.assertTrue(validate_parent(self.state,self.provenance))
        self.assertEqual(self.state,before)

    def test_data_model_and_optimizer_structure_changes_are_rejected(self):
        for key in ('features','shards','base_config','base_weights','validation','versions'):
            state = copy.deepcopy(self.state)
            state['provenance'][key] = 'changed'
            with self.assertRaisesRegex(ValueError,key):
                validate_parent(state,self.provenance)
        self.state['provenance']['config']['loraplus_b_to_a_ratio'] = 4
        with self.assertRaisesRegex(ValueError,'structure'):
            validate_parent(self.state,self.provenance)

    def test_optimizer_and_rng_are_required(self):
        for key in ('optimizer','rng_cpu','rng_cuda'):
            state = copy.deepcopy(self.state)
            del state[key]
            with self.assertRaisesRegex(ValueError,key):
                validate_parent(state,self.provenance)

    def test_historical_checkpoint_remains_supported(self):
        self.assertFalse(validate_parent(dict(config=dict(features='f',rank=16)),self.provenance))


if __name__ == '__main__':
    unittest.main()
