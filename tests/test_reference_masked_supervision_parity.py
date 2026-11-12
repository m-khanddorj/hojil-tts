import copy
import unittest

import torch

from training.reference_masked_supervision_parity import compare_logits, validate_metrics, LIMITS


class NumericalParityTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        gen = torch.Generator().manual_seed(92417)
        self.native = torch.randn(7, 15, 64, generator=gen)
        self.labels = torch.randint(0, 64, (7, 15), generator=gen)

    def test_identical_and_small_roundoff_across_all_codebooks(self):
        exact = compare_logits(self.native, self.native.clone(), self.labels)
        self.assertEqual(exact['maximum_forward_kl'], 0.)
        changed = self.native.clone()
        changed[:, ::2, ::2] += 1e-5
        result = compare_logits(self.native, changed, self.labels)
        self.assertEqual(result['distributions'], 105)
        self.assertGreater(result['maximum_absolute_logit_error'], 0.)

    def test_main_codebook_and_eos_labels(self):
        result = compare_logits(self.native[:, 0], self.native[:, 0], self.labels[:, 0])
        self.assertEqual(result['shape'], [7, 64])
        self.assertEqual(result['labels'], 7)

    def test_localized_error_not_hidden_by_average(self):
        changed = self.native.clone()
        changed[3, 7, 21] += .01
        with self.assertRaisesRegex(ValueError, 'limit exceeded'):
            compare_logits(self.native, changed, self.labels)

    def test_frame_or_codebook_permutation_rejected(self):
        for changed in (self.native.roll(1, 0), self.native.roll(1, 1)):
            with self.assertRaises(ValueError):
                compare_logits(self.native, changed, self.labels)

    def test_constant_shift_probability_equivalence_still_has_raw_guard(self):
        with self.assertRaisesRegex(ValueError, 'logit_error'):
            compare_logits(self.native, self.native + .001, self.labels)
        with self.assertRaisesRegex(ValueError, 'mean_absolute_logit_error'):
            compare_logits(self.native, self.native + 6e-5, self.labels)

    def test_rare_chosen_token_cannot_hide_behind_small_kl(self):
        a = torch.zeros(1, 2048); a[0, 2047] = -15.
        b = a.clone(); b[0, 2047] += 3e-4
        with self.assertRaisesRegex(ValueError, 'maximum_chosen_log_probability_error'):
            compare_logits(a, b, torch.tensor([2047]))

    def test_rare_unchosen_token_still_checked(self):
        a = torch.zeros(1, 2048); a[0, 2047] = -15.
        b = a.clone(); b[0, 2047] += .001
        with self.assertRaisesRegex(ValueError, 'maximum_absolute_logit_error'):
            compare_logits(a, b, torch.tensor([0]))

    def test_numerical_near_tie_can_flip_without_semantic_failure(self):
        a = torch.tensor([[0., 0., -2.]])
        b = torch.tensor([[0., 1e-5, -2.]])
        result = compare_logits(a, b, torch.tensor([1]))
        self.assertEqual(result['argmax_disagreements'], 1)
        self.assertEqual(result['stable_argmax_disagreements'], 0)

    def test_nonfinite_wrong_dtype_shape_and_labels_rejected(self):
        nan = self.native.clone(); nan[0, 0, 0] = float('nan')
        bad = [(nan, self.labels), (self.native.double(), self.labels),
               (self.native[:, :-1], self.labels), (self.native, self.labels.float()),
               (self.native, torch.full_like(self.labels, -1)),
               (self.native, torch.full_like(self.labels, 64))]
        for changed, labels in bad:
            with self.subTest(dtype=changed.dtype, shape=changed.shape), self.assertRaises(ValueError):
                compare_logits(self.native, changed, labels)

    def test_argmax_regret_checks_both_winners_with_a_fixed_limit(self):
        a = torch.zeros(1, 64); a[0, 0] = .00015
        b = a.clone(); b[0, 0] = -.00015; b[0, 1] = .00015
        with self.assertRaisesRegex(ValueError, 'maximum_reverse_argmax_regret'):
            compare_logits(a, b, torch.tensor([2]))

    def test_receipt_cannot_relax_or_remove_gates(self):
        result = compare_logits(self.native, self.native, self.labels)
        for key, maximum in LIMITS.items():
            changed = copy.deepcopy(result); changed[key] = maximum * 1.01
            with self.assertRaises(ValueError): validate_metrics(changed)
            changed = copy.deepcopy(result); changed[key] = float('nan')
            with self.assertRaises(ValueError): validate_metrics(changed)
            changed = copy.deepcopy(result); del changed[key]
            with self.assertRaises(ValueError): validate_metrics(changed)
        changed = copy.deepcopy(result); changed['limits']['maximum_forward_kl'] = 1.
        with self.assertRaises(ValueError): validate_metrics(changed)
        for key, value in [('labels', 1), ('finite', False), ('stable_argmax_disagreements', 1)]:
            changed = copy.deepcopy(result); changed[key] = value
            with self.assertRaises(ValueError): validate_metrics(changed)


if __name__ == '__main__':
    unittest.main()
