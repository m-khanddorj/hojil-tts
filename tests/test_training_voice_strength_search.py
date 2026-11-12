import unittest

from training.voice_strength_search import goal_screen, scaled_adapter


class StrengthTests(unittest.TestCase):
    def test_scales_update_once_without_mutating_source(self):
        import torch
        torch.manual_seed(12)
        state = {f'text_projection.linear_fc{i}.lora_{side}.weight': torch.randn(3, 3)
                 for i in (1, 2) for side in ('A', 'B')}
        original = {k:v.clone() for k,v in state.items()}
        for strength in (0, 0.25, 0.5, 0.75, 1):
            scaled = scaled_adapter(state, strength)
            for i in (1, 2):
                a, b = (f'text_projection.linear_fc{i}.lora_{s}.weight' for s in ('A', 'B'))
                torch.testing.assert_close(scaled[b] @ scaled[a], strength * (state[b] @ state[a]))
                self.assertTrue(torch.equal(scaled[a], state[a]))
            self.assertTrue(all(torch.equal(state[k], original[k]) for k in state))

    def test_invalid_strength_rejected(self):
        for strength in (-0.1, 1.1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                scaled_adapter({}, strength)

    def test_both_groups_and_strict_thresholds_required(self):
        passing = dict(pooled_cer=0.029, speaker_macro_cosine=0.985)
        self.assertTrue(goal_screen(dict(custom=passing, development=passing)))
        for group in ('custom', 'development'):
            for changed in (dict(pooled_cer=0.03), dict(speaker_macro_cosine=0.984),
                            dict(pooled_cer=float('nan')), dict(speaker_macro_cosine=float('nan'))):
                summary = dict(custom=passing, development=passing)
                summary[group] = dict(passing, **changed)
                self.assertFalse(goal_screen(summary))
        with self.assertRaises(ValueError):
            goal_screen(dict(custom=passing))


if __name__ == '__main__':
    unittest.main()
