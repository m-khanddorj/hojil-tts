import unittest
from types import SimpleNamespace

from training.preference_rollouts import capture_generation, select_cases


class PreferenceRolloutTests(unittest.TestCase):
    def test_selection_balances_speakers_and_separates_preference_validation(self):
        rows = [dict(id=f'{s}-{n}', speaker_id=str(s), text=f'target {s} {n}',
            ref_text=f'reference {s}', duration=4) for s in range(20) for n in range(3)]
        selected = select_cases(rows, ['target 0 0', 'reference 1'])
        self.assertEqual(selected, select_cases(list(reversed(rows)), ['target 0 0', 'reference 1']))
        self.assertEqual(len(selected), 32)
        self.assertFalse(any(r['text']=='target 0 0' or r['speaker_id']=='1' for r in selected))
        train = {r['speaker_id'] for r in selected if r['preference_split']=='train'}
        val = {r['speaker_id'] for r in selected if r['preference_split']=='validation'}
        self.assertEqual((len(train), len(val)), (12,4))
        self.assertFalse(train & val)
        self.assertTrue(all(sum(r['speaker_id']==s for r in selected)==2 for s in train|val))

    def model(self):
        import torch
        codes = torch.zeros(3,16,dtype=torch.long)
        codes[:,0] = torch.tensor([8,9,10])
        talker = SimpleNamespace(generate=lambda: SimpleNamespace(sequences=torch.tensor([[8,9,10,99]])))
        def generate():
            talker.generate()
            return [codes], [torch.zeros(3,2)]
        return SimpleNamespace(talker=talker, generate=generate,
            config=SimpleNamespace(talker_config=SimpleNamespace(codec_eos_token_id=99)))

    def test_capture_preserves_tokens_and_restores_methods(self):
        import torch
        model = self.model()
        original, talker = model.generate, model.talker.generate
        rng = torch.get_rng_state().clone()
        with capture_generation(model) as captured:
            result = model.generate()
        self.assertTrue(torch.equal(captured['codes'], result[0][0]))
        self.assertTrue(captured['terminated'])
        self.assertIs(model.generate, original)
        self.assertIs(model.talker.generate, talker)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        with self.assertRaisesRegex(RuntimeError, 'synthetic'):
            with capture_generation(model):
                raise RuntimeError('synthetic')
        self.assertIs(model.generate, original)
        self.assertIs(model.talker.generate, talker)

    def test_rejects_misaligned_sampled_codes(self):
        import torch
        model = self.model()
        model.talker.generate = lambda: SimpleNamespace(sequences=torch.tensor([[8,7,10,99]]))
        with self.assertRaisesRegex(ValueError, 'disagree'):
            with capture_generation(model):
                model.generate()


if __name__ == '__main__':
    unittest.main()
