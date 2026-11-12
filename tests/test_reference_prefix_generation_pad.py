import unittest
from unittest.mock import patch

import torch

from test_training_sequence import AVAILABLE, tiny
from training.reference_prefix_routing import ReferencePrefixRouting, weight_modules
from training.reference_prefix_generation_pad import ReferencePrefixGenerationPad, tensor_sha256


@unittest.skipUnless(AVAILABLE, 'Qwen required')
class GenerationPadTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.model = tiny().eval().requires_grad_(False)
        self.weights = {n + '.weight': m.weight.detach().clone() * .8
                        for n, m in weight_modules(self.model).items()}
        self.kwargs = dict(input_ids=[torch.tensor([[1, 2, 3, 10, 11, 12, 4, 5, 6, 7, 8]])],
            ref_ids=[torch.tensor([[1, 2, 3, 20, 21, 4, 5]])],
            voice_clone_prompt=dict(ref_spk_embedding=[torch.randn(32)],
                ref_code=[torch.randint(0, 64, (2, 16))], icl_mode=[True], x_vector_only_mode=[False]),
            languages=['Auto'], non_streaming_mode=True, max_new_tokens=4,
            do_sample=False, subtalker_dosample=False)
        self.custom, self.other = 'a' * 64, 'b' * 64

    def generate(self, wrapper=None, sha=None):
        with torch.inference_mode():
            if wrapper is None:
                return self.model.generate(**self.kwargs)
            with wrapper.use_reference(sha):
                return self.model.generate(**self.kwargs)

    def equal_codes(self, first, second):
        for a, b in zip(first[0], second[0]):
            self.assertTrue(torch.equal(a, b))

    def test_dispatch_old_mode_parity_and_inactive_full_policy(self):
        full = self.generate()
        router = ReferencePrefixRouting(self.model, self.weights)
        old = self.generate()
        original = self.model.talker.generate
        wrapper = ReferencePrefixGenerationPad(router, self.custom, full_generation_pad=False)
        self.equal_codes(old, self.generate(wrapper, self.custom))
        self.assertEqual(wrapper.last_trace['mode'], 'custom_original_half_pad')
        self.assertFalse(wrapper.last_trace['pad_replaced'])
        self.equal_codes(full, self.generate(wrapper, self.other))
        self.assertEqual(wrapper.last_trace['mode'], 'inactive_full_policy')
        self.assertFalse(wrapper.last_trace['protected'])
        self.assertTrue(router.protected)
        wrapper.remove()
        self.assertEqual(self.model.talker.generate, original)
        enabled = ReferencePrefixGenerationPad(router, self.custom, full_generation_pad=True)
        self.equal_codes(full, self.generate(enabled, self.other))
        self.assertFalse(enabled.last_trace['pad_replaced'])
        enabled.remove()
        router.remove()

    def test_only_generation_pads_change_prefix_first_frame_and_parameters_exact(self):
        router = ReferencePrefixRouting(self.model, self.weights)
        baseline = []
        native = self.model.talker.generate
        def capture(**kwargs):
            baseline.append({k: v.clone() if isinstance(v, torch.Tensor) else v
                             for k, v in kwargs.items()})
            return native(**kwargs)
        with patch.object(self.model.talker, 'generate', side_effect=capture):
            old = self.generate()
        wrapper = ReferencePrefixGenerationPad(router, self.custom)
        before = {n: tensor_sha256(p) for n, p in self.model.named_parameters()}
        actual = []
        def outgoing(**kwargs):
            actual.append(kwargs)
            return native(**kwargs)
        with patch.object(wrapper, 'original_generate', side_effect=outgoing):
            new = self.generate(wrapper, self.custom)
        self.assertTrue(torch.equal(old[0][0][0], new[0][0][0]))
        self.assertEqual(set(actual[0]), set(baseline[0]))
        for name in baseline[0]:
            if name in ('trailing_text_hidden', 'tts_pad_embed'):
                self.assertFalse(torch.equal(actual[0][name], baseline[0][name]))
            elif isinstance(actual[0][name], torch.Tensor):
                self.assertTrue(torch.equal(actual[0][name], baseline[0][name]), name)
            else:
                self.assertEqual(actual[0][name], baseline[0][name], name)
        self.assertTrue(wrapper.last_trace['input_tensors_unchanged'])
        self.assertTrue(wrapper.last_trace['non_pad_kwargs_identity_verified'])
        self.assertEqual(wrapper.last_trace['talker_calls'], 1)
        self.assertEqual(before, {n: tensor_sha256(p) for n, p in self.model.named_parameters()})
        _, full, _ = wrapper._pads()
        self.assertTrue(torch.equal(actual[0]['tts_pad_embed'], full))
        wrapper.remove()
        router.remove()

    def test_missing_nested_wrong_context_and_remove_guards(self):
        router = ReferencePrefixRouting(self.model, self.weights)
        wrapper = ReferencePrefixGenerationPad(router, self.custom)
        with self.assertRaisesRegex(ValueError, 'explicit reference context'):
            self.generate()
        self.assertFalse(router.active)
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            with wrapper.use_reference(self.custom):
                with self.assertRaisesRegex(ValueError, 'cannot nest'):
                    with wrapper.use_reference(self.other):
                        pass
                with self.assertRaisesRegex(ValueError, 'Cannot remove'):
                    wrapper.remove()
        with self.assertRaisesRegex(ValueError, 'one native talker call'):
            with wrapper.use_reference(self.custom):
                router.protected = False
                self.model.generate(**self.kwargs)
        self.assertFalse(wrapper.active)
        self.assertTrue(router.protected)
        with self.assertRaisesRegex(ValueError, 'SHA-256'):
            with wrapper.use_reference('wrong'):
                pass
        wrapper.remove()
        with self.assertRaises(ValueError):
            wrapper.remove()
        router.remove()

    def test_bad_pad_shape_value_and_double_call_fail_closed(self):
        router = ReferencePrefixRouting(self.model, self.weights)
        wrapper = ReferencePrefixGenerationPad(router, self.custom)
        _, full, half = wrapper._pads()
        good = dict(inputs_embeds=torch.zeros(1, 4, 32), attention_mask=torch.ones(1, 4, dtype=torch.long),
                    trailing_text_hidden=half.clone(), tts_pad_embed=half.clone(), marker=object())
        for replacement in (half.expand(1, 2, 32), full, half + float('nan')):
            with self.assertRaisesRegex(ValueError, 'PAD argument'):
                with wrapper.use_reference(self.custom):
                    router.active = True
                    router.layout = dict(prefix_length=4, target_start=1, target_stop=2)
                    try:
                        wrapper._generate((), dict(good, tts_pad_embed=replacement))
                    finally:
                        router.active = False
        with wrapper.use_reference(self.custom):
            router.active = True
            router.layout = dict(prefix_length=4, target_start=1, target_stop=2)
            try:
                with patch.object(wrapper, 'original_generate', return_value='done') as called:
                    self.assertEqual(wrapper._generate((), good), 'done')
                    outgoing = called.call_args.kwargs
                    for name in ('inputs_embeds', 'attention_mask', 'marker'):
                        self.assertIs(outgoing[name], good[name])
                    self.assertTrue(torch.equal(good['tts_pad_embed'], half))
                    with self.assertRaisesRegex(ValueError, 'one native talker call'):
                        wrapper._generate((), good)
            finally:
                router.active = False
        wrapper.remove()
        router.remove()

    def test_decoder_exception_restores_context_and_allows_retry(self):
        router = ReferencePrefixRouting(self.model, self.weights, protected=False)
        wrapper = ReferencePrefixGenerationPad(router, self.custom)
        with patch.object(router, 'original_main', side_effect=RuntimeError('decoder failure')):
            with self.assertRaisesRegex(RuntimeError, 'decoder failure'):
                self.generate(wrapper, self.custom)
        self.assertEqual(wrapper.last_trace['status'], 'failed')
        self.assertFalse(wrapper.active)
        self.assertFalse(router.active)
        self.assertFalse(router.protected)
        self.assertIsNone(router.main_mask)
        self.generate(wrapper, self.custom)
        self.assertEqual(wrapper.last_trace['status'], 'complete')
        wrapper.remove()
        router.remove()

    def test_nonfinite_computed_full_pad_and_training_mode_rejected(self):
        router = ReferencePrefixRouting(self.model, self.weights)
        wrapper = ReferencePrefixGenerationPad(router, self.custom)
        with patch.object(router, 'original_projection', return_value=torch.full((1, 3, 32), float('nan'))):
            with self.assertRaisesRegex(ValueError, 'Nonfinite full or half'):
                wrapper._pads()
        self.model.train()
        with self.assertRaisesRegex(ValueError, 'evaluation router'):
            with wrapper.use_reference(self.custom):
                pass
        self.model.eval()
        wrapper.remove()
        router.remove()


if __name__ == '__main__':
    unittest.main()
