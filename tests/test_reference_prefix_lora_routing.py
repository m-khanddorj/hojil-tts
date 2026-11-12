import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from test_training_sequence import AVAILABLE, tiny
from training.middle_attention_adapter import install, modules
from training.reference_prefix_lora_routing import ReferencePrefixLoRARouting, weight_modules, base_linear
from training.reference_prefix_generation_pad import ReferencePrefixGenerationPad, tensor_sha256


@unittest.skipUnless(AVAILABLE, 'Qwen required')
class PrefixLoRARoutingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.model = tiny().eval()
        self.model.speaker_preference_adapter = torch.nn.Identity()
        install(self.model, [0, 1], rank=4)
        with torch.no_grad():
            for module in modules(self.model).values():
                module.lora_B.normal_(std=.03)
        self.model.requires_grad_(False)
        self.registry = weight_modules(self.model, (0, 1))
        self.weights = {name + '.weight': base_linear(module).weight.detach().clone() * .8
                        for name, module in self.registry.items()}
        self.kwargs = dict(input_ids=[torch.tensor([[1, 2, 3, 10, 11, 12, 4, 5, 6, 7, 8]])],
            ref_ids=[torch.tensor([[1, 2, 3, 20, 21, 4, 5]])],
            voice_clone_prompt=dict(ref_spk_embedding=[torch.randn(32)],
                ref_code=[torch.randint(0, 64, (2, 16))], icl_mode=[True], x_vector_only_mode=[False]),
            languages=['Auto'], non_streaming_mode=True, max_new_tokens=4,
            do_sample=False, subtalker_dosample=False)

    def router(self, protected=True):
        return ReferencePrefixLoRARouting(self.model, self.weights, protected, expected_layers=(0, 1))

    def generate(self, pad=None, sha='a' * 64):
        with torch.inference_mode():
            if pad is None:
                return self.model.generate(**self.kwargs)
            with pad.use_reference(sha):
                return self.model.generate(**self.kwargs)

    def equal_codes(self, first, second):
        self.assertEqual(len(first[0]), len(second[0]))
        for a, b in zip(first[0], second[0]):
            self.assertTrue(torch.equal(a, b))

    def test_canonical_registry_rejects_wrong_outer_coverage(self):
        self.assertEqual(len(self.registry), 16)
        self.assertEqual(len(modules(self.model)), 4)
        self.assertFalse(any('.base_layer' in name for name in self.registry))
        for name, module in modules(self.model).items():
            self.assertIs(self.registry[name], module)
            self.assertIs(base_linear(module), module.base_layer)
        with self.assertRaisesRegex(ValueError, 'Unexpected wrapped'):
            weight_modules(self.model, (0,))
        with self.assertRaisesRegex(ValueError, 'Invalid expected'):
            weight_modules(self.model)
        wrong = dict(self.weights)
        name = next(iter(modules(self.model)))
        wrong[name + '.base_layer.weight'] = wrong.pop(name + '.weight')
        with self.assertRaisesRegex(ValueError, 'coverage differs'):
            ReferencePrefixLoRARouting(self.model, wrong, expected_layers=(0, 1))
        next(iter(modules(self.model).values())).lora_A.requires_grad_(True)
        with self.assertRaisesRegex(ValueError, 'frozen CPU FP32'):
            self.router()

    def test_outer_half_branch_excludes_lora_and_full_branch_is_exact(self):
        name, module = next(iter(modules(self.model).items()))
        original = module.forward
        x = torch.randn(1, 4, module.base_layer.in_features)
        with torch.inference_mode():
            full = original(x)
            half = F.linear(x, self.weights[name + '.weight'], module.base_layer.bias)
            residual = F.linear(F.linear(x, module.lora_A), module.lora_B)
        router = self.router()
        mask = torch.tensor([[True, False, False, True]])
        router.main_mask = mask
        with torch.inference_mode():
            actual = module(x)
        self.assertTrue(torch.equal(actual, torch.where(mask[..., None], half, full)))
        self.assertFalse(torch.equal(actual[:, mask[0]], (half + residual)[:, mask[0]]))
        router.main_mask = None
        with torch.inference_mode():
            self.assertTrue(torch.equal(module(x), full))
        self.assertIs(router.modules[name], module)
        self.assertNotIn(name + '.base_layer', router.original_linears)
        router.remove()
        self.assertEqual(module.forward, original)

    def test_disabled_native_generation_exact_and_frozen_parameters(self):
        original = {n: tensor_sha256(p) for n, p in self.model.named_parameters()}
        expected = self.generate()
        router = self.router(protected=False)
        self.equal_codes(expected, self.generate())
        self.assertEqual(router.last_trace['main_matrices'], 14)
        self.assertEqual(router.last_trace['prefills'], 1)
        self.assertGreater(router.last_trace['decode_steps'], 0)
        self.assertEqual(original, {n: tensor_sha256(p) for n, p in self.model.named_parameters()})
        router.remove()

    def test_pad_compatibility_inactive_exact_and_first_frame_prefix_preserved(self):
        baseline = self.generate()
        router = self.router()
        old_pad = ReferencePrefixGenerationPad(router, 'a' * 64, full_generation_pad=False)
        old = self.generate(old_pad)
        old_trace = old_pad.last_trace
        old_pad.remove()
        new_pad = ReferencePrefixGenerationPad(router, 'a' * 64, full_generation_pad=True)
        new = self.generate(new_pad)
        new_trace = new_pad.last_trace
        for name in ('inputs_embeds', 'attention_mask', 'trailing_text_hidden', 'tts_pad_embed'):
            self.assertEqual(old_trace['input_tensor_sha256'][name], new_trace['input_tensor_sha256'][name])
        self.assertTrue(torch.equal(old[0][0][0], new[0][0][0]))
        self.assertNotEqual(old_trace['output_pad_sha256'], new_trace['output_pad_sha256'])
        self.equal_codes(baseline, self.generate(new_pad, 'b' * 64))
        self.assertEqual(new_pad.last_trace['mode'], 'inactive_full_policy')
        new_pad.remove()
        router.remove()

    def test_half_text_control_projection_and_full_target_projection(self):
        router = self.router()
        projection = self.model.talker.text_projection
        x = torch.randn(1, 5, projection.linear_fc1.in_features)
        router.active = True
        router.text_span = (2, 5)
        with torch.inference_mode():
            actual = projection(x)
            full = router.original_projection(x)
            half = F.linear(projection.act_fn(F.linear(x,
                router.weights['talker.text_projection.linear_fc1'], projection.linear_fc1.bias)),
                router.weights['talker.text_projection.linear_fc2'], projection.linear_fc2.bias)
        self.assertTrue(torch.equal(actual[:, :2], half[:, :2]))
        self.assertTrue(torch.equal(actual[:, 2:], full[:, 2:]))
        router.active = False
        router.text_span = None
        router.remove()

    def test_decoder_exception_restores_outer_wrappers_and_context(self):
        originals = {name: module.forward for name, module in modules(self.model).items()}
        router = self.router()
        pad = ReferencePrefixGenerationPad(router, 'a' * 64)
        with patch.object(router, 'original_main', side_effect=RuntimeError('forced decoder failure')):
            with self.assertRaisesRegex(RuntimeError, 'forced decoder failure'):
                self.generate(pad)
        self.assertFalse(router.active)
        self.assertIsNone(router.main_mask)
        self.assertIsNone(router.text_span)
        self.assertFalse(pad.active)
        self.generate(pad)
        pad.remove()
        router.remove()
        for name, module in modules(self.model).items():
            self.assertEqual(module.forward, originals[name])


if __name__ == '__main__':
    unittest.main()
