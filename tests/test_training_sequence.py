"""Check alignment against the installed Qwen inference implementation."""
import importlib.util
import unittest
from unittest.mock import patch

AVAILABLE = (importlib.util.find_spec('torch') is not None
             and importlib.util.find_spec('qwen_tts') is not None)


def tiny():
    import torch
    from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSConfig
    from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSForConditionalGeneration
    torch.manual_seed(6)
    dims = dict(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                num_attention_heads=4, num_key_value_heads=2, head_dim=8, num_code_groups=16)
    cfg = Qwen3TTSConfig(tts_model_type='base', tts_model_size='0b6',
        tokenizer_type='qwen3_tts_tokenizer_12hz', tts_pad_token_id=100,
        tts_bos_token_id=101, tts_eos_token_id=102,
        talker_config=dict(**dims, text_hidden_size=32, text_vocab_size=128, vocab_size=1100,
            codec_eos_token_id=1000, codec_pad_id=1001, codec_bos_id=1002,
            codec_nothink_id=1003, codec_think_bos_id=1004, codec_think_eos_id=1005,
            spk_id={}, spk_is_dialect={}, codec_language_id={},
            rope_scaling={'rope_type': 'default', 'mrope_section': [2, 1, 1], 'interleaved': True},
            code_predictor_config=dict(**dims, vocab_size=64)),
        speaker_encoder_config=dict(enc_dim=32, enc_channels=[16, 16, 16, 16, 48],
            enc_attention_channels=8, enc_se_channels=8))
    cfg._attn_implementation = 'sdpa'
    return Qwen3TTSForConditionalGeneration(cfg).eval()


@unittest.skipUnless(AVAILABLE, 'optional Torch/Qwen training dependencies required')
class SequenceTests(unittest.TestCase):
    def setUp(self):
        import torch
        torch.set_num_threads(1)
        self.model = tiny()

    def inputs(self, length=5, frames=3):
        import torch
        return (torch.tensor([[1, 2, 3] + list(range(10, 10+length)) + [4, 5, 6, 7, 8]]),
                torch.tensor([[1, 2, 3, 20, 21, 4, 5]]),
                torch.randint(0, 64, (frames, 16)), torch.randn(32))

    def test_prefixes_match_generation_and_labels_predict_next_frame(self):
        import torch
        from training.sequence import build
        class Captured(Exception):
            pass
        for icl in (False, True):
            for non_streaming in (False, True):
                ids, ref, codes, speaker = self.inputs()
                seq = build(self.model, ids, codes, speaker, ref if icl else None,
                            codes if icl else None, non_streaming)
                captured = {}
                def intercept(**kwargs):
                    captured.update(kwargs)
                    raise Captured()
                prompt = {'ref_spk_embedding': [speaker], 'ref_code': [codes] if icl else [None],
                          'icl_mode': [icl], 'x_vector_only_mode': [not icl]}
                with patch.object(self.model.talker, 'generate', side_effect=intercept):
                    with self.assertRaises(Captured):
                        self.model.generate(input_ids=[ids], ref_ids=[ref] if icl else None,
                            voice_clone_prompt=prompt, languages=['Auto'],
                            non_streaming_mode=non_streaming, max_new_tokens=2)
                torch.testing.assert_close(seq.embeddings[:seq.prefix_length], captured['inputs_embeds'][0])
                self.assertEqual(seq.frame_positions[0], seq.prefix_length-1)
                torch.testing.assert_close(seq.labels[seq.frame_positions], codes[:, 0])
                self.assertEqual(seq.labels[-1], 1000)

    def test_microbatch_partition_preserves_loss_and_gradient(self):
        import torch
        from training.sequence import build, loss_sums
        params = [p for p in self.model.talker.parameters()]
        inputs = [self.inputs(n, f) for n, f in ((3, 2), (7, 4))]
        def make(values):
            return [build(self.model, ids, codes, spk, ref, codes, True)
                    for ids, ref, codes, spk in values]
        first, residual, n, m = loss_sums(self.model, make(inputs))
        loss = first/n + 0.3*residual/m
        loss.backward()
        expected = [p.grad.clone() if p.grad is not None else None for p in params]
        self.model.zero_grad(set_to_none=True)
        total = 0
        for values in inputs:
            a, b, _, _ = loss_sums(self.model, make([values]))
            part = a/n + 0.3*b/m
            total += part.item()
            part.backward()
        self.assertAlmostEqual(loss.item(), total, places=5)
        for param, grad in zip(params, expected):
            if grad is not None:
                torch.testing.assert_close(param.grad, grad, atol=2e-5, rtol=2e-4)

    def test_injection_trains_only_fp32_adapters(self):
        import torch
        from training.engine import inject
        model = self.model.to(torch.bfloat16)
        trainable = inject(model.talker, rank=2, dropout=0)
        self.assertTrue(trainable)
        for name, param in model.talker.named_parameters():
            self.assertEqual(param.requires_grad, 'lora_' in name)
            self.assertEqual(param.dtype, torch.float32 if param.requires_grad else torch.bfloat16)

    def test_full_training_updates_talker_and_freezes_speaker_encoder(self):
        import torch
        from training.engine import configure
        from training.sequence import build, loss_sums
        model = self.model.to(torch.bfloat16)
        params = configure(model, 'full', gradient_checkpointing=True)
        self.assertTrue(all(p.requires_grad and p.dtype == torch.float32 for p in model.talker.parameters()))
        self.assertTrue(all(not p.requires_grad for p in model.speaker_encoder.parameters()))
        self.assertFalse(any('lora_' in n for n, _ in model.named_parameters()))
        before = model.talker.codec_head.weight.detach().clone()
        model.talker.train()
        ids, ref, codes, speaker = self.inputs()
        seq = build(model, ids, codes, speaker, ref, codes, True)
        a, b, n, m = loss_sums(model, [seq])
        (a/n + 0.3*b/m).backward()
        optimizer = torch.optim.AdamW(params, lr=1e-4)
        optimizer.step()
        self.assertFalse(torch.equal(before, model.talker.codec_head.weight))
        self.assertTrue(all(p.grad is None for p in model.speaker_encoder.parameters()))


if __name__ == '__main__':
    unittest.main()
