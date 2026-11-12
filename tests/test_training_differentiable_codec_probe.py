import unittest

import torch
from qwen_tts.core.tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2DecoderConfig
from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2Decoder

from training.differentiable_codec_probe import expected_quantized, waveform_from_quantized, straight_through_waveform


class DifferentiableCodecProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2); torch.manual_seed(4073)
        config = Qwen3TTSTokenizerV2DecoderConfig(codebook_size=8,codebook_dim=16,
            hidden_size=16,latent_dim=16,intermediate_size=32,num_attention_heads=2,
            num_key_value_heads=2,num_hidden_layers=1,num_quantizers=16,head_dim=8,
            upsample_rates=(2,),upsampling_ratios=(2,),decoder_dim=16)
        config._attn_implementation='eager'
        cls.decoder=Qwen3TTSTokenizerV2Decoder(config).eval().requires_grad_(False)
        with torch.no_grad():
            for name,parameter in cls.decoder.named_parameters():
                if name.endswith('embedding_sum'): parameter.normal_(std=.15)
                if name.endswith('cluster_usage'): parameter.uniform_(.5,2.)

    def test_one_hot_latents_and_hard_forward_match_native(self):
        codes=torch.randint(0,8,(1,16,5))
        probs=torch.nn.functional.one_hot(codes,8).float()
        soft=expected_quantized(self.decoder,probs)
        hard=self.decoder.quantizer.decode(codes)
        torch.testing.assert_close(soft,hard,atol=1e-6,rtol=1e-6)
        arbitrary=torch.randn(1,16,5,8).softmax(-1)
        native=self.decoder(codes)
        candidate=straight_through_waveform(self.decoder,codes,arbitrary)
        self.assertTrue(torch.equal(native,candidate))
        self.assertTrue(torch.equal(waveform_from_quantized(self.decoder,hard),native))

    def test_surrogate_gradient_matches_independent_latent_vjp(self):
        codes=torch.randint(0,8,(1,16,5))
        logits=torch.randn(1,16,5,8,requires_grad=True)
        wave=straight_through_waveform(self.decoder,codes,logits.softmax(-1))
        direction=torch.randn_like(wave)
        actual,=torch.autograd.grad((wave*direction).sum(),logits)
        latent=self.decoder.quantizer.decode(codes).detach().requires_grad_(True)
        native_wave=waveform_from_quantized(self.decoder,latent)
        latent_grad,=torch.autograd.grad((native_wave*direction).sum(),latent)
        independent_logits=logits.detach().clone().requires_grad_(True)
        expected,=torch.autograd.grad((expected_quantized(self.decoder,independent_logits.softmax(-1))*latent_grad).sum(),independent_logits)
        torch.testing.assert_close(actual,expected,atol=1e-7,rtol=1e-5)
        self.assertTrue(torch.isfinite(actual).all())
        self.assertGreater(float(actual.norm()),1e-9)
        torch.testing.assert_close(actual.sum(-1),torch.zeros_like(actual[...,0]),atol=1e-7,rtol=0)
        self.assertTrue(all(p.grad is None for p in self.decoder.parameters()))

    def test_rejects_missing_codebooks_invalid_codes_and_trainable_codec(self):
        codes=torch.zeros(1,16,5,dtype=torch.long)
        probs=torch.full((1,16,5,8),1/8)
        with self.assertRaises(ValueError): expected_quantized(self.decoder,probs[:,:15])
        with self.assertRaises(ValueError): expected_quantized(self.decoder,probs*2)
        with self.assertRaises(ValueError): straight_through_waveform(self.decoder,codes+8,probs)
        parameter=next(self.decoder.parameters()); parameter.requires_grad_(True)
        try:
            with self.assertRaises(ValueError): expected_quantized(self.decoder,probs)
        finally: parameter.requires_grad_(False)


if __name__=='__main__': unittest.main()
