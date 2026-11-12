import unittest
import torch
from tests.test_training_sequence import tiny
from training.speaker_preference_adapter import install as install_speaker
from training import middle_attention_adapter as adapters
from training.codec_policy_replay import policy_outputs, combine_reference_probabilities, backward_probability_gradient
from training.differentiable_codec_probe import straight_through_waveform
from qwen_tts.core.tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2DecoderConfig
from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2Decoder


class CodecPolicyReplayTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.model=tiny().requires_grad_(False)
        install_speaker(self.model,rank=2); adapters.install(self.model,[0,1],rank=2)
        with torch.no_grad():
            for m in adapters.modules(self.model).values(): m.lora_B.normal_(std=.002)
        self.trajectory=dict(input_ids=torch.tensor([[1,2,3,10,11,12,4,5,6,7,8]]),
            ref_ids=torch.tensor([[1,2,3,20,21,4,5]]),codes=torch.randint(0,64,(3,16)),
            ref_codes=torch.randint(0,64,(2,16)),ref_xvec=torch.randn(32),terminated=True)

    def test_posterior_alignment_preserves_full_likelihood_and_eos(self):
        result=policy_outputs(self.model,self.trajectory)
        expected,tokens=adapters.log_probability(self.model,self.trajectory)
        torch.testing.assert_close(result['log_probability'],expected,atol=1e-5,rtol=0)
        self.assertEqual(result['token_count'],tokens)
        self.assertEqual(result['probabilities'].shape,(1,16,3,64))
        torch.testing.assert_close(result['probabilities'].sum(-1),torch.ones(1,16,3))
        self.assertTrue(((result['main_acoustic_mass']>0)&(result['main_acoustic_mass']<1)).all())
        with self.assertRaises(ValueError): policy_outputs(self.model,dict(self.trajectory,terminated=False))

    def test_separate_codec_and_policy_passes_match_full_graph_gradient(self):
        config=Qwen3TTSTokenizerV2DecoderConfig(codebook_size=64,codebook_dim=16,hidden_size=16,
            latent_dim=16,intermediate_size=32,num_attention_heads=2,num_key_value_heads=2,
            num_hidden_layers=1,num_quantizers=16,head_dim=8,upsample_rates=(2,),
            upsampling_ratios=(2,),decoder_dim=16)
        config._attn_implementation='eager'
        codec=Qwen3TTSTokenizerV2Decoder(config).eval().requires_grad_(False)
        with torch.no_grad():
            for name,p in codec.named_parameters():
                if name.endswith('embedding_sum'): p.normal_(std=.2)
        frozen,speaker=adapters.frozen_digest(self.model),adapters.speaker_digest(self.model)
        for frames in (3,5):
            trajectory=dict(self.trajectory,codes=torch.randint(0,64,(frames,16)))
            codes=torch.cat([trajectory['ref_codes'],trajectory['codes']],dim=0).T[None]
            self.model.zero_grad(set_to_none=True)
            probabilities=policy_outputs(self.model,trajectory)['probabilities']
            wave=straight_through_waveform(codec,codes,combine_reference_probabilities(trajectory['ref_codes'],probabilities))
            direction=torch.randn_like(wave[...,8:])
            (wave[...,8:]*direction).sum().backward()
            direct={n:p.grad.clone() for n,p in self.model.named_parameters() if p.requires_grad}
            self.assertTrue(all(g.norm()>0 for g in direct.values()))
            self.model.zero_grad(set_to_none=True)
            with torch.no_grad(): cached=policy_outputs(self.model,trajectory)['probabilities'].detach().clone()
            leaf=cached.clone().requires_grad_(True)
            wave=straight_through_waveform(codec,codes,combine_reference_probabilities(trajectory['ref_codes'],leaf))
            gradient,=torch.autograd.grad((wave[...,8:]*direction).sum(),leaf)
            backward_probability_gradient(self.model,trajectory,cached,gradient.detach())
            for name,p in self.model.named_parameters():
                if p.requires_grad: torch.testing.assert_close(p.grad,direct[name],atol=1e-7,rtol=2e-4)
                else: self.assertIsNone(p.grad)
            self.assertTrue(all(p.grad is None for p in codec.parameters()))
        self.assertEqual(adapters.frozen_digest(self.model),frozen)
        self.assertEqual(adapters.speaker_digest(self.model),speaker)

    def test_mismatched_cache_is_rejected_before_backward(self):
        with torch.no_grad(): cached=policy_outputs(self.model,self.trajectory)['probabilities']
        for invalid in (cached+.01,cached[:,:,:-1]):
            with self.assertRaises(ValueError):
                backward_probability_gradient(self.model,self.trajectory,invalid,torch.ones_like(invalid))
        with self.assertRaises(ValueError):
            backward_probability_gradient(self.model,self.trajectory,cached,torch.full_like(cached,float('nan')))
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))


if __name__=='__main__': unittest.main()
