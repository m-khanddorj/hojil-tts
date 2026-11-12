import unittest
import torch
from tests.test_training_sequence import tiny
from training.speaker_preference_adapter import install as install_speaker
from training import middle_attention_adapter as attention
from training.codec_policy_replay import policy_outputs,backward_probability_gradient
from training.voice_gradient_geometry import enable_adapter_gradients,capture_gradients,compare_gradients


class VoiceGradientGeometryTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(31517);torch.set_num_threads(1)
        self.model=tiny().requires_grad_(False)
        install_speaker(self.model,rank=2);attention.install(self.model,[0,1],rank=2)
        with torch.no_grad():
            self.model.speaker_preference_adapter.up.normal_(std=.02)
            for m in attention.modules(self.model).values():m.lora_B.normal_(std=.002)
        self.trajectory=dict(input_ids=torch.tensor([[1,2,3,10,11,12,4,5,6,7,8]]),
            ref_ids=torch.tensor([[1,2,3,20,21,4,5]]),codes=torch.randint(0,64,(3,16)),
            ref_codes=torch.randint(0,64,(2,16)),ref_xvec=torch.randn(32),terminated=True)

    def test_speaker_and_attention_replay_match_direct_graph_without_weight_changes(self):
        original={n:p.detach().clone() for n,p in self.model.named_parameters()}
        enable_adapter_gradients(self.model)
        probabilities=policy_outputs(self.model,self.trajectory)['probabilities']
        gradient=torch.randn_like(probabilities)
        (probabilities*gradient).sum().backward();direct=capture_gradients(self.model)
        self.model.zero_grad(set_to_none=True)
        with torch.no_grad():cached=policy_outputs(self.model,self.trajectory)['probabilities']
        backward_probability_gradient(self.model,self.trajectory,cached,gradient)
        replay=capture_gradients(self.model)
        self.assertEqual(set(direct),set(replay))
        self.assertTrue(all(v.norm()>0 for v in replay.values()))
        for name in direct:torch.testing.assert_close(replay[name],direct[name],atol=1e-7,rtol=2e-4)
        self.model.zero_grad(set_to_none=True)
        logp,tokens=attention.log_probability(self.model,self.trajectory)
        (-logp/tokens).backward();anchor=capture_gradients(self.model)
        report=compare_gradients(replay,anchor)
        self.assertEqual(report['all_adapters']['parameters'],
                         report['speaker_adapter']['parameters']+report['attention_adapter']['parameters'])
        self.assertEqual(report['speaker_adapter']['tensors'],2)
        for row in report.values():
            self.assertLessEqual(row['projected_anchor_change'],1e-10)
            self.assertLessEqual(row['projected_voice_change'],1e-10)
        for name,p in self.model.named_parameters():self.assertTrue(torch.equal(p,original[name]))

    def test_alignment_and_opposition_are_distinguished(self):
        names=['speaker_preference_adapter.up','talker.model.layers.0.self_attn.q_proj.lora_A']
        voice={n:torch.tensor([1.,0.]) for n in names}
        anchor={names[0]:torch.tensor([1.,0.]),names[1]:torch.tensor([-1.,0.])}
        report=compare_gradients(voice,anchor)
        self.assertEqual(report['speaker_adapter']['cosine'],1)
        self.assertEqual(report['speaker_adapter']['projected_voice_gradient_fraction'],1)
        self.assertEqual(report['attention_adapter']['cosine'],-1)
        self.assertEqual(report['attention_adapter']['projected_voice_gradient_fraction'],0)
        with self.assertRaises(ValueError):compare_gradients(voice,{names[0]:anchor[names[0]]})


if __name__=='__main__':unittest.main()
