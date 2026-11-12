import unittest
import torch
from tests.test_training_differentiable_codec_probe import DifferentiableCodecProbeTests
from training.direct_voice_objective import checkpointed_waveform, constrained_direction
from training.differentiable_codec_probe import straight_through_waveform


class DirectVoiceObjectiveTests(unittest.TestCase):
    def test_checkpointed_chunking_preserves_forward_and_gradient(self):
        DifferentiableCodecProbeTests.setUpClass();decoder=DifferentiableCodecProbeTests.decoder
        codes=torch.randint(0,8,(1,16,9));raw=torch.randn(1,16,9,8)
        leaf=raw.clone().requires_grad_(True)
        candidate=checkpointed_waveform(decoder,codes,leaf.softmax(-1),chunk_size=4,left_context=2)
        with torch.no_grad():native=decoder.chunked_decode(codes,chunk_size=4,left_context_size=2)
        self.assertTrue(torch.equal(candidate.detach(),native))
        direction=torch.randn_like(candidate)
        actual,=torch.autograd.grad((candidate*direction).sum(),leaf)
        other=raw.clone().requires_grad_(True);probs=other.softmax(-1);waves=[]
        for start in range(0,9,4):
            context=min(start,2);end=min(start+4,9)
            wave=straight_through_waveform(decoder,codes[...,start-context:end],probs[:,:,start-context:end])
            waves.append(wave[...,context*decoder.total_upsample:])
        expected,=torch.autograd.grad((torch.cat(waves,-1)*direction).sum(),other)
        torch.testing.assert_close(actual,expected,atol=1e-7,rtol=1e-5)
        self.assertGreater(float(actual.norm()),0)
        self.assertTrue(all(p.grad is None for p in decoder.parameters()))

    def test_constraint_handles_adam_direction_and_opposing_objectives(self):
        proposal=torch.tensor([-2.,1.]);voice=torch.tensor([1.,-1.]);anchor=torch.tensor([-1.,1.])
        with self.assertRaises(ValueError): constrained_direction(proposal,voice,anchor)
        for proposal,voice,anchor in (
            (torch.tensor([1.,-1.]),torch.tensor([1.,2.]),torch.tensor([1.,0.])),
            (torch.tensor([3.,-1.]),torch.tensor([1.,2.]),torch.tensor([0.,0.])),
            (torch.tensor([-2.,1.]),torch.tensor([1.,1.]),torch.tensor([-1.,0.])),
        ):
            d,receipt=constrained_direction(proposal,voice,anchor)
            self.assertLess(float(d.dot(voice)),0)
            self.assertLessEqual(float(d.dot(anchor)),1e-6)
            self.assertLessEqual(float(d.norm()),float(proposal.norm())+1e-6)
            self.assertLess(receipt['predicted_voice_change'],0)


if __name__=='__main__':unittest.main()
