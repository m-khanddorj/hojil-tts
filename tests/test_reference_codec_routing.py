import unittest
from unittest.mock import patch
import torch
from test_training_sequence import AVAILABLE,tiny
from training.reference_codec_routing import ReferenceCodecRouting,weight_modules


@unittest.skipUnless(AVAILABLE,'Qwen required')
class ReferenceCodecRoutingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(118)
        self.model=tiny().eval().requires_grad_(False)
        self.weights={n+'.weight':m.weight.detach().clone()*.8 for n,m in weight_modules(self.model).items()}
        self.kwargs=dict(input_ids=[torch.tensor([[1,2,3,10,11,12,4,5,6,7,8]])],
            ref_ids=[torch.tensor([[1,2,3,20,21,4,5]])],
            voice_clone_prompt=dict(ref_spk_embedding=[torch.randn(32)],ref_code=[torch.randint(0,64,(2,16))],
                                    icl_mode=[True],x_vector_only_mode=[False]),
            languages=['Auto'],non_streaming_mode=True,max_new_tokens=4,
            do_sample=False,subtalker_dosample=False)

    def test_noop_native_codecs_and_unchanged_parameters(self):
        before={n:p.clone() for n,p in self.model.named_parameters()}
        with torch.inference_mode():expected=self.model.generate(**self.kwargs)
        router=ReferenceCodecRouting(self.model,self.weights,protected=False)
        with torch.inference_mode():actual=self.model.generate(**self.kwargs)
        for a,b in zip(actual[0],expected[0]):self.assertTrue(torch.equal(a,b))
        self.assertEqual(router.last_trace['prefills'],1)
        self.assertEqual(router.last_trace['acoustic_start'],14)
        self.assertEqual(router.last_trace['prefix_length'],17)
        self.assertEqual(router.last_trace['reference_positions'],3)
        self.assertFalse(router.last_trace['text_prefix_exact'])
        for n,p in self.model.named_parameters():self.assertTrue(torch.equal(p,before[n]))
        router.remove()

    def test_protected_audio_preserves_complete_earlier_hidden_states_and_full_decode(self):
        hidden=[]
        def capture(module,args,output):
            if output.last_hidden_state.shape[1]>1:hidden.append(output.last_hidden_state.detach().clone())
        handle=self.model.talker.model.register_forward_hook(capture)
        with torch.inference_mode():self.model.generate(**self.kwargs)
        router=ReferenceCodecRouting(self.model,self.weights)
        first=next(iter(router.modules.values()));decoded=[]
        def check_decode(module,args,output):
            if args[0].shape[1]==1:
                expected=torch.nn.functional.linear(args[0],module.weight,module.bias)
                self.assertTrue(torch.equal(output,expected));decoded.append(True)
        decode_handle=first.register_forward_hook(check_decode)
        with torch.inference_mode():self.model.generate(**self.kwargs)
        handle.remove();decode_handle.remove()
        start=router.last_trace['acoustic_start']
        self.assertEqual(len(hidden),2)
        self.assertTrue(torch.equal(hidden[0][:,:start],hidden[1][:,:start]))
        self.assertFalse(torch.equal(hidden[0][:,start:],hidden[1][:,start:]))
        self.assertTrue(decoded)
        self.assertTrue(router.last_trace['text_prefix_exact'])
        router.remove()

    def test_exception_does_not_leave_routing_active(self):
        router=ReferenceCodecRouting(self.model,self.weights)
        with patch.object(router,'original_main',side_effect=RuntimeError('forced failure')):
            with self.assertRaisesRegex(RuntimeError,'forced failure'):
                with torch.inference_mode():self.model.generate(**self.kwargs)
        self.assertFalse(router.active);self.assertIsNone(router.main_mask)
        router.remove()
        self.assertFalse(hasattr(self.model,'reference_codec_routing'))


if __name__=='__main__':unittest.main()
