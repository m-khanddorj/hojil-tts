import unittest
from unittest.mock import patch
import torch
from test_training_sequence import AVAILABLE,tiny
from training.reference_prefix_routing import ReferencePrefixRouting,weight_modules


@unittest.skipUnless(AVAILABLE,'Qwen required')
class ReferencePrefixRoutingTests(unittest.TestCase):
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

    def test_noop_native_codec_parity_and_one_prefill(self):
        with torch.inference_mode():expected=self.model.generate(**self.kwargs)
        router=ReferencePrefixRouting(self.model,self.weights,protected=False)
        with torch.inference_mode():actual=self.model.generate(**self.kwargs)
        self.assertEqual(len(actual[0]),len(expected[0]))
        for a,b in zip(actual[0],expected[0]):self.assertTrue(torch.equal(a,b))
        self.assertEqual(router.last_trace['prefills'],1)
        self.assertGreater(router.last_trace['decode_steps'],0)
        self.assertEqual(router.last_trace['target_start'],10)
        self.assertEqual(router.last_trace['target_stop'],13)
        self.assertEqual(router.last_trace['prefix_length'],17)
        router.remove()
        self.assertFalse(hasattr(self.model,'reference_prefix_routing'))

    def test_reference_rows_switch_weights_target_and_decode_rows_stay_full(self):
        import torch.nn.functional as F
        router=ReferencePrefixRouting(self.model,self.weights)
        name=next(iter(router.original_linears));module=router.modules[name]
        x=torch.randn(1,4,module.in_features)
        mask=torch.tensor([[True,False,False,True]])
        router.main_mask=mask
        with torch.inference_mode():
            full=F.linear(x,module.weight,module.bias)
            other=F.linear(x,self.weights[name+'.weight'],module.bias)
            actual=module(x)
        self.assertTrue(torch.equal(actual,torch.where(mask[...,None],other,full)))
        router.main_mask=None
        self.assertTrue(torch.equal(module(x),full))
        router.active=True;router.text_span=(2,5)
        projection=self.model.talker.text_projection
        z=torch.randn(1,5,projection.linear_fc1.in_features)
        with torch.inference_mode():
            full=router.original_projection(z);actual=projection(z)
        self.assertTrue(torch.equal(actual[:,2:],full[:,2:]))
        self.assertFalse(torch.equal(actual[:,:2],full[:,:2]))
        router.active=False;router.text_span=None
        router.remove()

    def test_protected_generation_keeps_parameters_and_clears_exception_state(self):
        before={n:p.clone() for n,p in self.model.named_parameters()}
        router=ReferencePrefixRouting(self.model,self.weights)
        with torch.inference_mode():self.model.generate(**self.kwargs)
        self.assertTrue(router.last_trace['protected'])
        for name,value in self.model.named_parameters():self.assertTrue(torch.equal(value,before[name]))
        with patch.object(router,'original_main',side_effect=RuntimeError('forced decoder failure')):
            with self.assertRaisesRegex(RuntimeError,'forced decoder failure'):
                with torch.inference_mode():self.model.generate(**self.kwargs)
        self.assertFalse(router.active)
        self.assertIsNone(router.main_mask)
        self.assertIsNone(router.text_span)
        router.remove()


if __name__=='__main__':unittest.main()
