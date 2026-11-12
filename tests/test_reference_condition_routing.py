import unittest
from unittest.mock import patch
import torch
from test_training_sequence import AVAILABLE,tiny
from training.reference_condition_routing import ReferenceConditionRouting,weight_modules


@unittest.skipUnless(AVAILABLE,'Qwen required')
class ReferenceConditionRoutingTests(unittest.TestCase):
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
        router=ReferenceConditionRouting(self.model,self.weights,protected=False)
        with torch.inference_mode():actual=self.model.generate(**self.kwargs)
        for a,b in zip(actual[0],expected[0]):self.assertTrue(torch.equal(a,b))
        self.assertEqual(router.last_trace['speaker_position'],6)
        self.assertEqual(router.last_trace['acoustic_start'],14)
        self.assertEqual(router.last_trace['prefix_length'],17)
        self.assertEqual(router.last_trace['reference_positions'],4)
        for n,p in self.model.named_parameters():self.assertTrue(torch.equal(p,before[n]))
        router.remove()

    def test_native_masks_text_projections_and_cached_decode(self):
        hidden=[];projections=[]
        def capture(module,args,output):
            if output.last_hidden_state.shape[1]>1:hidden.append(output.last_hidden_state.detach().clone())
        handle=self.model.talker.model.register_forward_hook(capture)
        ph=self.model.talker.text_projection.register_forward_hook(lambda m,a,o:projections.append(o.detach().clone()))
        with torch.inference_mode():self.model.generate(**self.kwargs)
        baseline_projections=list(projections);projections.clear()
        router=ReferenceConditionRouting(self.model,self.weights)
        routed=[];decoded=[];hooks=[]
        before={n:p.clone() for n,p in self.model.named_parameters()}
        def check(name):
            def hook(module,args,output):
                x=args[0];full=torch.nn.functional.linear(x,module.weight,module.bias)
                if router.main_mask is not None:
                    expected_mask=torch.zeros(x.shape[:2],dtype=torch.bool)
                    expected_mask[:,6]=True;expected_mask[:,14:]=True
                    self.assertTrue(torch.equal(router.main_mask,expected_mask))
                    other=torch.nn.functional.linear(x,router.weights[name],module.bias)
                    self.assertTrue(torch.equal(output,torch.where(expected_mask[...,None],other,full)))
                    routed.append(name)
                elif x.shape[1]==1:
                    self.assertTrue(torch.equal(output,full));decoded.append(name)
            return hook
        for name,module in router.modules.items():hooks.append(module.register_forward_hook(check(name)))
        with torch.inference_mode():self.model.generate(**self.kwargs)
        for hook in hooks+[handle,ph]:hook.remove()
        self.assertEqual(set(routed),set(router.modules))
        self.assertEqual(set(decoded),set(router.modules))
        self.assertEqual(len(baseline_projections),len(projections))
        for a,b in zip(baseline_projections,projections):self.assertTrue(torch.equal(a,b))
        self.assertEqual(len(hidden),2)
        self.assertTrue(torch.equal(hidden[0][:,:6],hidden[1][:,:6]))
        self.assertFalse(torch.equal(hidden[0][:,6:],hidden[1][:,6:]))
        self.assertTrue(router.last_trace['leading_control_exact'])
        for n,p in self.model.named_parameters():self.assertTrue(torch.equal(p,before[n]))
        router.remove()

    def test_exception_cleanup(self):
        router=ReferenceConditionRouting(self.model,self.weights)
        with patch.object(router,'original_main',side_effect=RuntimeError('forced failure')):
            with self.assertRaisesRegex(RuntimeError,'forced failure'):
                with torch.inference_mode():self.model.generate(**self.kwargs)
        self.assertFalse(router.active);self.assertIsNone(router.main_mask)
        router.remove()
        self.assertFalse(hasattr(self.model,'reference_condition_routing'))


if __name__=='__main__':unittest.main()
