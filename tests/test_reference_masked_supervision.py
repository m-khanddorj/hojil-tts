import unittest
from unittest.mock import patch
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from test_training_sequence import AVAILABLE, tiny
from training.speaker_preference_adapter import install as install_speaker
from training.middle_attention_adapter import install as install_old, modules as old_modules
from training.reference_prefix_routing import ReferencePrefixRouting, weight_modules as plain_modules
from training.reference_prefix_lora_routing import ReferencePrefixLoRARouting, weight_modules as lora_modules, base_linear
from training.reference_prefix_generation_pad import ReferencePrefixGenerationPad, tensor_sha256
from training.reference_masked_supervision import MaskedReferenceSupervision, NativeReference


@unittest.skipUnless(AVAILABLE,'Qwen required')
class MaskedSupervisionTests(unittest.TestCase):
    def make(self, old=True):
        model=tiny().eval().requires_grad_(False)
        install_speaker(model,rank=2)
        with torch.no_grad():
            model.speaker_preference_adapter.up.normal_(std=.02)
        if old:
            install_old(model,[0,1],rank=4)
            with torch.no_grad():
                for module in old_modules(model).values():
                    module.lora_B.normal_(std=.02)
        model.requires_grad_(False)
        modules=lora_modules(model,(0,1)) if old else plain_modules(model)
        weights={n+'.weight':base_linear(m).weight.detach().clone()*.8 for n,m in modules.items()}
        return model,weights

    def setUp(self):
        torch.set_num_threads(1)
        self.model,self.weights=self.make()
        self.ids=torch.tensor([[1,2,3,10,11,12,4,5,6,7,8]])
        self.codes=torch.randint(0,64,(3,16))
        self.reference=NativeReference(torch.tensor([[1,2,3,20,21,4,5]]),
            torch.randint(0,64,(2,16)),torch.randn(32),'a'*64,'tiny native FP32 fixture')

    def kwargs(self):
        r=self.reference
        return dict(input_ids=[self.ids],ref_ids=[r.ref_ids],
            voice_clone_prompt=dict(ref_spk_embedding=[r.speaker],ref_code=[r.ref_codes],icl_mode=[True],x_vector_only_mode=[False]),
            languages=['Auto'],non_streaming_mode=True,max_new_tokens=4,do_sample=False,subtalker_dosample=False)

    def core(self,model=None,weights=None):
        return MaskedReferenceSupervision(model or self.model,weights or self.weights,layers=(0,1))

    def native(self,model,core,half):
        with core.native(half),torch.inference_mode():
            return model.generate(**self.kwargs())

    def equal_codes(self,a,b):
        self.assertEqual(len(a[0]),len(b[0]))
        for left,right in zip(a[0],b[0]):
            self.assertTrue(torch.equal(left,right))

    def test_zero_native_parity_plain_and_nonzero_old_outer_both_modes(self):
        for old in (False,True):
            model,weights=self.make(old)
            originals={n:tensor_sha256(p) for n,p in model.named_parameters()}
            controls={}
            for half in (False,True):
                router=(ReferencePrefixLoRARouting(model,weights,expected_layers=(0,1)) if old
                        else ReferencePrefixRouting(model,weights))
                pad=ReferencePrefixGenerationPad(router,'a'*64)
                with pad.use_reference('a'*64 if half else 'b'*64),torch.inference_mode():
                    controls[half]=model.generate(**self.kwargs())
                pad.remove();router.remove()
            core=self.core(model,weights)
            for half in (False,True):
                self.equal_codes(controls[half],self.native(model,core,half))
            self.assertEqual(originals,{n:tensor_sha256(p) for n,p in model.named_parameters() if n in originals})
            self.assertEqual(len(core.parameters()),8)
            core.remove()

    def test_teacher_prefix_native_exact_speaker_once_full_continuation_pad(self):
        core=self.core()
        for half in (False,True):
            captured={}
            original=core.original_talker_generate
            def capture(**kwargs):
                captured.update({k:v.clone() if isinstance(v,torch.Tensor) else v for k,v in kwargs.items()})
                return original(**kwargs)
            with patch.object(core,'original_talker_generate',side_effect=capture):
                self.native(self.model,core,half)
            with core.replay(self.ids,self.codes,self.reference,half,terminated=True) as seq:
                self.assertTrue(torch.equal(seq.embeddings[:seq.prefix_length],captured['inputs_embeds'][0]))
                full_pad=core._full_special()[2]
                self.assertTrue(torch.equal(full_pad,captured['tts_pad_embed']))
                self.assertTrue(torch.equal(full_pad,captured['trailing_text_hidden']))
                # The first native cached frame sums all16 codec embeddings.
                talker=self.model.talker
                parts=[talker.get_input_embeddings()(self.codes[:1,:1])]
                parts += [talker.code_predictor.get_input_embeddings()[i-1](self.codes[:1,i:i+1]) for i in range(1,16)]
                expected=torch.cat(parts,dim=1).sum(1,keepdim=True)+full_pad
                self.assertTrue(torch.equal(seq.embeddings[seq.prefix_length],expected[0,0]))
                self.assertEqual(seq.frame_positions.tolist(),[seq.prefix_length-1+i for i in range(3)])
                self.assertEqual(seq.labels[seq.prefix_length-1:].tolist(),self.codes[:,0].tolist()+[1000])
        core.remove()

    def test_cached_and_full_teacher_all16_codec_logits_and_eos_match(self):
        core=self.core()
        with torch.no_grad():
            for adapter in core.adapters.values():
                adapter.B.normal_(std=.02)
        for half in (False,True):
            with torch.no_grad():
                cached=core.cached_logits(self.ids,self.codes,self.reference,half,terminated=True)
                with core.replay(self.ids,self.codes,self.reference,half,terminated=True) as seq:
                    full=core.outputs(seq)
                torch.testing.assert_close(cached['main_logits'],full['main_logits'],atol=2e-6,rtol=2e-5)
                torch.testing.assert_close(cached['residual_logits'],full['residual_logits'],atol=2e-6,rtol=2e-5)
                torch.testing.assert_close(cached['log_probability'],full['log_probability'],atol=2e-5,rtol=0)
                self.assertEqual(full['main_logits'].shape,(4,1100))
                self.assertEqual(full['residual_logits'].shape,(3,15,64))
                self.assertEqual(full['token_count'],49)
        core.remove()

    def test_native_forced_codec_frames_independently_match_teacher_logits_and_suffix(self):
        core=self.core()
        with torch.no_grad():
            for adapter in core.adapters.values():adapter.B.normal_(std=.01)
        class Captured(Exception):pass
        for half in (False,True):
            native={}
            def forced(**kwargs):
                talker=self.model.talker
                predictor=talker.code_predictor
                P=kwargs['inputs_embeds'].shape[1]
                saved_main=core.original_main
                inputs=[]
                def capture_main(**arguments):
                    inputs.append(arguments['inputs_embeds'].clone())
                    return saved_main(**arguments)
                with patch.object(core,'original_main',side_effect=capture_main):
                    state=talker(inputs_embeds=kwargs['inputs_embeds'],attention_mask=kwargs['attention_mask'],
                        trailing_text_hidden=kwargs['trailing_text_hidden'],tts_pad_embed=kwargs['tts_pad_embed'],
                        cache_position=torch.arange(P),use_cache=True,return_dict=True)
                    main=[state.logits[:,-1]]
                    residual=[]
                    for frame in range(len(self.codes)):
                        first=torch.cat([state.past_hidden,talker.get_input_embeddings()(self.codes[frame:frame+1,:1])],dim=1)
                        sub=predictor(inputs_embeds=first,use_cache=True,return_dict=True)
                        row=[sub.logits[:,-1]]
                        for group in range(1,15):
                            sub=predictor(input_ids=self.codes[frame:frame+1,group:group+1],past_key_values=sub.past_key_values,
                                generation_steps=sub.generation_steps,use_cache=True,return_dict=True)
                            row.append(sub.logits[:,-1])
                        residual.append(torch.stack(row,dim=1))
                        with patch.object(predictor,'generate',return_value=SimpleNamespace(sequences=self.codes[frame:frame+1,1:])):
                            state=talker(input_ids=self.codes[frame:frame+1,:1],past_hidden=state.past_hidden,
                                past_key_values=state.past_key_values,generation_step=state.generation_step,
                                attention_mask=torch.ones(1,P+frame+1,dtype=torch.long),cache_position=torch.tensor([P+frame]),
                                trailing_text_hidden=kwargs['trailing_text_hidden'],tts_pad_embed=kwargs['tts_pad_embed'],
                                use_cache=True,return_dict=True)
                        main.append(state.logits[:,-1])
                native.update(main=torch.cat(main,dim=0),residual=torch.cat(residual,dim=0),inputs=torch.cat(inputs,dim=1)[0])
                raise Captured()
            with patch.object(core,'original_talker_generate',side_effect=forced):
                with self.assertRaises(Captured):self.native(self.model,core,half)
            with torch.no_grad(),core.replay(self.ids,self.codes,self.reference,half,terminated=True) as seq:
                full=core.outputs(seq)
                self.assertTrue(torch.equal(seq.embeddings,native['inputs']))
                torch.testing.assert_close(full['main_logits'],native['main'],atol=2e-6,rtol=2e-5)
                torch.testing.assert_close(full['residual_logits'],native['residual'],atol=2e-6,rtol=2e-5)
        core.remove()

    def test_mask_excludes_direct_new_lora_but_preserves_protected_input_gradients(self):
        core=self.core()
        with torch.no_grad():
            for adapter in core.adapters.values():adapter.B.normal_(std=.02)
        name=next(iter(core.adapter_keys))
        module=core.modules[name]
        for half in (False,True):
            with core.replay(self.ids,self.codes,self.reference,half,terminated=True) as seq:
                x=torch.randn(1,len(seq.embeddings),32,requires_grad=True)
                active,protected=core.scope['layout'].masks(core.positions)
                expected=core.original_linears[name](x)
                if half:
                    expected=F.linear(x,core._half(name),base_linear(module).bias)
                out=module(x)
                self.assertTrue(torch.equal(out[:,protected],expected[:,protected]))
                self.assertFalse(torch.equal(out[:,active],core.original_linears[name](x)[:,active]))
                gradients=torch.autograd.grad(out[:,protected].square().sum(),[x]+core.parameters(),allow_unused=True,retain_graph=True)
                self.assertGreater(gradients[0].abs().sum().item(),0)
                self.assertTrue(all(g is None or not torch.count_nonzero(g) for g in gradients[1:]))
                active_grads=torch.autograd.grad(out[:,active].square().sum(),core.parameters(),allow_unused=True)
                self.assertGreater(sum(g.abs().sum().item() for g in active_grads if g is not None),0)
        core.remove()

    def test_protected_reference_hidden_retains_indirect_target_gradients_and_old_frozen(self):
        core=self.core()
        old={n:tensor_sha256(p) for n,p in self.model.named_parameters() if not p.requires_grad}
        halves={n:tensor_sha256(v) for n,v in core.bank.named_buffers()}
        for half in (False,True):
            self.model.zero_grad(set_to_none=True)
            with core.replay(self.ids,self.codes,self.reference,half,terminated=True) as seq:
                output=core.outputs(seq)
                output['hidden'][seq.prefix_length-1].square().sum().backward()
            self.assertGreater(sum(p.grad.abs().sum().item() for p in core.parameters() if p.grad is not None),0)
            self.assertTrue(all(p.grad is None for p in self.model.parameters() if not p.requires_grad))
        self.assertEqual(old,{n:tensor_sha256(p) for n,p in self.model.named_parameters() if not p.requires_grad})
        self.assertEqual(halves,{n:tensor_sha256(v) for n,v in core.bank.named_buffers()})
        core.remove()

    def test_nonreentrant_checkpointed_loss_and_gradients_match(self):
        core=self.core()
        with torch.no_grad():
            for adapter in core.adapters.values():adapter.B.normal_(std=.01)
        self.model.train()
        for half in (False,True):
            expected=None
            for checkpointing in (False,True):
                for decoder in (self.model.talker.model,self.model.talker.code_predictor.model):
                    if checkpointing:decoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
                    else:decoder.gradient_checkpointing_disable()
                self.model.zero_grad(set_to_none=True)
                with core.replay(self.ids,self.codes,self.reference,half,terminated=True) as seq:
                    output=core.outputs(seq)
                    loss=-output['log_probability']/output['token_count']
                    loss.backward()
                gradients=[p.grad.clone() for p in core.parameters()]
                if expected is None:expected=(loss.detach(),gradients)
                else:
                    torch.testing.assert_close(loss,expected[0],atol=1e-7,rtol=0)
                    for actual,wanted in zip(gradients,expected[1]):
                        torch.testing.assert_close(actual,wanted,atol=1e-7,rtol=3e-5)
        self.model.eval()
        core.remove()

    def test_validation_scope_exceptions_and_outside_backward_fail_closed(self):
        core=self.core()
        bad=NativeReference(self.reference.ref_ids,self.reference.ref_codes,self.reference.speaker.half(),'a'*64,'cached')
        with self.assertRaisesRegex(ValueError,'FP32 speaker'):
            with core.replay(self.ids,self.codes,bad,terminated=True):pass
        with self.assertRaisesRegex(ValueError,'terminated'):
            with core.replay(self.ids,self.codes,self.reference):pass
        with self.assertRaisesRegex(ValueError,'Explicit native/replay'):
            self.model.generate(**self.kwargs())
        with self.assertRaisesRegex(RuntimeError,'forced'):
            with core.replay(self.ids,self.codes,self.reference,True,terminated=True):
                with self.assertRaisesRegex(ValueError,'nonnested'):
                    with core.native():pass
                raise RuntimeError('forced')
        self.assertIsNone(core.scope)
        self.assertIsNone(core.positions)
        with core.replay(self.ids,self.codes,self.reference,True,terminated=True) as seq:
            with self.assertRaisesRegex(ValueError,'without cache'):
                self.model.talker.model(inputs_embeds=seq.embeddings[None],use_cache=False,past_key_values=object())
            loss=-core.outputs(seq)['log_probability']
        with self.assertRaisesRegex(ValueError,'Backward must finish'):
            loss.backward()
        self.assertTrue(all(p.grad is None for p in core.parameters()))
        self.native(self.model,core,False)
        core.remove()


if __name__=='__main__':unittest.main()
