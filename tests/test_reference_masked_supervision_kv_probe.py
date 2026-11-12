"""Tiny CPU tensors and pure metadata only; no pretrained models/audio."""
from dataclasses import asdict
from types import SimpleNamespace
from contextlib import contextmanager
import copy
import unittest

import torch
from torch import nn
from training.reference_masked_supervision import Layout
from training import reference_masked_supervision_kv_probe as m


def fixture():
    layout=Layout(2,4,7)
    layers=[SimpleNamespace(self_attn=SimpleNamespace(k_proj=nn.Linear(3,3,bias=False),v_proj=nn.Linear(3,3,bias=False))) for _ in range(2)]
    for layer in layers:
        for module in (layer.self_attn.k_proj,layer.self_attn.v_proj):
            with torch.no_grad():module.weight.copy_(torch.eye(3))
    core=SimpleNamespace(model=SimpleNamespace(talker=SimpleNamespace(model=SimpleNamespace(layers=layers))),
        scope=dict(kind='native',layout=layout,half_reference=False),positions=torch.arange(7))
    binding=dict(half_reference=False,layout=asdict(layout),positions=list(range(7)),teacher_checkpoint_sha256='0',text='context')
    return core,layout,binding


def outputs(core,x):
    return {f'{i}.{kind}':getattr(layer.self_attn,kind+'_proj')(x)
        for i,layer in enumerate(core.model.talker.model.layers) for kind in ('k','v')}


def donor(core,layout,binding):
    x=torch.arange(21,dtype=torch.float32).reshape(1,7,3)
    with m.ProjectionBank(core,layout,binding) as bank,torch.inference_mode():outputs(core,x)
    return bank.bank(0),x


class KVProbeTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def test_predetermined_probes_use_text_and_reference_shapes_only(self):
        fixed=[str(i) for i in range(16)];fitting={};shapes={}
        for i in fixed:
            fitting[i]=dict(id=i,ref_id='r'+i,partition='fitting',text='target',ref_audio='ref.wav',
                ref_audio_sha256='a'*64,ref_text='reference',speaker_id='speaker')
            shapes[i]={'input_ids':{'shape':[1,12]}}  # deliberately no target codes/audio
            shapes['r'+i]={'ref_ids':{'shape':[1,10]},'codes':{'shape':[2 if i!='5' else 20,16]}}
        result=m.selected_probes(fixed,fitting,shapes)
        self.assertEqual([r['pair_id'] for r in result],['0','5']);self.assertEqual(result[1]['layout']['prefix_length'],39)
        shapes['r6']['codes']['shape'][0]=20
        self.assertEqual(m.selected_probes(fixed,fitting,shapes)[1]['pair_id'],'5')
        fitting['5']['partition']='calibration'
        with self.assertRaises(ValueError):m.selected_probes(fixed,fitting,shapes)

    def test_capture_is_cloned_protected_only_and_never_adds_future_rows(self):
        core,layout,binding=fixture();x=torch.randn(1,7,3)
        with m.ProjectionBank(core,layout,binding) as hook,torch.inference_mode():
            pre=outputs(core,x);saved=hook.values['0.k'].clone();x.fill_(999)
            self.assertTrue(torch.equal(saved,hook.values['0.k']))
            core.positions=torch.tensor([7]);outputs(core,torch.ones(1,1,3))
        bank=hook.bank(1);self.assertEqual(len(bank['values']),4);self.assertEqual(bank['protected_positions'],[0,1,4,5,6])
        self.assertEqual(bank['values']['0.k'].shape,(1,5,3));self.assertEqual(bank['generated_rows_added_to_donor'],0)

    def test_graft_changes_only_protected_rows_and_decode_returns_original(self):
        core,layout,binding=fixture();bank,x=donor(core,layout,binding)
        y=x+100
        with m.ProjectionBank(core,layout,binding,bank) as hook,torch.inference_mode():
            actual=outputs(core,y);allowed,protected=layout.masks(core.positions)
            self.assertTrue(torch.equal(actual['0.k'][:,allowed],y[:,allowed]))
            self.assertTrue(torch.equal(actual['0.k'][:,protected],x[:,protected]))
            core.positions=torch.tensor([7]);value=torch.randn(1,1,3)
            self.assertIs(hook.hook('0.k')(None,(),value),value)

    def test_hook_is_after_routed_delta_before_single_normalization(self):
        core,layout,binding=fixture();bank,x=donor(core,layout,binding)
        module=core.model.talker.model.layers[0].self_attn.k_proj
        original=module.forward
        def routed(value):return original(value)+50  # represents completed outer route/LoRA forward
        module.forward=routed;norm_calls=[]
        def norm(value):norm_calls.append(value.clone());return value*2
        with m.ProjectionBank(core,layout,binding,bank),torch.inference_mode():actual=norm(module(x))
        allowed,protected=layout.masks(core.positions)
        self.assertEqual(len(norm_calls),1);self.assertTrue(torch.equal(actual[:,protected],x[:,protected]*2))
        self.assertTrue(torch.equal(actual[:,allowed],(x[:,allowed]+50)*2))

    def test_wrong_context_duplicate_prefill_and_nested_hooks_rejected(self):
        core,layout,binding=fixture();bank,x=donor(core,layout,binding)
        wrong=copy.deepcopy(bank);wrong['binding']['text']='other'
        with self.assertRaises(ValueError):m.ProjectionBank(core,layout,binding,wrong)
        wrong=copy.deepcopy(bank);wrong['protected_positions']=[0]
        with self.assertRaises(ValueError):m.ProjectionBank(core,layout,binding,wrong)
        with m.ProjectionBank(core,layout,binding) as hook,torch.inference_mode():
            outputs(core,x)
            with self.assertRaises(ValueError):outputs(core,x)
            with self.assertRaises(ValueError):hook.__enter__()
        core.positions=torch.arange(1,8)
        with m.ProjectionBank(core,layout,binding),torch.inference_mode(),self.assertRaises(ValueError):outputs(core,x)

    def test_exceptions_remove_hooks_and_restore_adapters_rng_and_gradients(self):
        core,layout,binding=fixture()
        with self.assertRaises(RuntimeError):
            with m.ProjectionBank(core,layout,binding):raise RuntimeError('synthetic failure')
        self.assertTrue(all(not getattr(l.self_attn,k+'_proj')._forward_hooks for l in core.model.talker.model.layers for k in ('k','v')))
        parameter=nn.Parameter(torch.tensor([1.]));parameter.grad=torch.tensor([2.]);core.parameters=lambda:[parameter];core.scope=None
        def restore(owner,state):
            with torch.no_grad():parameter.copy_(state['p'])
        policy=SimpleNamespace(snapshot=lambda owner:{'p':parameter.detach().clone()},restore=restore,
            frozen_identity=lambda model,owner:'fixed')
        trainer=SimpleNamespace(rng_state=lambda:torch.get_rng_state().clone(),restore_rng=torch.set_rng_state)
        rng=torch.get_rng_state().clone()
        with self.assertRaises(RuntimeError):
            with m.preserve_state(core,policy,trainer):
                with torch.no_grad():parameter.fill_(9)
                parameter.grad=None;torch.rand(3);raise RuntimeError('failure')
        self.assertEqual(parameter.item(),1);self.assertEqual(parameter.grad.item(),2);self.assertTrue(torch.equal(rng,torch.get_rng_state()))

    def test_dummy_prefix_cannot_depend_on_target_codec_constant(self):
        layout=Layout(2,4,7);seen=[]
        class Core:
            model=SimpleNamespace(config=SimpleNamespace(talker_config=SimpleNamespace(hidden_size=3)))
            @contextmanager
            def _context(self,*args):yield
            def _build(self,ids,codes,reference):
                seen.append(int(codes[0,0]));return SimpleNamespace(embeddings=torch.cat([torch.ones(7,3),codes[:,:3].float()]))
        reference=SimpleNamespace(validate=lambda *args:None)
        result=m.dummy_prefix(Core(),torch.tensor([[1]]),reference,False,layout)
        self.assertEqual(seen,[0,1]);self.assertEqual(result.shape,(7,3))
        class Bad(Core):
            def _build(self,ids,codes,reference):return SimpleNamespace(embeddings=torch.ones(8,3)*int(codes[0,0]))
        with self.assertRaises(ValueError):m.dummy_prefix(Bad(),torch.tensor([[1]]),reference,False,layout)

    def test_self_graft_requires_exact_wav_codes_logits_inputs_and_kv(self):
        value=dict(audio_sha256='wave',codes=torch.ones(2,16,dtype=torch.long),first_sequence=torch.ones(1,3,dtype=torch.long),
            inputs=[torch.ones(1,7,3)],contexts=[{'position_ids':torch.arange(7)}],logits={'main':[torch.ones(1,1,5)]},
            bank={'values':{'0.k':torch.ones(1,5,3)}})
        self.assertTrue(m.identity_control(value,copy.deepcopy(value))['wav_exact'])
        for key in ('audio_sha256','codes','logits','contexts','bank'):
            other=copy.deepcopy(value)
            if key=='audio_sha256':other[key]='wrong'
            elif key=='codes':other[key][0,0]+=1
            elif key=='logits':other[key]['main'][0][0,0,0]+=1e-6
            elif key=='contexts':other[key][0]['position_ids'][0]+=1
            else:other[key]['values']['0.k'][0,0,0]+=1
            with self.subTest(key=key),self.assertRaises(ValueError):m.identity_control(value,other)

    def test_drift_is_per_projection_finite_same_context(self):
        a=dict(binding={'context':'same'},values={'0.k':torch.ones(1,2,3),'0.v':torch.ones(1,2,3)})
        b=copy.deepcopy(a);b['values']['0.v']+=1
        result=m.drift(a,b);self.assertEqual(result['0.k']['l2'],0);self.assertEqual(result['0.v']['relative_l2'],1)
        b['binding']['context']='different'
        with self.assertRaises(ValueError):m.drift(a,b)
        b=copy.deepcopy(a);b['values']['0.k'][0,0,0]=float('nan')
        with self.assertRaises(ValueError):m.drift(a,b)
        from training.evaluate import ASR_REV
        from scripts.score_latest_audio import score
        original=[dict(label=f'{arm}-{half}-{i}',arm=arm,half_reference=half,pair_id=str(i),case=str(i),
            text='a b',group='fitting',speaker_id=str(i),seed=2026,cosine=.985)
            for arm in m.SETTINGS['arms'] for half in (False,True) for i in range(2)]
        scored=[dict(r,hypothesis=r['text'],score=score(r['text'],r['text'],True),asr_revision=ASR_REV,decoding={}) for r in original]
        self.assertEqual(m.scalar_summary(original,scored,{})['graft32']['half']['pooled_cer'],0)
        with self.assertRaises(ValueError):m.scalar_summary(original,scored[:-1],{})
        changed=copy.deepcopy(scored);changed[0]['score']['character_errors']+=1
        with self.assertRaises(ValueError):m.scalar_summary(original,changed,{})
        changed=copy.deepcopy(scored);changed[0]['cosine']=.999
        with self.assertRaises(ValueError):m.scalar_summary(original,changed,{})

    def test_capture_requires_no_grad_finite_fp32_and_complete_coverage(self):
        core,layout,binding=fixture();x=torch.ones(1,7,3)
        with m.ProjectionBank(core,layout,binding) as hook:
            with self.assertRaises(ValueError):outputs(core,x)
        with self.assertRaises(ValueError):hook.bank(0)
        with m.ProjectionBank(core,layout,binding),torch.inference_mode(),self.assertRaises(ValueError):
            x[0,0,0]=float('nan');outputs(core,x)


if __name__=='__main__':unittest.main()
