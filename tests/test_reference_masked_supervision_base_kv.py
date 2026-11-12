import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from training import reference_masked_supervision_base_kv as m
from training.reference_masked_supervision import Layout


def cases():
    unique=[dict(case=f'd{i}',group='development',speaker_id=f's{i//2}',text=f't{i}',
        reference='ref',reference_text='reference',half_reference=False) for i in range(32)]
    unique += [dict(case=f'c{i}',group='custom',speaker_id='custom',text=f't{i}',
        reference='customref',reference_text='reference',half_reference=True) for i in range(8)]
    return [dict(c,seed=s) for s in (2026,2027) for c in unique]


def summary():
    return {g:dict(recordings=n,speakers=s,pooled_cer=.02,pooled_wer=.1,speaker_macro_cosine=.985)
            for g,n,s in [('development',64,16),('custom',16,1)]}


class Decoder(nn.Module):
    def __init__(self):
        super().__init__();self.layers=nn.ModuleList()
        for _ in range(2):
            a=nn.Module();a.k_proj=nn.Linear(2,2,bias=False);a.v_proj=nn.Linear(2,2,bias=False)
            nn.init.eye_(a.k_proj.weight);nn.init.eye_(a.v_proj.weight)
            layer=nn.Module();layer.self_attn=a;self.layers.append(layer)

    def forward(self,inputs_embeds,cache_position,position_ids=None,attention_mask=None,past_key_values=None):
        x=inputs_embeds
        for layer in self.layers:x=layer.self_attn.k_proj(x)+layer.self_attn.v_proj(x)
        return x


def model():
    decoder=Decoder().eval();return SimpleNamespace(talker=SimpleNamespace(model=decoder)),decoder


class BaseKVTests(unittest.TestCase):
    def test_reference_cache_allowlist_tamper_and_verify_once(self):
        from training import reference_masked_supervision_features as f
        records=[dict(id=x) for x in ('ref1','ref2','target-do-not-load')]
        e=dict(features='fake',contexts=[dict(feature_id='ref1'),dict(feature_id='ref2')])
        def metadata(path):
            return {'feature_sha256':{'ref1':'ok','ref2':'ok'}} if path.name=='verification.json' else {'asset_receipts':{'ref1':'sha','ref2':'sha'}}
        def asset(run,descriptor,protocol):return {'id':descriptor['id']},{'feature_sha256':'ok'}
        wrapper=SimpleNamespace()
        with patch.object(f,'verify_complete',return_value={'records':records}) as verify, \
                patch.object(f,'read_asset',side_effect=asset) as load,patch.object(m,'read',side_effect=metadata),patch.object(m,'digest',return_value='sha'):
            self.assertEqual(set(m.reference_assets(wrapper,e)),{'ref1','ref2'})
            self.assertIs(m.reference_assets(wrapper,e),wrapper._base_kv_reference_assets)
            verify.assert_called_once();self.assertEqual([c.args[1]['id'] for c in load.call_args_list],['ref1','ref2'])
        with patch.object(f,'verify_complete',return_value={'records':records}), \
                patch.object(f,'read_asset',return_value=({}, {'feature_sha256':'changed'})), \
                patch.object(m,'read',side_effect=metadata),patch.object(m,'digest',return_value='sha'):
            with self.assertRaises(ValueError):m.reference_assets(SimpleNamespace(),e)

    def test_panel_exact40_and_both_seeds(self):
        self.assertEqual(len(m.contexts(cases())),40)
        for change in ('missing','metadata','route'):
            rows=cases()
            if change=='missing':rows.pop()
            elif change=='metadata':rows[-1]['text']='changed'
            else:rows[0]['half_reference']=True
            with self.assertRaises(ValueError):m.contexts(rows)

    def test_strict_two_group_quality(self):
        self.assertTrue(m.eligible(summary(),0))
        self.assertFalse(m.eligible(summary(),1))
        for g in ('development','custom'):
            for k,v in [('pooled_cer',.03),('speaker_macro_cosine',.984)]:
                r=summary();r[g][k]=v;self.assertFalse(m.eligible(r,0))

    def test_invalid_metrics_and_scope(self):
        for k,v in [('pooled_cer',float('nan')),('speaker_macro_cosine',float('inf')),('recordings',63),('speakers',15)]:
            r=summary();r['development'][k]=v
            with self.assertRaises(ValueError):m.eligible(r,0)
        r=summary();del r['custom']
        with self.assertRaises(ValueError):m.eligible(r,0)

    def test_authoritative_closure(self):
        bind={'protocol_sha256':'p','results_sha256':'r','driver_sha256':'d','terminal_status_sha256':'s'}
        valid=dict(status='closed',exit_code=0,original_session_id=35473,**bind);m.closure_ok(valid,bind)
        for k,v in [('status','running'),('exit_code',1),('exit_code',False),('original_session_id',999),('results_sha256','x')]:
            with self.assertRaises(ValueError):m.closure_ok(dict(valid,**{k:v}),bind)

    def test_no_effect_scope_and_identity(self):
        rows=[dict(pair_id=p,half_reference=h,arm=a,audio_sha256=f'{p}-{h}-{a in ("student32","graft32")}',
            cosine=.98,frames=4,terminated=True) for p in ('a','b') for h in (False,True)
            for a in ('teacher0','self_graft0','student32','graft32')]
        m.no_effect(rows)
        altered=copy.deepcopy(rows);altered[-1]['audio_sha256']='different'
        with self.assertRaises(ValueError):m.no_effect(altered)
        with self.assertRaises(ValueError):m.no_effect(rows[:-1])

    def test_exact_control_includes_all_logits(self):
        x=torch.ones(1);a=dict(audio_sha256='a',codes=x,first_sequence=x,inputs=[x],contexts=[x],logits={'main':[x]},bank={'values':{'0.k':x}})
        m.exact_control(a,copy.deepcopy(a));b=copy.deepcopy(a);b['logits']['main'][0]+=1
        with self.assertRaises(ValueError):m.exact_control(a,b)

    def test_graft_only_protected_rows_and_generated_unchanged(self):
        mod,decoder=model();layout=Layout(1,3,4);binding={'text':'legitimate target','ref':'original'}
        x=torch.arange(8,dtype=torch.float32).reshape(1,4,2)
        with torch.inference_mode(),m.Bank(mod,layout,binding) as donor:
            decoder(inputs_embeds=x,cache_position=torch.arange(4))
        d=donor.receipt()
        with torch.inference_mode(),m.Bank(mod,layout,binding,d) as student:
            output=decoder(inputs_embeds=x+10,cache_position=torch.arange(4))
            generated=decoder(inputs_embeds=torch.ones(1,1,2)*7,cache_position=torch.tensor([4]))
        r=student.receipt();self.assertEqual(len(r['values']),4);self.assertEqual(r['protected_positions'],[0,3])
        self.assertTrue(torch.equal(output[:,[0,3]],x[:,[0,3]]*4))
        self.assertTrue(torch.equal(output[:,[1,2]],(x[:,[1,2]]+10)*4))
        self.assertTrue(torch.equal(generated,torch.ones(1,1,2)*28))
        self.assertTrue(all(c['allowed_before_sha256']==c['allowed_after_sha256'] for c in r['checks'].values()))
        self.assertTrue(all(n==2 for n in r['calls'].values()))

    def test_prefix_abort_exception_cleanup(self):
        mod,decoder=model();layout=Layout(1,3,4)
        with self.assertRaises(m.PrefixComplete),torch.inference_mode(),m.Bank(mod,layout,{},prefix_only=True) as bank:
            decoder(inputs_embeds=torch.ones(1,4,2),cache_position=torch.arange(4))
        self.assertEqual(len(bank.receipt()['values']),4)
        self.assertFalse(decoder._forward_hooks);self.assertFalse(decoder._forward_pre_hooks)
        self.assertTrue(all(not x._forward_hooks for x in bank.modules.values()))

    def test_invalid_donor_context_and_nonfinite_tensor(self):
        mod,decoder=model();layout=Layout(1,3,4)
        with torch.inference_mode(),m.Bank(mod,layout,{'case':'a'}) as bank:
            decoder(inputs_embeds=torch.ones(1,4,2),cache_position=torch.arange(4))
        d=bank.receipt()
        with self.assertRaises(ValueError):m.Bank(mod,layout,{'case':'b'},d)
        d['values']['0.k']=d['values']['0.k'].clone();d['values']['0.k'][0,0,0]=float('nan')
        with self.assertRaises(ValueError),torch.inference_mode(),m.Bank(mod,layout,{'case':'a'},d):
            decoder(inputs_embeds=torch.ones(1,4,2),cache_position=torch.arange(4))
        self.assertFalse(decoder._forward_pre_hooks)

    def test_bad_positions_and_grad_mode_rejected(self):
        mod,decoder=model();layout=Layout(1,3,4)
        with self.assertRaises(ValueError),m.Bank(mod,layout,{}):
            decoder(inputs_embeds=torch.ones(1,4,2),cache_position=torch.arange(4))
        with self.assertRaises(ValueError),torch.inference_mode(),m.Bank(mod,layout,{}):
            decoder(inputs_embeds=torch.ones(1,4,2),cache_position=torch.arange(1,5))
        self.assertFalse(decoder._forward_pre_hooks)


if __name__=='__main__':unittest.main()
