import copy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from training import reference_masked_supervision_broad_features as f


def feature():
    d=dict(id='fit',text='legitimate fitting text',audio_sha256='a')
    v=dict(id='fit',audio_sha256='a',protocol_sha256='new',descriptor_sha256=f.fingerprint(d),
        input_ids=torch.arange(9)[None],ref_ids=torch.arange(6)[None],codes=torch.zeros(2,16,dtype=torch.long),
        speaker=torch.ones(1024,dtype=torch.float32))
    return d,v


def scoped():
    assets=[dict(id=f'f{i}') for i in range(512)]
    p=dict(selected_fitting_pair_ids=[f'f{i}' for i in range(256)],fitting_probe_pair_ids=[f'f{i}' for i in range(16)],
        fitting_feature_ids=sorted(r['id'] for r in assets),calibration_feature_source=dict(ids=['heldout','__custom_reference__']),
        previous_cache_overlap_ids=[f'f{i}' for i in range(52)],previous_cache_overlap_is_readiness=False)
    return assets,p


class BroadFeaturesTests(unittest.TestCase):
    def test_nonpersistent_buffer_content_and_registry_restoration_guard(self):
        model=torch.nn.Linear(2,2);model.register_buffer('scratch',torch.ones(1),persistent=False)
        model.speech_tokenizer=SimpleNamespace(model=torch.nn.Linear(2,2));wrapper=SimpleNamespace(model=model)
        state=f.full_state(wrapper);registry=f.native_tensors(wrapper)
        self.assertIn('native.buffer.scratch',state);self.assertTrue(f.same_registry(wrapper,registry))
        model.scratch.add_(1);self.assertNotEqual(state,f.full_state(wrapper))
        model.scratch=model.scratch.clone();self.assertFalse(f.same_registry(wrapper,registry))

    def test_exact_process_closure_and_priority(self):
        r=dict(status='closed',exit_code=0,original_session_id=14712,protocol_sha256='p');f.closed(r,14712,{'protocol_sha256':'p'})
        for k,v in [('status','running'),('exit_code',1),('exit_code',False),('original_session_id',51327),('protocol_sha256','other')]:
            with self.assertRaises(ValueError):f.closed(dict(r,**{k:v}),14712,{'protocol_sha256':'p'})
        f.require_feature_priority(False)
        for value in (True,None,0):
            with self.assertRaises(ValueError):f.require_feature_priority(value)

    def test_fitting_scope_and_no_calibration_inclusion(self):
        a,p=scoped();f.scope(a,p)
        for action in ('duplicate','calibration','probe','readiness'):
            aa,pp=copy.deepcopy(a),copy.deepcopy(p)
            if action=='duplicate':aa[-1]=aa[0]
            elif action=='calibration':pp['calibration_feature_source']['ids'].append('f400')
            elif action=='probe':pp['fitting_probe_pair_ids'][-1]='f500'
            else:pp['previous_cache_overlap_is_readiness']=True
            with self.assertRaises(ValueError):f.scope(aa,pp)

    def test_valid_payload_preserves_native_contract(self):
        d,v=feature();shapes=f.old.validate_feature(v,d,'new')
        self.assertEqual(shapes['codes']['shape'],[2,16]);self.assertEqual(shapes['speaker']['dtype'],'torch.float32')
        self.assertEqual(set(shapes),set(f.old.TENSORS))

    def test_bad_dtype_codes_vectors_and_identity_rejected(self):
        d,v=feature()
        for name,value in [('speaker',torch.ones(1024,dtype=torch.float16)),('speaker',torch.full((1024,),float('nan'))),
                           ('speaker',torch.zeros(1024)),('codes',torch.zeros(2,15,dtype=torch.long)),
                           ('codes',torch.full((2,16),2048,dtype=torch.long)),('input_ids',torch.ones(1,8,dtype=torch.long)),
                           ('audio_sha256','wrong')]:
            x=copy.deepcopy(v);x[name]=value
            with self.assertRaises(ValueError):f.old.validate_feature(x,d,'new')

    def test_overlap_allows_provenance_difference_only(self):
        _,a=feature();b=copy.deepcopy(a);b['protocol_sha256']='old';b['descriptor_sha256']='old descriptor'
        result=f.compare_overlap(a,b);self.assertEqual(set(result),set(f.old.TENSORS))
        self.assertNotEqual(a['protocol_sha256'],b['protocol_sha256'])

    def test_overlap_catches_each_tensor_and_asset(self):
        _,a=feature()
        for key in f.old.TENSORS:
            b=copy.deepcopy(a);b[key].flatten()[0]+=1
            with self.assertRaises(ValueError):f.compare_overlap(a,b)
        b=copy.deepcopy(a);b['speaker']=b['speaker'].double()
        with self.assertRaises(ValueError):f.compare_overlap(a,b)
        b=copy.deepcopy(a);b['audio_sha256']='other'
        with self.assertRaises(ValueError):f.compare_overlap(a,b)

    def test_extraction_complete_freshness_and_exact_overlap_coverage(self):
        assets,p=scoped();p.update(records=assets,overlap_ids=p['previous_cache_overlap_ids'],old_feature_run='old')
        native={'speaker_encoder':{'tensor':'sha'},'speech_tokenizer':{'tensor':'sha'}}
        receipt=dict(protocol_sha256='p',records=512,asset_receipts={r['id']:'sha' for r in assets},all512_fresh=True,
            cached_features_used_as_outputs=False,all_parameters_buffers_unchanged=True,all_gradients_absent=True,
            overlap_tensor_comparisons={i:{} for i in p['overlap_ids']},overlap_comparisons=52,native_components=native,**f.FLAGS)
        def reader(path):return {'native_components':native} if str(path).startswith('old') else receipt
        with patch.object(f,'read',side_effect=reader):
            f.extraction_evidence(Path('new'),p,'p')
            receipt['cached_features_used_as_outputs']=True
            with self.assertRaises(ValueError):f.extraction_evidence(Path('new'),p,'p')
            receipt['cached_features_used_as_outputs']=False;receipt['overlap_tensor_comparisons'].pop(p['overlap_ids'][0])
            with self.assertRaises(ValueError):f.extraction_evidence(Path('new'),p,'p')


if __name__=='__main__':unittest.main()
