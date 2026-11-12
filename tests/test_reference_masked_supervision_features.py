import json
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from training import reference_masked_supervision_features as features


def pair(identity, reference):
    def sha(value):
        import hashlib
        return hashlib.sha256(value.encode()).hexdigest()
    return dict(id=identity,ref_id=reference,text='text '+identity,ref_text='text '+reference,
        audio=identity+'.wav',ref_audio=reference+'.wav',audio_sha256=sha(identity),
        ref_audio_sha256=sha(reference),speaker_id='speaker',sample_rate=24000,source='cv27')


def descriptor():
    return dict(id='a',audio_sha256='a'*64,text='sample')


def payload(d):
    return dict(id=d['id'],audio_sha256=d['audio_sha256'],protocol_sha256='p'*64,
        descriptor_sha256=features.fingerprint(d),input_ids=torch.arange(10)[None],
        ref_ids=torch.arange(7)[None],codes=torch.arange(48).reshape(3,16),
        speaker=torch.ones(1024,dtype=torch.float32))


class FeatureTests(unittest.TestCase):
    def test_descriptors_preserve_one_asset_per_utterance_and_reject_conflicts(self):
        a,b=pair('a','b'),pair('b','a')
        plan=dict(selected_fitting_pair_ids=['a','b'],real_feature_ids=['a','b'])
        custom=dict(reference='custom.wav',reference_sha256=features.CUSTOM_SHA,
                    reference_text='original reference transcript',speaker_id='user')
        records=features.descriptors(plan,[a,b],[],custom)
        self.assertEqual(len(records),3)
        self.assertEqual({d['id'] for d in records},{'a','b',features.CUSTOM_ID})
        self.assertEqual(records,features.descriptors(plan,[b,a],[],custom))
        for changed in [dict(b,ref_text='changed'),dict(b,ref_audio_sha256='f'*64),dict(b,speaker_id='other')]:
            with self.assertRaises(ValueError):
                features.descriptors(plan,[a,changed],[],custom)
        with self.assertRaises(ValueError):
            features.descriptors(dict(plan,real_feature_ids=['a']),[a,b],[],custom)

    def test_tensor_contract_rejects_half_vectors_nonfinite_and_invalid_codes(self):
        d=descriptor();value=payload(d)
        result=features.validate_feature(value,d,'p'*64)
        self.assertEqual(result['codes']['shape'],[3,16])
        for key,tensor in [('speaker',torch.ones(1024,dtype=torch.float16)),
                           ('speaker',torch.full((1024,),float('nan'))),
                           ('speaker',torch.zeros(1024)),
                           ('codes',torch.full((3,16),2048)),
                           ('codes',torch.zeros((0,16),dtype=torch.long)),
                           ('codes',torch.zeros((3,15),dtype=torch.long)),
                           ('ref_ids',torch.arange(7).float()[None])]:
            with self.subTest(key=key,shape=tensor.shape),self.assertRaises(ValueError):
                features.validate_feature(dict(value,**{key:tensor}),d,'p'*64)
        with self.assertRaises(ValueError):
            features.validate_feature(value,dict(d,audio_sha256='x'*64),'p'*64)

    def test_committed_receipt_binds_tensor_file_and_all_payload_identities(self):
        d=descriptor();value=payload(d)
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp)/'assets'/features.asset_key(d['id']);directory.mkdir(parents=True)
            torch.save(value,directory/'features.pt')
            receipt=dict(id=d['id'],protocol_sha256='p'*64,descriptor_sha256=features.fingerprint(d),
                feature_sha256=features.digest(directory/'features.pt'),
                tensors=features.validate_feature(value,d,'p'*64))
            features.write_new(directory/'receipt.json',receipt)
            loaded,_=features.read_asset(tmp,d,'p'*64)
            self.assertTrue(torch.equal(loaded['codes'],value['codes']))
            with self.assertRaises(ValueError):
                features.read_asset(tmp,d,'q'*64)
            with (directory/'features.pt').open('ab') as stream:stream.write(b'changed')
            with self.assertRaises(ValueError):
                features.read_asset(tmp,d,'p'*64)
            with self.assertRaises(ValueError):
                features.write_new(directory/'receipt.json',receipt)

    def test_non_talker_byte_proof_excludes_acoustic_policy_and_detects_encoder_change(self):
        def create(path,speaker,talker):
            header=json.dumps({'speaker_encoder.weight':dict(dtype='F32',shape=[1],data_offsets=[0,4]),
                'talker.weight':dict(dtype='F32',shape=[1],data_offsets=[4,8])}).encode()
            path.write_bytes(struct.pack('<Q',len(header))+header+struct.pack('<ff',speaker,talker))
        with tempfile.TemporaryDirectory() as tmp:
            a,b=Path(tmp)/'a.safetensors',Path(tmp)/'b.safetensors'
            create(a,1,2);create(b,1,9)
            self.assertEqual(features.safetensor_non_talker(a),features.safetensor_non_talker(b))
            create(b,2,9)
            self.assertNotEqual(features.safetensor_non_talker(a),features.safetensor_non_talker(b))

    def test_native_role_frames_are_separate_and_component_identity_ignores_talker(self):
        seen=[]
        class Native:
            def _build_assistant_text(self,text):return 'target:'+text
            def _build_ref_text(self,text):return 'reference:'+text
        def processor(**kwargs):
            self.assertTrue(kwargs['padding'])
            seen.append(kwargs['text']);return dict(input_ids=torch.arange(len(kwargs['text'])))
        with patch.dict('sys.modules',{'qwen_tts':SimpleNamespace(Qwen3TTSModel=Native)}):
            ids=features.native_ids(processor,'sample')
        self.assertEqual(seen,['target:sample','reference:sample'])
        self.assertNotEqual(ids['input_ids'].shape,ids['ref_ids'].shape)
        calls=[]
        class Processor:
            @staticmethod
            def from_pretrained(path,**kwargs):
                calls.append((path,kwargs));return processor
        with patch.dict('sys.modules',{'qwen_tts.core.models':SimpleNamespace(Qwen3TTSProcessor=Processor)}):
            self.assertIs(features.load_native_processor('native-base'),processor)
        self.assertEqual(calls,[('native-base',dict(local_files_only=True,fix_mistral_regex=True))])
        wrapper=SimpleNamespace(model=SimpleNamespace(speaker_encoder=torch.nn.Linear(2,2),
            speech_tokenizer=SimpleNamespace(model=torch.nn.Linear(2,2)),talker=torch.nn.Linear(2,2)))
        before=features.native_component_identity(wrapper)
        with torch.no_grad():wrapper.model.talker.weight.add_(1)
        self.assertEqual(before,features.native_component_identity(wrapper))
        with torch.no_grad():wrapper.model.speaker_encoder.weight.add_(1)
        self.assertNotEqual(before,features.native_component_identity(wrapper))


if __name__ == '__main__':
    unittest.main()
