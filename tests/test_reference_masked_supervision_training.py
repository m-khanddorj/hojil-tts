from contextlib import contextmanager
import copy
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from test_training_sequence import tiny
from training.speaker_preference_adapter import install as install_speaker
from training.reference_prefix_routing import weight_modules
from training.reference_masked_supervision import MaskedReferenceSupervision,NativeReference
from training import reference_masked_supervision_training as fit
from training import reference_masked_supervision_policy as policy

ROOT=Path('C:/Users/Khan/.codex/visualizations/2026/11/09/01a1bf77-0b17-7391-9781-f859f1fbb0b1')


class AlgebraCore:
    def __init__(self):self.theta=torch.nn.Parameter(torch.tensor(.7));self.scope=None
    @contextmanager
    def replay(self,ids,codes,reference,half_reference,terminated):
        assert self.scope is None and terminated
        self.scope=(int(ids[0,0]),len(codes),half_reference)
        try:yield SimpleNamespace(codes=codes)
        finally:self.scope=None
    def outputs(self,seq):
        factor,T,half=self.scope
        return dict(main_nll=self.theta.square()*(T+1)*(factor+half),
            residual_nll=self.theta.pow(3)*(15*T)*(factor+2*half),token_count=16*T+1,
            residual_logits=torch.zeros(T,15,2))


class TrainingTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def tiny_core(self):
        model=tiny().eval().requires_grad_(False);install_speaker(model,rank=2);model.requires_grad_(False)
        model.speaker_preference_adapter.register_buffer('fixture_coordinates',torch.ones(2))
        halves={n+'.weight':m.weight.detach().clone()*.8 for n,m in weight_modules(model).items()}
        core=MaskedReferenceSupervision(model,halves,layers=(0,1))
        reference=NativeReference(torch.tensor([[1,2,3,20,21,4,5]]),torch.randint(0,64,(2,16)),
            torch.randn(32),'a'*64,'tiny original native fixture')
        item=dict(id='train',partition='fitting',speaker_id='speaker',input_ids=torch.tensor([[1,2,3,10,11,12,4,5,6,7,8]]),
            codes=torch.randint(0,64,(3,16)),reference=reference)
        return model,core,item

    def update(self,core,optimizer,item):
        optimizer.zero_grad(set_to_none=True)
        fit.backward_batch(core,[item],['full','half'])
        torch.nn.utils.clip_grad_norm_(core.parameters(),1.,error_if_nonfinite=True)
        optimizer.step()

    def fixture(self):
        assets={};rows=[]
        for name,T,R,partition in [('a',2,20,'fitting'),('b',10,2,'calibration'),('c',1,2,'fitting'),('d',2,20,'fitting')]:
            ref=name+'-ref';sha=(name*64)[:64]
            rows.append(dict(id=name,ref_id=ref,audio_sha256=sha,ref_audio_sha256='f'*64,speaker_id=name,partition=partition))
            assets[name]=dict(id=name,audio_sha256=sha,protocol_sha256='p',input_ids=torch.zeros(1,11,dtype=torch.long),
                ref_ids=torch.zeros(1,8,dtype=torch.long),codes=torch.zeros(T,16,dtype=torch.long),speaker=torch.ones(32))
            assets[ref]=dict(id=ref,audio_sha256='f'*64,protocol_sha256='p',input_ids=torch.zeros(1,10,dtype=torch.long),
                ref_ids=torch.zeros(1,7,dtype=torch.long),codes=torch.zeros(R,16,dtype=torch.long),speaker=torch.ones(32))
        return rows,assets

    def audit(self,step,passing=False):
        return dict(step=step,status='verified',recordings=80,
            frozen_weights_verified=True,native_conditioning_verified=True,all_artifact_hashes_verified=True,
            all_recomputed_cosines_finite=True,max_cosine_recomputation_error=0.,truncated=0,
            adapter_movement=dict(maximum=0. if step==0 else .001),
            summary={g:dict(recordings=n,speakers=s,pooled_cer=.02 if passing else .04,pooled_wer=.08,speaker_macro_cosine=.985)
                     for g,n,s in [('development',64,16),('custom',16,1)]})

    def test_equal_pair_mode_normalization_with_unequal_lengths(self):
        core=AlgebraCore()
        pairs=[dict(id=str(factor),partition='fitting',input_ids=torch.tensor([[factor]]),codes=torch.zeros(T,16,dtype=torch.long),reference=None)
               for factor,T in ((2,2),(5,7))]
        report=fit.backward_batch(core,pairs,['half','full'])
        theta=torch.tensor(.7,requires_grad=True)
        expected=sum(theta.square()*(factor+half)+.3*theta.pow(3)*(factor+2*half)
                     for factor in (2,5) for half in (0,1))/4
        expected.backward()
        torch.testing.assert_close(core.theta.grad,theta.grad,atol=1e-6,rtol=0)
        self.assertAlmostEqual(report['summary']['objective'],float(expected.detach()),places=6)
        self.assertEqual([r['mode'] for r in report['examples']],['half','full','half','full'])
        self.assertEqual(report['summary']['modes']['full']['main_tokens'],11)
        self.assertEqual(report['summary']['modes']['full']['residual_tokens'],135)

    def test_calibration_and_duplicate_inputs_rejected_before_any_gradient(self):
        core=AlgebraCore()
        item=dict(id='x',partition='fitting',input_ids=torch.tensor([[2]]),codes=torch.zeros(2,16,dtype=torch.long),reference=None)
        with self.assertRaisesRegex(ValueError,'Calibration'):
            fit.backward_batch(core,[item,dict(item,id='y',partition='calibration')],['full','half'])
        self.assertIsNone(core.theta.grad)
        with self.assertRaisesRegex(ValueError,'Invalid pair'):
            fit.backward_batch(core,[item,item],['full','half'])
        rows,assets=self.fixture();assets['a-ref']['audio_sha256']='x'*64
        with self.assertRaisesRegex(ValueError,'provenance'):fit.pair(rows[0],assets)

    def test_checkpoint_restore_matches_uninterrupted_optimizer_and_rejects_tamper(self):
        model,core,item=self.tiny_core();optimizer=fit.make_optimizer(core)
        frozen=policy.frozen_identity(model,core)
        model.speaker_preference_adapter.fixture_coordinates.add_(1)
        self.assertNotEqual(frozen,policy.frozen_identity(model,core))
        model.speaker_preference_adapter.fixture_coordinates.sub_(1)
        with tempfile.TemporaryDirectory() as directory:
            run=Path(directory);fit.write_new(run/'protocol.json',dict(test=True))
            fit.save_checkpoint(run,0,core,None,frozen)
            for _ in range(8):self.update(core,optimizer,item)
            optimizer.zero_grad(set_to_none=True)
            fit.save_checkpoint(run,8,core,optimizer,frozen)
            state=fit.read_checkpoint(run,8)
            self.update(core,optimizer,item);expected=policy.snapshot(core);expected_optimizer=copy.deepcopy(optimizer.state_dict())
            model2,core2,_=self.tiny_core();policy.restore(core2,state['adapter']);resumed=fit.make_optimizer(core2)
            resumed.load_state_dict(state['optimizer']);fit.optimizer_check(resumed,8);fit.restore_rng(state['rng'])
            self.update(core2,resumed,item)
            for name,value in expected.items():self.assertTrue(torch.equal(value,policy.snapshot(core2)[name]))
            for key,values in expected_optimizer['state'].items():
                for name,value in values.items():self.assertTrue(torch.equal(value,resumed.state_dict()['state'][key][name]))
            self.assertEqual(frozen,policy.frozen_identity(model2,core2))
            self.assertGreater(policy.movement(core2)['maximum'],0)
            with self.assertRaisesRegex(ValueError,'Checkpoint exists'):fit.save_checkpoint(run,8,core2,resumed,frozen)
            first=next(iter(resumed.state.values()));first['exp_avg']=first['exp_avg'].half()
            with self.assertRaisesRegex(ValueError,'FP32 optimizer'):fit.optimizer_check(resumed,9)
            with fit.checkpoint_path(run,8).open('ab') as stream:stream.write(b'corrupt')
            with self.assertRaisesRegex(ValueError,'hash/protocol'):fit.read_checkpoint(run,8)
            core2.remove()
        core.remove()

    def test_rng_state_restores_python_numpy_and_torch(self):
        random.seed(17);np.random.seed(23);torch.manual_seed(31)
        state=fit.rng_state();expected=(random.random(),np.random.random(),torch.rand(4))
        fit.restore_rng(state)
        self.assertEqual(expected[0],random.random());self.assertEqual(expected[1],np.random.random())
        self.assertTrue(torch.equal(expected[2],torch.rand(4)))
        cuda_state=dict(state,cuda=[torch.zeros(8,dtype=torch.uint8)])
        with patch.object(torch.cuda,'is_available',side_effect=AssertionError('CPU inference must not query CUDA RNG')):
            fit.restore_rng(cuda_state,include_cuda=False)

    def test_memory_frontier_covers_decoder_and_residual_extremes(self):
        rows,assets=self.fixture();frontier=fit.memory_frontier(rows,assets)
        self.assertEqual({r['pair_id'] for r in frontier},{'a','b'})
        indexed={r['pair_id']:r for r in frontier}
        self.assertGreater(indexed['a']['total_positions'],indexed['b']['total_positions'])
        self.assertLess(indexed['a']['target_frames'],indexed['b']['target_frames'])

    def test_memory_preflight_backwards_both_modes_without_optimizer_or_state_change(self):
        from training import reference_masked_supervision_features as feature_api
        model,core,_=self.tiny_core();wrapper=SimpleNamespace(model=model,device=torch.device('cpu'))
        rows,assets=self.fixture();before=policy.snapshot(core);frozen=policy.frozen_identity(model,core)
        with self.assertRaisesRegex(ValueError,'Teacher diagnostics require'):
            fit.save_teacher(ROOT,0,wrapper,core,dict(training_device='cpu'),assets)
        with tempfile.TemporaryDirectory() as directory:
            directory=Path(directory);plan=directory/'plan.json';features=directory/'features';features.mkdir()
            fit.write_new(plan,{});fit.write_new(features/'protocol.json',{});fit.write_new(features/'verification.json',{})
            output=directory/'memory.json';fitting={r['id']:r for r in rows if r['partition']=='fitting'}
            calibration=[r for r in rows if r['partition']=='calibration']
            with patch.object(fit,'plan_data',return_value=({},fitting,calibration)),\
                    patch.object(feature_api,'load_features',return_value=({'plan':str(plan)},assets)),\
                    patch.object(feature_api,'assert_native_components'),\
                    patch.object(policy,'baseline_evidence',return_value={'fixture':True}),\
                    patch.object(policy,'load_base',return_value=(wrapper,core)),\
                    patch.object(fit,'make_optimizer',side_effect=AssertionError('No optimizer permitted')):
                fit.preflight(plan,features,'centered16',directory,output,device='cpu')
            receipt=fit.read(output)
            self.assertEqual(receipt['optimizer_updates'],0);self.assertEqual(len(receipt['measurements']),4)
            self.assertEqual({r['mode'] for r in receipt['measurements']},{'full','half'})
            self.assertEqual(frozen,policy.frozen_identity(model,core))
            for name,value in before.items():self.assertTrue(torch.equal(value,policy.snapshot(core)[name]))
            with self.assertRaisesRegex(ValueError,'CUDA memory preflight'):
                fit.validate_memory(receipt,receipt['frontier'])
            accepted=copy.deepcopy(receipt);accepted.update(device='cuda',hardware=dict(name='fixture',total_memory=1<<30,compute_capability=[8,0]))
            for value in accepted['measurements']:value.update(peak_allocated_bytes=4096,peak_reserved_bytes=8192)
            fit.validate_memory(accepted,accepted['frontier'])
            accepted['measurements'][-1]=accepted['measurements'][0]
            with self.assertRaisesRegex(ValueError,'frontier/mode'):fit.validate_memory(accepted,accepted['frontier'])
        core.remove()

    def test_policy_native_forced_controls_independently_match_teacher(self):
        model,core,item=self.tiny_core();before=policy.frozen_identity(model,core)
        with torch.no_grad():
            for delta in core.adapters.values():delta.B.normal_(std=.01)
        generation=dict(max_new_tokens=4,do_sample=False,subtalker_dosample=False)
        for half in (False,True):
            receipt=policy.verify_replay_pair(model,core,item['input_ids'],item['codes'],item['reference'],generation,half)
            self.assertTrue(receipt['prefix_and_consumed_frames_exact']);self.assertEqual(receipt['token_count'],49)
        self.assertEqual(before,policy.frozen_identity(model,core));core.remove()

    def test_prior_calibration_gate_uses_verified_receipts_and_stops_on_qualified_step(self):
        from training import reference_masked_supervision_speech as speech
        with tempfile.TemporaryDirectory() as directory:
            run=Path(directory);fit.write_new(run/'protocol.json',{})
            fit.checkpoint_path(run,0).write_bytes(b'checkpoint-zero')
            controls=[]
            for group in ('development','custom'):
                path=run/(group+'.wav');path.write_bytes(group.encode())
                controls.append(dict(case=group,seed=2026,group=group,wav=path.name,audio_sha256=fit.digest(path)))
            p=dict(feature_audit_sha256='f',frozen_identity={'frozen':True},native_probe_pair_id='probe',policy=dict(controls=controls))
            proof=dict(status='verified',protocol_sha256=fit.digest(run/'protocol.json'),
                checkpoint_sha256=fit.digest(fit.checkpoint_path(run,0)),features_audit_sha256='f',frozen_identity=p['frozen_identity'],
                all_new_parameters_restored=True,optimizer_updates=0,probe_pair_id='probe',historical_waveform_controls=controls,
                forced_trajectory=[dict(half_reference=a,nonzero_adapter=b,frames=1,token_count=17,prefix_and_consumed_frames_exact=True,
                    atol=2e-5,rtol=2e-5,adapter_movement=dict(maximum=.001 if b else 0),maximum_logit_error=dict(main_logits=0.,residual_logits=0.))
                    for a in (False,True) for b in (False,True)])
            fit.write_new(run/'native-verification.json',proof)
            audits={0:self.audit(0),8:self.audit(8,passing=True)}
            with patch.object(speech,'verify',return_value=(p,{})),\
                    patch.object(speech,'verified_receipt',side_effect=lambda run,p,e,step:audits[step]) as checked,\
                    patch.object(fit,'read_checkpoint',side_effect=lambda run,step:dict(adapter_movement=audits[step]['adapter_movement'])):
                self.assertEqual(len(fit.permitted(run,p,0)),1)
                with self.assertRaisesRegex(ValueError,'calibration_qualified'):fit.permitted(run,p,8)
                self.assertGreaterEqual(checked.call_count,3)
            proof['forced_trajectory'][0]['prefix_and_consumed_frames_exact']=False
            (run/'native-verification.json').write_text(json.dumps(proof))
            with self.assertRaisesRegex(ValueError,'replay proof incomplete'):fit.permitted(run,p,0)

    def test_pinned_real_metadata_plan_scope_and_tamper_fail_closed(self):
        schedule,selected,calibration=fit.plan_data(ROOT/'reference-masked-supervision-plan-v1.json')
        self.assertEqual((len(selected),len(calibration),len(schedule['real_feature_ids'])),(204,32,323))
        self.assertEqual(sum(len(r['pair_ids']) for r in schedule['updates']),256)
        self.assertFalse({r['speaker_id'] for r in selected.values()}&{r['speaker_id'] for r in calibration})
        for row in schedule['updates']:
            self.assertEqual(len({selected[i]['speaker_id'] for i in row['pair_ids']}),8)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'tampered.json';path.write_text('{}')
            with self.assertRaisesRegex(ValueError,'Pinned pre-quality'):fit.plan_data(path)


if __name__=='__main__':unittest.main()
