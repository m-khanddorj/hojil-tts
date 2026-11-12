"""Shared-control and strict qualification tests; no speech/model jobs."""
import copy
import unittest
import torch
from training import reference_masked_supervision_broad_speech as speech
from training import reference_masked_supervision_broad_training as fit


class BroadSpeechTests(unittest.TestCase):
    def fixture(self):
        p={k:'same' for k in ('policy','frozen_identity','settings','runtime','numeric_config','config','generation',
            'decoding','reference_sha256','custom_reference','training_hardware')}
        e={k:'same' for k in ('cases','generation','decoding','features_audit_sha256','asr_manifest_sha256')}
        state=dict(step=0,optimizer=None,adapter_movement=dict(maximum=0.),frozen_identity='same',
            adapter={'x.A':torch.ones(2),'x.B':torch.zeros(2)},rng=fit.rng_state())
        return p,e,state

    def test_exact_shared_baseline_is_accepted(self):
        p,e,s=self.fixture();speech.baseline_equivalence(p,e,copy.deepcopy(p),copy.deepcopy(e),s,copy.deepcopy(s))

    def test_changed_cases_route_backend_or_reference_rejected(self):
        p,e,s=self.fixture()
        for name in e:
            changed=copy.deepcopy(e);changed[name]='changed'
            with self.assertRaises(ValueError):speech.baseline_equivalence(p,changed,p,e,s,s)
        for name in p:
            changed=copy.deepcopy(p);changed[name]='changed'
            with self.assertRaises(ValueError):speech.baseline_equivalence(changed,e,p,e,s,s)

    def test_zero_tensor_rng_and_movement_tampering_rejected(self):
        p,e,s=self.fixture()
        for name in ('tensor','rng','movement','optimizer'):
            t=copy.deepcopy(s)
            if name=='tensor':t['adapter']['x.A'][0]+=1
            elif name=='rng':t['rng']['torch'][0]^=1
            elif name=='movement':t['adapter_movement']['maximum']=float('nan')
            else:t['optimizer']={}
            with self.assertRaises(ValueError):speech.baseline_equivalence(p,e,p,e,t,s)

    def receipt(self,step,cer=.02,cos=.985):
        return dict(status='verified',step=step,recordings=80,truncated=0,frozen_weights_verified=True,
            native_conditioning_verified=True,all_artifact_hashes_verified=True,all_recomputed_cosines_finite=True,
            max_cosine_recomputation_error=0.,adapter_movement=dict(maximum=0. if step==0 else .001),
            summary={g:dict(recordings=n,speakers=s,pooled_cer=cer,pooled_wer=.05,speaker_macro_cosine=cos)
                for g,n,s in [('development',64,16),('custom',16,1)]})

    def test_zero_shared_control_never_qualifies_as_candidate(self):
        result=speech.progression([self.receipt(0)])
        self.assertFalse(result['stop']);self.assertFalse(result['goal_achieved'])
        result=speech.progression([self.receipt(0),self.receipt(8)])
        self.assertEqual(result['reason'],'calibration_qualified_needs_original114');self.assertFalse(result['goal_achieved'])

    def test_strict_both_group_gate_and_complete_finite_audit(self):
        for metric,value in [('pooled_cer',.03),('speaker_macro_cosine',.984)]:
            a=self.receipt(8);a['summary']['development'][metric]=value
            self.assertNotEqual(speech.progression([self.receipt(0),a])['reason'],'calibration_qualified_needs_original114')
        for value in (float('nan'),float('inf')):
            a=self.receipt(8);a['summary']['custom']['pooled_cer']=value
            with self.assertRaises(ValueError):speech.progression([self.receipt(0),a])
        a=self.receipt(8);a['recordings']=79
        with self.assertRaises(ValueError):speech.progression([self.receipt(0),a])


if __name__=='__main__':unittest.main()
