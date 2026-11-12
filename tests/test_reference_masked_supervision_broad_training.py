"""Tiny/pure data-only registration tests; no native model loading."""
import copy
from contextlib import contextmanager
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
from training import reference_masked_supervision_broad_training as fit


def schedule():
    rows={str(i):dict(id=str(i),speaker_id='s'+str(i),partition='fitting') for i in range(256)}
    plan=dict(settings=copy.deepcopy(fit.SETTINGS),updates=[dict(update=u,pair_ids=[str(i) for i in range((u-1)*8,u*8)],
        modes=['full','half'] if u%2 else ['half','full'],pair_weight=.125,mode_weight=.5) for u in range(1,33)],
        fitting_probe_pair_ids=[str(i*16) for i in range(16)])
    return plan,rows


class Algebra:
    def __init__(self):self.x=torch.nn.Parameter(torch.tensor(.7));self.scope=None
    @contextmanager
    def replay(self,ids,codes,reference,half_reference,terminated):
        self.scope=(int(ids[0,0]),len(codes),half_reference)
        try:yield SimpleNamespace(codes=codes)
        finally:self.scope=None
    def outputs(self,seq):
        n,T,h=self.scope
        return dict(main_nll=self.x.square()*(T+1)*(n+h),residual_nll=self.x.pow(3)*15*T*(n+2*h),
            token_count=16*T+1,residual_logits=torch.zeros(T,15,2))


class BroadTrainingTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def test_exact_schedule_and_exposure(self):
        p,r=schedule();fit.validate_schedule(p,r)
        self.assertEqual([fit.exposure(p,s)['seen_pairs'] for s in fit.CHECKPOINTS],[0,4,8,16])
        self.assertEqual(fit.exposure(p,8)['mode_visits']['0'],dict(full=1,half=1))
        self.assertEqual(fit.exposure(p,8)['mode_visits']['128'],dict(full=0,half=0))

    def test_duplicates_partition_modes_and_probe_rejected(self):
        for mutation in ('duplicate','partition','modes','probe'):
            p,r=schedule()
            if mutation=='duplicate':p['updates'][1]['pair_ids'][0]='0'
            elif mutation=='partition':r['0']['partition']='calibration'
            elif mutation=='modes':p['updates'][1]['modes']=['full','half']
            else:p['fitting_probe_pair_ids'][-1]='not-scheduled'
            with self.assertRaises(ValueError):fit.validate_schedule(p,r)

    def test_unchanged_objective_and_no_calibration_gradients(self):
        core=Algebra();items=[dict(id=str(n),partition='fitting',input_ids=torch.tensor([[n]]),
            codes=torch.zeros(T,16,dtype=torch.long),reference=None) for n,T in ((2,2),(5,7))]
        fit.backward_batch(core,items,['half','full'])
        x=torch.tensor(.7,requires_grad=True);loss=sum(x.square()*(n+h)+.3*x.pow(3)*(n+2*h) for n in (2,5) for h in (0,1))/4
        loss.backward();torch.testing.assert_close(core.x.grad,x.grad)
        core.x.grad=None
        with self.assertRaises(ValueError):fit.backward_batch(core,[dict(items[0],partition='calibration')],['full','half'])
        self.assertIsNone(core.x.grad)
        self.assertIs(fit.make_optimizer,fit.old.make_optimizer)

    def test_memory_receipt_requires_fitting_grad_and_calibration_no_grad(self):
        frontier=[dict(pair_id='fit',partition='fitting',target_frames=2),dict(pair_id='cal',partition='calibration',target_frames=3)]
        rows=[dict(pair_id=r['pair_id'],mode=m,partition=r['partition'],frames=r['target_frames'],
            main_tokens=r['target_frames']+1,residual_tokens=15*r['target_frames'],objective=1.,main_nll=2.,residual_nll=3.,
            backward_performed=r['partition']=='fitting',gradient_norm=1. if r['partition']=='fitting' else None,
            seconds=1.,peak_allocated_bytes=20,peak_reserved_bytes=30) for r in frontier for m in ('full','half')]
        value=dict(device='cuda',hardware=dict(total_memory=100),measurements=rows);fit.validate_preflight(value,frontier)
        for changed in (dict(rows[2],backward_performed=True,gradient_norm=1.),dict(rows[2],objective=float('nan'))):
            bad=copy.deepcopy(value);bad['measurements'][2]=changed
            with self.assertRaises(ValueError):fit.validate_preflight(bad,frontier)
        bad=copy.deepcopy(value);bad['measurements'].pop()
        with self.assertRaises(ValueError):fit.validate_preflight(bad,frontier)

    def test_frontiers_keep_calibration_separate(self):
        fitting={'a':dict(id='a',partition='fitting')};cal=[dict(id='c',partition='calibration')]
        with patch.object(fit,'memory_frontier',side_effect=lambda rows,assets:[r['partition'] for r in rows]) as call:
            self.assertEqual(fit.preflight_frontiers(fitting,cal,{}),['fitting','calibration'])
            self.assertEqual(call.call_count,2)

    def test_exact_rng_comparison_and_cpu_native_limit(self):
        r=fit.rng_state();self.assertTrue(fit.same_rng(r,copy.deepcopy(r)))
        changed=copy.deepcopy(r);changed['torch'][0]^=1;self.assertFalse(fit.same_rng(r,changed))
        changed=copy.deepcopy(r);changed['cuda']=[torch.tensor([1],dtype=torch.uint8)]
        self.assertFalse(fit.same_rng(r,changed));self.assertTrue(fit.same_rng(r,changed,include_cuda=False))

    def test_exact_zero_restore_rejects_positive_or_optimizer_state(self):
        state=dict(step=0,optimizer=None,adapter_movement=dict(maximum=0.),adapter={'a.A':torch.ones(2),'a.B':torch.zeros(2)},rng=fit.rng_state())
        with patch.object(fit.policy,'restore') as restore,patch.object(fit.policy,'snapshot',return_value=state['adapter']):
            fit.restore_zero(object(),state);restore.assert_called_once()
            with self.assertRaises(ValueError):fit.restore_zero(object(),dict(state,step=32))
            with self.assertRaises(ValueError):fit.restore_zero(object(),dict(state,optimizer={}))

    def test_feature_partition_collision_rejected(self):
        with patch.object(fit,'read',return_value=dict(fitting_feature_ids=list(map(str,range(512))))),\
             patch.object(fit.fresh,'load_features',return_value=({},dict.fromkeys(map(str,range(512))))),\
             patch.object(fit,'calibration_assets',return_value={'0':{}}):
            with self.assertRaisesRegex(ValueError,'boundary'):fit.load_assets(dict(plan='x',features='y'))


if __name__=='__main__':unittest.main()
