"""Synthetic CPU-only diagnostics; no native models or retained experiment files."""
from contextlib import contextmanager
import copy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from training import reference_masked_supervision_gradient_probe as probe
from training import reference_masked_supervision_policy_v2 as policy
from training import reference_masked_supervision_training_v2 as fit
from training.reference_masked_supervision_evaluation import progression


def panel_fixture():
    rows = {str(i): dict(id=str(i), ref_id='ref'+str(i), speaker_id='voice'+str(i), partition='fitting') for i in range(30)}
    schedule = dict(fitting_probe_pair_ids=[str(i) for i in range(16)])
    flags = dict(outlier_pairs=[dict(pair_id=str(i), reference_id='ref'+str(i), speaker_id='voice'+str(i),
        positive_minus_strongest_negative=-.01) for i in range(15, 30)])
    return schedule, rows, flags


class AlgebraCore:
    def __init__(self):
        self.adapters = torch.nn.Linear(2, 3, bias=False)
        self.model = SimpleNamespace(config=SimpleNamespace(talker_config=SimpleNamespace(codec_eos_token_id=2)))
        self.scope = None
    def parameters(self):
        return list(self.adapters.parameters())
    @contextmanager
    def replay(self, ids, codes, reference, half, terminated):
        assert self.scope is None and terminated
        frames = len(codes); self.scope = (frames, half)
        seq = SimpleNamespace(codes=codes, labels=torch.cat([codes[:,0],torch.tensor([2])]),
                              embeddings=torch.ones(frames+1,2), prefix_length=1)
        try:
            yield seq
        finally:
            self.scope = None
    def outputs(self, seq):
        assert self.scope is not None
        frames, half = self.scope
        x = torch.tensor([[1., 2.+half]])
        main = self.adapters(x).expand(frames+1, -1)
        residual = self.adapters(x*.7).expand(frames*15,-1).reshape(frames,15,3)
        return dict(main_logits=main,residual_logits=residual,
            main_nll=F.cross_entropy(main,seq.labels,reduction='sum'),
            residual_nll=F.cross_entropy(residual.reshape(-1,3),seq.codes[:,1:].reshape(-1),reduction='sum'),
            token_count=16*frames+1)


class GradientProbeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1); torch.manual_seed(91)
        self.core = AlgebraCore()
        self.item = dict(id='fit', partition='fitting',input_ids=torch.tensor([[1,2]]),
            codes=torch.zeros(3,16,dtype=torch.long), reference=SimpleNamespace(ref_ids=torch.tensor([[1]]),
                ref_codes=torch.ones(2,16,dtype=torch.long), speaker=torch.tensor([.2,.3]),reference_sha256='a'*64))

    def test_inherited_feature_runtime_and_registered_protocol_cannot_redirect(self):
        parent=dict(features='registered-assets',feature_protocol_sha256='p',feature_audit_sha256='a',
            training_hardware={'gpu':'original'},numeric_config={'dtype':'float32'},runtime={'torch':'pinned'})
        registered=dict(parent,training_protocol_sha256=probe.PROTOCOL_SHA)
        probe.inherited_fields(registered,parent)
        for key in registered:
            changed=dict(registered);changed[key]='redirected'
            with self.assertRaisesRegex(ValueError,'identity changed'):probe.inherited_fields(changed,parent)
        with patch.object(probe,'digest',return_value='a'*64):
            probe.require_protocol_hash('out','a'*64)
            with self.assertRaisesRegex(ValueError,'protocol changed'):probe.require_protocol_hash('out','b'*64)

    def test_fixed_panel_preserves_overlap_and_rejects_evaluation_or_flag_drift(self):
        schedule, rows, flags = panel_fixture()
        values=probe.panel(schedule,rows,flags)
        self.assertEqual(len(values),30)
        self.assertEqual(sum(v['fixed_probe'] for v in values),16)
        self.assertEqual(sum(v['flagged'] for v in values),15)
        self.assertEqual(sum(v['flagged'] and v['fixed_probe'] for v in values),1)
        bad=copy.deepcopy(rows);bad['0']['partition']='calibration'
        with self.assertRaisesRegex(ValueError,'prohibited'):probe.panel(schedule,bad,flags)
        bad=copy.deepcopy(flags);bad['outlier_pairs'][0]['reference_id']='wrong'
        with self.assertRaisesRegex(ValueError,'metadata'):probe.panel(schedule,rows,bad)
        with self.assertRaisesRegex(ValueError,'flagged15'):probe.panel(schedule,rows,dict(outlier_pairs=flags['outlier_pairs'][:-1]))

    def test_terminal_requires_failed32_closed0_and_complete_history(self):
        history=[]
        for step in (0,8,16,32):
            history.append(dict(step=step,status='verified',recordings=80,truncated=0,
                frozen_weights_verified=True,native_conditioning_verified=True,all_artifact_hashes_verified=True,
                all_recomputed_cosines_finite=True,max_cosine_recomputation_error=0.,adapter_movement=dict(maximum=step*.001),
                summary={g:dict(recordings=n,speakers=s,pooled_cer=.05-step*.0001,pooled_wer=.1,speaker_macro_cosine=.983)
                         for g,n,s in [('development',64,16),('custom',16,1)]}))
        status=dict(status='complete',step=32,progression=progression(history),reason=progression(history)['reason'],
                    goal_achieved=False,global_model_selection_changed=False)
        closed=dict(status='closed',step=32,exit_code=0,original_session_id=33720)
        probe.failed_terminal(status,closed,history)
        for changed in [dict(closed,exit_code=1),dict(closed,original_session_id=1),dict(closed,status='running')]:
            with self.assertRaises(ValueError):probe.failed_terminal(status,changed,history)
        with self.assertRaises(ValueError):probe.failed_terminal(dict(status,status='running'),closed,history)
        with self.assertRaises(ValueError):probe.failed_terminal(status,closed,history[:-1])
        changed=copy.deepcopy(history)
        for group in changed[-1]['summary'].values():group.update(pooled_cer=.02,speaker_macro_cosine=.985)
        with self.assertRaisesRegex(ValueError,'Failed final'):probe.failed_terminal(status,closed,changed)

    def test_full_vocabulary_forward_kl_is_mean_over_valid_rows(self):
        student=torch.tensor([[.2,.7,-.1],[1.,-.2,.3]],requires_grad=True)
        teacher=torch.tensor([[.1,.8,-.1],[.7,0.,.3]],requires_grad=True)
        value=probe.forward_kl(student,teacher)
        t=teacher.detach().double().softmax(-1); expected=(t*(t.log()-student.double().log_softmax(-1))).sum()/2
        torch.testing.assert_close(value,expected)
        value.backward();self.assertIsNone(teacher.grad)
        duplicate=student.detach().repeat(4,1).requires_grad_()
        torch.testing.assert_close(probe.forward_kl(duplicate,teacher.detach().repeat(4,1)),value.detach())
        with self.assertRaises(ValueError):probe.forward_kl(student.double(),teacher)
        with self.assertRaises(ValueError):probe.forward_kl(student,teacher.detach()*float('nan'))

    def test_component_normalization_and_combined_gradient_match(self):
        _,main,_=probe.component_gradient(self.core,self.item,'full','main_ce')
        _,residual,_=probe.component_gradient(self.core,self.item,'full','residual_ce')
        _,task,_=probe.component_gradient(self.core,self.item,'full','task')
        torch.testing.assert_close(task,main+.3*residual)
        self.assertIsNone(self.core.scope)
        self.assertTrue(all(p.grad is None for p in self.core.parameters()))
        for partition in ['calibration','evaluation']:
            with self.assertRaises(ValueError):probe.component_gradient(self.core,dict(self.item,partition=partition),'full','main_ce')

    def test_exception_restores_adapter_grads_rng_and_replay_scope(self):
        parameter=self.core.parameters()[0];parameter.grad=torch.ones_like(parameter)
        state=policy.snapshot(self.core);rng=fit.rng_state();gradient=parameter.grad.clone()
        with self.assertRaisesRegex(RuntimeError,'fixture'):
            with probe.preserved(self.core):
                with torch.no_grad():parameter.add_(.2)
                parameter.grad=None;torch.rand(5)
                raise RuntimeError('fixture')
        self.assertTrue(probe.equal(state,policy.snapshot(self.core)))
        self.assertTrue(probe.equal(rng,fit.rng_state()))
        torch.testing.assert_close(parameter.grad,gradient)
        with self.assertRaisesRegex(ValueError,'conditioning'):
            probe.component_gradient(self.core,self.item,'half','main_ce',expected_trace={'wrong':True})
        self.assertIsNone(self.core.scope)

    def test_zero_teacher_control_and_student_restore_precede_graph(self):
        student=policy.snapshot(self.core);teacher={k:v.clone() for k,v in student.items()}
        next(iter(teacher.values()))[0].add_(.1)
        controls=probe.controls(self.core,self.item,teacher,student)
        self.assertEqual([r['mode'] for r in controls],['full','half'])
        self.assertTrue(probe.equal(student,policy.snapshot(self.core)))
        for row in controls:
            for result in row['zero'].values():
                self.assertLess(abs(result['value']),1e-10)
                self.assertLess(result['gradient_norm'],1e-7)
        logits,trace=probe.teacher_outputs(self.core,self.item,'full',teacher,student)
        self.assertTrue(probe.equal(student,policy.snapshot(self.core)))
        value,gradient,_=probe.component_gradient(self.core,self.item,'full','main_kl',logits,trace)
        self.assertTrue(torch.isfinite(gradient).all());self.assertGreater(value,0)
        self.assertGreater(float(gradient.norm()),0)

    def test_geometry_reports_unit_and_weighted_residual_without_nan(self):
        g=dict(main_ce=torch.tensor([1.,0.]),residual_ce=torch.tensor([-2.,0.]),
               main_kl=torch.tensor([0.,0.]),residual_kl=torch.tensor([0.,1.]))
        result=probe.geometry(g)
        self.assertAlmostEqual(result['weighted_residual_to_main_norm'],.6)
        self.assertAlmostEqual(result['norms']['task'],.4)
        self.assertIsNone(result['comparisons']['main_ce__main_kl']['cosine'])
        self.assertTrue(result['comparisons']['main_ce__residual_ce']['opposing'])

    def test_aggregate_equal_pairs_modes_and_overlapping_strata(self):
        schedule,rows,flags=panel_fixture();panel=probe.panel(schedule,rows,flags)
        records=[dict(id=r['row']['id'],mode=m,losses={k:1.+i for k in probe.COMPONENTS},
            gradients={k:torch.tensor([float(int(r['row']['id'])+1),float(i)]) for k in probe.COMPONENTS})
            for r in panel for i,m in enumerate(probe.MODES)]
        result=probe.aggregate(records,panel)
        self.assertEqual(result['all']['both']['pairs'],30)
        self.assertEqual(result['all']['both']['weight'],1/60)
        self.assertEqual(result['fixed_probe']['both']['pairs'],16)
        self.assertEqual(result['flagged']['both']['pairs'],15)
        self.assertAlmostEqual(result['all']['both']['geometry']['norms']['main_ce'],(15.5**2+.5**2)**.5)
        self.assertEqual(result['all']['both']['mean_component_values']['main_ce'],1.5)
        with self.assertRaises(ValueError):probe.aggregate(records[:-1],panel)


if __name__ == '__main__':
    unittest.main()
