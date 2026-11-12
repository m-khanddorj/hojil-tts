import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from training.reference_prefix_generation_pad_evaluation import (
    terminal_decision,validate_records,parity,paired_prefill,validate_trace,prepare)
from training.reference_profile_preference_evaluation import progression


class GenerationPadEvaluationTests(unittest.TestCase):
    def checkpoint(self,step,cer=.04,cosine=.98):
        return dict(status='verified',step=step,recordings=50,inactive_development_exact_parity=True,
            screen_summary=dict(custom=dict(pooled_cer=cer,speaker_macro_cosine=cosine),
                development=dict(pooled_cer=.02,speaker_macro_cosine=.986)),
            screen_truncated=0,goal_screen=cer<.03 and cosine>.984)

    def records(self):
        cases=[dict(case=f'c{i}',seed=seed,speaker_id='custom' if i<3 else str(i),group='custom' if i<3 else 'development',
            text=f'text {i}',reference='reference',reference_sha256=('a' if i<3 else 'b')*64,reference_text='ref',row_id=i)
            for seed in (2026,2027) for i in range(13)]
        rows=[dict(c,cosine=.98,terminated=True,audio_sha256=str(i),score=dict(cer=.01,wer=.1,
            character_errors=1,characters=100,word_errors=1,words=10)) for i,c in enumerate(cases)]
        return cases,rows

    def test_terminal_failed_complete_dpo_only(self):
        history=[self.checkpoint(0),self.checkpoint(4,.05),self.checkpoint(8,.06)]
        decision=progression(history);status=dict(status='complete',reason=decision['reason'],step=8)
        terminal_decision(status,history,decision)
        with self.assertRaises(ValueError):terminal_decision(dict(status,status='running'),history,decision)
        pending=[self.checkpoint(0),self.checkpoint(4,.04,.981)]
        with self.assertRaises(ValueError):terminal_decision(dict(status='complete',step=4,reason='maximum_updates_reached'),pending,progression(pending))
        passing=[self.checkpoint(0),self.checkpoint(4,.02,.985)]
        with self.assertRaises(ValueError):terminal_decision(dict(status='complete',step=4,reason='needs_broader_confirmation'),passing,progression(passing))
        maximum=[self.checkpoint(s,.04,.98+i*.001) for i,s in enumerate((0,4,8,16))]
        terminal_decision(dict(status='complete',step=16,reason='maximum_updates_reached'),maximum,progression(maximum))

    def test_complete_scope_metadata_and_finite_scores(self):
        cases,rows=self.records();validate_records(cases,rows)
        for bad in (rows[:-1],rows[:-1]+[rows[0]]):
            with self.assertRaises(ValueError):validate_records(cases,bad)
        for field,value in (('row_id',999),('cosine',float('nan')),('reference_sha256','c'*64)):
            bad=copy.deepcopy(rows);bad[0][field]=value
            with self.assertRaises(ValueError):validate_records(cases,bad)

    def test_exact_control_and_inactive_fallback_parity(self):
        _,rows=self.records();self.assertEqual(parity(rows,copy.deepcopy(rows),'control'),26)
        changed=copy.deepcopy(rows);changed[0]['audio_sha256']='changed'
        with self.assertRaises(ValueError):parity(rows,changed,'control')
        self.assertEqual(parity(rows,changed,'full-pad'),20)
        changed[3]['score']['character_errors']=2
        with self.assertRaises(ValueError):parity(rows,changed,'full-pad')
        changed=copy.deepcopy(rows);changed[3]['cosine']=float('nan')
        with self.assertRaises(ValueError):parity(rows,changed,'full-pad')

    def test_prefill_and_all_first_frame_codebooks_match(self):
        row=dict(pad_trace=dict(input_tensor_sha256=dict(inputs_embeds='x',attention_mask='m',tts_pad_embed='p',trailing_text_hidden='p')))
        trajectory=dict(codes=torch.arange(32).reshape(2,16));paired_prefill(row,trajectory,copy.deepcopy(row),copy.deepcopy(trajectory))
        changed=copy.deepcopy(row);changed['pad_trace']['input_tensor_sha256']['attention_mask']='different'
        with self.assertRaises(ValueError):paired_prefill(changed,trajectory,row,trajectory)
        changed=copy.deepcopy(trajectory);changed['codes'][0,15]+=1
        with self.assertRaises(ValueError):paired_prefill(row,changed,row,trajectory)
        changed=copy.deepcopy(trajectory);changed['codes'][1,15]+=1
        paired_prefill(row,changed,row,trajectory)  # Later decoded frames may change.

    def test_trace_gate_layout_and_pad_boundary(self):
        p=dict(reference_sha256='a'*64)
        for active in (True,False):
            for stage in ('control','full-pad'):
                replace=active and stage=='full-pad';sha=('a' if active else 'b')*64
                routing=dict(target_start=10,target_stop=13,prefix_length=19,prefills=1,decode_steps=2,protected=active,main_matrices=196,reference_positions=16)
                trace=dict(status='complete',reference_sha256=sha,registered_reference_sha256='a'*64,custom_reference=active,
                    protected=active,full_generation_pad=stage=='full-pad',talker_calls=1,pad_replaced=replace,
                    non_pad_kwargs_identity_verified=True,input_tensors_unchanged=True,
                    mode='custom_full_generation_pad' if replace else 'custom_original_half_pad' if active else 'inactive_full_policy',
                    layout=dict(target_start=10,target_stop=13,prefix_length=19),half_pad_sha256='half',full_pad_sha256='full',
                    input_tensor_sha256={k:'half' if active else 'full' for k in ('trailing_text_hidden','tts_pad_embed')},
                    output_pad_sha256={k:'full' if replace or not active else 'half' for k in ('trailing_text_hidden','tts_pad_embed')})
                row=dict(reference_sha256=sha,pad_trace=trace,routing=routing)
                trace['input_tensor_sha256'].update(inputs_embeds='prefill',attention_mask='mask')
                trajectory=dict(pad_trace=copy.deepcopy(trace),routing=copy.deepcopy(routing),ref_ids=torch.zeros(1,7),
                    input_ids=torch.zeros(1,11),ref_codes=torch.zeros(4,16),codes=torch.zeros(2,16))
                validate_trace(row,trajectory,p,stage)
                bad=copy.deepcopy(row);bad['pad_trace']['pad_replaced']=not replace
                with self.assertRaises(ValueError):validate_trace(bad,trajectory,p,stage)

    def test_early_prepare_never_creates_output(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)/'new'
            with patch('training.reference_prefix_generation_pad_evaluation.dpo_evidence',side_effect=ValueError('not terminal')):
                with self.assertRaises(ValueError):prepare(Path('old'),Path('dpo'),out,Path('driver'))
            self.assertFalse(out.exists())


if __name__=='__main__':unittest.main()
