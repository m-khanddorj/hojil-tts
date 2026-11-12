"""Verified frozen-policy loading and native controls for masked supervision.

This module selects no backbone and starts no job on import. All model work
is explicit, after external registration and feature verification.
"""
from pathlib import Path
import hashlib
import json
from types import SimpleNamespace
from unittest.mock import patch

import torch

from .reference_masked_supervision import MaskedReferenceSupervision, base_linear


def require(value, message):
    if not value:raise ValueError(message)


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def tensor_digest(items):
    h=hashlib.sha256()
    for name,value in sorted(items):
        value=value.detach().cpu().contiguous()
        h.update(name.encode());h.update(str((tuple(value.shape),value.dtype)).encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()


def frozen_identity(model,core):
    old=[(n,p) for n,p in model.named_parameters() if not n.startswith('reference_masked_adapters.')]
    require(all(not p.requires_grad and p.grad is None for _,p in old),'Frozen policy received gradients')
    core.parameters()
    return dict(original_parameters_sha256=tensor_digest(old),
        half_weights_sha256=tensor_digest((n,core._half(n)) for n in core.modules),
        speaker_state_sha256=tensor_digest(model.speaker_preference_adapter.state_dict().items())
            if hasattr(model,'speaker_preference_adapter') else None)


def snapshot(core):
    core.parameters()
    return {n:p.detach().cpu().clone() for n,p in core.adapters.state_dict().items()}


def restore(core,state):
    current=core.adapters.state_dict()
    require(set(state)==set(current),'NEW adapter coverage changed')
    for name,value in state.items():
        require(value.shape==current[name].shape and value.dtype==torch.float32
                and torch.isfinite(value).all(),'Invalid NEW adapter state')
    core.adapters.load_state_dict(state,strict=True)


def movement(core):
    rows={}
    with torch.no_grad():
        for name,key in core.adapter_keys.items():
            delta=core.adapters[key]
            # Canonical CPU FP32 measurement is reproducible after GPU training
            # and CPU speech reload; no merged matrix is installed anywhere.
            A,B=delta.A.detach().cpu(),delta.B.detach().cpu()
            base=base_linear(core.modules[name]).weight.detach().cpu()
            value=float((B@A).norm()/base.norm().clamp_min(1e-12))
            require(__import__('math').isfinite(value),'Nonfinite NEW adapter movement')
            rows[name]=value
    return dict(per_matrix=rows,maximum=max(rows.values()),mean=sum(rows.values())/len(rows))


def baseline_evidence(kind,run):
    """Bind an explicit, fully audited failed114 candidate without choosing one."""
    run=Path(run)
    if kind=='centered16':
        from . import reference_prefix_generation_pad_confirmation as previous
        p,_,_,_=previous.verify(run)
        result_path=run/'speech/results.json';audit_path=run/'verification.json'
        terminal_path=run.parent/'reference-prefix-generation-pad-confirmation-status.json'
    elif kind=='attention4':
        from . import reference_prefix_lora_pad_evaluation as previous
        p,_,_=previous.verify(run)
        previous.completed(run,p,'candidate')
        result_path=run/'candidate/results.json';audit_path=run/'candidate/verification.json'
        terminal_path=run.parent/'reference-prefix-lora-pad-status.json'
    else:raise ValueError('Explicit centered16 or attention4 policy required')
    report=read(result_path);audit=read(audit_path)
    require(audit['status']=='verified' and audit['recordings']==114 and audit['confirmation_pass'] is False
        and audit['all_weights_unchanged'] and audit['all_recomputed_cosines_finite']
        and audit['native_conditioning_verified'] and audit['results_sha256']==digest(result_path)
        and audit['protocol_sha256']==digest(run/'protocol.json')
        and report['summary']==audit['summary'],'Complete audited failed114 policy required')
    rows=report['records']
    terminal=read(terminal_path)
    require(terminal['status']=='complete' and terminal['optimizer_updates']==0 and terminal['goal_achieved'] is False,
            'Selected policy driver must have terminated before preparation')
    expected={(r['case'],r['seed']):r for r in p['cases']}
    require(len(rows)==114 and len(expected)==114 and {(r['case'],r['seed']) for r in rows}==set(expected)
        and sum(r['group']=='development' for r in rows)==88 and sum(r['group']=='custom' for r in rows)==26,
        'Incomplete baseline scope')
    require(audit['pad_traces_verified']==114 and audit['truncated']==report['truncated']==0
        and audit['optimizer_updates']==0 and audit['goal_achieved'] is False and audit['global_model_selection_changed'] is False
        and __import__('math').isfinite(audit['maximum_cosine_recomputation_error'])
        and 0<=audit['maximum_cosine_recomputation_error']<=2e-6,'Failed-policy native audit coverage differs')
    for row in rows:
        require(all(row.get(k)==v for k,v in expected[row['case'],row['seed']].items()) and row['terminated']
            and digest(result_path.parent/row['wav'])==row['audio_sha256']
            and digest(result_path.parent/row['trajectory'])==row['trajectory_sha256']
            and digest(row['reference'])==row['reference_sha256'],'Historical policy artifact/condition changed')
    if kind=='centered16':
        proof_path=run.parent/'prefix-pad-goal-audit.json'
        require(digest(proof_path)=='4b743778ea3bb71afb277867721f0db480a60904580349ca1fb29bf759dc377d',
                'Independent failed114 proof changed')
        proof=read(proof_path);bindings=proof['evidence']
        require(proof['status']=='verified' and proof['goal_criteria_met'] is False and proof['recordings']==114
            and bindings['protocol_sha256']==digest(run/'protocol.json') and bindings['results_sha256']==digest(result_path)
            and bindings['native_audit_sha256']==digest(audit_path) and bindings['terminal_status_sha256']==digest(terminal_path)
            and audit['screening_recordings_reproduced_exactly']==audit['screening_scores_reproduced']==26,
            'Independent failed114 evidence binding differs')
    controls=[min((r for r in rows if r['group']==group),key=lambda r:(r['case'],r['seed']))
              for group in ('development','custom')]
    return dict(kind=kind,run=str(run.resolve()),protocol_sha256=digest(run/'protocol.json'),
        results=str(result_path.resolve()),results_sha256=digest(result_path),audit=str(audit_path.resolve()),
        audit_sha256=digest(audit_path),terminal_status=str(terminal_path.resolve()),terminal_status_sha256=digest(terminal_path),
        model_identity=audit['model_identity'],controls=controls,
        config=p['config'],generation=p['generation'],decoding=p['decoding'],reference_sha256=p['reference_sha256'])


def load_base(descriptor,settings):
    require(descriptor==baseline_evidence(descriptor['kind'],Path(descriptor['run'])),'Selected baseline evidence changed')
    if descriptor['kind']=='centered16':
        from . import reference_prefix_generation_pad_confirmation as previous
        _,_,_,wrapper,controller,identity=previous.load(Path(descriptor['run']))
    else:
        from . import reference_prefix_lora_pad_evaluation as previous
        p,old,_=previous.verify(Path(descriptor['run']))
        wrapper,controller,identity=previous.load(p,old,'candidate')
    require(identity==descriptor['model_identity'],'Loaded frozen policy differs')
    router=controller.router
    halves={name+'.weight':value.detach().clone() for name,value in router.weights.items()}
    controller.remove();router.remove()
    # No W+BA merges: original outer computations remain installed intact.
    torch.manual_seed(settings['seed'])
    core=MaskedReferenceSupervision(wrapper.model,halves,layers=settings['layers'],rank=settings['rank'])
    wrapper.model.eval()
    expected=descriptor['model_identity']['alternative_weights_sha256']
    require(frozen_identity(wrapper.model,core)['half_weights_sha256']==expected,'Historical halfway matrices changed')
    return wrapper,core


def move_for_training(wrapper,core,device):
    """Move inference-relevant train modules, retaining codec/encoder on CPU."""
    device=torch.device(device)
    wrapper.model.talker.to(device)
    core.adapters.to(device);core.bank.to(device)
    if hasattr(wrapper.model,'speaker_preference_adapter'):
        wrapper.model.speaker_preference_adapter.to(device)
    wrapper.device=device
    wrapper.model.eval()
    for decoder in (wrapper.model.talker.model,wrapper.model.talker.code_predictor.model):
        require(decoder.config.attention_dropout==0,'Stochastic attention unsupported')
        decoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        decoder.train()
    core.parameters()


def native_kwargs(input_ids,reference,generation):
    return dict(input_ids=[input_ids],ref_ids=[reference.ref_ids],languages=['Auto'],non_streaming_mode=True,
        voice_clone_prompt=dict(ref_spk_embedding=[reference.speaker],ref_code=[reference.ref_codes],
                               icl_mode=[True],x_vector_only_mode=[False]),**generation)


def forced_native_outputs(model,core,input_ids,codes,reference,half_reference,generation):
    """Independent native prefix and actual cached talker/code-predictor replay.

    Recorded real codec IDs replace sampling only. Native talker.forward builds
    every consumed frame embedding itself; native predictor.forward uses caches.
    No teacher-builder suffix or forward_finetune residual logits are reused.
    """
    require(not model.training and not torch.is_grad_enabled(),'Native replay requires eval/no_grad')
    captured={}
    class Finished(Exception):pass
    def forced(**kwargs):
        talker=model.talker;predictor=talker.code_predictor;P=kwargs['inputs_embeds'].shape[1]
        native_inputs=[];original_main=core.original_main
        def capture_main(**arguments):
            native_inputs.append(arguments['inputs_embeds'].clone())
            return original_main(**arguments)
        with patch.object(core,'original_main',side_effect=capture_main):
            state=talker(inputs_embeds=kwargs['inputs_embeds'],attention_mask=kwargs['attention_mask'],
                trailing_text_hidden=kwargs['trailing_text_hidden'],tts_pad_embed=kwargs['tts_pad_embed'],
                cache_position=torch.arange(P,device=codes.device),use_cache=True,return_dict=True)
            main=[state.logits[:,-1]];residual=[]
            for frame in range(len(codes)):
                initial=torch.cat([state.past_hidden,talker.get_input_embeddings()(codes[frame:frame+1,:1])],dim=1)
                sub=predictor(inputs_embeds=initial,use_cache=True,return_dict=True);row=[sub.logits[:,-1]]
                for group in range(1,codes.shape[1]-1):
                    sub=predictor(input_ids=codes[frame:frame+1,group:group+1],past_key_values=sub.past_key_values,
                        generation_steps=sub.generation_steps,use_cache=True,return_dict=True)
                    row.append(sub.logits[:,-1])
                residual.append(torch.stack(row,dim=1))
                with patch.object(predictor,'generate',return_value=SimpleNamespace(sequences=codes[frame:frame+1,1:])):
                    state=talker(input_ids=codes[frame:frame+1,:1],past_hidden=state.past_hidden,
                        past_key_values=state.past_key_values,generation_step=state.generation_step,
                        attention_mask=torch.ones(1,P+frame+1,device=codes.device,dtype=torch.long),
                        cache_position=torch.tensor([P+frame],device=codes.device),
                        trailing_text_hidden=kwargs['trailing_text_hidden'],tts_pad_embed=kwargs['tts_pad_embed'],
                        use_cache=True,return_dict=True)
                main.append(state.logits[:,-1])
        captured.update(main_logits=torch.cat(main,dim=0),residual_logits=torch.cat(residual,dim=0),
                        embeddings=torch.cat(native_inputs,dim=1)[0],prefix_length=P)
        raise Finished()
    try:
        with core.native(half_reference),patch.object(core,'original_talker_generate',side_effect=forced):
            model.generate(**native_kwargs(input_ids,reference,generation))
    except Finished:pass
    require(bool(captured),'Native forced replay did not complete')
    return captured


def verify_replay_pair(model,core,input_ids,codes,reference,generation,half_reference,atol=2e-5,rtol=2e-5):
    with torch.no_grad():
        native=forced_native_outputs(model,core,input_ids,codes,reference,half_reference,generation)
        with core.replay(input_ids,codes,reference,half_reference,terminated=True) as seq:
            require(seq.prefix_length==native['prefix_length'] and torch.equal(seq.embeddings,native['embeddings']),
                    'Native prefix/consumed-code embeddings are not exact')
            teacher=core.outputs(seq)
            errors={}
            for key in ('main_logits','residual_logits'):
                require(torch.isfinite(native[key]).all() and torch.isfinite(teacher[key]).all()
                    and torch.allclose(native[key],teacher[key],atol=atol,rtol=rtol),'Native cached/teacher logits differ: '+key)
                errors[key]=float((native[key]-teacher[key]).abs().max())
            require(teacher['token_count']==codes.numel()+1,'Complete16-codebook plus EOS coverage differs')
    return dict(half_reference=half_reference,frames=len(codes),token_count=codes.numel()+1,
                prefix_and_consumed_frames_exact=True,maximum_logit_error=errors,atol=atol,rtol=rtol)
