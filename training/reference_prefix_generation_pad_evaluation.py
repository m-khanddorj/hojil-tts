"""Conditional, no-training recurrent PAD ablation with an exact 26-clip control.

The original half-routed prefill is retained for the requested reference. Only
the recurrent generation PAD is switched; other references use centered16 full.
Every speech stage is independently audited before any screening conclusion.
"""
import argparse
import math
import os
from pathlib import Path
import numpy as np
import torch
from .engine import read,write,save,digest,ids
from .policy import external
from . import reference_prefix_comparison as prefix
from . import reference_profile_preference_evaluation as dpo
from . import reference_profile_preference_training as trainer
from .reference_profile_collection import numeric_config
from .reference_prefix_generation_pad import ReferencePrefixGenerationPad,tensor_sha256
from .targeted_voice import aggregate
from .voice_strength_search import goal_screen

STAGES=('control','full-pad')
NUMERIC=dict(cpu_threads=4,matmul_tf32=False,cudnn_tf32=False)


def require(ok,message):
    if not ok:raise ValueError(message)


def sources():
    result=dict(trainer.sources(),**prefix.sources())
    for name in ('reference_prefix_generation_pad.py','reference_prefix_generation_pad_evaluation.py'):
        result[name]=digest(Path(__file__).with_name(name))
    for name in ('test_reference_prefix_generation_pad.py','test_training_reference_prefix_generation_pad_evaluation.py'):
        result[name]=digest(Path(__file__).parent.parent/'tests'/name)
    return result


def terminal_decision(status,history,stored):
    require(status['status']=='complete','DPO must finish before this separate ablation')
    decision=dpo.progression(history)
    reason=decision['reason'] if decision['stop'] else 'maximum_updates_reached'
    require(decision==stored and status['step']==decision['step'] and status['reason']==reason,'DPO terminal decision differs')
    require(reason in ('maximum_updates_reached','consecutive_pareto_regressions'),'A passing DPO screen requires confirmation first')
    require(decision['stop'] or decision['step']==16,'DPO is not terminal')
    require(not any(r['goal_screen'] for r in history),'A passing DPO checkpoint requires confirmation first')
    return decision


def dpo_evidence(run):
    status_path=run.parent/'reference-profile-preference-training-status.json';status=read(status_path)
    require(status['status']=='complete','DPO must finish before this separate ablation')
    p,e,_=dpo.verify(run);step=status['step'];require(step in p['checkpoints'],'Unregistered terminal DPO step')
    history=[];paths=[status_path,run/'protocol.json',run/'evaluation-protocol.json',run/'progression.json']
    for step in p['checkpoints'][:p['checkpoints'].index(step)+1]:
        final_path=run/f'verification-{step:06d}.json';final=read(final_path)
        checkpoint=trainer.checkpoint_path(run,step);trainer.read_checkpoint(run,step,p)
        sha=digest(checkpoint);evaluation_sha=digest(run/'evaluation-protocol.json');combined=[]
        require(final['status']=='verified' and final['step']==step and final['recordings']==50
            and final['all_weights_unchanged'] and final['inactive_development_exact_parity']
            and (step!=0 or final['all_baseline_waveforms_exact'])
            and final['checkpoint_sha256']==sha and final['evaluation_protocol_sha256']==evaluation_sha,
            'Complete bound DPO audit required')
        for kind,count in (('probes',24),('screen',26)):
            dest=dpo.directory(run,step,kind);audit_path=dest/'verification.json';report_path=dest/'results.json'
            audit=read(audit_path);report=read(report_path);rows=report['records'];summary=dpo.validate_records(e['cases'][kind],rows)
            require(len(rows)==count and audit['status']=='verified' and audit['step']==step and audit['kind']==kind
                and audit['recordings']==count and audit['all_weights_unchanged']
                and digest(audit_path)==final['partition_audit_sha256'][kind] and digest(report_path)==audit['results_sha256']
                and audit['checkpoint_sha256']==sha and audit['evaluation_protocol_sha256']==evaluation_sha
                and audit['summary']==report['summary']==summary
                and audit['truncated']==report['truncated']==sum(not r['terminated'] for r in rows)
                and math.isfinite(audit['maximum_cosine_recomputation_error']) and audit['maximum_cosine_recomputation_error']<=2e-6,
                'DPO partition evidence differs')
            for r in rows:
                require(digest(dest/r['wav'])==r['audio_sha256'] and digest(dest/r['trajectory'])==r['trajectory_sha256']
                    and digest(r['reference'])==r['reference_sha256'],'DPO speech evidence changed')
            combined.extend(rows);paths.extend((audit_path,report_path))
            if kind=='screen':
                require(final['screen_summary']==summary and final['screen_truncated']==report['truncated'],'DPO screen receipt differs')
        require(final['summary']==aggregate(combined) and final['truncated']==sum(not r['terminated'] for r in combined),'DPO complete audit differs')
        history.append(final);paths.extend((final_path,checkpoint,checkpoint.with_suffix('.json')))
    terminal_decision(status,history,read(run/'progression.json'))
    return {str(path):digest(path) for path in paths}


def baseline_evidence(old):
    p=prefix.verify(old);report=read(old/'results.json');audit=read(old/'verification.json')
    centered_path=Path(p['baseline']);centered=read(centered_path);centered_audit=read(p['initial_audit'])
    require(audit['status']=='verified' and audit['recordings']==26 and audit['report_sha256']==digest(old/'results.json')
        and audit['reference_matrices_verified']==198 and audit['acoustic_and_speaker_adapter_verified']
        and audit['summary']==report['summary']==aggregate(report['records']),'Original routed baseline audit missing')
    require(centered_audit['status']=='verified' and centered_audit['recordings']==26
        and centered_audit['frozen_original_adapter_and_coordinates_verified']
        and centered_audit['summary']==centered['summary']==aggregate(centered['records']),'Centered16 audit missing')
    rows=[r for r in report['records'] if r['group']=='custom']+[r for r in centered['records'] if r['group']=='development']
    cases=[dict(c,seed=s) for s in p['seeds'] for c in p['cases']];validate_records(cases,rows)
    target={r['reference_sha256'] for r in rows if r['group']=='custom'}
    require(len(target)==1,'One requested reference required');target=next(iter(target))
    require(all(r['reference_sha256']!=target for r in rows if r['group']=='development'),'Custom and inactive references overlap')
    paths=[old/'protocol.json',old/'results.json',old/'verification.json',old/'model.json',
        centered_path,Path(p['initial_audit']),Path(p['initial_run'])/'protocol.json',
        prefix.initial_trainer.checkpoint(Path(p['initial_run']),16),old.parent/'verify-reference-prefix-comparison.py']
    require(digest(paths[-1])==audit['source_sha256'],'Historical prefix audit source differs')
    for r in rows:
        directory=old if r['group']=='custom' else centered_path.parent
        require(digest(directory/r['wav'])==r['audio_sha256'] and digest(r['reference'])==r['reference_sha256'],'Historical baseline audio changed')
    return p,cases,rows,target,{str(path):digest(path) for path in paths}


def validate_records(cases,rows):
    require(len(cases)==len(rows)==26,'Complete 26-recording screen required')
    expected={(r['case'],r['seed']):r for r in cases}
    require(len(expected)==26 and {(r['case'],r['seed']) for r in rows}==set(expected),'Duplicate or missing screen condition')
    for r in rows:
        require(all(r[k]==v for k,v in expected[r['case'],r['seed']].items()),'Screen condition metadata changed')
        require(math.isfinite(r['cosine']) and -1<=r['cosine']<=1,'Invalid stored cosine')
    require(sum(r['group']=='custom' for r in rows)==6 and sum(r['group']=='development' for r in rows)==20,'Both screen groups required')
    return aggregate(rows)


def parity(before,after,stage):
    old={(r['case'],r['seed']):r for r in before};checked=0
    for r in after:
        if stage=='control' or r['group']=='development':
            b=old[r['case'],r['seed']]
            require(r['audio_sha256']==b['audio_sha256'] and r['score']==b['score']
                and math.isfinite(r['cosine']) and math.isfinite(b['cosine']) and abs(r['cosine']-b['cosine'])<=2e-6,
                'Required exact baseline waveform/score parity failed')
            checked+=1
    require(checked==(26 if stage=='control' else 20),'Incomplete baseline parity')
    return checked


def prepare(old,dpo_run,out,driver):
    hashes=dpo_evidence(dpo_run);p,cases,_,target,baseline_hashes=baseline_evidence(old)
    asr_manifest=Path(p['config']['asr'])/'download-manifest.json'
    hashes[str(asr_manifest)]=digest(asr_manifest)
    require(numeric_config()==NUMERIC,'CPU FP32 numeric configuration required')
    logs={}
    for name in ('prefix-generation-pad-tests.log','reference-prefix-generation-pad-evaluation-tests.log'):
        path=out.parent/name;content=path.read_text(encoding='utf-8-sig')
        require('Ran 6 tests' in content and '\nOK\n' in content and 'skipped' not in content.lower(),'New CPU tests must pass')
        logs[str(path)]=digest(path)
    out=external(out.resolve());out.mkdir(exist_ok=False)
    write(out/'protocol.json',dict(old=str(old),dpo_run=str(dpo_run),evidence=dict(hashes,**baseline_hashes),
        source=sources(),runtime=prefix.runtime(),numeric_config=NUMERIC,tests=logs,driver=str(driver),driver_sha256=digest(driver),
        cases=cases,reference_sha256=target,config=p['config'],generation=p['generation'],decoding=p['decoding'],stages=list(STAGES),
        intervention='Requested reference only: original half-routed prefill; recurrent trailing_text_hidden and tts_pad_embed use full text projection',
        scope='26 clips per stage: six requested-reference clips and twenty exact centered16 development fallback clips',
        optimizer_updates=0,goal_achieved=False,global_model_selection_changed=False,
        limitation='Complete, independently audited screen only; any pass still needs a separate complete 114-recording confirmation'))


def verify(out):
    p=read(out/'protocol.json')
    require(p['source']==sources() and p['runtime']==prefix.runtime() and p['numeric_config']==numeric_config()==NUMERIC
        and p['stages']==list(STAGES) and digest(p['driver'])==p['driver_sha256'],'PAD ablation registration changed')
    for path,sha in dict(p['tests'],**p['evidence']).items():require(digest(path)==sha,'Registered evidence changed')
    old,cases,rows,target,_=baseline_evidence(Path(p['old']))
    require(p['cases']==cases and p['reference_sha256']==target and p['generation']==old['generation']
        and p['decoding']==old['decoding'] and p['config']==old['config'],'Registered screen configuration changed')
    return p,old,rows


def prior_control(out,p):
    receipt=read(out/'control/verification.json');report=read(out/'control/results.json')
    require(receipt['status']=='verified' and receipt['stage']=='control' and receipt['recordings']==26
        and receipt['baseline_waveforms_checked']==26 and receipt['all_weights_unchanged']
        and receipt['results_sha256']==digest(out/'control/results.json') and receipt['protocol_sha256']==digest(out/'protocol.json')
        and receipt['summary']==validate_records(p['cases'],report['records']),'Complete independently audited control required')
    for r in report['records']:
        require(digest(out/'control'/r['wav'])==r['audio_sha256'] and digest(out/'control'/r['trajectory'])==r['trajectory_sha256'],'Control artifact changed')
    return report['records']


def load(p,old,stage):
    require(stage in STAGES,'Unregistered stage');wrapper=prefix.load(old);wrapper.model.requires_grad_(False);wrapper.model.eval()
    require(all(v.device.type=='cpu' and (not v.is_floating_point() or v.dtype==torch.float32) for v in wrapper.model.parameters()),'CPU FP32 model required')
    require(wrapper.model.talker.config._attn_implementation=='sdpa','Registered SDPA backend required')
    identity=prefix.model_identity(wrapper);expected=read(Path(p['old'])/'model.json')
    require(all(expected[k]==v for k,v in identity.items()),'Original centered16/routing identity changed')
    patch=ReferencePrefixGenerationPad(wrapper.model.reference_prefix_routing,p['reference_sha256'],full_generation_pad=stage=='full-pad')
    return wrapper,patch,identity


def unchanged(wrapper,patch,identity):
    require(prefix.model_identity(wrapper)==identity and not wrapper.model.reference_prefix_routing.active
        and all(v.grad is None and not v.requires_grad for v in wrapper.model.parameters()),'Original weights or inference state changed')
    patch.remove()


def generate(out,stage):
    import soundfile as sf
    from .preference_rollouts import capture_generation
    p,old,baseline=verify(out);control=prior_control(out,p) if stage=='full-pad' else None
    wrapper,patch,identity=load(p,old,stage);dest=out/stage;dest.mkdir(exist_ok=False);records=[];prompts={}
    for c in p['cases']:
        require(digest(c['reference'])==c['reference_sha256'],'Reference changed')
        if c['reference'] not in prompts:
            with torch.inference_mode():prompts[c['reference']]=wrapper.create_voice_clone_prompt(ref_audio=c['reference'],ref_text=c['reference_text'],x_vector_only_mode=False)
        prompt=prompts[c['reference']];torch.manual_seed(c['seed']);np.random.seed(c['seed'])
        with patch.use_reference(c['reference_sha256']),torch.inference_mode(),capture_generation(wrapper.model) as captured:
            waves,sr=wrapper.generate_voice_clone(text=c['text'],language='Auto',voice_clone_prompt=prompt,non_streaming_mode=True,**p['generation'])
        require(sr==24000 and len(waves[0])>0 and np.isfinite(waves[0]).all() and len(waves[0])/sr<=30,'Invalid generated waveform')
        path=dest/f"{c['seed']}-{c['case']}.wav";sf.write(path,waves[0],sr,subtype='PCM_16');wave,sr=sf.read(path,dtype='float32')
        with torch.inference_mode():
            vector=wrapper.model.extract_speaker_embedding(audio=wave,sr=sr).flatten();ref=prompt[0].ref_spk_embedding.flatten()
            cosine=float(torch.nn.functional.cosine_similarity(vector[None],ref[None]))
        require(math.isfinite(cosine),'Nonfinite generated cosine')
        trajectory=path.with_suffix('.pt');trace=patch.last_trace;routing=wrapper.model.reference_prefix_routing.last_trace
        save(trajectory,dict(codes=captured['codes'],first_sequence=captured['first_sequence'],terminated=captured['terminated'],
            ref_codes=prompt[0].ref_code.detach().cpu(),ref_xvec=ref.detach().cpu(),input_ids=ids(wrapper,c['text']).cpu(),
            ref_ids=ids(wrapper,c['reference_text'],True).cpu(),reference_sha256=c['reference_sha256'],pad_trace=trace,routing=routing))
        row=dict(c,wav=path.name,audio_sha256=digest(path),trajectory=trajectory.name,trajectory_sha256=digest(trajectory),
            cosine=cosine,seconds=len(wave)/sr,frames=len(captured['codes']),terminated=captured['terminated'],
            stage=stage,pad_trace=trace,routing=routing,model_identity=identity,protocol_sha256=digest(out/'protocol.json'))
        write(path.with_suffix('.json'),row);records.append(row);print(f'{stage}: {len(records)}/26',flush=True)
    unchanged(wrapper,patch,identity)
    baseline_map={(r['case'],r['seed']):r for r in baseline}
    for r in records:
        if stage=='control' or r['group']=='development':require(r['audio_sha256']==baseline_map[r['case'],r['seed']]['audio_sha256'],'Native baseline waveform parity failed')
    write(dest/'generated.json',dict(records=records,all_weights_unchanged=True,model_identity=identity,protocol_sha256=digest(out/'protocol.json')))


def score(out,stage):
    import soundfile as sf
    import torchaudio
    from transformers import WhisperForConditionalGeneration,WhisperProcessor
    from .evaluate import ASR_REV
    from scripts.score_latest_audio import score as score_text
    p,_,_=verify(out);dest=out/stage;require(stage in STAGES and not (dest/'results.json').exists(),'No overwrite or unregistered stage')
    if stage=='full-pad':prior_control(out,p)
    generated=read(dest/'generated.json');require(generated['all_weights_unchanged'] and generated['protocol_sha256']==digest(out/'protocol.json'),'Generation receipt differs')
    asr=Path(p['config']['asr']);manifest=read(asr/'download-manifest.json')
    require(manifest['repo']=='BuzzASR/mongolian' and manifest['revision']==ASR_REV,'ASR mismatch')
    for name,item in manifest['files'].items():require(digest(asr/name)==item['sha256'],'ASR weights changed')
    processor=WhisperProcessor.from_pretrained(asr,local_files_only=True)
    model=WhisperForConditionalGeneration.from_pretrained(asr,torch_dtype=torch.float32,local_files_only=True).to('cuda').eval()
    records=generated['records']
    for r in records:
        require(digest(dest/r['wav'])==r['audio_sha256'],'Audio changed')
        wave,sr=sf.read(dest/r['wav'],dtype='float32');wave=torchaudio.functional.resample(torch.from_numpy(wave),sr,16000).numpy()
        batch=processor(wave,sampling_rate=16000,return_tensors='pt',return_attention_mask=True).to('cuda')
        with torch.inference_mode():tokens=model.generate(batch.input_features,attention_mask=batch.attention_mask,**p['decoding'])
        hypothesis=processor.batch_decode(tokens,skip_special_tokens=True)[0].strip()
        r.update(hypothesis=hypothesis,score=score_text(r['text'],hypothesis,True),decoding=p['decoding'],asr_revision=ASR_REV)
    write(dest/'results.json',dict(records=records,summary=validate_records(p['cases'],records),truncated=sum(not r['terminated'] for r in records)))


def audit(out,stage):
    import soundfile as sf
    from .evaluate import ASR_REV
    from scripts.score_latest_audio import score as score_text
    p,old,baseline=verify(out);dest=out/stage;require(not (dest/'verification.json').exists(),'Do not overwrite an audit')
    control=prior_control(out,p) if stage=='full-pad' else None
    wrapper,patch,identity=load(p,old,stage);report=read(dest/'results.json');records=report['records'];summary=validate_records(p['cases'],records)
    special_ids,full_pad,half_pad=patch._pads()
    refs={};maximum=0.;paired=0;control_map={} if control is None else {(r['case'],r['seed']):r for r in control}
    for r in records:
        require(r['stage']==stage and r['protocol_sha256']==digest(out/'protocol.json') and r['model_identity']==identity
            and r['decoding']==p['decoding'] and r['asr_revision']==ASR_REV and digest(dest/r['wav'])==r['audio_sha256']
            and digest(dest/r['trajectory'])==r['trajectory_sha256'] and digest(r['reference'])==r['reference_sha256'],'Speech provenance changed')
        require(r['score']==score_text(r['text'],r['hypothesis'],True),'ASR edit counts differ')
        t=torch.load(dest/r['trajectory'],map_location='cpu',weights_only=True);codes=t['codes'];seq=t['first_sequence'][0];eos=wrapper.model.config.talker_config.codec_eos_token_id
        require(codes.shape==(r['frames'],16) and len(codes)>0 and len(seq)==len(codes)+1
            and torch.equal(seq[:len(codes)],codes[:,0]) and bool(seq[-1]==eos)==r['terminated']==t['terminated']
            and not bool((seq[:-1]==eos).any()) and t['reference_sha256']==r['reference_sha256'],'Native trajectory/EOS differs')
        require(torch.equal(t['input_ids'],ids(wrapper,r['text']).cpu()) and torch.equal(t['ref_ids'],ids(wrapper,r['reference_text'],True).cpu()),'Native transcript IDs differ')
        if r['reference'] not in refs:
            with torch.inference_mode():
                prompt=wrapper.create_voice_clone_prompt(ref_audio=r['reference'],ref_text=r['reference_text'],x_vector_only_mode=False)
                ref,refsr=sf.read(r['reference'],dtype='float32');embedding=wrapper.model.extract_speaker_embedding(audio=ref,sr=refsr).flatten()
            refs[r['reference']]=(prompt[0],embedding)
        prompt,embedding=refs[r['reference']]
        require(torch.equal(t['ref_codes'],prompt.ref_code.cpu()) and torch.equal(t['ref_xvec'],prompt.ref_spk_embedding.flatten().cpu()),'Native original reference conditioning differs')
        validate_trace(r,t,p,stage)
        require(r['pad_trace']['special_token_ids_sha256']==tensor_sha256(special_ids)
            and r['pad_trace']['full_pad_sha256']==tensor_sha256(full_pad)
            and r['pad_trace']['half_pad_sha256']==tensor_sha256(half_pad),'Independent native PAD reconstruction differs')
        if control is not None:
            b=control_map[r['case'],r['seed']];bt=torch.load(out/'control'/b['trajectory'],map_location='cpu',weights_only=True)
            paired_prefill(r,t,b,bt);paired+=1
        wave,sr=sf.read(dest/r['wav'],dtype='float32');require(sr==24000 and wave.ndim==1 and len(wave)>0 and np.isfinite(wave).all(),'Invalid audit waveform')
        with torch.inference_mode():
            vector=wrapper.model.extract_speaker_embedding(audio=wave,sr=sr).flatten()
            cosine=float(torch.nn.functional.cosine_similarity(vector[None],embedding[None]))
        require(math.isfinite(cosine),'Nonfinite independently recomputed cosine');maximum=max(maximum,abs(cosine-r['cosine']))
    checked=parity(baseline,records,stage);truncated=sum(not r['terminated'] for r in records)
    require(maximum<=2e-6 and summary==report['summary'] and truncated==report['truncated'],'Independent score summary differs')
    unchanged(wrapper,patch,identity)
    write(dest/'verification.json',dict(status='verified',stage=stage,recordings=26,summary=summary,truncated=truncated,
        goal_screen=goal_screen(summary) and truncated==0,goal_achieved=False,global_model_selection_changed=False,
        optimizer_updates=0,all_weights_unchanged=True,model_identity=identity,baseline_waveforms_checked=checked,
        paired_prefill_and_first_frame_verified=paired,all_recomputed_cosines_finite=True,maximum_cosine_recomputation_error=maximum,
        results_sha256=digest(dest/'results.json'),protocol_sha256=digest(out/'protocol.json'),
        control_audit_sha256=digest(out/'control/verification.json') if stage=='full-pad' else None,
        paired_changes=dpo.paired_changes(baseline,records),limitation='Any full-pad screen pass requires complete 114-recording confirmation'))


def validate_trace(r,t,p,stage):
    require(t['pad_trace']==r['pad_trace'] and t['routing']==r['routing'],'Trajectory trace differs')
    trace=r['pad_trace'];active=r['reference_sha256']==p['reference_sha256']
    replace=active and stage=='full-pad'
    require(trace['status']=='complete' and trace['reference_sha256']==r['reference_sha256']
        and trace['registered_reference_sha256']==p['reference_sha256'] and trace['custom_reference']==active
        and trace['protected']==active and trace['full_generation_pad']==(stage=='full-pad') and trace['talker_calls']==1
        and trace['pad_replaced']==replace and trace['non_pad_kwargs_identity_verified'] and trace['input_tensors_unchanged']
        and trace['mode']==('custom_full_generation_pad' if replace else 'custom_original_half_pad' if active else 'inactive_full_policy'),
        'Reference-gated PAD trace differs')
    nr=t['ref_ids'].shape[1]-5;nt=t['input_ids'].shape[1]-8;length=8+nr+nt+2+len(t['ref_codes'])
    expected=dict(target_start=8+nr,target_stop=8+nr+nt,prefix_length=length,prefills=1,decode_steps=len(t['codes']),
        protected=active,main_matrices=196,reference_positions=length-nt)
    require(r['routing']==expected,'Reference routing layout differs')
    require(trace['layout']=={k:expected[k] for k in ('target_start','target_stop','prefix_length')},'PAD boundary layout differs')
    require({'inputs_embeds','attention_mask','trailing_text_hidden','tts_pad_embed'}<=set(trace['input_tensor_sha256']),
        'Native prefill/PAD tensor identities missing')
    for key in ('trailing_text_hidden','tts_pad_embed'):
        require(trace['input_tensor_sha256'][key]==trace['half_pad_sha256' if active else 'full_pad_sha256']
            and trace['output_pad_sha256'][key]==trace['full_pad_sha256' if replace or not active else 'half_pad_sha256'],
            'Native incoming or outgoing PAD identity differs')


def paired_prefill(row,trajectory,control,control_trajectory):
    require(row['pad_trace']['input_tensor_sha256']==control['pad_trace']['input_tensor_sha256'],
        'Intervention changed prefill inputs, mask or incoming native PAD')
    require(torch.equal(trajectory['codes'][0],control_trajectory['codes'][0]),'Intervention changed first generated frame')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=['prepare','generate','score','audit'])
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--old',type=Path);parser.add_argument('--dpo-run',type=Path)
    parser.add_argument('--driver',type=Path);parser.add_argument('--stage',choices=STAGES);args=parser.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if args.phase=='prepare':prepare(args.old,args.dpo_run,args.out,args.driver);return
    p=read(args.out/'protocol.json');os.environ['NUMBA_CACHE_DIR']=p['config']['numba_cache'];os.environ['TEMP']=os.environ['TMP']=p['config']['tmp']
    os.environ['HF_HUB_OFFLINE']=os.environ['TRANSFORMERS_OFFLINE']='1'
    {'generate':generate,'score':score,'audit':audit}[args.phase](args.out,args.stage)


if __name__=='__main__':main()
