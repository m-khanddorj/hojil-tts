"""Conditional original114 for the separately registered all-half route.

No old confirmation gate is relaxed. Both ablation arms and authoritative
process closure precede deterministic selection; all114 then need fresh audit.
"""
from dataclasses import asdict
import math
from pathlib import Path

from . import reference_masked_supervision_routing_ablation as ablation
from .reference_masked_supervision_confirmation import (
    PANEL_SHA, expanded_panel, measurement, fresh_reference, complete_audit)
from .engine import ids, save

read, write, digest, require = ablation.read, ablation.write, ablation.digest, ablation.require
ABLATION_SHA = 'faaca2f70c9b47dd3a3373c5de0605a8a6d2bd1efe0cf2c0f952cba50c024ee9'
RULE = dict(eligibility='complete80, no truncation, both pooled CER < .03 and original-reference speaker-macro cosine > .984',
    order=['minimize maximum group CER','maximize minimum group cosine','lower checkpoint step'],
    comparison_checkpoints=[0,32], eligible_checkpoints=[32],
    excluded_steps={'0':'Unchanged custom branch of previously failed original114 centered16/fullPAD policy'},
    require_both_arms_complete=True, half_reference=True, continuation_pad='full')
FLAGS = dict(optimizer_updates=0,goal_achieved=False,global_model_selection_changed=False)
KNOWN_BASE_RESULTS_SHA = 'f6ac5ed48e2681fb91589e0e1a9737b7df9678c85f834aa5e85beaeb857200d8'


def excluded_zero_evidence(p,e):
    """Prior-policy identity supports exclusion; this is not a fresh26 replay."""
    train=Path(e['train']);base=p['policy'];zero=read(train/'checkpoint-000000.json')
    native=read(train/'native-verification.json');prior=read(base['results'])
    require(base['kind']=='centered16' and digest(base['results'])==base['results_sha256']==KNOWN_BASE_RESULTS_SHA
        and zero['adapter_movement']['maximum']==0 and zero['frozen_identity']==p['frozen_identity']
        and base['model_identity']['speaker_adapter_sha256']==p['frozen_identity']['speaker_state_sha256']
        and base['model_identity']['alternative_weights_sha256']==p['frozen_identity']['half_weights_sha256']
        and prior['summary']['custom']['recordings']==26
        and prior['summary']['custom']['pooled_cer']==.03279631760644419
        and prior['summary']['custom']['speaker_macro_cosine']==.9835558739992288,
        'Prior unchanged custom failure proof differs')
    controls=[r for r in native['historical_waveform_controls'] if r['group']=='custom']
    expected=[r for r in base['controls'] if r['group']=='custom']
    require(len(controls)==len(expected)==1 and all(controls[0][k]==expected[0][k] for k in
        ('case','seed','group','audio_sha256')) and controls[0]['audio_sha256']==
        '839ec5cab008dacff3586ef35696a2dbc43e3d825f23b54b5ffa1d8358dd215a',
        'Exact historical custom zero-control proof differs')


def selection(arms):
    require(set(arms) == {'0','32'}, 'Both complete ablation endpoints required before selection')
    eligible=[]; metrics={}
    for step in (0,32):
        arm=arms[str(step)];summary=arm['half_development_shared_custom_summary']
        require(set(summary)=={'development','custom'}, 'Both complete groups required')
        for group,count,speakers in (('development',64,16),('custom',16,1)):
            r=summary[group]
            require(r['recordings']==count and r['speakers']==speakers
                and all(math.isfinite(r[k]) for k in ('pooled_cer','pooled_wer','speaker_macro_cosine'))
                and r['pooled_cer']>=0 and r['pooled_wer']>=0 and -1<=r['speaker_macro_cosine']<=1,
                'Invalid complete calibration metrics')
        require(type(arm['truncated']) is int and 0<=arm['truncated']<=80, 'Invalid truncation count')
        passed=arm['truncated']==0 and all(r['pooled_cer']<.03 and r['speaker_macro_cosine']>.984 for r in summary.values())
        require(arm['calibration_thresholds_met'] is passed, 'Recomputed qualification differs')
        if passed and step in RULE['eligible_checkpoints']:eligible.append(step)
        metrics[str(step)]=dict(maximum_group_cer=max(r['pooled_cer'] for r in summary.values()),
            minimum_group_cosine=min(r['speaker_macro_cosine'] for r in summary.values()))
    require(eligible, 'No eligible all-half policy; original114 is prohibited')
    chosen=min(eligible,key=lambda s:(metrics[str(s)]['maximum_group_cer'],-metrics[str(s)]['minimum_group_cosine'],s))
    return dict(step=chosen,eligible_steps=eligible,metrics=metrics,rule=RULE,**FLAGS)


def validate_closure(status,closure,bindings,session):
    require(status.get('status')=='complete' and all(status.get(k)==v for k,v in FLAGS.items()),
        'Complete unpromoted parent driver required')
    require(closure.get('status')=='closed' and type(closure.get('exit_code')) is int and closure['exit_code']==0
        and type(closure.get('original_session_id')) is int and closure['original_session_id']>0
        and (session is None or closure['original_session_id']==session)
        and all(closure.get(k)==v for k,v in bindings.items()), 'Authoritative successful bound process closure required')


def parent_evidence(run):
    """Stdlib-only entry: recheck both complete native arms and shared controls."""
    run=Path(run);p,e=ablation.verify(run,ABLATION_SHA)
    excluded_zero_evidence(p,e)
    require(run.name=='masked-supervision-routing-ablation-v1', 'Exact registered routing ablation required')
    status_path=run.parent/(run.name+'-status.json');closed_path=run.parent/(run.name+'-process-closed.json')
    driver=run.parent/'run-masked-supervision-routing-ablation-v1.py'
    status=read(status_path);result=read(run/'results.json')
    bindings=dict(protocol_sha256=ABLATION_SHA,driver_sha256=digest(driver),
        terminal_status_sha256=digest(status_path),results_sha256=digest(run/'results.json'))
    validate_closure(status,read(closed_path),bindings,23429)
    require(status['protocol_sha256']==ABLATION_SHA and status['results_sha256']==bindings['results_sha256']
        and result['status']=='verified' and result['protocol_sha256']==ABLATION_SHA
        and result['new_recordings']==128 and result['reused_recordings']==160
        and all(result.get(k)==v for k,v in ablation.FLAGS.items()) and set(result['arms'])=={'0','32'},
        'Complete fixed128 ablation report required')
    phases={f'{phase}-{step:06d}' for step in (0,32) for phase in ('generate','score','audit')}|{'finalize'}
    require(set(status['logs'])==phases, 'Complete serial ablation phase history required')
    hashes=dict(e['evidence']);paths=[run/'protocol.json',run/'results.json',status_path,closed_path,driver]
    for item in status['logs'].values():
        require(type(item['exit_code']) is int and item['exit_code']==0 and digest(item['path'])==item['sha256'],
            'Ablation phase failed or log changed')
        paths.append(Path(item['path']))
    for step in (0,32):
        audit,new=ablation.audited_arm(run,p,e,step)
        old_path=Path(e['train'])/f'eval-{step:06d}/results.json';old=read(old_path)['records']
        comparison=ablation.combined(e['original_cases'],old,new);arm=result['arms'][str(step)]
        require(all(arm.get(k)==v for k,v in comparison.items()) and arm['shared_controls'] is True
            and arm['independent_control_repeat'] is False and arm['custom_records_reused']==16
            and arm['full_development_records_reused']==64 and arm['original_results_sha256']==digest(old_path)
            and arm['original_audit_sha256']==digest(Path(e['train'])/f'verification-{step:06d}.json')
            and arm['new_results_sha256']==audit['results_sha256']
            and arm['new_audit_sha256']==digest(run/f'verification-{step:06d}.json'), 'Combined arm evidence differs')
        dest=ablation.directory(run,step)
        paths.extend((run/f'verification-{step:06d}.json',dest/'generated.json',dest/'results.json'))
        for r in new:
            paths.extend((dest/r['wav'],dest/r['trajectory'],Path(r['reference'])))
    for path,sha in result['evidence'].items():require(digest(path)==sha,'Final ablation evidence changed')
    chosen=selection(result['arms'])
    for path in paths:hashes[str(path.resolve())]=digest(path)
    return p,e,chosen,hashes


def sources():
    here=Path(__file__)
    paths=[here,here.parents[1]/'tests/test_reference_masked_supervision_routing_confirmation.py',
        here.with_name('reference_masked_supervision_confirmation.py'),here.with_name('confirm_joint_candidate.py'),
        here.with_name('joint_confirmation.py')]
    return {str(path.resolve()):digest(path) for path in paths}


def evidence(run):
    p,e,chosen,hashes=parent_evidence(run);train=Path(e['train']);step=chosen['step']
    panel_path=train.parent/'joint-goal-confirmation/protocol.json'
    require(digest(panel_path)==PANEL_SHA,'Exact original114 panel required')
    panel=read(panel_path);original_cases=expanded_panel(panel,p['custom_reference'])
    baseline=read(Path(p['policy']['run'])/'protocol.json')
    require(original_cases==baseline['cases'] and panel['generation']==p['generation']==baseline['generation']
        and panel['decoding']==p['decoding']==baseline['decoding']
        and panel['preparation_source_sha256']==digest(Path(__file__).with_name('joint_confirmation.py')),
        'Original114 metadata or generation/ASR settings changed')
    cases=[dict(c,half_reference=True) for c in original_cases]
    state=read(train/f'checkpoint-{step:06d}.json')
    require(step==32 and state['step']==step and state['adapter_movement']['maximum']>0
        and state['protocol_sha256']==ablation.PROTOCOL_SHA
        and state['sha256']==digest(train/f'checkpoint-{step:06d}.pt') and state['frozen_identity']==p['frozen_identity'],
        'Selected checkpoint identity differs')
    for c in cases:
        require(digest(c['reference'])==c['reference_sha256'],'Original114 reference changed')
        hashes[str(Path(c['reference']).resolve())]=c['reference_sha256']
    hashes[str(panel_path.resolve())]=PANEL_SHA
    return p,e,chosen,panel,cases,state,hashes


def prepare(run,out,driver,test_log,note):
    run,out,driver=Path(run).resolve(),Path(out).resolve(),Path(driver).resolve()
    require(out.parent==run.parent and out.name=='masked-supervision-routing-confirmation-v1' and not out.exists(),
        'New external confirmation directory required')
    p,e,chosen,panel,cases,state,hashes=evidence(run)
    ablation.validate_test_log(Path(test_log).read_text(encoding='utf-8-sig'))
    for path in (driver,Path(test_log),Path(note)):hashes[str(path.resolve())]=digest(path)
    protocol=dict(ablation=str(run),ablation_protocol_sha256=ABLATION_SHA,selection=chosen,rule=RULE,
        train_run=e['train'],step=chosen['step'],evidence=hashes,source=sources(),
        panel=str(run.parent/'joint-goal-confirmation/protocol.json'),panel_sha256=PANEL_SHA,cases=cases,
        config=p['config'],generation=p['generation'],decoding=p['decoding'],features=p['features'],
        checkpoint_sha256=state['sha256'],frozen_identity=state['frozen_identity'],adapter_movement=state['adapter_movement'],
        numeric_config=p['numeric_config'],runtime=p['runtime'],asr_manifest=e['asr_manifest'],
        asr_manifest_sha256=e['asr_manifest_sha256'],driver=str(driver),driver_sha256=digest(driver),
        scope='All-half reference/full continuation PAD, original114 unchanged88/26 and seeds2026/2027',**FLAGS)
    out.mkdir();write(out/'protocol.json',protocol);return protocol


def verify(out,protocol_sha):
    out=Path(out);require(digest(out/'protocol.json')==protocol_sha,'Registered confirmation protocol changed')
    p=read(out/'protocol.json');original,e,chosen,panel,cases,state,hashes=evidence(p['ablation'])
    require(p['source']==sources() and p['rule']==RULE and p['selection']==chosen and p['step']==chosen['step']
        and p['ablation_protocol_sha256']==ABLATION_SHA and p['cases']==cases
        and p['train_run']==e['train'] and p['panel_sha256']==digest(p['panel'])==PANEL_SHA
        and Path(p['panel']).resolve()==out.parent/'joint-goal-confirmation/protocol.json'
        and all(p[k]==original[k] for k in ('config','generation','decoding','features','numeric_config','runtime'))
        and all(p[k]==state[k] for k in ('frozen_identity','adapter_movement'))
        and p['checkpoint_sha256']==state['sha256'] and p['asr_manifest']==e['asr_manifest']
        and p['asr_manifest_sha256']==e['asr_manifest_sha256']==digest(p['asr_manifest'])
        and digest(p['driver'])==p['driver_sha256'] and all(p.get(k)==v for k,v in FLAGS.items())
        and all(p['evidence'].get(k)==v for k,v in hashes.items()),'Confirmation provenance changed')
    for path,sha in p['evidence'].items():require(digest(path)==sha,'Bound confirmation evidence changed')
    return p,original,panel


def bound_generation(out,p,scored=None):
    out=Path(out);dest=out/'speech';generated=read(dest/'generated.json');protocol_sha=digest(out/'protocol.json')
    require(generated['all_weights_unchanged'] is True and generated['protocol_sha256']==protocol_sha
        and generated['checkpoint_sha256']==p['checkpoint_sha256'],'Confirmation generation receipt differs')
    expected={(c['case'],c['seed']):c for c in p['cases']};records=generated['records']
    require(len(records)==len(expected)==114 and {(r['case'],r['seed']) for r in records}==set(expected),
        'Incomplete or duplicate114 generation')
    current={} if scored is None else {(r['case'],r['seed']):r for r in scored}
    if scored is not None:require(len(scored)==114 and set(current)==set(expected),'Incomplete scored114 coverage')
    for r in records:
        key=r['case'],r['seed']
        require(all(r.get(k)==v for k,v in expected[key].items()) and r['step']==p['step']
            and r['checkpoint_sha256']==p['checkpoint_sha256'] and r['protocol_sha256']==protocol_sha
            and r['half_reference'] is True and digest(dest/r['wav'])==r['audio_sha256']
            and digest(dest/r['trajectory'])==r['trajectory_sha256']
            and digest(r['reference'])==r['reference_sha256'],'Confirmation artifact or route changed')
        if scored is not None:
            require(set(current[key])==set(r)|{'hypothesis','score','asr_revision','decoding'}
                and all(current[key].get(k)==v for k,v in r.items()),'Confirmation scoring altered generated fields')
    return generated


def load(out,protocol_sha):
    from . import reference_masked_supervision_training_v2 as trainer
    from . import reference_masked_supervision_features as features
    p,original,panel=verify(out,protocol_sha)
    ablation.native_verify(Path(p['ablation']),ABLATION_SHA,p['step'])
    _,wrapper,core,state=trainer.load(p['train_run'],p['step'],device='cpu',training=False)
    features.assert_native_components(wrapper,original['features'])
    return p,original,panel,wrapper,core,state



def generate(out,protocol_sha):
    import numpy as np
    import soundfile as sf
    import torch
    from .preference_rollouts import capture_generation
    from .reference_masked_supervision_speech_v2 import capture_inputs,unchanged
    out=Path(out);p,_,_,wrapper,core,state=load(out,protocol_sha);dest=out/'speech';dest.mkdir(exist_ok=False)
    protocol_sha=digest(out/'protocol.json');references={};records=[]
    for c in p['cases']:
        key=c['reference'],c['reference_text'],c['reference_sha256']
        if key not in references:references[key]=fresh_reference(wrapper,c)
        prompt,ref=references[key];half=True
        torch.manual_seed(c['seed']);np.random.seed(c['seed'])
        with core.native(half),torch.inference_mode(),capture_generation(wrapper.model) as captured,\
                capture_inputs(wrapper.model.talker.model) as chunks:
            waves,sr=wrapper.generate_voice_clone(text=c['text'],language='Auto',voice_clone_prompt=prompt,
                non_streaming_mode=True,**p['generation'])
        require(sr==24000 and len(waves)==1 and len(waves[0])>0 and np.isfinite(waves[0]).all() and len(waves[0])/sr<=30,
            'Invalid confirmation waveform')
        path=dest/f"{c['seed']}-{c['case']}.wav";sf.write(path,waves[0],sr,subtype='PCM_16');wave,sr=sf.read(path,dtype='float32')
        with torch.inference_mode():
            vector=wrapper.model.extract_speaker_embedding(audio=wave,sr=sr).flatten()
            cosine=float(torch.nn.functional.cosine_similarity(vector[None],ref.speaker[None]))
        require(math.isfinite(cosine),'Nonfinite original-reference cosine')
        trace=dict(core.last_trace,layout=asdict(core.last_trace['layout']));trajectory=path.with_suffix('.pt')
        save(trajectory,dict(codes=captured['codes'],first_sequence=captured['first_sequence'],terminated=captured['terminated'],
            native_inputs=torch.cat(chunks,dim=1)[0],input_ids=ids(wrapper,c['text']).cpu(),ref_ids=ref.ref_ids,
            ref_codes=ref.ref_codes,ref_xvec=ref.speaker,reference_sha256=c['reference_sha256'],trace=trace))
        record=dict(c,wav=path.name,audio_sha256=digest(path),trajectory=trajectory.name,trajectory_sha256=digest(trajectory),
            cosine=cosine,seconds=len(wave)/sr,frames=len(captured['codes']),terminated=captured['terminated'],half_reference=half,
            step=p['step'],checkpoint_sha256=p['checkpoint_sha256'],protocol_sha256=protocol_sha,trace=trace)
        write(path.with_suffix('.json'),record);records.append(record);print(f'Original114 confirmation: {len(records)}/114',flush=True)
    unchanged(wrapper.model,core,state)
    write(dest/'generated.json',dict(records=records,all_weights_unchanged=True,
        checkpoint_sha256=p['checkpoint_sha256'],protocol_sha256=protocol_sha))


def score(out,protocol_sha):
    import soundfile as sf
    import torch,torchaudio
    from transformers import WhisperForConditionalGeneration,WhisperProcessor
    from .evaluate import ASR_REV
    from scripts.score_latest_audio import score as score_text
    out=Path(out);p,_,panel=verify(out,protocol_sha);dest=out/'speech'
    ablation.native_verify(Path(p['ablation']),ABLATION_SHA,p['step'])
    require(not (dest/'results.json').exists(),'No confirmation score overwrite')
    rows=bound_generation(out,p)['records'];manifest=read(p['asr_manifest']);asr=Path(p['config']['asr'])
    require(manifest['repo']=='BuzzASR/mongolian' and manifest['revision']==ASR_REV,'Original ASR required')
    for name,item in manifest['files'].items():require(digest(asr/name)==item['sha256'],'Original ASR file changed')
    processor=WhisperProcessor.from_pretrained(asr,local_files_only=True)
    model=WhisperForConditionalGeneration.from_pretrained(asr,torch_dtype=torch.float32,local_files_only=True).to('cuda').eval()
    for r in rows:
        wave,sr=sf.read(dest/r['wav'],dtype='float32');wave=torchaudio.functional.resample(torch.from_numpy(wave),sr,16000).numpy()
        batch=processor(wave,sampling_rate=16000,return_tensors='pt',return_attention_mask=True).to('cuda')
        with torch.inference_mode():tokens=model.generate(batch.input_features,attention_mask=batch.attention_mask,**p['decoding'])
        hypothesis=processor.batch_decode(tokens,skip_special_tokens=True)[0].strip()
        r.update(hypothesis=hypothesis,score=score_text(r['text'],hypothesis,True),asr_revision=ASR_REV,decoding=p['decoding'])
    write(dest/'results.json',dict(records=rows,**measurement(panel,rows)))


def audit(out,protocol_sha):
    import numpy as np
    import soundfile as sf
    import torch
    from .evaluate import ASR_REV
    from .reference_masked_supervision_speech_v2 import unchanged
    from .reference_masked_supervision_policy_v2 import movement
    out=Path(out);require(not (out/'verification.json').exists() and not (out/'completion.json').exists(),'No confirmation audit overwrite')
    p,_,panel,wrapper,core,state=load(out,protocol_sha);dest=out/'speech';report=read(dest/'results.json');rows=report['records']
    measured=measurement(panel,rows);bound_generation(out,p,rows);references={};maximum=0.
    for r in rows:
        require(r['asr_revision']==ASR_REV and r['decoding']==p['decoding'],'Confirmation ASR settings changed')
        t=torch.load(dest/r['trajectory'],map_location='cpu',weights_only=True)
        codes=t['codes'];seq=t['first_sequence'][0];eos=wrapper.model.config.talker_config.codec_eos_token_id
        require(codes.dtype==torch.long and seq.dtype==torch.long and codes.shape==(r['frames'],16) and len(codes)>0
            and bool(((codes>=0)&(codes<2048)).all()) and seq.shape==(len(codes)+1,)
            and torch.equal(seq[:-1],codes[:,0]) and bool(seq[-1]==eos)==r['terminated']==t['terminated']
            and t['reference_sha256']==r['reference_sha256'],'Confirmation native codec/EOS differs')
        key=r['reference'],r['reference_text'],r['reference_sha256']
        if key not in references:references[key]=fresh_reference(wrapper,r)[1]
        ref=references[key]
        require(all(t[k].dtype==torch.long for k in ('input_ids','ref_ids','ref_codes'))
            and t['ref_xvec'].dtype==t['native_inputs'].dtype==torch.float32
            and torch.isfinite(t['native_inputs']).all() and torch.isfinite(t['ref_xvec']).all()
            and torch.equal(t['input_ids'],ids(wrapper,r['text']).cpu()) and torch.equal(t['ref_ids'],ref.ref_ids)
            and torch.equal(t['ref_codes'],ref.ref_codes) and torch.equal(t['ref_xvec'],ref.speaker),
            'Independent original native reference conditioning differs')
        layout=core._layout(ref.ref_ids.shape[1]-5,t['input_ids'].shape[1]-8,len(ref.ref_codes))
        with torch.no_grad(),core._context('conditioning-audit',r['half_reference'],layout):
            rebuilt=core._build(t['input_ids'],codes,ref)
            require(torch.equal(rebuilt.embeddings,t['native_inputs']),'Actual original114 prefix/continuation inputs differ')
        require(t['trace']==r['trace']==dict(kind='native',half_reference=r['half_reference'],layout=asdict(layout),calls=1),
            'Original114 native route trace differs')
        wave,sr=sf.read(dest/r['wav'],dtype='float32')
        require(sr==24000 and wave.ndim==1 and len(wave)>0 and np.isfinite(wave).all()
            and r['seconds']==len(wave)/sr and r['seconds']<=30,'Invalid original114 waveform or duration')
        with torch.inference_mode():
            vector=wrapper.model.extract_speaker_embedding(audio=wave,sr=sr).flatten()
            cosine=float(torch.nn.functional.cosine_similarity(vector[None],ref.speaker[None]))
        require(math.isfinite(cosine) and math.isfinite(r['cosine']),'Nonfinite independent original-reference cosine')
        maximum=max(maximum,abs(cosine-r['cosine']))
    unchanged(wrapper.model,core,state)
    require(maximum<=2e-6 and all(report[k]==v for k,v in measured.items()) and movement(core)==p['adapter_movement'],
        'Independent original114 metrics or canonical adapter identity differ')
    receipt=dict(status='verified',step=p['step'],recordings=114,native_inputs_verified=114,**measured,
        frozen_weights_verified=True,native_conditioning_verified=True,all_artifact_hashes_verified=True,
        all_recomputed_cosines_finite=True,max_cosine_recomputation_error=maximum,
        protocol_sha256=digest(out/'protocol.json'),checkpoint_sha256=p['checkpoint_sha256'],panel_sha256=PANEL_SHA,
        results_sha256=digest(dest/'results.json'),generated_sha256=digest(dest/'generated.json'),
        frozen_identity=p['frozen_identity'],adapter_movement=p['adapter_movement'],
        optimizer_updates=0,goal_achieved=False,global_model_selection_changed=False)
    complete_audit(receipt);write(out/'verification.json',receipt)



def verified_result(out,protocol_sha):
    """Read-only final bindings; root still confirms the confirmation driver exit."""
    from .evaluate import ASR_REV
    out=Path(out);p,_,panel=verify(out,protocol_sha);report=read(out/'speech/results.json');receipt=read(out/'verification.json')
    bound_generation(out,p,report['records']);measured=measurement(panel,report['records'])
    require(all(receipt[k]==report[k]==v for k,v in measured.items())
        and receipt['step']==p['step'] and receipt['protocol_sha256']==digest(out/'protocol.json')
        and receipt['checkpoint_sha256']==p['checkpoint_sha256'] and receipt['panel_sha256']==PANEL_SHA
        and receipt['results_sha256']==digest(out/'speech/results.json') and receipt['generated_sha256']==digest(out/'speech/generated.json')
        and receipt['frozen_identity']==p['frozen_identity'] and receipt['adapter_movement']==p['adapter_movement']
        and receipt['optimizer_updates']==0 and receipt['goal_achieved'] is False and receipt['global_model_selection_changed'] is False
        and all(r['asr_revision']==ASR_REV and r['decoding']==p['decoding'] for r in report['records']),
        'Complete confirmation provenance differs')
    passed=complete_audit(receipt)
    return receipt


def complete(out,protocol_sha,closure_path):
    """Root-only post-exit receipt; never called by the still-running driver."""
    out=Path(out);p,_,_=verify(out,protocol_sha);audit=verified_result(out,protocol_sha)
    require(audit['confirmation_pass'] is True,'Failed114 cannot create a completion receipt')
    status_path=out.parent/(out.name+'-status.json');status=read(status_path)
    bindings=dict(protocol_sha256=protocol_sha,driver_sha256=p['driver_sha256'],
        terminal_status_sha256=digest(status_path),verification_sha256=digest(out/'verification.json'),
        results_sha256=digest(out/'speech/results.json'))
    require(Path(closure_path).resolve()==out.parent/(out.name+'-process-closed.json'),
        'Canonical root-authored confirmation closure required')
    validate_closure(status,read(closure_path),bindings,None)
    require(status['step']==32 and status['confirmation_pass'] is True
        and status['protocol_sha256']==protocol_sha and status['verification_sha256']==bindings['verification_sha256']
        and status['summary']==audit['summary'] and status['truncated']==0
        and set(status['logs'])=={'generate','score','audit','verify'},'Terminal confirmation evidence differs')
    for item in status['logs'].values():
        require(type(item['exit_code']) is int and item['exit_code']==0
            and digest(item['path'])==item['sha256'],'Confirmation phase log differs')
    write(out/'completion.json',dict(status='verified_complete',confirmation_pass=True,recordings=114,
        protocol_sha256=protocol_sha,verification_sha256=bindings['verification_sha256'],
        process_closed_sha256=digest(closure_path),terminal_status_sha256=bindings['terminal_status_sha256'],
        checkpoint_sha256=p['checkpoint_sha256'],panel_sha256=PANEL_SHA,**FLAGS))


def main():
    import argparse,os
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('prepare','generate','score','audit','verify','complete'))
    parser.add_argument('--out',required=True,type=Path);parser.add_argument('--ablation',type=Path)
    parser.add_argument('--driver',type=Path);parser.add_argument('--test-log',type=Path);parser.add_argument('--note',type=Path)
    parser.add_argument('--protocol-sha');parser.add_argument('--closure',type=Path);args=parser.parse_args()
    if args.phase=='prepare':
        require(all((args.ablation,args.driver,args.test_log,args.note)),'Preparation bindings required')
        prepare(args.ablation,args.out,args.driver,args.test_log,args.note);return
    require(args.protocol_sha is not None,'Explicit registered protocol hash required')
    p,_,_=verify(args.out,args.protocol_sha)
    if args.phase=='verify':verified_result(args.out,args.protocol_sha);return
    if args.phase=='complete':
        require(args.closure is not None,'Root-authored closure required');complete(args.out,args.protocol_sha,args.closure);return
    os.environ['NUMBA_CACHE_DIR']=p['config']['numba_cache'];os.environ['TEMP']=os.environ['TMP']=p['config']['tmp']
    os.environ['HF_HUB_OFFLINE']=os.environ['TRANSFORMERS_OFFLINE']='1'
    import torch
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    write(args.out/f'{args.phase}-started.json',dict(protocol_sha256=args.protocol_sha))
    {'generate':generate,'score':score,'audit':audit}[args.phase](args.out,args.protocol_sha)


if __name__=='__main__':main()
