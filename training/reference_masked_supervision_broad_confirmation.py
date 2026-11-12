"""Conditional original114 confirmation of a closed broad-data checkpoint.

The original panel, routes and measurement helpers are unchanged. This module
only replaces the parent evidence/loader binding; no global monkeypatching.
"""
from dataclasses import asdict
import math
from pathlib import Path
from .engine import read,write,save,digest,ids
from .reference_masked_supervision_evaluation import progression,require
from . import reference_masked_supervision_confirmation as original

PANEL_SHA,QUALIFICATION=original.PANEL_SHA,original.QUALIFICATION
expanded_panel,measurement=original.expanded_panel,original.measurement
bound_generation,fresh_reference,complete_audit=original.bound_generation,original.fresh_reference,original.complete_audit
TEST_LOG='reference-masked-supervision-broad-confirmation-tests-v1.log'


def sources():
    result=dict(original.sources());folder=Path(__file__).parent
    for name in ('reference_masked_supervision_broad_confirmation.py',
                 '../tests/test_reference_masked_supervision_broad_confirmation.py',
                 'reference_masked_supervision_broad_training.py','reference_masked_supervision_broad_speech.py'):
        path=(folder/name).resolve();result[str(path)]=digest(path)
    return result


def validate_parent(status,closure,history,step,bindings):
    require(step in (8,16,32),'Positive registered checkpoint required')
    decision=progression(history)
    require(decision['stop'] and decision['reason']==QUALIFICATION and decision['step']==step,
        'Positive complete80 strict two-group qualification required')
    require(status['status']=='complete' and status['phase']=='complete' and status['step']==step
        and status['progression']==decision and status['summary']==history[-1]['summary']
        and status['original114_confirmation_required'] is True and status['goal_achieved'] is False
        and status['global_model_selection_changed'] is False
        and status['verification_sha256']==bindings['verification_sha256'] and status['inputs_sha256']==bindings['inputs_sha256'],
        'Terminal qualified broad-data driver required')
    require(closure['status']=='closed' and type(closure['exit_code']) is int and closure['exit_code']==0
        and type(closure['original_session_id']) is int and closure['original_session_id']==41410 and closure['step']==step
        and all(closure.get(k)==v for k,v in bindings.items()), 'Authoritative successful41410 process-closed receipt required')
    return decision


def phases_for(step):
    require(step in (8,16,32),'Positive registered checkpoint required')
    phases={'memory-preflight','prepare','speech-prepare','initialize','native-controls','reuse-baseline'}
    phases.update(f'{name}-{s:06d}' for s in (8,16,32) if s<=step for name in ('train','generate','score','audit'))
    return phases


def evidence(train_run,step):
    from . import reference_masked_supervision_broad_training as trainer
    from . import reference_masked_supervision_broad_speech as speech
    from .evaluate import ASR_REV
    train_run=Path(train_run).resolve()
    require(train_run.name=='reference-masked-supervision-broad-training-v1','Explicit registered broader pilot required')
    root=train_run.parent;status_path=root/(train_run.name+'-status.json')
    closure_path=root/(train_run.name+'-process-closed.json');inputs_path=root/(train_run.name+'-inputs.json')
    status,closure=read(status_path),read(closure_path)
    require(status['status']=='complete' and closure['status']=='closed' and closure['exit_code']==0,
        'Closed successful parent required before confirmation preparation')
    p,e=speech.verify(train_run);speech.permitted(train_run,p,e,step);state=trainer.read_checkpoint(train_run,step)
    expected=[s for s in trainer.CHECKPOINTS if s<=step]
    history=[speech.verified_receipt(train_run,p,e,s) for s in expected]
    bindings=dict(protocol_sha256=digest(train_run/'protocol.json'),driver_sha256=digest(p['driver']),
        terminal_status_sha256=digest(status_path),inputs_sha256=digest(inputs_path),
        verification_sha256=digest(train_run/f'verification-{step:06d}.json'),
        checkpoint_sha256=digest(trainer.checkpoint_path(train_run,step)))
    validate_parent(status,closure,history,step,bindings)
    require(state['adapter_movement']['maximum']>0 and state['adapter_movement']==history[-1]['adapter_movement'],
        'Positive canonical adapter movement required')
    require({q.name for q in train_run.glob('checkpoint-*.pt')}=={f'checkpoint-{s:06d}.pt' for s in expected}
        and {q.name for q in train_run.glob('update-*.json')}=={f'update-{s:06d}.json' for s in range(1,step+1)},
        'Fitting continued outside qualified checkpoint history')
    require(history[0]['measurement_origin']=='shared_original_v2_checkpoint0'
        and history[0]['fresh_recordings']==0 and history[0]['reused_recordings']==80,
        'Explicit independently rebound shared zero80 required')
    hashes=dict(read(inputs_path)['files'])
    for path,sha in hashes.items():require(digest(path)==sha,'Parent driver input changed')
    require(set(status['logs'])==phases_for(step),'Complete terminal broad phase history required')
    for item in status['logs'].values():
        require(item['exit_code']==0 and digest(item['path'])==item['sha256'],'Parent phase log failed or changed')
        hashes[item['path']]=item['sha256']
    paths=[status_path,closure_path,inputs_path,Path(p['driver']),train_run/'protocol.json',train_run/'speech-protocol.json',
        train_run/'native-verification.json',Path(p['memory_receipt']),Path(p['plan']),
        Path(p['features'])/'protocol.json',Path(p['features'])/'verification.json']
    for s in expected:
        paths.extend((trainer.checkpoint_path(train_run,s),trainer.checkpoint_path(train_run,s).with_suffix('.json'),
            train_run/f'verification-{s:06d}.json',train_run/f'teacher-{s:06d}.json'))
        if s:paths.extend((train_run/f'eval-{s:06d}/generated.json',train_run/f'eval-{s:06d}/results.json'))
    previous=Path(p['comparator']['train'])
    paths.extend((previous/'protocol.json',previous/'speech-protocol.json',previous/'checkpoint-000000.pt',
        previous/'verification-000000.json',previous/'eval-000000/generated.json',previous/'eval-000000/results.json'))
    paths.extend(train_run/f'update-{s:06d}.json' for s in range(1,step+1))
    panel_path=root/'joint-goal-confirmation/protocol.json'
    require(digest(panel_path)==PANEL_SHA,'Original registered114 panel changed')
    panel=read(panel_path);cases=expanded_panel(panel,p['custom_reference']);baseline=read(Path(p['policy']['run'])/'protocol.json')
    require(cases==baseline['cases'] and panel['generation']==p['generation']==baseline['generation']
        and panel['decoding']==p['decoding']==baseline['decoding']
        and panel['preparation_source_sha256']==digest(Path(__file__).with_name('joint_confirmation.py')),
        'Original114 metadata or measurement settings changed')
    manifest_path=Path(p['config']['asr'])/'download-manifest.json';manifest=read(manifest_path)
    require(manifest['repo']=='BuzzASR/mongolian' and manifest['revision']==ASR_REV
        and digest(manifest_path)==e['asr_manifest_sha256'],'Original ASR identity changed')
    for c in cases:require(digest(c['reference'])==c['reference_sha256'],'Original reference WAV changed')
    paths.extend((panel_path,manifest_path))
    for path in paths:hashes[str(path.resolve())]=digest(path)
    return p,panel,cases,state,hashes


def test_evidence(root):
    path=Path(root)/TEST_LOG;text=path.read_text(encoding='utf-8-sig')
    require('Ran 8 tests' in text and text.strip().endswith('OK') and 'skipped' not in text.lower() and 'FAILED' not in text,
        'Eight complete confirmation tests required')
    return path,digest(path)


def prepare(train_run,step,out,driver):
    from .policy import external
    from .preference_pilot import runtime
    from . import reference_masked_supervision_broad_training as trainer
    train_run=Path(train_run).resolve();out=external(Path(out).resolve());driver=Path(driver).resolve()
    require(trainer.numeric_config()==trainer.NUMERIC,'Original FP32 numerical settings required')
    p,panel,cases,state,hashes=evidence(train_run,step);tests,test_sha=test_evidence(out.parent)
    require(out.parent==train_run.parent and out.name=='reference-masked-supervision-broad-confirmation-v1'
        and not out.exists(),'Fresh external broad confirmation required')
    protocol=dict(train_run=str(train_run),step=step,evidence=hashes,source=sources(),runtime=runtime(),
        panel=str(train_run.parent/'joint-goal-confirmation/protocol.json'),panel_sha256=PANEL_SHA,cases=cases,
        config=p['config'],generation=p['generation'],decoding=p['decoding'],reference_sha256=p['reference_sha256'],
        checkpoint_sha256=digest(trainer.checkpoint_path(train_run,step)),frozen_identity=state['frozen_identity'],
        adapter_movement=state['adapter_movement'],numeric_config=trainer.NUMERIC,
        asr_manifest=str(Path(p['config']['asr'])/'download-manifest.json'),
        asr_manifest_sha256=digest(Path(p['config']['asr'])/'download-manifest.json'),
        tests=str(tests),tests_sha256=test_sha,driver=str(driver),driver_sha256=digest(driver),
        scope='Original114 unchanged:88development/26custom,10/1speakers,seeds2026/2027,full-development/half-custom,full-PAD',
        goal_achieved=False,global_model_selection_changed=False,optimizer_updates=0)
    out.mkdir();write(out/'protocol.json',protocol);return protocol


def verify(out):
    from .preference_pilot import runtime
    from . import reference_masked_supervision_broad_training as trainer
    out=Path(out);p=read(out/'protocol.json')
    require(p['source']==sources() and p['runtime']==runtime() and p['numeric_config']==trainer.numeric_config()==trainer.NUMERIC
        and digest(p['tests'])==p['tests_sha256'] and digest(p['driver'])==p['driver_sha256'],
        'Confirmation execution registration changed')
    original,panel,cases,state,hashes=evidence(Path(p['train_run']),p['step'])
    require(p['evidence']==hashes and p['cases']==cases and p['panel_sha256']==digest(p['panel'])==PANEL_SHA
        and Path(p['panel']).resolve()==Path(p['train_run']).parent/'joint-goal-confirmation/protocol.json'
        and all(p[k]==original[k] for k in ('config','generation','decoding','reference_sha256'))
        and p['frozen_identity']==state['frozen_identity'] and p['adapter_movement']==state['adapter_movement']
        and p['checkpoint_sha256']==digest(trainer.checkpoint_path(p['train_run'],p['step']))
        and Path(p['asr_manifest']).resolve()==(Path(p['config']['asr'])/'download-manifest.json').resolve()
        and p['asr_manifest_sha256']==digest(p['asr_manifest']), 'Confirmation candidate or original panel changed')
    return p,original,panel


def load(out):
    from . import reference_masked_supervision_broad_training as trainer
    from . import reference_masked_supervision_broad_features as features
    p,original,panel=verify(out)
    _,wrapper,core,state=trainer.load(p['train_run'],p['step'],device='cpu',training=False)
    features.assert_native_components(wrapper,original['features'])
    return p,original,panel,wrapper,core,state


def generate(out):
    import numpy as np
    import soundfile as sf
    import torch
    from .preference_rollouts import capture_generation
    from .reference_masked_supervision_speech_v2 import capture_inputs,unchanged
    out=Path(out);p,_,_,wrapper,core,state=load(out);dest=out/'speech';dest.mkdir(exist_ok=False)
    protocol_sha=digest(out/'protocol.json');references={};records=[]
    for c in p['cases']:
        key=c['reference'],c['reference_text'],c['reference_sha256']
        if key not in references:references[key]=fresh_reference(wrapper,c)
        prompt,ref=references[key];half=c['reference_sha256']==p['reference_sha256']
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


def score(out):
    import soundfile as sf
    import torch,torchaudio
    from transformers import WhisperForConditionalGeneration,WhisperProcessor
    from .evaluate import ASR_REV
    from scripts.score_latest_audio import score as score_text
    out=Path(out);p,_,panel=verify(out);dest=out/'speech'
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


def audit(out):
    import numpy as np
    import soundfile as sf
    import torch
    from .evaluate import ASR_REV
    from .reference_masked_supervision_speech_v2 import unchanged
    from .reference_masked_supervision_policy_v2 import movement
    out=Path(out);require(not (out/'verification.json').exists() and not (out/'completion.json').exists(),'No confirmation audit overwrite')
    p,_,panel,wrapper,core,state=load(out);dest=out/'speech';report=read(dest/'results.json');rows=report['records']
    measured=measurement(panel,rows);bound_generation(out,p,rows);references={};maximum=0.
    for r in rows:
        require(r['asr_revision']==ASR_REV and r['decoding']==p['decoding'],'Confirmation ASR settings changed')
        t=torch.load(dest/r['trajectory'],map_location='cpu',weights_only=True)
        codes=t['codes'];seq=t['first_sequence'][0];eos=wrapper.model.config.talker_config.codec_eos_token_id
        require(codes.dtype==torch.long and codes.shape==(r['frames'],16) and len(codes)>0
            and bool(((codes>=0)&(codes<2048)).all()) and seq.shape==(len(codes)+1,)
            and torch.equal(seq[:-1],codes[:,0]) and bool(seq[-1]==eos)==r['terminated']==t['terminated']
            and t['reference_sha256']==r['reference_sha256'],'Confirmation native codec/EOS differs')
        key=r['reference'],r['reference_text'],r['reference_sha256']
        if key not in references:references[key]=fresh_reference(wrapper,r)[1]
        ref=references[key]
        require(torch.equal(t['input_ids'],ids(wrapper,r['text']).cpu()) and torch.equal(t['ref_ids'],ref.ref_ids)
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
    if receipt['confirmation_pass']:
        write(out/'completion.json',dict(status='verified_complete',confirmation_pass=True,recordings=114,
            protocol_sha256=digest(out/'protocol.json'),verification_sha256=digest(out/'verification.json'),
            checkpoint_sha256=p['checkpoint_sha256'],panel_sha256=PANEL_SHA,
            goal_achieved=False,global_model_selection_changed=False,optimizer_updates=0))


def verified_result(out):
    """Read-only final bindings; root still confirms the confirmation driver exit."""
    from .evaluate import ASR_REV
    out=Path(out);p,_,panel=verify(out);report=read(out/'speech/results.json');receipt=read(out/'verification.json')
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
    if passed:
        completion=read(out/'completion.json')
        require(completion==dict(status='verified_complete',confirmation_pass=True,recordings=114,
            protocol_sha256=digest(out/'protocol.json'),verification_sha256=digest(out/'verification.json'),
            checkpoint_sha256=p['checkpoint_sha256'],panel_sha256=PANEL_SHA,
            goal_achieved=False,global_model_selection_changed=False,optimizer_updates=0),'Confirmation completion receipt differs')
    else:require(not (out/'completion.json').exists(),'Failed confirmation cannot have a completion receipt')
    return receipt


def main():
    import argparse,os
    from . import reference_masked_supervision_broad_training as trainer
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=('prepare','generate','score','audit','verify'))
    parser.add_argument('--out',required=True,type=Path);parser.add_argument('--train-run',type=Path)
    parser.add_argument('--step',type=int);parser.add_argument('--driver',type=Path);args=parser.parse_args()
    config=read((args.train_run if args.phase=='prepare' else args.out)/'protocol.json')['config']
    os.environ['NUMBA_CACHE_DIR']=config['numba_cache'];os.environ['TEMP']=os.environ['TMP']=config['tmp'];trainer.configure()
    if args.phase=='prepare':prepare(args.train_run,args.step,args.out,args.driver)
    else:{'generate':generate,'score':score,'audit':audit,'verify':verified_result}[args.phase](args.out)


if __name__=='__main__':main()
