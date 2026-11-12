"""Native 80-recording calibration generation, ASR and independent voice audit.

Training and features have their own registrations. This module never trains
or promotes a model; a passing calibration checkpoint still needs original114.
"""
from contextlib import contextmanager
from dataclasses import asdict
import math
from pathlib import Path

from .engine import read, write, save, digest, ids
from .reference_masked_supervision_evaluation import calibration_cases, validate_records, verified_summary, progression, require


def sources():
    return {name: digest(Path(__file__).parent / name) for name in (
        'reference_masked_supervision_speech.py', 'reference_masked_supervision_evaluation.py',
        '../tests/test_reference_masked_supervision_evaluation.py',
        '../tests/test_reference_masked_supervision_speech.py', 'engine.py', 'policy.py',
        'preference_rollouts.py', 'targeted_voice.py', 'evaluate.py',
        '../scripts/score_latest_audio.py', '../data/pipeline.py')}


def prepare(run):
    from . import reference_masked_supervision_training as trainer
    run = Path(run); p = trainer.verify(run)
    path = run / 'speech-protocol.json'
    require(not path.exists(), 'Do not overwrite registered calibration')
    plan = read(p['plan'])
    cases = calibration_cases(plan, p['calibration'], p['custom_reference'])
    manifest = Path(p['config']['asr']) / 'download-manifest.json'
    write(path, dict(training_protocol_sha256=digest(run/'protocol.json'), source=sources(),
        plan_sha256=digest(p['plan']), features_audit_sha256=digest(Path(p['features'])/'verification.json'),
        cases=cases, checkpoints=[0,8,16,32], asr_manifest=str(manifest), asr_manifest_sha256=digest(manifest),
        generation=p['generation'], decoding=p['decoding'], goal_achieved=False,
        scope='Complete80 calibration only; qualifying positive update requires unchanged original114 confirmation'))


def verify(run):
    import torch
    from . import reference_masked_supervision_training as trainer
    require(torch.get_num_threads()==4 and not torch.backends.cuda.matmul.allow_tf32
        and not torch.backends.cudnn.allow_tf32, 'Registered four-thread FP32 numerical settings required')
    run = Path(run); p = trainer.verify(run); e = read(run/'speech-protocol.json')
    require(e['training_protocol_sha256'] == digest(run/'protocol.json') and e['source'] == sources()
        and e['plan_sha256'] == digest(p['plan'])
        and e['features_audit_sha256'] == digest(Path(p['features'])/'verification.json')
        and e['asr_manifest_sha256'] == digest(e['asr_manifest']), 'Calibration registration changed')
    require(e['cases'] == calibration_cases(read(p['plan']), p['calibration'], p['custom_reference'])
        and e['checkpoints'] == [0,8,16,32] and e['generation'] == p['generation']
        and e['decoding'] == p['decoding'], 'Calibration conditions changed')
    return p, e


def permitted(run, p, e, step):
    from . import reference_masked_supervision_training as trainer
    require(step in e['checkpoints'], 'Unregistered speech checkpoint')
    native = read(run/'native-verification.json')
    require(native['status'] == 'verified' and native['protocol_sha256'] == digest(run/'protocol.json')
        and native['checkpoint_sha256'] == digest(trainer.checkpoint_path(run,0))
        and native['features_audit_sha256'] == e['features_audit_sha256'], 'Production native controls required')
    return trainer.read_checkpoint(run,step)


def directory(run, step):
    return Path(run)/f'eval-{step:06d}'


def diagnostic_bindings(run, p, step):
    """Bind logged training diagnostics; this does not independently rerun them."""
    from . import reference_masked_supervision_training as trainer
    teacher_path=run/f'teacher-{step:06d}.json'; teacher=read(teacher_path)
    protocol_sha=digest(run/'protocol.json')
    require(teacher['step']==step and teacher['protocol_sha256']==protocol_sha
        and teacher['checkpoint_sha256']==digest(trainer.checkpoint_path(run,step))
        and teacher['frozen_identity']==p['frozen_identity'], 'Teacher diagnostic provenance differs')
    updates={}
    for number in range(1,step+1):
        path=run/f'update-{number:06d}.json'; record=read(path)
        require(record['step']==number and record['protocol_sha256']==protocol_sha
            and record['frozen_identity']==p['frozen_identity'], 'Update diagnostic provenance differs')
        updates[str(number)]=digest(path)
    return dict(teacher_sha256=digest(teacher_path),update_sha256=updates)


def bound_generation(run, e, step, scored=None):
    from . import reference_masked_supervision_training as trainer
    dest=directory(run,step); generated=read(dest/'generated.json')
    checkpoint_sha=digest(trainer.checkpoint_path(run,step)); speech_sha=digest(run/'speech-protocol.json')
    require(generated['all_weights_unchanged'] is True and generated['checkpoint_sha256']==checkpoint_sha
        and generated['speech_protocol_sha256']==speech_sha, 'Generation receipt changed')
    expected={(c['case'],c['seed']):c for c in e['cases']}; records=generated['records']
    require(len(records)==80 and {(r['case'],r['seed']) for r in records}==set(expected), 'Incomplete generation receipt')
    current={} if scored is None else {(r['case'],r['seed']):r for r in scored}
    if scored is not None: require(len(scored)==80 and set(current)==set(expected), 'Scored receipt coverage differs')
    for r in records:
        key=r['case'],r['seed']
        require(all(r.get(k)==v for k,v in expected[key].items()) and r['step']==step
            and r['checkpoint_sha256']==checkpoint_sha and r['speech_protocol_sha256']==speech_sha
            and digest(dest/r['wav'])==r['audio_sha256'] and digest(dest/r['trajectory'])==r['trajectory_sha256']
            and digest(r['reference'])==r['reference_sha256'], 'Generation artifact or metadata changed')
        if scored is not None:
            require(set(current[key])==set(r)|{'hypothesis','score','asr_revision','decoding'}
                and all(current[key].get(k)==v for k,v in r.items()), 'Scoring changed generated record fields')
    return generated


def verified_receipt(run, p, e, step):
    """Recheck full prior evidence before a quality decision or further fitting."""
    from . import reference_masked_supervision_training as trainer
    from .evaluate import ASR_REV
    run=Path(run); state=trainer.read_checkpoint(run,step)
    audit=read(run/f'verification-{step:06d}.json'); dest=directory(run,step)
    report=read(dest/'results.json'); records=report['records']
    require(audit['step']==step and audit['checkpoint_sha256']==digest(trainer.checkpoint_path(run,step))
        and audit['protocol_sha256']==digest(run/'protocol.json')
        and audit['speech_protocol_sha256']==digest(run/'speech-protocol.json')
        and audit['results_sha256']==digest(dest/'results.json')
        and audit['generated_sha256']==digest(dest/'generated.json')
        and audit['native_verification_sha256']==digest(run/'native-verification.json')
        and audit['features_audit_sha256']==e['features_audit_sha256']
        and audit['native_inputs_verified']==80 and audit['frozen_identity']==state['frozen_identity']==p['frozen_identity']
        and audit['adapter_movement']==state['adapter_movement'] and audit['goal_achieved'] is False,
        'Prior native audit provenance changed')
    bound_generation(run,e,step,records)
    summary=validate_records(e['cases'],records)
    require(summary==report['summary']==audit['summary']
        and sum(not r['terminated'] for r in records)==report['truncated']==audit['truncated']
        and all(r['asr_revision']==ASR_REV and r['decoding']==e['decoding'] for r in records), 'Prior score evidence differs')
    verified_summary(audit)
    require(all(audit.get(k)==v for k,v in diagnostic_bindings(run,p,step).items()), 'Bound training diagnostics changed')
    return audit


def unchanged(model, core, state):
    import torch
    from . import reference_masked_supervision_policy as policy
    require(policy.frozen_identity(model,core) == state['frozen_identity'], 'Frozen policy changed')
    current = policy.snapshot(core)
    require(set(current) == set(state['adapter']) and all(torch.equal(current[k],state['adapter'][k].cpu()) for k in current),
        'Checkpoint adapter changed during speech')
    require(core.scope is None and all(v.grad is None for v in model.parameters()), 'Leaked generation state or gradients')


@contextmanager
def capture_inputs(decoder):
    chunks = []
    def capture(module, args, kwargs):
        require(not args and 'inputs_embeds' in kwargs, 'Unexpected native decoder invocation')
        chunks.append(kwargs['inputs_embeds'].detach().cpu().clone())
    handle = decoder.register_forward_pre_hook(capture, with_kwargs=True)
    try:
        yield chunks
    finally:
        handle.remove()


def generate(run, step):
    import numpy as np
    import soundfile as sf
    import torch
    from . import reference_masked_supervision_training as trainer
    from . import reference_masked_supervision_features as features
    from .preference_rollouts import capture_generation
    run=Path(run); p,e=verify(run); permitted(run,p,e,step)
    _,wrapper,core,state=trainer.load(run,step,device='cpu',training=False)
    features.assert_native_components(wrapper,p['features'])
    _,assets=features.load_features(p['features'])
    dest=directory(run,step); dest.mkdir(exist_ok=True)
    require(not (dest/'generated.json').exists(), 'Complete generation already exists')
    checkpoint_sha=digest(trainer.checkpoint_path(run,step)); protocol_sha=digest(run/'speech-protocol.json')
    prompts={}; records=[]
    for c in e['cases']:
        path=dest/f"{c['seed']}-{c['case']}.wav"; meta=path.with_suffix('.json')
        if meta.exists():
            r=read(meta)
            require(all(r.get(k)==v for k,v in c.items()) and r['checkpoint_sha256']==checkpoint_sha
                and r['speech_protocol_sha256']==protocol_sha and digest(path)==r['audio_sha256']
                and digest(dest/r['trajectory'])==r['trajectory_sha256'], 'Partial generation changed')
            records.append(r); continue
        require(not path.exists() and not path.with_suffix('.pt').exists(), 'Orphan generation artifact requires inspection')
        require(digest(c['reference'])==c['reference_sha256'], 'Reference WAV changed')
        asset=assets[c['feature_id']]
        if c['reference'] not in prompts:
            with torch.inference_mode():
                prompt=wrapper.create_voice_clone_prompt(ref_audio=c['reference'],ref_text=c['reference_text'],x_vector_only_mode=False)
            require(torch.equal(prompt[0].ref_code.cpu(),asset['codes'])
                and torch.equal(prompt[0].ref_spk_embedding.flatten().cpu(),asset['speaker'])
                and torch.equal(ids(wrapper,c['reference_text'],True).cpu(),asset['ref_ids']), 'Fresh reference feature parity failed')
            prompts[c['reference']]=prompt
        prompt=prompts[c['reference']]
        torch.manual_seed(c['seed']); np.random.seed(c['seed'])
        with core.native(c['half_reference']), torch.inference_mode(), capture_generation(wrapper.model) as captured, \
                capture_inputs(wrapper.model.talker.model) as chunks:
            waves,sr=wrapper.generate_voice_clone(text=c['text'],language='Auto',voice_clone_prompt=prompt,
                non_streaming_mode=True,**e['generation'])
        require(sr==24000 and len(waves)==1 and len(waves[0])>0 and np.isfinite(waves[0]).all()
            and len(waves[0])/sr<=30, 'Invalid generated waveform')
        sf.write(path,waves[0],sr,subtype='PCM_16'); wave,sr=sf.read(path,dtype='float32')
        with torch.inference_mode():
            vector=wrapper.model.extract_speaker_embedding(audio=wave,sr=sr).flatten()
            cosine=float(torch.nn.functional.cosine_similarity(vector[None],asset['speaker'][None]))
        require(math.isfinite(cosine), 'Nonfinite generated voice cosine')
        trace=dict(core.last_trace,layout=asdict(core.last_trace['layout']))
        trajectory=path.with_suffix('.pt')
        save(trajectory,dict(codes=captured['codes'],first_sequence=captured['first_sequence'],
            terminated=captured['terminated'],native_inputs=torch.cat(chunks,dim=1)[0],
            input_ids=ids(wrapper,c['text']).cpu(),ref_ids=asset['ref_ids'],ref_codes=asset['codes'],
            ref_xvec=asset['speaker'],reference_sha256=c['reference_sha256'],trace=trace))
        r=dict(c,wav=path.name,audio_sha256=digest(path),trajectory=trajectory.name,trajectory_sha256=digest(trajectory),
            cosine=cosine,seconds=len(wave)/sr,frames=len(captured['codes']),terminated=captured['terminated'],
            step=step,checkpoint_sha256=checkpoint_sha,speech_protocol_sha256=protocol_sha,trace=trace)
        write(meta,r); records.append(r); print(f'checkpoint {step}: {len(records)}/80',flush=True)
    unchanged(wrapper.model,core,state)
    require(len(records)==80, 'Incomplete calibration generation')
    write(dest/'generated.json',dict(records=records,all_weights_unchanged=True,checkpoint_sha256=checkpoint_sha,
        speech_protocol_sha256=protocol_sha))


def score(run, step):
    import soundfile as sf
    import torch, torchaudio
    from transformers import WhisperForConditionalGeneration,WhisperProcessor
    from .evaluate import ASR_REV
    from scripts.score_latest_audio import score as score_text
    from . import reference_masked_supervision_training as trainer
    run=Path(run); p,e=verify(run); permitted(run,p,e,step); dest=directory(run,step)
    require(not (dest/'results.json').exists(), 'Do not overwrite scored calibration')
    generated=bound_generation(run,e,step); records=generated['records']
    manifest=read(e['asr_manifest']); asr=Path(p['config']['asr'])
    require(manifest['repo']=='BuzzASR/mongolian' and manifest['revision']==ASR_REV, 'Original ASR required')
    for name,item in manifest['files'].items(): require(digest(asr/name)==item['sha256'], 'ASR file changed')
    processor=WhisperProcessor.from_pretrained(asr,local_files_only=True)
    model=WhisperForConditionalGeneration.from_pretrained(asr,torch_dtype=torch.float32,local_files_only=True).to('cuda').eval()
    for r in records:
        require(digest(dest/r['wav'])==r['audio_sha256'], 'Generated waveform changed')
        wave,sr=sf.read(dest/r['wav'],dtype='float32')
        wave=torchaudio.functional.resample(torch.from_numpy(wave),sr,16000).numpy()
        batch=processor(wave,sampling_rate=16000,return_tensors='pt',return_attention_mask=True).to('cuda')
        with torch.inference_mode(): tokens=model.generate(batch.input_features,attention_mask=batch.attention_mask,**e['decoding'])
        hypothesis=processor.batch_decode(tokens,skip_special_tokens=True)[0].strip()
        r.update(hypothesis=hypothesis,score=score_text(r['text'],hypothesis,True),asr_revision=ASR_REV,decoding=e['decoding'])
    summary=validate_records(e['cases'],records)
    write(dest/'results.json',dict(records=records,summary=summary,truncated=sum(not r['terminated'] for r in records)))


def audit(run, step):
    import numpy as np
    import soundfile as sf
    import torch
    from .evaluate import ASR_REV
    from . import reference_masked_supervision_training as trainer
    from . import reference_masked_supervision_policy as policy
    from . import reference_masked_supervision_features as features
    from .reference_masked_supervision import NativeReference
    run=Path(run); p,e=verify(run); permitted(run,p,e,step); dest=directory(run,step)
    path=run/f'verification-{step:06d}.json'; require(not path.exists(), 'Do not overwrite native speech audit')
    _,wrapper,core,state=trainer.load(run,step,device='cpu',training=False)
    features.assert_native_components(wrapper,p['features']); _,assets=features.load_features(p['features'])
    report=read(dest/'results.json'); records=report['records']; summary=validate_records(e['cases'],records)
    bound_generation(run,e,step,records)
    checkpoint_sha=digest(trainer.checkpoint_path(run,step)); speech_sha=digest(run/'speech-protocol.json')
    maximum=0.; references={}
    for r in records:
        require(r['step']==step and r['checkpoint_sha256']==checkpoint_sha and r['speech_protocol_sha256']==speech_sha
            and r['asr_revision']==ASR_REV and r['decoding']==e['decoding']
            and digest(dest/r['wav'])==r['audio_sha256'] and digest(dest/r['trajectory'])==r['trajectory_sha256']
            and digest(r['reference'])==r['reference_sha256'], 'Speech artifact identity changed')
        t=torch.load(dest/r['trajectory'],map_location='cpu',weights_only=True)
        codes=t['codes']; seq=t['first_sequence'][0]; eos=wrapper.model.config.talker_config.codec_eos_token_id
        require(codes.dtype==torch.long and codes.shape==(r['frames'],16) and len(codes)>0
            and bool(((codes>=0)&(codes<2048)).all()) and seq.shape==(len(codes)+1,)
            and torch.equal(seq[:-1],codes[:,0]) and bool(seq[-1]==eos)==r['terminated']==t['terminated']
            and t['reference_sha256']==r['reference_sha256'], 'Native codec/termination record differs')
        asset=assets[r['feature_id']]
        require(torch.equal(t['input_ids'],ids(wrapper,r['text']).cpu())
            and torch.equal(t['ref_ids'],ids(wrapper,r['reference_text'],True).cpu())
            and torch.equal(t['ref_ids'],asset['ref_ids']) and torch.equal(t['ref_codes'],asset['codes'])
            and torch.equal(t['ref_xvec'],asset['speaker']), 'Native conditioning feature differs')
        if r['reference'] not in references:
            with torch.inference_mode():
                prompt=wrapper.create_voice_clone_prompt(ref_audio=r['reference'],ref_text=r['reference_text'],x_vector_only_mode=False)
            require(torch.equal(prompt[0].ref_code.cpu(),asset['codes'])
                and torch.equal(prompt[0].ref_spk_embedding.flatten().cpu(),asset['speaker']), 'Independent reference extraction differs')
            references[r['reference']]=asset['speaker']
        ref=NativeReference(asset['ref_ids'],asset['codes'],asset['speaker'],r['reference_sha256'],e['features_audit_sha256'])
        layout=core._layout(t['ref_ids'].shape[1]-5,t['input_ids'].shape[1]-8,len(t['ref_codes']))
        # Conditioning-only reconstruction, including for truncated outputs:
        # no likelihood is scored and no synthetic EOS is treated as observed.
        with torch.no_grad(),core._context('conditioning-audit',r['half_reference'],layout):
            rebuilt=core._build(t['input_ids'],codes,ref)
            require(torch.equal(rebuilt.embeddings,t['native_inputs']), 'Actual native prefix/continuation inputs differ')
        require(t['trace']==r['trace'] and r['trace']==dict(kind='native',half_reference=r['half_reference'],
            layout=asdict(layout),calls=1), 'Native routing trace differs')
        wave,sr=sf.read(dest/r['wav'],dtype='float32')
        require(sr==24000 and wave.ndim==1 and len(wave)>0 and np.isfinite(wave).all()
            and r['seconds']==len(wave)/sr and r['seconds']<=30, 'Invalid audited waveform or duration')
        with torch.inference_mode():
            vector=wrapper.model.extract_speaker_embedding(audio=wave,sr=sr).flatten()
            value=float(torch.nn.functional.cosine_similarity(vector[None],references[r['reference']][None]))
        require(math.isfinite(value) and math.isfinite(r['cosine']), 'Nonfinite independently extracted cosine')
        maximum=max(maximum,abs(value-r['cosine']))
    truncated=sum(not r['terminated'] for r in records)
    require(maximum<=2e-6 and summary==report['summary'] and truncated==report['truncated'], 'Independent metrics differ')
    unchanged(wrapper.model,core,state)
    movement=policy.movement(core)
    require(movement==state['adapter_movement'] and state['frozen_identity']==p['frozen_identity'],
        'Canonical checkpoint movement or policy identity differs')
    receipt=dict(status='verified',step=step,recordings=80,summary=summary,truncated=truncated,
        frozen_weights_verified=True,native_conditioning_verified=True,all_artifact_hashes_verified=True,
        all_recomputed_cosines_finite=True,max_cosine_recomputation_error=maximum,
        adapter_movement=movement,checkpoint_sha256=checkpoint_sha,
        protocol_sha256=digest(run/'protocol.json'),speech_protocol_sha256=speech_sha,
        results_sha256=digest(dest/'results.json'),generated_sha256=digest(dest/'generated.json'),
        native_verification_sha256=digest(run/'native-verification.json'),frozen_identity=state['frozen_identity'],
        features_audit_sha256=e['features_audit_sha256'],
        native_inputs_verified=80,goal_achieved=False,**diagnostic_bindings(run,p,step))
    history=[verified_receipt(run,p,e,s) for s in e['checkpoints'] if s<step]+[receipt]
    receipt['progression']=progression(history)
    write(path,receipt)


def main():
    import argparse, os
    parser=argparse.ArgumentParser(); parser.add_argument('phase',choices=('prepare','generate','score','audit'))
    parser.add_argument('--run',required=True,type=Path); parser.add_argument('--step',type=int,default=0); args=parser.parse_args()
    config=read(args.run/'protocol.json')['config']
    os.environ['NUMBA_CACHE_DIR']=config['numba_cache']; os.environ['TEMP']=os.environ['TMP']=config['tmp']
    os.environ['HF_HUB_OFFLINE']=os.environ['TRANSFORMERS_OFFLINE']='1'
    import torch
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    if args.phase=='prepare': prepare(args.run)
    else: {'generate':generate,'score':score,'audit':audit}[args.phase](args.run,args.step)


if __name__=='__main__': main()
