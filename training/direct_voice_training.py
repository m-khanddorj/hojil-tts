"""Bounded fresh-rollout speaker-gradient pilot with an anchor-loss constraint.

Phases run in separate processes. All model/data/audio artifacts are outside Git.
The full CPU speech panel, original-reference cosine and ASR remain authoritative.
"""
import argparse,copy,gc,os,time
from pathlib import Path
import numpy as np
import torch
from .engine import digest,read,write,save,ids,check_optimizer
from .policy import external,fingerprint
from . import middle_attention_training as initial
from . import middle_attention_adapter as adapters
from .codec_policy_replay import policy_outputs,combine_reference_probabilities,backward_probability_gradient
from .direct_voice_objective import checkpointed_waveform,constrained_direction
from .preference_objective import deterministic_training
from .targeted_voice import aggregate
from .voice_strength_search import goal_screen

SETTINGS=dict(updates=4,recordings_per_update=16,seed=31517,learning_rate=2e-5,
    max_backtracks=8,anchor_tolerance=1e-6,weight_radius=.01,gradient_clip=1.)


def sources():
    import importlib.util
    s=initial.source_hashes()
    for name in ('direct_voice_training.py','direct_voice_objective.py','codec_policy_replay.py',
                 'differentiable_codec_probe.py','preference_rollouts.py','confirm_joint_candidate.py'):
        s[name]=digest(Path(__file__).with_name(name))
    for name in ('qwen_tts.core.models.modeling_qwen3_tts','qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2'):
        s[name]=digest(importlib.util.find_spec(name).origin)
    for name in ('test_training_direct_voice_objective.py','test_training_direct_voice_training.py'):
        s[name]=digest(Path(__file__).parent.parent/'tests'/name)
    return s


def trajectory(pool,record):
    path=pool/record['trajectory']
    if digest(path)!=record['trajectory_sha256']:raise ValueError('Trajectory changed')
    return torch.load(path,map_location='cpu',weights_only=True)


def select_anchors(pool,records):
    eligible=[];validation=[]
    for r in records:
        if not r['terminated'] or r['score']['cer']!=0:continue
        if r['preference_split']=='validation':validation.append(r);continue
        if r['preference_split']!='train' or not .96<=r['cosine']<.984:continue
        t=trajectory(pool,r)
        if len(t['codes'])+len(t['ref_codes'])<=172:eligible.append(r)
    chosen={}
    for r in sorted(eligible,key=lambda r:fingerprint(['direct-voice-anchor-v1',r['sample_id']])):
        chosen.setdefault(r['speaker_id'],r)
    if len(chosen)<16 or len({r['speaker_id'] for r in validation})!=16:
        raise ValueError('Insufficient fitting/validation coverage')
    anchors=sorted(chosen.values(),key=lambda r:fingerprint([SETTINGS['seed'],r['speaker_id']]))
    return anchors,validation


def prepare(candidate,run,panel):
    p=initial.verify(candidate);state=initial.read_checkpoint(candidate,4,p)
    audit=read(candidate/'verification-000004.json')
    if audit['status']!='verified' or audit['recordings']!=26:raise ValueError('Initial audit missing')
    root=run.parent;proof=read(root/'policy-codec-replay-probe/results.json')
    if proof['status']!='verified_read_only_policy_gradient' or not proof['replay']['all_parameters_unchanged']:
        raise ValueError('Real policy gradient not verified')
    log=root/'direct-voice-objective-tests.log';test=log.read_text(encoding='utf-8-sig')
    if 'Ran 5 tests' not in test or '\nOK\n' not in test or 'skipped' in test.lower():raise ValueError('Objective tests missing')
    integration=root/'direct-voice-training-tests.log';test=integration.read_text(encoding='utf-8-sig')
    if 'Ran 2 tests' not in test or '\nOK\n' not in test or 'skipped' in test.lower():raise ValueError('Update integration tests missing')
    from .confirm_joint_candidate import validate_panel
    expanded=read(panel);validate_panel(expanded)
    if any(expanded[k]!=p[k] for k in ('seeds','generation','decoding')):raise ValueError('Confirmation settings differ')
    pool=Path(p['pool']);records=read(pool/'results.json')['records']
    anchors,validation=select_anchors(pool,records)
    from data.pipeline import text_key
    for r in anchors+validation:
        if digest(r['reference'])!=r['reference_sha256']:raise ValueError('Reference changed')
        for c in expanded['cases']:
            if (r['speaker_id']==c['speaker_id'] or r['reference_sha256']==c['reference_sha256']
                    or {text_key(r['text']),text_key(r['reference_text'])}&{text_key(c['text']),text_key(c['reference_text'])}):
                raise ValueError('Training/diagnostic data overlap the confirmation panel')
    run=external(run.resolve());run.mkdir(exist_ok=False)
    write(run/'protocol.json',dict(source=sources(),runtime=initial.collection.runtime(),training=SETTINGS,
        candidate=str(candidate),candidate_protocol_sha256=digest(candidate/'protocol.json'),
        initial_checkpoint_sha256=digest(candidate/'checkpoint-000004.pt'),
        initial_audit_sha256=digest(candidate/'verification-000004.json'),
        initial_speech_sha256=digest(candidate/'eval-000004/results.json'),
        pool=str(pool),pool_protocol_sha256=digest(pool/'protocol.json'),pool_results_sha256=digest(pool/'results.json'),
        config=p['config'],base_sha256=p['base_sha256'],adapted_sha256=p['adapted_sha256'],
        codec_sha256=digest(Path(p['config']['base'])/'speech_tokenizer/model.safetensors'),
        cases=p['cases'],seeds=p['seeds'],generation=p['generation'],decoding=p['decoding'],
        parent_protocol_sha256=p['parent_protocol_sha256'],panel=str(panel),panel_sha256=digest(panel),
        anchors=anchors,validation=validation,tests=str(log),tests_sha256=digest(log),
        integration_tests=str(integration),integration_tests_sha256=digest(integration),
        gradient_proof_sha256=digest(root/'policy-codec-replay-probe/results.json'),
        stopping='Incomplete rollout or failed update constraint stops; two consecutive verified speech-harm warnings or a joint screen pass stops; screen pass requires all114 confirmation recordings',
        objective='Fresh generated speaker-loss straight-through gradient; projected Adam proposal with full fixed-anchor NLL backtracking; reference codes and original weights frozen',
        telemetry='Fresh training audio is scored before updates; fixed anchor/validation NLL is diagnostic; no fixed-hard-code speaker-loss trend is interpreted as learning',
        goal_achieved=False,global_model_selection_changed=False))
    save(run/'checkpoint-000000.pt',dict(step=0,adapter=state['adapter'],optimizer=None,
        frozen_backbone_sha256=state['frozen_backbone_sha256'],frozen_speaker_sha256=state['frozen_speaker_sha256'],
        protocol_sha256=digest(run/'protocol.json')))
    write(run/'checkpoint-000000.json',dict(sha256=digest(run/'checkpoint-000000.pt')))


def verify(run):
    p=read(run/'protocol.json');candidate=Path(p['candidate']);pool=Path(p['pool'])
    if (p['source']!=sources() or p['runtime']!=initial.collection.runtime() or p['training']!=SETTINGS
            or digest(candidate/'protocol.json')!=p['candidate_protocol_sha256']
            or digest(candidate/'checkpoint-000004.pt')!=p['initial_checkpoint_sha256']
            or digest(candidate/'verification-000004.json')!=p['initial_audit_sha256']
            or digest(candidate/'eval-000004/results.json')!=p['initial_speech_sha256']
            or digest(pool/'protocol.json')!=p['pool_protocol_sha256'] or digest(pool/'results.json')!=p['pool_results_sha256']
            or digest(p['panel'])!=p['panel_sha256'] or digest(p['tests'])!=p['tests_sha256']):
        raise ValueError('Registered pilot provenance changed')
    if (digest(p['integration_tests'])!=p['integration_tests_sha256']
            or digest(run.parent/'policy-codec-replay-probe/results.json')!=p['gradient_proof_sha256']):
        raise ValueError('Registered gradient/update tests changed')
    return p


def read_checkpoint(run,step,p):
    path=run/f'checkpoint-{step:06d}.pt'
    if digest(path)!=read(path.with_suffix('.json'))['sha256']:raise ValueError('Checkpoint changed')
    state=torch.load(path,map_location='cpu',weights_only=True)
    if state['step']!=step or state['protocol_sha256']!=digest(run/'protocol.json'):raise ValueError('Checkpoint identity differs')
    return state


def load(run,step,device='cpu',training=False):
    p=verify(run);state=read_checkpoint(run,step,p)
    original=initial.verify(Path(p['candidate']));wrapper=initial.load(original)
    adapters.restore(wrapper.model,state['adapter'])
    if (adapters.frozen_digest(wrapper.model)!=state['frozen_backbone_sha256']
            or adapters.speaker_digest(wrapper.model)!=state['frozen_speaker_sha256']):raise ValueError('Frozen model mismatch')
    if device=='cuda':
        wrapper.model.to('cuda');wrapper.device=torch.device('cuda')
    if training:
        for decoder in (wrapper.model.talker.model,wrapper.model.talker.code_predictor.model):
            decoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        deterministic_training(wrapper.model)
    else:wrapper.model.eval()
    return p,wrapper,state


def unchanged(model,state):
    if (adapters.frozen_digest(model)!=state['frozen_backbone_sha256']
            or adapters.speaker_digest(model)!=state['frozen_speaker_sha256']
            or any(not torch.equal(v,state['adapter'][k]) for k,v in adapters.snapshot(model).items())):
        raise ValueError('Unexpected model mutation')


def scheduled(p,step):
    if step not in range(1,p['training']['updates']+1):raise ValueError('Unregistered update')
    n=p['training']['recordings_per_update'];anchors=p['anchors']
    return [anchors[((step-1)*n+i)%len(anchors)] for i in range(n)]


def generate(run,step,evaluation=False):
    import soundfile as sf
    from .preference_rollouts import capture_generation
    p,wrapper,state=load(run,step if evaluation else step-1,'cpu' if evaluation else 'cuda')
    out=run/(f'eval-{step:06d}' if evaluation else f'rollout-{step:06d}');out.mkdir(exist_ok=False)
    if evaluation:requests=[(c,s) for s in p['seeds'] for c in p['cases']]
    else:
        requests=[(dict(case=r['case'],speaker_id=r['speaker_id'],group='fitting',text=r['text'],
            reference=r['reference'],reference_sha256=r['reference_sha256'],reference_text=r['reference_text']),
            p['training']['seed']+step*1000+i) for i,r in enumerate(scheduled(p,step))]
    records=[];prompts={}
    for c,seed in requests:
        if digest(c['reference'])!=c['reference_sha256']:raise ValueError('Reference changed')
        if c['reference'] not in prompts:
            with torch.inference_mode():prompts[c['reference']]=wrapper.create_voice_clone_prompt(
                ref_audio=c['reference'],ref_text=c['reference_text'],x_vector_only_mode=False)
        prompt=prompts[c['reference']];torch.manual_seed(seed);np.random.seed(seed)
        with torch.inference_mode(),capture_generation(wrapper.model) as captured:
            waves,sr=wrapper.generate_voice_clone(text=c['text'],language='Auto',voice_clone_prompt=prompt,
                non_streaming_mode=True,**p['generation'])
        if sr!=24000 or not len(waves[0]) or not np.isfinite(waves[0]).all() or len(waves[0])/sr>30:raise ValueError('Invalid speech')
        wav=out/f"{seed}-{c['case']}.wav";sf.write(wav,waves[0],sr,subtype='PCM_16');wave,sr=sf.read(wav,dtype='float32')
        with torch.inference_mode():
            vec=wrapper.model.extract_speaker_embedding(audio=wave,sr=sr).flatten();ref=prompt[0].ref_spk_embedding.flatten()
            cosine=float(torch.nn.functional.cosine_similarity(vec[None],ref[None]))
        path=wav.with_suffix('.pt');save(path,dict(codes=captured['codes'],first_sequence=captured['first_sequence'],
            terminated=captured['terminated'],ref_codes=prompt[0].ref_code.detach().cpu(),ref_xvec=ref.detach().cpu(),
            input_ids=ids(wrapper,c['text']).cpu(),ref_ids=ids(wrapper,c['reference_text'],True).cpu()))
        r=dict(c,seed=seed,wav=wav.name,audio_sha256=digest(wav),trajectory=path.name,trajectory_sha256=digest(path),
            cosine=cosine,seconds=len(wave)/sr,terminated=captured['terminated'],
            checkpoint_sha256=digest(run/f"checkpoint-{step if evaluation else step-1:06d}.pt"),
            protocol_sha256=digest(run/'protocol.json'))
        write(wav.with_suffix('.json'),r);records.append(r);print(f'{out.name}: {len(records)}/{len(requests)}',flush=True)
    unchanged(wrapper.model,state)
    if evaluation and step==0:
        old=read(Path(p['candidate'])/'eval-000004/results.json')['records']
        expected={(r['case'],r['seed']):r['audio_sha256'] for r in old}
        if any(expected[r['case'],r['seed']]!=r['audio_sha256'] for r in records):raise ValueError('Initial speech parity failed')
    write(out/'generated.json',dict(records=records,all_weights_unchanged=True))
    if not evaluation and any(not r['terminated'] for r in records):raise ValueError('Incomplete fitting rollout; no sample discarded')


def score(run,step,evaluation=False):
    import soundfile as sf
    import torchaudio
    from transformers import WhisperForConditionalGeneration,WhisperProcessor
    from .evaluate import ASR_REV
    from scripts.score_latest_audio import score as score_text
    p=verify(run);out=run/(f'eval-{step:06d}' if evaluation else f'rollout-{step:06d}')
    generated=read(out/'generated.json');records=generated['records']
    if not generated['all_weights_unchanged']:raise ValueError('Generation state changed')
    asr=Path(p['config']['asr']);manifest=read(asr/'download-manifest.json')
    if manifest['repo']!='BuzzASR/mongolian' or manifest['revision']!=ASR_REV:raise ValueError('ASR mismatch')
    for name,v in manifest['files'].items():
        if digest(asr/name)!=v['sha256']:raise ValueError('ASR weights changed')
    processor=WhisperProcessor.from_pretrained(asr,local_files_only=True)
    model=WhisperForConditionalGeneration.from_pretrained(asr,torch_dtype=torch.float32,local_files_only=True).to('cuda').eval()
    for r in records:
        if digest(out/r['wav'])!=r['audio_sha256']:raise ValueError('Audio changed')
        wave,sr=sf.read(out/r['wav'],dtype='float32');wave=torchaudio.functional.resample(torch.from_numpy(wave),sr,16000).numpy()
        batch=processor(wave,sampling_rate=16000,return_tensors='pt',return_attention_mask=True).to('cuda')
        with torch.inference_mode():tokens=model.generate(batch.input_features,attention_mask=batch.attention_mask,**p['decoding'])
        hypothesis=processor.batch_decode(tokens,skip_special_tokens=True)[0].strip()
        r.update(hypothesis=hypothesis,score=score_text(r['text'],hypothesis,True),decoding=p['decoding'],asr_revision=ASR_REV)
    report=dict(records=records,summary=aggregate(records),truncated=sum(not r['terminated'] for r in records))
    write(out/'results.json',report);print(report['summary'],flush=True)


def posteriors(run,step):
    p,wrapper,state=load(run,step-1,'cuda',True);out=run/f'rollout-{step:06d}'
    report=read(out/'results.json');receipts=[]
    if report['truncated']:raise ValueError('No truncated fitting speech')
    for r in report['records']:
        t=trajectory(out,r)
        with torch.no_grad():v=policy_outputs(wrapper.model,t)
        path=out/f"posterior-{r['seed']}-{r['case']}.pt";save(path,v['probabilities'].cpu())
        receipts.append(dict(case=r['case'],seed=r['seed'],path=path.name,sha256=digest(path),
            raw_log_probability=float(v['log_probability']),main_acoustic_mass_min=float(v['main_acoustic_mass'].min())))
    unchanged(wrapper.model,state);write(out/'posteriors.json',dict(records=receipts,checkpoint_sha256=digest(run/f'checkpoint-{step-1:06d}.pt')))


def load_codec(p):
    from safetensors import safe_open
    from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSConfig
    from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSSpeakerEncoder
    from qwen_tts.core.tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2Config
    from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2Decoder
    base=Path(p['config']['base']);directory=base/'speech_tokenizer'
    if digest(base/'model.safetensors')!=p['base_sha256']:raise ValueError('Speaker source changed')
    if digest(directory/'model.safetensors')!=p['codec_sha256']:raise ValueError('Codec source changed')
    config=Qwen3TTSTokenizerV2Config.from_pretrained(directory,local_files_only=True).decoder_config
    config._attn_implementation='sdpa';decoder=Qwen3TTSTokenizerV2Decoder(config).eval().requires_grad_(False)
    with safe_open(directory/'model.safetensors',framework='pt',device='cpu') as f:
        weights={k.removeprefix('decoder.'):f.get_tensor(k).float() for k in f.keys() if k.startswith('decoder.')}
    decoder.load_state_dict(weights,strict=True);del weights
    speaker=Qwen3TTSSpeakerEncoder(Qwen3TTSConfig.from_pretrained(base,local_files_only=True).speaker_encoder_config).eval().requires_grad_(False)
    with safe_open(base/'model.safetensors',framework='pt',device='cpu') as f:
        weights={k.removeprefix('speaker_encoder.'):f.get_tensor(k).float() for k in f.keys() if k.startswith('speaker_encoder.')}
    speaker.load_state_dict(weights,strict=True)
    return decoder.to('cuda'),speaker.to('cuda')


def codec_gradients(run,step):
    from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram
    from .targeted_voice import frozen_digest
    p=verify(run);out=run/f'rollout-{step:06d}';report=read(out/'results.json')
    cached=read(out/'posteriors.json')
    if cached['checkpoint_sha256']!=digest(run/f'checkpoint-{step-1:06d}.pt'):raise ValueError('Stale rollout posterior')
    entries={(r['case'],r['seed']):r for r in cached['records']};decoder,speaker=load_codec(p)
    before=(frozen_digest(decoder),frozen_digest(speaker));receipts=[]
    for r in report['records']:
        t=trajectory(out,r);item=entries[r['case'],r['seed']];path=out/item['path']
        if digest(path)!=item['sha256']:raise ValueError('Posterior changed')
        probabilities=torch.load(path,map_location='cpu',weights_only=True).to('cuda').requires_grad_(True)
        codes=torch.cat([t['ref_codes'],t['codes']]).T[None].to('cuda');ref_frames=len(t['ref_codes'])
        torch.cuda.reset_peak_memory_stats();started=time.monotonic()
        with torch.no_grad():native=decoder.chunked_decode(codes)
        joined=combine_reference_probabilities(t['ref_codes'],probabilities)
        wave=checkpointed_waveform(decoder,codes,joined)
        if not torch.equal(native,wave.detach()):raise ValueError('Checkpointed native waveform differs')
        cut=int(ref_frames/codes.shape[-1]*wave.shape[-1])
        mel=mel_spectrogram(wave[0,0,cut:].reshape(1,-1),n_fft=1024,num_mels=128,sampling_rate=24000,
            hop_size=256,win_size=1024,fmin=0,fmax=12000).transpose(1,2)
        vector=speaker(mel).flatten();target=t['ref_xvec'].to('cuda').flatten()
        loss=1-torch.nn.functional.cosine_similarity(vector[None],target[None]).mean()
        gradient,=torch.autograd.grad(loss,probabilities)
        if not torch.isfinite(gradient).all() or gradient.norm()==0:raise ValueError('Speaker gradient invalid')
        target_path=out/f"gradient-{r['seed']}-{r['case']}.pt";save(target_path,gradient.cpu())
        receipts.append(dict(case=r['case'],seed=r['seed'],path=target_path.name,sha256=digest(target_path),
            loss=float(loss.detach()),gradient_norm=float(gradient.norm()),seconds=time.monotonic()-started,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),frames=codes.shape[-1],reference_probabilities_constant=True))
        del probabilities,codes,native,joined,wave,mel,vector,target,loss,gradient
    if (frozen_digest(decoder),frozen_digest(speaker))!=before:raise ValueError('Frozen codec or scorer changed')
    if any(v.grad is not None for m in (decoder,speaker) for v in m.parameters()):raise ValueError('Frozen parameter gradient')
    write(out/'gradients.json',dict(records=receipts,frozen_weights_unchanged=True,posteriors_sha256=digest(out/'posteriors.json')))


def mean_nll(model,p,records,backward=False):
    values=[]
    for r in records:
        logp,tokens=adapters.log_probability(model,trajectory(Path(p['pool']),r))
        loss=-logp/tokens;values.append(float(loss.detach()))
        if backward:(loss/len(records)).backward()
    return sum(values)/len(values)


def update(run,step):
    p,wrapper,state=load(run,step-1,'cuda',True);model=wrapper.model;out=run/f'rollout-{step:06d}'
    if (run/f'checkpoint-{step:06d}.pt').exists():raise ValueError('Checkpoint already exists')
    gradients=read(out/'gradients.json');post=read(out/'posteriors.json');report=read(out/'results.json')
    if (not gradients['frozen_weights_unchanged'] or gradients['posteriors_sha256']!=digest(out/'posteriors.json')
            or post['checkpoint_sha256']!=digest(run/f'checkpoint-{step-1:06d}.pt')):raise ValueError('Stale/mismatched gradients')
    named=[(n,v) for n,v in model.named_parameters() if v.requires_grad];params=[v for n,v in named]
    optimizer=torch.optim.AdamW(params,lr=SETTINGS['learning_rate'],weight_decay=0.,fused=True)
    if state['optimizer'] is not None:optimizer.load_state_dict(state['optimizer'])
    optimizer.zero_grad(set_to_none=True);events=[]
    posterior={(r['case'],r['seed']):r for r in post['records']};grads={(r['case'],r['seed']):r for r in gradients['records']}
    n=len(report['records'])
    if n!=SETTINGS['recordings_per_update']:raise ValueError('Fitting coverage differs')
    for r in report['records']:
        key=r['case'],r['seed'];a,b=posterior[key],grads[key]
        if digest(out/a['path'])!=a['sha256'] or digest(out/b['path'])!=b['sha256']:raise ValueError('Probability/gradient cache changed')
        cached=torch.load(out/a['path'],map_location='cpu',weights_only=True)
        gradient=torch.load(out/b['path'],map_location='cpu',weights_only=True)
        events.append(backward_probability_gradient(model,trajectory(out,r),cached,gradient/n))
    voice=torch.cat([v.grad.detach().flatten().clone() for v in params])
    if not torch.isfinite(voice).all() or voice.norm()==0:raise ValueError('Invalid voice policy gradient')
    optimizer.zero_grad(set_to_none=True)
    anchor_before=mean_nll(model,p,p['anchors'],backward=True)
    anchor=torch.cat([v.grad.detach().flatten().clone() for v in params]);optimizer.zero_grad(set_to_none=True)
    offset=0
    for v in params:v.grad=voice[offset:offset+v.numel()].view_as(v).clone();offset+=v.numel()
    gradient_norm=float(torch.nn.utils.clip_grad_norm_(params,SETTINGS['gradient_clip'],error_if_nonfinite=True))
    old=torch.cat([v.detach().flatten().clone() for v in params]);optimizer.step();check_optimizer(optimizer)
    proposal=torch.cat([v.detach().flatten() for v in params])-old
    direction,projection=constrained_direction(proposal,voice,anchor)
    accepted=False;trials=[]
    for attempt in range(SETTINGS['max_backtracks']):
        scale=2.**(-attempt);offset=0
        with torch.no_grad():
            for v in params:
                v.copy_((old[offset:offset+v.numel()]+scale*direction[offset:offset+v.numel()]).view_as(v));offset+=v.numel()
            movement=adapters.movement(model);after=mean_nll(model,p,p['anchors'])
        trials.append(dict(scale=scale,anchor_nll=after,max_matrix_movement=movement['maximum']))
        actual=torch.cat([v.detach().flatten() for v in params])-old
        predicted=float(actual.double().dot(voice.double()))
        trials[-1]['actual_update_norm']=float(actual.norm());trials[-1]['predicted_voice_change']=predicted
        if after<=anchor_before+SETTINGS['anchor_tolerance'] and movement['maximum']<=SETTINGS['weight_radius'] and predicted<0:
            accepted=True;break
    if not accepted:
        adapters.restore(model,state['adapter']);write(out/'rejected-update.json',dict(reason='anchor_backtracking_failed',trials=trials))
        raise ValueError('No candidate meets the registered anchor constraint; pilot stops')
    with torch.no_grad():validation_nll=mean_nll(model,p,p['validation'])
    if (adapters.frozen_digest(model)!=state['frozen_backbone_sha256']
            or adapters.speaker_digest(model)!=state['frozen_speaker_sha256']):raise ValueError('Original weights changed')
    if any(v.grad is not None for v in model.parameters() if not v.requires_grad):raise ValueError('Frozen parameter gradient')
    metrics=dict(anchor_nll_before=anchor_before,anchor_nll_after=after,validation_nll=validation_nll,
        voice_gradient_norm=float(voice.norm()),anchor_gradient_norm=float(anchor.norm()),gradient_norm=gradient_norm,
        gradient_cosine=float(torch.nn.functional.cosine_similarity(voice[None],anchor[None])),
        projection=projection,backtracking=trials,accepted_scale=scale,weight_movement=movement,
        replay_max_error=max(e['maximum_probability_replay_error'] for e in events),
        fresh_rollout_summary=report['summary'],fresh_codec_losses=[r['loss'] for r in gradients['records']])
    path=run/f'checkpoint-{step:06d}.pt'
    save(path,dict(step=step,adapter=adapters.snapshot(model),optimizer=optimizer.state_dict(),
        frozen_backbone_sha256=state['frozen_backbone_sha256'],frozen_speaker_sha256=state['frozen_speaker_sha256'],
        protocol_sha256=digest(run/'protocol.json'),metrics=metrics,rollout_results_sha256=digest(out/'results.json'),
        gradients_sha256=digest(out/'gradients.json')))
    write(path.with_suffix('.json'),dict(sha256=digest(path),metrics=metrics));print(metrics,flush=True)


def audit(run,step):
    import soundfile as sf
    from scripts.score_latest_audio import score as score_text
    p,wrapper,state=load(run,step);out=run/f'eval-{step:06d}';report=read(out/'results.json')
    expected={(c['case'],s) for c in p['cases'] for s in p['seeds']};records=report['records']
    if len(records)!=26 or {(r['case'],r['seed']) for r in records}!=expected:raise ValueError('Incomplete speech screen')
    cases={c['case']:c for c in p['cases']};refs={};error=0.
    for r in records:
        if any(r[k]!=v for k,v in cases[r['case']].items()):raise ValueError('Evaluation condition changed')
        if (digest(out/r['wav'])!=r['audio_sha256'] or r['checkpoint_sha256']!=digest(run/f'checkpoint-{step:06d}.pt')
                or r['protocol_sha256']!=digest(run/'protocol.json') or r['decoding']!=p['decoding']):raise ValueError('Audio provenance changed')
        if r['score']!=score_text(r['text'],r['hypothesis'],True):raise ValueError('Edit counts differ')
        t=trajectory(out,r);codes,seq=t['codes'],t['first_sequence'];eos=wrapper.model.config.talker_config.codec_eos_token_id
        if codes.ndim!=2 or codes.shape[1]!=16 or bool(seq[0,-1]==eos)!=r['terminated']:raise ValueError('Trajectory/EOS differs')
        if r['terminated'] and not torch.equal(seq[0,-(len(codes)+1):-1],codes[:,0]):raise ValueError('Main-code trajectory differs')
        wave,sr=sf.read(out/r['wav'],dtype='float32')
        with torch.inference_mode():
            if r['reference'] not in refs:
                ref,refsr=sf.read(r['reference'],dtype='float32');refs[r['reference']]=wrapper.model.extract_speaker_embedding(audio=ref,sr=refsr).flatten()
            vec=wrapper.model.extract_speaker_embedding(audio=wave,sr=sr).flatten()
            cosine=float(torch.nn.functional.cosine_similarity(vec[None],refs[r['reference']][None]))
        error=max(error,abs(cosine-r['cosine']))
    if error>2e-6 or aggregate(records)!=report['summary']:raise ValueError('Independent scores differ')
    if step:
        m=state['metrics']
        if m['anchor_nll_after']>m['anchor_nll_before']+SETTINGS['anchor_tolerance']:raise ValueError('Anchor constraint violated')
        if digest(run/f'rollout-{step:06d}/results.json')!=state['rollout_results_sha256'] or digest(run/f'rollout-{step:06d}/gradients.json')!=state['gradients_sha256']:
            raise ValueError('Update data changed')
    unchanged(wrapper.model,state)
    passed=goal_screen(report['summary']) and all(r['terminated'] for r in records)
    write(run/f'verification-{step:06d}.json',dict(status='verified',step=step,recordings=26,
        summary=report['summary'],max_cosine_recomputation_error=error,goal_screen=passed,
        frozen_acoustic_and_speaker_verified=True,all_artifact_hashes_verified=True,
        source_sha256=digest(Path(__file__)),goal_achieved=False))
    print(report['summary'],flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--phase',choices=['prepare','rollout','rollout-score','posteriors','codec','update','speech','score','audit'],required=True)
    parser.add_argument('--step',type=int,default=0);parser.add_argument('--candidate',type=Path);parser.add_argument('--panel',type=Path)
    args=parser.parse_args();source=args.candidate if args.phase=='prepare' else args.output
    config=read(source/'protocol.json')['config'];os.environ['NUMBA_CACHE_DIR']=config['numba_cache']
    os.environ['TEMP']=os.environ['TMP']=config['tmp'];os.environ['HF_HUB_OFFLINE']=os.environ['TRANSFORMERS_OFFLINE']='1'
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if args.phase=='prepare':prepare(args.candidate,args.output,args.panel)
    elif args.phase in ('rollout','speech'):generate(args.output,args.step,args.phase=='speech')
    elif args.phase in ('rollout-score','score'):score(args.output,args.step,args.phase=='score')
    else:{'posteriors':posteriors,'codec':codec_gradients,'update':update,'audit':audit}[args.phase](args.output,args.step)


if __name__=='__main__':main()
