"""Bounded reference-only speaker-input adaptation on a frozen full LoRA model.

This is a training-speaker calibration experiment, not a new global model or
an ASRRL reproduction. Adaptation sees reference audio/text only; generated
different-text speech is scored separately at every registered checkpoint.
"""
import argparse
from dataclasses import replace
import os
from pathlib import Path
import shutil
import subprocess
import sys

from .engine import digest, ids, read, save, write
from .policy import external, fingerprint
from .sequence import build, loss_sums


def speaker_sequence(model, input_ids, codes, speaker):
    """Keep the established layout; restore differentiation at its speaker slot.

    build intentionally detaches cached speaker features. Position 6 is three
    role tokens plus three codec tags. Replace only this one embedding, keeping
    its text padding contribution. Tests compare this to actual generation.
    """
    import torch
    seq = build(model, input_ids, codes, speaker, non_streaming=True)
    if speaker.ndim != 1 or speaker.numel() != seq.embeddings.shape[-1]:
        raise ValueError('Expected one speaker vector')
    residual = (speaker-speaker.detach()).reshape(1, -1)
    slot = seq.embeddings[6:7] + residual
    return replace(seq, embeddings=torch.cat([seq.embeddings[:6], slot, seq.embeddings[7:]]))


def bounded_step(vector, origin, gradient, radius_fraction=0.05, step_fraction=1/16):
    """Normalized gradient step, projected onto an L2 ball about original input."""
    import torch
    if (not 0 < radius_fraction < 1 or not 0 < step_fraction <= 1
            or vector.shape != origin.shape or vector.shape != gradient.shape
            or not all(torch.isfinite(x).all() for x in (vector, origin, gradient))):
        raise ValueError('Invalid bounded speaker update')
    radius = origin.norm()*radius_fraction
    if radius <= 0:
        raise ValueError('Zero speaker vector')
    delta = vector-origin-radius*step_fraction*gradient/gradient.norm().clamp_min(1e-12)
    delta = delta*torch.clamp(radius/delta.norm().clamp_min(1e-12), max=1)
    return (origin+delta).detach()


def sources():
    folder = Path(__file__).parent
    return {n:digest(folder/n) for n in ('speaker_reference_adaptation.py','sequence.py',
        'engine.py','policy.py','targeted_voice.py','preference_rollouts.py','evaluate.py')}


def prepare(pool, run):
    from .preference_rollouts import verify as verify_pool
    from data.pipeline import text_key
    collection, parent = verify_pool(pool)
    # Deterministic selection before seeing any results from this intervention.
    selected = {}
    for case in sorted(collection['cases'],key=lambda c:fingerprint(['speaker-reference-v1',c['case']])):
        if case['preference_split']=='train' and case['speaker_id'] not in selected:
            if text_key(case['text'])==text_key(case['reference_text']):
                raise ValueError('Reference and evaluation transcript must differ')
            selected[case['speaker_id']]=dict(case,group='training-calibration')
        if len(selected)==4:break
    if len(selected)!=4:raise ValueError('Four training speakers required')
    run=external(run.resolve());run.mkdir(exist_ok=False)
    p=dict(config=parent['config'],pool=str(pool),pool_protocol_sha256=digest(pool/'protocol.json'),
        source=sources(),base_sha256=parent['base_sha256'],adapted_sha256=parent['adapted_sha256'],
        cases=list(selected.values()),steps=[0,8,32],seeds=[2026,2027],
        generation=parent['generation'],decoding=parent['decoding'],
        radius_fraction=0.05,step_fraction=1/16,residual_weight=0.3,
        method='Projected normalized gradient descent on reference reconstruction CE; FP32; frozen epoch-2 full LoRA weights',
        objective='main mean codec CE + 0.3 * residual mean CE; no evaluation text/audio enters the gradient',
        selection='Four hash-selected original training speakers; calibration, not unseen/test evidence',
        gate='For expanding to the fixed development/custom screen: strictly improve mean voice cosine, no pooled CER or WER increase, no individual cosine decline >0.005 versus unadapted full model',
        goal_achieved=False,global_model_selection_changed=False)
    write(run/'protocol.json',p)
    shutil.copy2(Path(__file__),run/'source.py')
    write(run/'status.json',dict(status='prepared',goal_achieved=False))


def verify(run):
    p=read(run/'protocol.json')
    if p['source']!=sources() or digest(Path(p['pool'])/'protocol.json')!=p['pool_protocol_sha256']:
        raise ValueError('Registered source/provenance changed')
    for c in p['cases']:
        if digest(c['reference'])!=c['reference_sha256']:raise ValueError('Reference changed')
    return p


def load(p):
    import torch
    from safetensors import safe_open
    from qwen_tts import Qwen3TTSModel
    base,adapted=Path(p['config']['base']),Path(p['config']['adapted'])
    if digest(base/'model.safetensors')!=p['base_sha256'] or digest(adapted/'model.safetensors')!=p['adapted_sha256']:
        raise ValueError('Frozen model source changed')
    wrapper=Qwen3TTSModel.from_pretrained(str(base),device_map='cpu',torch_dtype=torch.float32,
        attn_implementation='sdpa',local_files_only=True)
    state=wrapper.model.state_dict()
    with safe_open(adapted/'model.safetensors',framework='pt',device='cpu') as donor,torch.no_grad():
        if set(state)!=set(donor.keys()):raise ValueError('Full model keys differ')
        for key in state:
            value=donor.get_tensor(key).float()
            if not key.startswith('talker.') and not torch.equal(state[key],value):
                raise ValueError('Speaker encoder must remain original')
            state[key].copy_(value)
    wrapper.model.eval().requires_grad_(False)
    return wrapper


def adapt(run):
    import torch
    from .targeted_voice import frozen_digest
    p=verify(run);wrapper=load(p);model=wrapper.model
    before=frozen_digest(model)
    # extract_speaker_embedding follows model.device; construct all references
    # before moving only the talker to CUDA, while encoder and codec are CPU.
    with torch.no_grad():
        prompts={c['case']:wrapper.create_voice_clone_prompt(ref_audio=c['reference'],
            ref_text=c['reference_text'],x_vector_only_mode=False) for c in p['cases']}
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    # Input-gradient graphs need training-mode checkpointing but no dropout.
    # Qwen attention dropout is zero; assert this rather than silently adding noise.
    for decoder in (model.talker.model,model.talker.code_predictor.model):
        if decoder.config.attention_dropout!=0:raise ValueError('Stochastic attention unsupported')
        decoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        decoder.train()
    model.talker.to('cuda')
    for case in p['cases']:
        path=run/f"{case['case']}-embeddings.pt"
        if path.exists():
            receipt=read(path.with_suffix('.json'))
            if digest(path)!=receipt['sha256'] or receipt['protocol_sha256']!=digest(run/'protocol.json'):
                raise ValueError('Adaptation cache changed')
            continue
        # Speaker encoder and codec stay on CPU; reference is the only fitted clip.
        prompt=prompts[case['case']]
        original=prompt[0].ref_spk_embedding.detach().float().flatten().to('cuda')
        codes=prompt[0].ref_code.detach().to('cuda')
        target_ids=ids(wrapper,case['reference_text']).to('cuda')
        current=original.clone();checkpoints={0:original.cpu()};metrics=[]
        for step in range(max(p['steps'])+1):
            current=current.detach().requires_grad_(True)
            seq=speaker_sequence(model,target_ids,codes,current)
            first,residual,n,m=loss_sums(model,[seq])
            loss=first/n+p['residual_weight']*residual/m
            gradient,=torch.autograd.grad(loss,current)
            if not torch.isfinite(loss) or not torch.isfinite(gradient).all() or gradient.norm()==0:
                raise ValueError('Nonfinite or disconnected reference gradient')
            if any(x.grad is not None for x in model.parameters()):
                raise ValueError('Frozen model acquired gradients')
            item=dict(step=step,loss=float(loss.detach()),main_ce=float(first.detach()/n),
                residual_ce=float(residual.detach()/m),gradient_norm=float(gradient.norm()),
                relative_displacement=float((current-original).norm()/original.norm()),
                gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated())
            metrics.append(item)
            if step in p['steps']:
                checkpoints[step]=current.detach().cpu()
            write(run/f"{case['case']}-fit.json",dict(case=case['case'],metrics=metrics))
            print(case['case'],item,flush=True)
            if step==max(p['steps']):break
            current=bounded_step(current.detach(),original,gradient,p['radius_fraction'],p['step_fraction'])
            del seq,first,residual,loss,gradient
        save(path,dict(checkpoints=checkpoints,original=original.cpu(),ref_codes=codes.cpu(),
            reference_sha256=case['reference_sha256'],reference_text=case['reference_text'],
            protocol_sha256=digest(run/'protocol.json'),frozen_backbone_sha256=before))
        write(path.with_suffix('.json'),dict(sha256=digest(path),protocol_sha256=digest(run/'protocol.json')))
        write(run/'status.json',dict(status='adapting',case=case['case']))
    model.talker.to('cpu')
    if frozen_digest(model)!=before:raise ValueError('Frozen weights changed during adaptation')
    write(run/'adaptation-verification.json',dict(frozen_backbone_sha256=before,
        frozen_weights_unchanged=True,model_parameters_trained=0,reference_only=True))


def generate(run):
    from dataclasses import replace as clone
    import numpy as np
    import soundfile as sf
    import torch
    from .preference_rollouts import capture_generation
    from .targeted_voice import frozen_digest
    p=verify(run);wrapper=load(p)
    expected=read(run/'adaptation-verification.json')['frozen_backbone_sha256']
    if frozen_digest(wrapper.model)!=expected:raise ValueError('Evaluation model differs')
    for case in p['cases']:
        path=run/f"{case['case']}-embeddings.pt"
        if digest(path)!=read(path.with_suffix('.json'))['sha256']:raise ValueError('Embedding changed')
        saved=__import__('torch').load(path,map_location='cpu',weights_only=True)
        if saved['protocol_sha256']!=digest(run/'protocol.json') or saved['frozen_backbone_sha256']!=expected:
            raise ValueError('Embedding provenance mismatch')
        with torch.inference_mode():
            prompt=wrapper.create_voice_clone_prompt(ref_audio=case['reference'],ref_text=case['reference_text'],x_vector_only_mode=False)
        refvec=prompt[0].ref_spk_embedding.detach().flatten()
        if not torch.equal(refvec,saved['original']) or not torch.equal(prompt[0].ref_code,saved['ref_codes']):
            raise ValueError('Reference conditioning differs')
        for step in p['steps']:
            # Cosine always uses ORIGINAL reference, never optimized conditioning.
            adapted=[clone(prompt[0],ref_spk_embedding=saved['checkpoints'][step])]
            for seed in p['seeds']:
                prefix=f"step-{step:03d}-{case['case']}-{seed}"
                wav=run/(prefix+'.wav');meta=run/(prefix+'.json')
                if meta.exists():
                    r=read(meta)
                    if r['protocol_sha256']!=digest(run/'protocol.json') or digest(wav)!=r['audio_sha256']:
                        raise ValueError('Generation cache changed')
                    continue
                torch.manual_seed(seed);np.random.seed(seed)
                with torch.inference_mode(),capture_generation(wrapper.model) as trajectory:
                    waves,sr=wrapper.generate_voice_clone(text=case['text'],language='Auto',
                        voice_clone_prompt=adapted,non_streaming_mode=True,**p['generation'])
                if not len(waves[0]) or not np.isfinite(waves[0]).all():raise ValueError('Invalid speech')
                sf.write(wav,waves[0],sr,subtype='PCM_16')
                wave,sr=sf.read(wav,dtype='float32')
                with torch.inference_mode():
                    vector=wrapper.model.extract_speaker_embedding(audio=wave,sr=sr).flatten()
                    cosine=float(torch.nn.functional.cosine_similarity(vector[None],refvec[None]))
                write(meta,dict(**case,step=step,seed=seed,wav=wav.name,audio_sha256=digest(wav),
                    cosine=cosine,seconds=len(wave)/sr,terminated=trajectory['terminated'],
                    embedding_sha256=digest(path),protocol_sha256=digest(run/'protocol.json')))
                write(run/'status.json',dict(status='generating',sample=prefix))
                print(prefix,cosine,flush=True)


def score_all(run):
    import soundfile as sf
    import torch,torchaudio
    from transformers import WhisperForConditionalGeneration,WhisperProcessor
    from scripts.score_latest_audio import score
    from .evaluate import ASR_REV
    from .targeted_voice import aggregate
    p=verify(run);asr=Path(p['config']['asr']);manifest=read(asr/'download-manifest.json')
    if manifest['repo']!='BuzzASR/mongolian' or manifest['revision']!=ASR_REV:raise ValueError('ASR mismatch')
    for name,item in manifest['files'].items():
        if digest(asr/name)!=item['sha256']:raise ValueError('ASR source changed')
    processor=WhisperProcessor.from_pretrained(asr,local_files_only=True)
    model=WhisperForConditionalGeneration.from_pretrained(asr,torch_dtype=torch.float32,local_files_only=True).to('cuda').eval()
    reports=[]
    for step in p['steps']:
        records=[]
        for case in p['cases']:
            for seed in p['seeds']:
                prefix=f"step-{step:03d}-{case['case']}-{seed}"
                r=read(run/(prefix+'.json'));path=run/('score-'+prefix+'.json')
                if r['protocol_sha256']!=digest(run/'protocol.json') or digest(run/r['wav'])!=r['audio_sha256']:
                    raise ValueError('Speech record changed')
                if path.exists():
                    result=read(path)
                    if any(result[k]!=v for k,v in r.items()) or result['asr_revision']!=ASR_REV or result['decoding']!=p['decoding']:
                        raise ValueError('ASR cache changed')
                else:
                    wave,sr=sf.read(run/r['wav'],dtype='float32')
                    wave=torchaudio.functional.resample(torch.from_numpy(wave),sr,16000).numpy()
                    batch=processor(wave,sampling_rate=16000,return_tensors='pt',return_attention_mask=True).to('cuda')
                    with torch.inference_mode():
                        tokens=model.generate(batch.input_features,attention_mask=batch.attention_mask,**p['decoding'])
                    hypothesis=processor.batch_decode(tokens,skip_special_tokens=True)[0].strip()
                    result=dict(r,hypothesis=hypothesis,score=score(r['text'],hypothesis,True),asr_revision=ASR_REV,decoding=p['decoding'])
                    write(path,result)
                records.append(result)
        summary=aggregate(records)['training-calibration']
        report=dict(step=step,summary=summary,records=records,truncated=sum(not r['terminated'] for r in records))
        if step:
            before=reports[0];old={(r['case'],r['seed']):r for r in before['records']}
            worst=min(r['cosine']-old[r['case'],r['seed']]['cosine'] for r in records)
            report['worst_clip_cosine_change']=worst
            report['eligible_for_development_screen']=bool(not report['truncated']
                and summary['speaker_macro_cosine']>before['summary']['speaker_macro_cosine']
                and summary['pooled_cer']<=before['summary']['pooled_cer']
                and summary['pooled_wer']<=before['summary']['pooled_wer'] and worst>=-0.005)
        reports.append(report)
        write(run/f'results-{step:03d}.json',report)
        print(step,summary,flush=True)
    write(run/'results.json',dict(reports=reports,goal_achieved=False,global_model_selection_changed=False,
        scope='Training-speaker calibration only; no development/custom results'))
    write(run/'status.json',dict(status='complete',goal_achieved=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pool',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--phase',choices=['prepare','adapt','generate','score','run'],required=True)
    args=parser.parse_args()
    if args.phase=='prepare':prepare(args.pool,args.output);return
    p=verify(args.output);cfg=p['config']
    os.environ['NUMBA_CACHE_DIR']=cfg['numba_cache']
    os.environ['TEMP']=os.environ['TMP']=cfg['tmp']
    os.environ['HF_HUB_OFFLINE']=os.environ['TRANSFORMERS_OFFLINE']='1'
    import torch
    torch.set_num_threads(4)
    if args.phase=='run':
        for phase in ('adapt','generate','score'):
            subprocess.run([sys.executable,'-m','training.speaker_reference_adaptation','--output',str(args.output),'--phase',phase],check=True)
    else:{'adapt':adapt,'generate':generate,'score':score_all}[args.phase](args.output)


if __name__=='__main__':main()
