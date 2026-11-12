"""Small text-only LoRA pilot on the exact FP32 halfway voice model.

Separate from the historical trainer: no old run, test cache or selection is
modified. Every scheduled checkpoint gets generated-speech and voice checks.
"""
import argparse
from collections import defaultdict
import gc
import hashlib
import html
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

from .engine import (assignment, backward, check_optimizer, digest, evaluate,
                     ids, load_adapter, read, rows, save, write)
from .policy import external, fingerprint, learning_rate, validate_cohorts
from .observability import matched_probe

TEXT_TARGETS = r'^text_projection\.linear_fc[12]$'


def text_adapter(model, rank=16, dropout=0.05):
    from peft import LoraConfig, inject_adapter_in_model
    model.requires_grad_(False)
    inject_adapter_in_model(LoraConfig(r=rank, lora_alpha=rank,
        lora_dropout=dropout, target_modules=TEXT_TARGETS), model.talker)
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if len(trainable) != 4 or any(not n.startswith('talker.text_projection.') or '.lora_' not in n for n, _ in trainable):
        raise ValueError('Only the two text-projection LoRA modules may train')
    for _, p in trainable:
        p.data = p.data.float()
    return trainable


def frozen_digest(model):
    value = hashlib.sha256()
    for name, p in model.named_parameters():
        if not p.requires_grad:
            value.update(name.encode())
            value.update(str((tuple(p.shape), p.dtype)).encode())
            value.update(p.detach().cpu().contiguous().numpy().tobytes())
    return value.hexdigest()


def aggregate(records):
    from .evaluate import pooled
    if not records or len({(r['group'], r['seed'], r['case']) for r in records}) != len(records):
        raise ValueError('Empty or duplicate speech records')
    import math
    if any(not math.isfinite(r['cosine']) for r in records):
        raise ValueError('Nonfinite speaker measurement')
    result = {}
    for group in sorted({r['group'] for r in records}):
        subset = [r for r in records if r['group'] == group]
        speakers = defaultdict(list)
        for r in subset:
            speakers[r['speaker_id']].append(r['cosine'])
        result[group] = dict(**pooled([r['score'] for r in subset]),
            speaker_macro_cosine=sum(sum(v)/len(v) for v in speakers.values())/len(speakers),
            recordings=len(subset), speakers=len(speakers),
            per_seed={str(seed): pooled([r['score'] for r in subset if r['seed'] == seed])
                      for seed in sorted({r['seed'] for r in subset})})
    return result


def joint_gate(baseline, candidate, max_clip_cosine_drop=0.005):
    def keyed(records):
        keys = {(r['group'], r['seed'], r['case']): r for r in records}
        if len(keys) != len(records):
            raise ValueError('Duplicate comparison record')
        return keys
    a, b = keyed(baseline), keyed(candidate)
    if not a or set(a) != set(b):
        raise ValueError('Incomplete paired voice evaluation')
    for key in a:
        if any(a[key][f] != b[key][f] for f in ('text', 'reference_sha256', 'speaker_id')):
            raise ValueError('Evaluation reference changed')
    old, new = aggregate(baseline), aggregate(candidate)
    if set(old) != {'development', 'custom'}:
        raise ValueError('Both development and custom voice checks are required')
    voice_failures, accuracy_failures = [], []
    for group in old:
        if new[group]['speaker_macro_cosine'] < old[group]['speaker_macro_cosine'] - 1e-7:
            voice_failures.append(group + ': mean speaker cosine decreased')
        if new[group]['pooled_cer'] > old[group]['pooled_cer']:
            accuracy_failures.append(group + ': pooled CER increased')
        for seed, before in old[group]['per_seed'].items():
            if new[group]['per_seed'][seed]['pooled_wer'] > before['pooled_wer']:
                accuracy_failures.append(group + ': seed ' + seed + ' WER increased')
    worst = min(b[k]['cosine']-a[k]['cosine'] for k in a)
    if worst < -max_clip_cosine_drop:
        voice_failures.append('A recording exceeded the registered cosine-drop review threshold')
    improves = new['development']['pooled_wer'] < old['development']['pooled_wer']
    return dict(eligible=improves and not voice_failures and not accuracy_failures,
        voice_pass=not voice_failures, pronunciation_pass=improves and not accuracy_failures,
        voice_failures=voice_failures, accuracy_failures=accuracy_failures,
        worst_clip_cosine_change=worst, baseline=old, candidate=new,
        interpretation='Conservative diagnostic screen, not a perceptual noninferiority test; listening is required before promotion.')


def source_hashes():
    folder = Path(__file__).parent
    return {name: digest(folder/name) for name in ('targeted_voice.py', 'engine.py', 'sequence.py',
        'policy.py', 'observability.py', 'evaluate.py', 'speaker_diagnostics.py')}


def setup(cfg):
    run = external(Path(cfg['output']))
    run.mkdir(exist_ok=False)
    base, adapted, backup, features = map(Path, (cfg['base'], cfg['adapted'], cfg['backup'], cfg['features']))
    recovery = read(backup/'recovery-verification.json')['cohorts']
    for name in ('train', 'unseen-validation', 'familiar-validation'):
        if digest(backup/f'cohorts/{name}.jsonl') != recovery[name]['sha256']:
            raise ValueError('Changed cohort')
    signature = read(features/'signature.json')
    if signature['train_sha256'] != recovery['train']['sha256'] or signature['base_revision'] != base.name or signature['codec_sha256'] != digest(base/'speech_tokenizer/model.safetensors'):
        raise ValueError('Feature model mismatch')
    inventory = read(backup/'dataset-v1/dataset-inventory.json')['files']
    if digest(backup/'dataset-v1/manifest.jsonl') != inventory['manifest.jsonl']['sha256']:
        raise ValueError('Changed manifest')
    train = rows(backup/'cohorts/train.jsonl')
    unseen = rows(backup/'cohorts/unseen-validation.jsonl')
    canonical = validate_cohorts(train, unseen, rows(backup/'dataset-v1/manifest.jsonl'))
    development = {}
    for row in unseen:
        key = row['speaker_id']
        if key not in development or fingerprint([cfg['seed'], row['id']]) < fingerprint([cfg['seed'], development[key]['id']]):
            development[key] = row
    if len(development) != 10:
        raise ValueError('Expected ten development speakers')
    cases = []
    for i, row in enumerate(sorted(development.values(), key=lambda r: r['speaker_id']), 1):
        ref = (backup/'dataset-v1'/row['ref_audio']).resolve()
        if digest(ref) != inventory[row['ref_audio']]['sha256']:
            raise ValueError('Changed development reference')
        cases.append(dict(group='development', case=f'dev-{i:03d}', row_id=row['id'],
            text=row['text'], speaker_id=row['speaker_id'], reference=str(ref),
            reference_text=row['ref_text'], reference_sha256=digest(ref)))
    old = read(Path(cfg['custom_protocol']))
    if digest(Path(old['reference'])) != old['reference_sha256']:
        raise ValueError('Changed custom reference')
    for example in old['texts']:
        cases.append(dict(group='custom', case='custom-'+example['case'], text=example['text'],
            speaker_id='user-reference', reference=old['reference'], reference_text=old['reference_text'],
            reference_sha256=old['reference_sha256']))
    cases.sort(key=lambda c: (c['group'] != 'custom', c['case']))
    familiar = defaultdict(list)
    for row in rows(backup/'cohorts/familiar-validation.jsonl'):
        if row.get('eligible'):
            familiar[row['speaker_id']].append(row)
    full_selected = [dict(r, validation_group='unseen') for r in unseen]
    full_selected += [dict(min(group, key=lambda r: fingerprint([cfg['seed'], r['id']])), validation_group='familiar')
                      for _, group in sorted(familiar.items())]
    familiar_probe = sorted([r for r in full_selected if r['validation_group']=='familiar'], key=lambda r: fingerprint([cfg['seed'], r['id']]))[:10]
    loss_cases = [dict(r, validation_group='unseen') for r in development.values()] + familiar_probe
    validate_cohorts(train, loss_cases, list(canonical.values()))
    base_sha, adapted_sha = digest(base/'model.safetensors'), digest(adapted/'model.safetensors')
    if base_sha != old['base_model_sha256'] or adapted_sha != old['epoch2_model_sha256']:
        raise ValueError('Halfway parents changed')
    protocol = dict(config=cfg, source=source_hashes(), base_sha256=base_sha, adapted_sha256=adapted_sha,
        feature_signature=signature, feature_shards={p.name:digest(p) for p in sorted(features.glob('*.pt'))},
        validation_cache_sha256=digest(Path(cfg['validation_cache'])),
        validation_cache_signature=fingerprint([fingerprint(full_selected), signature]),
        cases=cases, loss_cases=loss_cases, generation=old['generation'], decoding=old['decoding'],
        seeds=[2026,2027], selection_scope='Small development screening pilot; custom voice is a personalization check, never reserved-test evidence',
        criterion='Lower development WER; no seedwise WER/CER regression or lower mean speaker cosine in either group; no clip cosine loss beyond 0.005. Listening still required.',
        waveform_baseline_example=str(Path(cfg['custom_protocol']).parent/'half_blend-001.wav'),
        waveform_baseline_sha256='cd32675b26cc6619a9e3d5d66ee4425a55b62a4eb10015c05de7495d13504a00')
    write(run/'protocol.json', protocol)
    (run/'source').mkdir()
    for name in protocol['source']:
        shutil.copy2(Path(__file__).parent/name, run/'source'/name)
    write(run/'status.json', dict(status='prepared', phase='initializing'))
    return protocol


def load_model(protocol):
    import torch
    from safetensors import safe_open
    from qwen_tts import Qwen3TTSModel
    cfg = protocol['config']; base, adapted = Path(cfg['base']), Path(cfg['adapted'])
    if source_hashes() != protocol['source'] or digest(base/'model.safetensors') != protocol['base_sha256'] or digest(adapted/'model.safetensors') != protocol['adapted_sha256']:
        raise ValueError('Frozen run inputs changed')
    torch.manual_seed(cfg['seed'])
    wrapper = Qwen3TTSModel.from_pretrained(str(base), device_map='cpu', torch_dtype=torch.float32,
        attn_implementation='sdpa', local_files_only=True)
    wrapper.model.requires_grad_(False)
    state = wrapper.model.state_dict(); changed = []
    with safe_open(base/'model.safetensors',framework='pt',device='cpu') as a, safe_open(adapted/'model.safetensors',framework='pt',device='cpu') as b, torch.no_grad():
        if set(a.keys()) != set(b.keys()):
            raise ValueError('Halfway parents have different keys')
        for key in a.keys():
            x,y = a.get_tensor(key),b.get_tensor(key)
            if not torch.equal(state[key],x.float()):
                raise ValueError('Base initialization differs')
            if not torch.equal(x,y):
                if not key.startswith('talker.'):
                    raise ValueError('Unexpected changed component')
                state[key].copy_(x.float().lerp(y.float(),0.5)); changed.append(key)
    if len(changed) != 233:
        raise ValueError('Halfway adapter coverage changed')
    text_adapter(wrapper.model, cfg['rank'], cfg['dropout'])
    return wrapper


def checkpoint_path(run, step):
    return run/f'checkpoint-{step:06d}.pt'


def train(protocol, until):
    import torch
    from peft import get_peft_model_state_dict
    cfg = protocol['config']; run = Path(cfg['output'])
    if until not in [0]+cfg['checkpoints']:
        raise ValueError('Unregistered checkpoint')
    wrapper = load_model(protocol); model = wrapper.model
    freeze = frozen_digest(model)
    if (run/'frozen-backbone.json').exists():
        if freeze != read(run/'frozen-backbone.json')['sha256']:
            raise ValueError('Frozen backbone differs')
    else:
        write(run/'frozen-backbone.json', dict(sha256=freeze,trainable=[n for n,p in model.named_parameters() if p.requires_grad],
            trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),dtype='float32'))
    free,total=torch.cuda.mem_get_info()
    fraction=min(0.84,(free-256*2**20)/total)
    if fraction<=0: raise ValueError('Insufficient free GPU memory')
    torch.cuda.set_per_process_memory_fraction(fraction)
    model.talker.to('cuda')
    params = [p for p in model.talker.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params,lr=cfg['peak_lr'],weight_decay=0.01,fused=True)
    latest = read(run/'latest.json')['step'] if (run/'latest.json').exists() else 0
    frame_limit, token_limit = cfg['frame_limit'], cfg['token_limit']
    if (run/'latest.json').exists():
        receipt = read(run/'latest.json'); path = checkpoint_path(run,latest)
        if digest(path) != receipt['sha256']:
            raise ValueError('Checkpoint changed')
        saved = torch.load(path,map_location='cpu',weights_only=True)
        if saved['protocol_sha256'] != digest(run/'protocol.json'):
            raise ValueError('Checkpoint protocol changed')
        load_adapter(model.talker,saved['adapter']); optimizer.load_state_dict(saved['optimizer'])
        torch.set_rng_state(saved['rng_cpu']); torch.cuda.set_rng_state(saved['rng_cuda'])
        frame_limit, token_limit = saved['frame_limit'], saved['token_limit']
    if latest > until or (latest == until and checkpoint_path(run,until).exists()):
        raise ValueError('Checkpoint already trained or sequence reversed')
    check_optimizer(optimizer)
    def persist(step, losses=None):
        if frozen_digest(model) != freeze:
            raise ValueError('Frozen speech backbone was modified')
        state = {k:v.detach().cpu() for k,v in get_peft_model_state_dict(model.talker).items()}
        if len(state) != 4 or any(not k.startswith('text_projection.') for k in state):
            raise ValueError('Unexpected checkpoint parameters')
        path = checkpoint_path(run,step)
        save(path, dict(step=step,protocol_sha256=digest(run/'protocol.json'),adapter=state,optimizer=optimizer.state_dict(),
            rng_cpu=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state(),frame_limit=frame_limit,token_limit=token_limit,
            frozen_backbone_sha256=freeze,losses=losses))
        write(run/'latest.json',dict(step=step,sha256=digest(path)))
        write(run/f'checkpoint-{step:06d}.json',dict(step=step,sha256=digest(path),frozen_backbone_verified=True,losses=losses))
    backup, features = Path(cfg['backup']), Path(cfg['features'])
    train_rows = rows(backup/'cohorts/train.jsonl')
    canonical = {r['id']:r for r in rows(backup/'dataset-v1/manifest.jsonl')}
    data = {}
    for name, expected in protocol['feature_shards'].items():
        if digest(features/name) != expected:
            raise ValueError('Feature shard changed')
        for row in torch.load(features/name,map_location='cpu',weights_only=True):
            if row['id'] in data:
                raise ValueError('Duplicate cached feature')
            data[row['id']] = row
    if set(data) != {r['id'] for r in train_rows}:
        raise ValueError('Incomplete training features')
    cache_path = Path(cfg['validation_cache'])
    if digest(cache_path) != protocol['validation_cache_sha256']:
        raise ValueError('Validation features changed')
    cache = torch.load(cache_path,map_location='cpu',weights_only=True)
    if cache['signature'] != protocol['validation_cache_signature']:
        raise ValueError('Validation feature signature mismatch')
    data.update(cache['data'])
    order = list(train_rows); random.Random(cfg['seed']+cfg['start_epoch']).shuffle(order)
    groups = [order[i*cfg['clips_per_update']:(i+1)*cfg['clips_per_update']] for i in range(cfg['updates'])]
    if any(len(g)!=cfg['clips_per_update'] for g in groups):
        raise ValueError('Pilot exceeds one training pass')
    probes = matched_probe(train_rows,protocol['loss_cases'],data,cfg['seed'])
    tokens = {}
    required = [r for group in groups[latest:until] for r in group]+protocol['loss_cases']+probes
    for r in required:
        for key,reference in ((r['id'],False),(r['ref_id'],True)):
            if (key,reference) not in tokens:
                tokens[key,reference] = ids(wrapper,canonical[key]['text'],reference)
    mode = {'icl':True,'non_streaming':True}
    eval_items = [assignment(r,data,tokens,mode) for r in protocol['loss_cases']]
    probe_items = [assignment(r,data,tokens,mode) for r in probes]
    if until == 0:
        losses=dict(validation=evaluate(model,eval_items),train_matched=evaluate(model,probe_items))
        persist(0,losses); return
    for decoder in (model.talker.model,model.talker.code_predictor.model):
        decoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.talker.train()
    for step in range(latest+1,until+1):
        begun=time.monotonic()
        assigned=[assignment(r,data,tokens,mode) for r in groups[step-1]]
        rate=learning_rate(step,cfg['updates'],cfg['peak_lr'],cfg['end_lr'],cfg['warmup_updates'])
        for group in optimizer.param_groups: group['lr']=rate
        cpu_rng,gpu_rng=torch.get_rng_state(),torch.cuda.get_rng_state()
        retries=0
        while True:
            optimizer.zero_grad(set_to_none=True)
            try:
                loss=backward(model,assigned,frame_limit,token_limit)
                break
            except torch.OutOfMemoryError:
                if frame_limit==1 and token_limit==1: raise
                frame_limit=max(1,frame_limit//2);token_limit=max(1,token_limit//2);retries+=1
                optimizer.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
                torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state(gpu_rng)
        norm=torch.nn.utils.clip_grad_norm_(params,1.0,error_if_nonfinite=True)
        optimizer.step();check_optimizer(optimizer)
        event=dict(step=step,lr=rate,gradient_norm=float(norm),seconds=time.monotonic()-begun,oom_retries=retries,
            frame_limit=frame_limit,token_limit=token_limit,clips_seen=step*cfg['clips_per_update'],
            gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),**loss)
        write(run/f'train-{step:06d}.json',event)
        print(json.dumps(event),flush=True)
    losses=dict(validation=evaluate(model,eval_items),train_matched=evaluate(model,probe_items))
    persist(until,losses)


def speech(protocol, step):
    import numpy as np
    import soundfile as sf
    import torch, torchaudio
    from transformers import WhisperForConditionalGeneration,WhisperProcessor
    from scripts.score_latest_audio import score
    from .evaluate import ASR_REV
    cfg=protocol['config'];run=Path(cfg['output']);out=run/f'eval-{step:06d}'
    out.mkdir(exist_ok=True)
    checkpoint=checkpoint_path(run,step);receipt=read(run/f'checkpoint-{step:06d}.json')
    if digest(checkpoint)!=receipt['sha256']: raise ValueError('Changed checkpoint')
    model=load_model(protocol)
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    load_adapter(model.model.talker,saved['adapter']);model.model.eval()
    if frozen_digest(model.model)!=saved['frozen_backbone_sha256']: raise ValueError('Evaluation backbone differs')
    prompts={};records=[]
    for seed in protocol['seeds']:
        for case in protocol['cases']:
            key=case['reference'];ref=Path(key)
            if digest(ref)!=case['reference_sha256']: raise ValueError('Reference changed')
            if key not in prompts:
                with torch.inference_mode(): prompts[key]=model.create_voice_clone_prompt(ref_audio=key,ref_text=case['reference_text'],x_vector_only_mode=False)
            wav_path=out/f"{seed}-{case['case']}.wav";meta=wav_path.with_suffix('.json')
            if meta.exists():
                record=read(meta)
                if record['checkpoint_sha256']!=receipt['sha256'] or digest(wav_path)!=record['audio_sha256'] or record['protocol_sha256']!=digest(run/'protocol.json'):
                    raise ValueError('Changed speech cache')
            else:
                torch.manual_seed(seed);np.random.seed(seed)
                with torch.inference_mode():
                    waves,sr=model.generate_voice_clone(text=case['text'],language='Auto',voice_clone_prompt=prompts[key],non_streaming_mode=True,**protocol['generation'])
                wave=np.asarray(waves[0])
                if not len(wave) or not np.isfinite(wave).all() or len(wave)/sr>30: raise ValueError('Invalid generated speech')
                sf.write(wav_path,wave,sr,subtype='PCM_16')
                wave,sr=sf.read(wav_path,dtype='float32')
                with torch.inference_mode(): vec=model.model.extract_speaker_embedding(audio=wave,sr=sr).detach().cpu().flatten()
                refvec=prompts[key][0].ref_spk_embedding.detach().cpu().flatten()
                cosine=torch.nn.functional.cosine_similarity(vec[None],refvec[None]).item()
                record=dict(**case,seed=seed,wav=wav_path.name,audio_sha256=digest(wav_path),seconds=len(wave)/sr,cosine=cosine,
                    checkpoint_sha256=receipt['sha256'],protocol_sha256=digest(run/'protocol.json'))
                write(meta,record)
            if step==0 and seed==2026 and case['case']=='custom-001':
                if record['audio_sha256']!=protocol['waveform_baseline_sha256']:
                    raise ValueError('Initial adapter does not reproduce the accepted halfway voice exactly')
                write(run/'halfway-parity.json',dict(exact_waveform_match=True,sha256=record['audio_sha256']))
            records.append(record)
            print(f"Speech checkpoint {step}: {len(records)}/{len(protocol['cases'])*2}",flush=True)
    del model,prompts,saved;gc.collect()
    asr=Path(cfg['asr']);manifest=read(asr/'download-manifest.json')
    if manifest['repo']!='BuzzASR/mongolian' or manifest['revision']!=ASR_REV: raise ValueError('ASR revision mismatch')
    for name,item in manifest['files'].items():
        if digest(asr/name)!=item['sha256']: raise ValueError('ASR file changed')
    processor=WhisperProcessor.from_pretrained(asr,local_files_only=True)
    model=WhisperForConditionalGeneration.from_pretrained(asr,torch_dtype=torch.float32,local_files_only=True).to('cuda').eval()
    for record in records:
        score_path=out/f"score-{record['seed']}-{record['case']}.json"
        if score_path.exists():
            result=read(score_path)
            if result['audio_sha256']!=record['audio_sha256'] or result['asr_revision']!=ASR_REV or result['decoding']!=protocol['decoding']: raise ValueError('ASR cache changed')
        else:
            wave,sr=sf.read(out/record['wav'],dtype='float32')
            wave=torchaudio.functional.resample(torch.from_numpy(wave),sr,16000).numpy()
            batch=processor(wave,sampling_rate=16000,return_tensors='pt',return_attention_mask=True).to('cuda')
            with torch.inference_mode(): generated=model.generate(batch.input_features,attention_mask=batch.attention_mask,**protocol['decoding'])
            hypothesis=processor.batch_decode(generated,skip_special_tokens=True)[0].strip()
            result=dict(record,hypothesis=hypothesis,score=score(record['text'],hypothesis,True),asr_revision=ASR_REV,decoding=protocol['decoding'])
            write(score_path,result)
        record.update(result)
    report=dict(step=step,summary=aggregate(records),records=records,checkpoint_sha256=receipt['sha256'])
    if step: report['gate']=joint_gate(read(run/'eval-000000/results.json')['records'],records,cfg['max_clip_cosine_drop'])
    write(out/'results.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='records'}),flush=True)


def render(run, reports, status):
    page='<!doctype html><html lang="en"><meta charset="utf-8"><title>Targeted voice-preserving retraining</title>'
    if status=='running':page+='<meta http-equiv="refresh" content="60">'
    page+='<style>body{font:17px system-ui;background:#111821;color:#e7eef8;max-width:1100px;margin:40px auto;padding:0 20px}td,th{padding:12px;text-align:left}a{color:#91caff}audio{width:100%}</style><h1>Targeted retraining</h1><p>'+html.escape(status)+'</p><p>Exact halfway backbone; only text-projection LoRA is trainable. Ten development speakers plus your custom reference, two seeds. Internal voice scores are a screen; listening is still required.</p><table><tr><th>Step</th><th>Dev CER</th><th>Dev WER</th><th>Dev voice</th><th>Custom WER</th><th>Custom voice</th><th>Joint screen</th></tr>'
    for report in reports:
        s=report['step'];d=report['summary']['development'];c=report['summary']['custom']
        gate='Baseline' if s==0 else ('Eligible for listening' if report['gate']['eligible'] else 'Not eligible')
        page+=f'<tr><td>{s}</td><td>{100*d["pooled_cer"]:.2f}%</td><td>{100*d["pooled_wer"]:.2f}%</td><td>{d["speaker_macro_cosine"]:.5f}</td><td>{100*c["pooled_wer"]:.2f}%</td><td>{c["speaker_macro_cosine"]:.5f}</td><td>{gate}</td></tr>'
    page+='</table><h2>Learning checks</h2><table><tr><th>Step</th><th>Matched train loss</th><th>Familiar validation loss</th><th>Unseen validation loss</th></tr>'
    for report in reports:
        metrics=read(run/f'checkpoint-{report["step"]:06d}.json')['losses']
        if metrics:
            page+=f'<tr><td>{report["step"]}</td><td>{metrics["train_matched"]["all"]["combined"]:.4f}</td><td>{metrics["validation"]["familiar"]["combined"]:.4f}</td><td>{metrics["validation"]["unseen"]["combined"]:.4f}</td></tr>'
    page+='</table><h2>Same listening example at each checkpoint</h2>'
    if reports:
        from urllib.parse import quote
        example=next(r for r in reports[0]['records'] if r['case']=='custom-001')
        reference=quote(os.path.relpath(example['reference'],run).replace('\\','/'),safe='/.:')
        page+='<h3>Reference voice</h3><audio controls preload="none" src="'+reference+'"></audio><p>'+html.escape(example['text'])+'</p>'
    for report in reports:
        row=next(r for r in report['records'] if r['case']=='custom-001' and r['seed']==2026)
        page+=f'<h3>Step {report["step"]}</h3><audio controls preload="none" src="eval-{report["step"]:06d}/{row["wav"]}"></audio>'
    page+='<p><a href="protocol.json">Registered protocol</a> · <a href="selection.json">Selection status</a></p></html>'
    (run/'report.html').write_text(page,encoding='utf-8')


def orchestrate(cfg, resume=False):
    run=external(Path(cfg['output']))
    protocol=read(run/'protocol.json') if resume else setup(cfg)
    if protocol['config']!=cfg or protocol['source']!=source_hashes(): raise ValueError('Run configuration/source changed')
    reports=[];voice_bad=0;eligible=[];phase='initializing'
    try:
        for step in [0]+cfg['checkpoints']:
            for stage in ('train','speech'):
                done=checkpoint_path(run,step).exists() if stage=='train' else (run/f'eval-{step:06d}/results.json').exists()
                if done: continue
                phase=f'{stage}-{step:06d}'
                write(run/'status.json',dict(status='running',phase=phase,updated=time.time()))
                with (run/f'{phase}.log').open('a',encoding='utf-8') as log:
                    process=subprocess.run([sys.executable,'-m','training.targeted_voice','--config',str(Path(cfg['config_path'])),
                        '--stage',stage,'--step',str(step)],cwd=Path(__file__).resolve().parents[1],stdout=log,stderr=subprocess.STDOUT)
                if process.returncode: raise RuntimeError(f'{phase} failed; see its log')
            report=read(run/f'eval-{step:06d}/results.json');reports.append(report)
            if step:
                voice_bad=0 if report['gate']['voice_pass'] else voice_bad+1
                if report['gate']['eligible']:eligible.append(report)
            selected=min(eligible,key=lambda r:(r['summary']['development']['pooled_wer'],r['summary']['development']['pooled_cer'])) if eligible else reports[0]
            write(run/'selection.json',dict(selected_step=selected['step'],status='candidate_needs_listening' if eligible else 'retain_halfway_baseline',
                evaluated_steps=[r['step'] for r in reports],voice_regression_streak=voice_bad,global_model_selection_changed=False))
            render(run,reports,'running')
            print(json.dumps(dict(completed_step=step,summary=report['summary'],gate=report.get('gate',{}).get('eligible'))),flush=True)
            if voice_bad>=cfg['voice_patience']:break
        stop='voice_guard' if voice_bad>=cfg['voice_patience'] else 'pilot_budget'
        write(run/'status.json',dict(status='complete',phase='complete',stop_reason=stop,evaluated_steps=[r['step'] for r in reports],updated=time.time()))
        render(run,reports,'complete: '+stop)
    except BaseException as error:
        write(run/'status.json',dict(status='failed',phase=phase,error=repr(error),updated=time.time()))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--stage',choices=('run','train','speech'),default='run')
    parser.add_argument('--step',type=int,default=0)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();cfg=read(args.config);cfg['config_path']=str(args.config.resolve())
    if cfg['checkpoints']!=list(range(cfg['eval_every'],cfg['updates']+1,cfg['eval_every'])): raise ValueError('Checkpoint schedule mismatch')
    os.environ['HF_HUB_OFFLINE']=os.environ['TRANSFORMERS_OFFLINE']='1'
    os.environ['NUMBA_CACHE_DIR']=cfg['numba_cache'];os.environ['TEMP']=os.environ['TMP']=cfg['tmp']
    import torch
    torch.set_num_threads(4)
    if args.stage=='run':orchestrate(cfg,args.resume)
    else:
        protocol=read(Path(cfg['output'])/'protocol.json')
        if protocol['config']!=cfg:raise ValueError('Configuration differs from registered run')
        (train if args.stage=='train' else speech)(protocol,args.step)


if __name__=='__main__':
    main()
