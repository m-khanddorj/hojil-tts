"""Fresh generated preferences from the audited update-8 speaker policy.

Generation/scoring follows the historical balanced collector, kept immutable.
The original-reference x-vector remains the reward target and stored input.
"""
import argparse
import copy
from collections import defaultdict
import hashlib
import os
from pathlib import Path
import shutil
import time

from .engine import digest,ids,read,save,write
from .policy import external
from .preference_pilot import runtime
from . import balanced_preference_rollouts as previous_collection
from . import speaker_preference_training as previous_fit
from .balanced_preference_rollouts import capture_batch

PLANNED_FIT=dict(rank=16,radius_fraction=0.05,seed=9017,updates=16,checkpoints=[4,8,16],
    pairs_per_update=4,peak_lr=.001,end_lr=.0001,warmup_updates=2,beta=.1,
    weight_decay=0.0,gradient_clip=1.0)

def refresh_cases(cases):
    if len({c['case'] for c in cases})!=len(cases):raise ValueError('Duplicate source cases')
    result=copy.deepcopy(cases)
    for i,c in enumerate(result):c['batch_seed']=17301+i
    if {c['batch_seed'] for c in result}&{c['batch_seed'] for c in cases}:
        raise ValueError('New batch seeds overlap original seeds')
    return result

def sources():
    return dict(previous_collection.sources(),**previous_fit.source_hashes(),
        **{'refreshed_speaker_rollouts.py':digest(Path(__file__))})

def adapter_digest(model):
    h=hashlib.sha256()
    for name,value in model.speaker_preference_adapter.state_dict().items():
        h.update(name.encode());h.update(str((tuple(value.shape),value.dtype)).encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def verify_initial(initial):
    run=Path(initial['run']);p=previous_fit.verify(run)
    if digest(run/'protocol.json')!=initial['protocol_sha256']:raise ValueError('Initial protocol changed')
    path=previous_fit.checkpoint(run,initial['step'])
    if digest(path)!=initial['checkpoint_sha256']:raise ValueError('Initial checkpoint changed')
    if digest(run/f"verification-{initial['step']:06d}.json")!=initial['audit_sha256']:
        raise ValueError('Initial audit changed')
    if digest(run/f"eval-{initial['step']:06d}/results.json")!=initial['speech_sha256']:
        raise ValueError('Initial speech results changed')
    state=previous_fit.read_checkpoint(run,initial['step'],p)
    return p,state

def prepare(parent,step,run):
    if step!=8:raise ValueError('This stage is registered from update 8 only')
    p=previous_fit.verify(parent)
    audit_path=parent/f'verification-{step:06d}.json';audit=read(audit_path)
    report_path=parent/f'eval-{step:06d}/results.json';report=read(report_path)
    if audit['status']!='verified' or audit['step']!=step or audit['recordings']!=26:
        raise ValueError('Initial screening audit missing')
    state=previous_fit.read_checkpoint(parent,step,p)
    if report['checkpoint_sha256']!=digest(previous_fit.checkpoint(parent,step)):
        raise ValueError('Initial speech checkpoint differs')
    pool=Path(p['pool']);source=previous_collection.verify(pool)
    control=next(r for r in report['records'] if r['case']=='custom-001' and r['seed']==2026)
    if digest(report_path.parent/control['wav'])!=control['audio_sha256']:raise ValueError('Control audio changed')
    initial=dict(run=str(parent),step=step,protocol_sha256=digest(parent/'protocol.json'),
        checkpoint_sha256=digest(previous_fit.checkpoint(parent,step)),audit_sha256=digest(audit_path),
        speech_sha256=digest(report_path),expected_control_sha256=control['audio_sha256'],
        frozen_backbone_sha256=state['frozen_backbone_sha256'])
    run=external(run.resolve());run.mkdir(exist_ok=False)
    protocol=dict(source,source=sources(),runtime=runtime(),cases=refresh_cases(source['cases']),
        initial_policy=initial,source_pool=str(pool),source_pool_protocol_sha256=digest(pool/'protocol.json'),
        planned_fit=PLANNED_FIT,optimizer_initialization='Fresh AdamW moments; frozen DPO reference is update 8',
        purpose='Fresh joint preference trajectories from the audited update-8 policy; collection only',
        sampling='Same 127 cases and speaker split; four draws with new fixed batch seeds 17301+i',
        goal_achieved=False,global_model_selection_changed=False)
    write(run/'protocol.json',protocol);shutil.copy2(Path(__file__),run/'source.py')
    write(run/'status.json',dict(status='prepared',total=len(protocol['cases'])*4))

def verify(run):
    p=read(run/'protocol.json')
    if p['source']!=sources() or p['runtime']!=runtime() or p['planned_fit']!=PLANNED_FIT:
        raise ValueError('Registered source, runtime or fitting settings changed')
    initial,_=verify_initial(p['initial_policy'])
    source=previous_collection.verify(Path(p['source_pool']))
    if digest(Path(p['source_pool'])/'protocol.json')!=p['source_pool_protocol_sha256']:
        raise ValueError('Source pool changed')
    if p['cases']!=refresh_cases(source['cases']):raise ValueError('Selection changed')
    if any(p[k]!=source[k] for k in ('config','draws','generation','decoding','base_sha256','adapted_sha256')):
        raise ValueError('Collection configuration changed')
    if initial['parent_protocol_sha256']!=p['parent_protocol_sha256']:raise ValueError('Evaluation parent differs')
    return p

def load_policy(p):
    import torch
    original,state=verify_initial(p['initial_policy'])
    wrapper=previous_fit.load(original)
    wrapper.model.speaker_preference_adapter.load_state_dict(state['adapter'],strict=True)
    if any(v.dtype!=torch.float32 or not torch.isfinite(v).all() for v in state['adapter'].values()):
        raise ValueError('Invalid initial adapter')
    wrapper.model.eval()
    return wrapper

def control(run):
    import torch
    import numpy as np
    import soundfile as sf
    from .targeted_voice import frozen_digest
    p=verify(run);wrapper=load_policy(p)
    original=read(Path(p['initial_policy']['run'])/'protocol.json')
    c=next(c for c in original['cases'] if c['case']=='custom-001')
    if digest(c['reference'])!=c['reference_sha256']:raise ValueError('Control reference changed')
    with torch.inference_mode():
        prompt=wrapper.create_voice_clone_prompt(ref_audio=c['reference'],ref_text=c['reference_text'],x_vector_only_mode=False)
        torch.manual_seed(2026);np.random.seed(2026)
        waves,sr=wrapper.generate_voice_clone(text=c['text'],language='Auto',voice_clone_prompt=prompt,
            non_streaming_mode=True,**p['generation'])
    path=run/'initial-control.wav';sf.write(path,waves[0],sr,subtype='PCM_16')
    if digest(path)!=p['initial_policy']['expected_control_sha256']:raise ValueError('Initial update-8 audio parity failed')
    if frozen_digest(wrapper.model)!=p['initial_policy']['frozen_backbone_sha256']:raise ValueError('Initial acoustic model differs')
    write(run/'control.json',dict(exact_waveform_parity=True,audio_sha256=digest(path),
        protocol_sha256=digest(run/'protocol.json'),adapter_sha256=adapter_digest(wrapper.model)))
    print('Update-8 CPU control reproduced exactly',flush=True)

def generate(run,limit=None):
    import numpy as np
    import soundfile as sf
    import torch
    from .targeted_voice import frozen_digest
    p=verify(run);receipt=read(run/'control.json')
    if (not receipt['exact_waveform_parity'] or receipt['protocol_sha256']!=digest(run/'protocol.json')
            or digest(run/'initial-control.wav')!=receipt['audio_sha256']):raise ValueError('Control missing or changed')
    wrapper=load_policy(p);initial_adapter=adapter_digest(wrapper.model)
    if initial_adapter!=receipt['adapter_sha256']:raise ValueError('Initial speaker adapter differs')
    frozen=frozen_digest(wrapper.model)
    write(run/'frozen-model.json',dict(sha256=frozen,trainable_parameters=32768,optimizer_updates=0,source_model_sha256=p['adapted_sha256'],initial_adapter_sha256=initial_adapter))
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(0.84)
    wrapper.model.to('cuda').eval();wrapper.device=torch.device('cuda')
    completed=0
    for ci,case in enumerate(p['cases']):
        if digest(case['reference'])!=case['reference_sha256']:raise ValueError('Reference changed')
        paths=[run/f"{case['case']}-draw-{d}.json" for d in range(p['draws'])]
        if all(path.exists() for path in paths):
            for path in paths:
                r=read(path)
                if r['protocol_sha256']!=digest(run/'protocol.json'):raise ValueError('Cache protocol mismatch')
                for field,sha in (('wav','audio_sha256'),('trajectory','trajectory_sha256')):
                    if digest(run/r[field])!=r[sha]:raise ValueError('Rollout cache changed')
        else:
            with torch.inference_mode():
                prompt=wrapper.create_voice_clone_prompt(ref_audio=case['reference'],ref_text=case['reference_text'],x_vector_only_mode=False)
            torch.manual_seed(case['batch_seed']);np.random.seed(case['batch_seed'])
            start=time.monotonic()
            with torch.inference_mode(),capture_batch(wrapper.model) as captured:
                waves,sr=wrapper.generate_voice_clone(text=[case['text']]*p['draws'],language='Auto',
                    voice_clone_prompt=prompt,non_streaming_mode=True,**p['generation'])
            elapsed=time.monotonic()-start
            if len(waves)!=p['draws'] or sr!=24000:raise ValueError('Batch output mismatch')
            refvec=prompt[0].ref_spk_embedding
            for draw,(wave,path) in enumerate(zip(waves,paths)):
                if not len(wave) or not np.isfinite(wave).all() or len(wave)/sr>30:raise ValueError('Invalid speech')
                sample=f"{case['case']}-draw-{draw}";wav=run/(sample+'.wav');trajectory=run/(sample+'.pt')
                if path.exists():
                    # Resuming a partially saved batch regenerates the complete
                    # RNG stream; never overwrite an already committed sample.
                    import tempfile
                    with tempfile.TemporaryDirectory(dir=run) as temp:
                        check=Path(temp)/'check.wav';sf.write(check,wave,sr,subtype='PCM_16')
                        old=read(path)
                        if digest(check)!=old['audio_sha256'] or digest(wav)!=old['audio_sha256'] or digest(trajectory)!=old['trajectory_sha256'] or old['protocol_sha256']!=digest(run/'protocol.json'):
                            raise ValueError('Partial-batch replay mismatch')
                    continue
                sf.write(wav,wave,sr,subtype='PCM_16');wave,sr=sf.read(wav,dtype='float32')
                with torch.inference_mode():vec=wrapper.model.extract_speaker_embedding(audio=wave,sr=sr)
                cosine=float(torch.nn.functional.cosine_similarity(vec.flatten()[None],refvec.flatten()[None]))
                save(trajectory,dict(codes=captured['codes'][draw],first_sequence=captured['sequences'][draw:draw+1],
                    terminated=captured['terminated'][draw],ref_codes=prompt[0].ref_code.detach().cpu(),
                    ref_xvec=refvec.detach().cpu(),input_ids=ids(wrapper,case['text']).cpu(),
                    ref_ids=ids(wrapper,case['reference_text'],True).cpu(),batch_seed=case['batch_seed'],draw=draw,
                    protocol_sha256=digest(run/'protocol.json')))
                write(path,dict(**case,sample_id=sample,draw=draw,origin='model_generated',wav=wav.name,
                    audio_sha256=digest(wav),trajectory=trajectory.name,trajectory_sha256=digest(trajectory),
                    terminated=captured['terminated'][draw],frames=len(captured['codes'][draw]),seconds=len(wave)/sr,
                    cosine=cosine,batch_generation_seconds=elapsed,protocol_sha256=digest(run/'protocol.json')))
        completed+=p['draws']
        write(run/'status.json',dict(status='generating',completed=completed,total=len(p['cases'])*p['draws'],
            last=case['case'],gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),updated=time.time()))
        print(f"Refreshed rollouts {completed}/{len(p['cases'])*p['draws']}",flush=True)
        if limit and ci+1>=limit:return
    if frozen_digest(wrapper.model)!=frozen or adapter_digest(wrapper.model)!=initial_adapter:raise ValueError('Model or speaker adapter changed during generation')
    write(run/'generation-verification.json',dict(frozen_model_unchanged=True,speaker_adapter_unchanged=True,initial_adapter_sha256=initial_adapter,recordings=completed,
        frozen_backbone_sha256=frozen,protocol_sha256=digest(run/'protocol.json')))
    write(run/'status.json',dict(status='generated',completed=completed,updated=time.time()))


def score_all(run):
    import soundfile as sf
    import torch,torchaudio
    from transformers import WhisperForConditionalGeneration,WhisperProcessor
    from scripts.score_latest_audio import score
    from .evaluate import ASR_REV,pooled
    from .voice_preferences import joint_pairs
    p=verify(run)
    if not read(run/'generation-verification.json')['frozen_model_unchanged']:raise ValueError('Generation incomplete')
    asr=Path(p['config']['asr']);manifest=read(asr/'download-manifest.json')
    if manifest['repo']!='BuzzASR/mongolian' or manifest['revision']!=ASR_REV:raise ValueError('ASR mismatch')
    for name,item in manifest['files'].items():
        if digest(asr/name)!=item['sha256']:raise ValueError('ASR file changed')
    processor=WhisperProcessor.from_pretrained(asr,local_files_only=True)
    model=WhisperForConditionalGeneration.from_pretrained(asr,torch_dtype=torch.float32,local_files_only=True).to('cuda').eval()
    records=[];pairs=[]
    for case in p['cases']:
        candidates=[]
        for draw in range(p['draws']):
            sample=f"{case['case']}-draw-{draw}";r=read(run/(sample+'.json'));path=run/('score-'+sample+'.json')
            if r['protocol_sha256']!=digest(run/'protocol.json') or any(r[k]!=v for k,v in case.items()):raise ValueError('Record provenance mismatch')
            for field,sha in (('wav','audio_sha256'),('trajectory','trajectory_sha256')):
                if digest(run/r[field])!=r[sha]:raise ValueError('Rollout changed')
            if path.exists():
                result=read(path)
                if any(result[k]!=v for k,v in r.items()) or result['asr_revision']!=ASR_REV or result['decoding']!=p['decoding']:raise ValueError('ASR cache changed')
            else:
                wave,sr=sf.read(run/r['wav'],dtype='float32')
                wave=torchaudio.functional.resample(torch.from_numpy(wave),sr,16000).numpy()
                batch=processor(wave,sampling_rate=16000,return_tensors='pt',return_attention_mask=True).to('cuda')
                with torch.inference_mode():tokens=model.generate(batch.input_features,attention_mask=batch.attention_mask,**p['decoding'])
                hypothesis=processor.batch_decode(tokens,skip_special_tokens=True)[0].strip()
                result=dict(r,hypothesis=hypothesis,score=score(r['text'],hypothesis,True),asr_revision=ASR_REV,decoding=p['decoding'])
                write(path,result)
            records.append(result)
            if result['terminated']:candidates.append(result)
        pairs.extend(dict(pair,preference_split=case['preference_split']) for pair in joint_pairs(candidates))
        write(run/'status.json',dict(status='scoring',completed=len(records),total=len(p['cases'])*p['draws'],pairs=len(pairs),updated=time.time()))
    summaries={}
    for split in ('train','validation'):
        subset=[r for r in records if r['preference_split']==split];speakers=defaultdict(list)
        for r in subset:speakers[r['speaker_id']].append(r['cosine'])
        summaries[split]=dict(**pooled([r['score'] for r in subset]),recordings=len(subset),speakers=len(speakers),
            speaker_macro_cosine=sum(sum(x)/len(x) for x in speakers.values())/len(speakers),
            usable_pairs=sum(pair['preference_split']==split for pair in pairs),
            cases_with_pairs=len({pair['case'] for pair in pairs if pair['preference_split']==split}))
    write(run/'results.json',dict(summary=summaries,records=records,pairs=pairs,
        truncated=sum(not r['terminated'] for r in records),goal_achieved=False,
        scope='Original training data; preference fitting/validation only, no development/custom quality claim'))
    write(run/'status.json',dict(status='complete',recordings=len(records),pairs=len(pairs),updated=time.time()))
    print(summaries,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent',type=Path);parser.add_argument('--step',type=int,default=8)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--phase',choices=['prepare','control','generate','score'],required=True)
    parser.add_argument('--limit',type=int)
    args=parser.parse_args()
    cfg=read((args.parent if args.phase=='prepare' else args.output)/'protocol.json')['config']
    os.environ['NUMBA_CACHE_DIR']=cfg['numba_cache'];os.environ['TEMP']=os.environ['TMP']=cfg['tmp']
    os.environ['HF_HUB_OFFLINE']=os.environ['TRANSFORMERS_OFFLINE']='1'
    import torch
    torch.set_num_threads(4)
    if args.phase=='prepare':prepare(args.parent,args.step,args.output)
    elif args.phase=='control':control(args.output)
    elif args.phase=='generate':generate(args.output,args.limit)
    else:score_all(args.output)

if __name__=='__main__':main()
