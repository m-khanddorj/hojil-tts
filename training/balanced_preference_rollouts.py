"""Speaker-balanced full-LoRA preference data with batched sampled trajectories.

Collection only: this does not change model weights or inference selection.
The four draws share a registered batch seed; they are not the old per-sample
seed protocol. CPU two-seed development/custom evaluation remains unchanged.
"""
import argparse
from collections import defaultdict
from contextlib import contextmanager
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .engine import digest,ids,read,rows,save,write
from .policy import external,validate_cohorts
from .preference_rollouts import select_cases
from .preference_pilot import runtime
from data.pipeline import text_key


def separate_transcripts(cases):
    """Keep speaker allocation; remove fitting cases overlapping validation text."""
    val=[c for c in cases if c['preference_split']=='validation']
    train=[c for c in cases if c['preference_split']=='train']
    if {c['speaker_id'] for c in val}&{c['speaker_id'] for c in train}:
        raise ValueError('Speaker overlap')
    excluded_text={text_key(c[k]) for c in val for k in ('text','ref_text')}
    excluded=[c['id'] for c in train if any(text_key(c[k]) in excluded_text for k in ('text','ref_text'))]
    return [c for c in cases if c['id'] not in excluded],excluded


@contextmanager
def capture_batch(model):
    """Capture sampled codecs and first-codebook EOS for each batch member."""
    import torch
    captured={};original=model.generate;talker_original=model.talker.generate
    def talker(*args,**kwargs):
        result=talker_original(*args,**kwargs)
        captured['sequences']=result.sequences.detach().cpu().clone()
        return result
    def full(*args,**kwargs):
        result=original(*args,**kwargs)
        captured['codes']=[c.detach().cpu().clone() for c in result[0]]
        return result
    model.generate=full;model.talker.generate=talker
    try:yield captured
    finally:model.generate=original;model.talker.generate=talker_original
    sequences,codes=captured['sequences'],captured['codes']
    if sequences.ndim!=2 or len(sequences)!=len(codes):raise ValueError('Batch size mismatch')
    eos=model.config.talker_config.codec_eos_token_id
    captured['terminated']=[]
    for seq,c in zip(sequences,codes):
        if c.ndim!=2 or c.shape[1]!=16 or not len(c):raise ValueError('Invalid codec trajectory')
        stops=(seq==eos).nonzero().flatten()
        terminated=bool(len(stops));captured['terminated'].append(terminated)
        if terminated:
            stop=int(stops[0])
            if stop!=len(c) or not torch.equal(seq[:stop],c[:,0]):
                raise ValueError('Sampled batch trajectory disagrees with first codebook/EOS')


def sources():
    folder=Path(__file__).parent
    return {n:digest(folder/n) for n in ('balanced_preference_rollouts.py','speaker_reference_adaptation.py',
        'preference_rollouts.py','preference_pilot.py','voice_preferences.py','targeted_voice.py',
        'sequence.py','engine.py','policy.py','evaluate.py','../scripts/score_latest_audio.py')}


def prepare(parent,run):
    from . import reference_protected as protected
    protected.configure_helpers()
    p=read(parent/'protocol.json');cfg=p['config']
    if p['source']!=protected.source_hashes():raise ValueError('Parent source changed')
    backup=Path(cfg['backup']);recovery=read(backup/'recovery-verification.json')['cohorts']
    for name in ('train','unseen-validation','familiar-validation'):
        if digest(backup/f'cohorts/{name}.jsonl')!=recovery[name]['sha256']:raise ValueError('Changed cohort')
    inventory=read(backup/'dataset-v1/dataset-inventory.json')['files']
    manifest=backup/'dataset-v1/manifest.jsonl'
    if digest(manifest)!=inventory['manifest.jsonl']['sha256']:raise ValueError('Manifest changed')
    train=rows(backup/'cohorts/train.jsonl')
    held=rows(backup/'cohorts/unseen-validation.jsonl')
    held += [r for r in rows(backup/'cohorts/familiar-validation.jsonl') if r.get('eligible')]
    canonical=validate_cohorts(train,held,rows(manifest))
    excluded=[t for c in p['cases'] for t in (c['text'],c['reference_text'])]
    chosen=select_cases(train,excluded,speakers=64,per_speaker=2)
    chosen,overlap=separate_transcripts(chosen)
    cases=[]
    for i,row in enumerate(chosen):
        ref=canonical[row['ref_id']];path=backup/'dataset-v1'/ref['audio']
        if digest(path)!=inventory[ref['audio']]['sha256']:raise ValueError('Reference changed')
        cases.append(dict(case=f'balanced-{i:03d}',row_id=row['id'],ref_id=row['ref_id'],
            speaker_id=row['speaker_id'],text=row['text'],reference=str(path),reference_text=ref['text'],
            reference_sha256=digest(path),partition='train',preference_split=row['preference_split'],
            batch_seed=7301+i))
    coverage={split:dict(cases=sum(c['preference_split']==split for c in cases),
        speakers=len({c['speaker_id'] for c in cases if c['preference_split']==split})) for split in ('train','validation')}
    if coverage['train']['speakers']<40 or coverage['train']['cases']<72:raise ValueError('Insufficient fitting diversity')
    run=external(run.resolve());run.mkdir(exist_ok=False)
    protocol=dict(parent=str(parent),parent_protocol_sha256=digest(parent/'protocol.json'),config=cfg,
        source=sources(),runtime=runtime(),base_sha256=p['base_sha256'],adapted_sha256=p['adapted_sha256'],
        canonical_manifest_sha256=digest(manifest),train_cohort_sha256=recovery['train']['sha256'],
        cases=cases,coverage=coverage,excluded_overlap_rows=overlap,draws=4,
        generation=p['generation'],decoding=p['decoding'],
        generation_backend='CUDA FP32 SDPA, TF32 disabled, four identical-condition batch members; CPU codec decode',
        sampling='One fixed seed per case batch, four indexed draws; not old per-sample-seed parity',
        selection='64 hash-selected original training speakers, two 2-9s targets each; first 16 speakers held out from future preference fitting',
        purpose='Broader joint CER/voice preferences from the full epoch-2 LoRA policy; collection only',
        goal_achieved=False,global_model_selection_changed=False)
    write(run/'protocol.json',protocol);shutil.copy2(Path(__file__),run/'source.py')
    write(run/'status.json',dict(status='prepared',total=len(cases)*4))


def verify(run):
    p=read(run/'protocol.json')
    if p['source']!=sources() or p['runtime']!=runtime():raise ValueError('Source/runtime changed')
    if digest(Path(p['parent'])/'protocol.json')!=p['parent_protocol_sha256']:raise ValueError('Parent changed')
    return p


def generate(run,limit=None):
    import numpy as np
    import soundfile as sf
    import torch
    from .speaker_reference_adaptation import load
    from .targeted_voice import frozen_digest
    p=verify(run);wrapper=load(p)
    frozen=frozen_digest(wrapper.model)
    write(run/'frozen-model.json',dict(sha256=frozen,trainable_parameters=0,source_model_sha256=p['adapted_sha256']))
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
        print(f"Balanced rollouts {completed}/{len(p['cases'])*p['draws']}",flush=True)
        if limit and ci+1>=limit:return
    if frozen_digest(wrapper.model)!=frozen:raise ValueError('Model changed during generation')
    write(run/'generation-verification.json',dict(frozen_model_unchanged=True,recordings=completed,
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
    parser.add_argument('--parent',type=Path);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--phase',choices=['prepare','generate','score','run'],required=True)
    parser.add_argument('--limit',type=int)
    args=parser.parse_args()
    if args.phase=='prepare':prepare(args.parent,args.output);return
    p=verify(args.output);cfg=p['config']
    os.environ['NUMBA_CACHE_DIR']=cfg['numba_cache'];os.environ['TEMP']=os.environ['TMP']=cfg['tmp']
    os.environ['HF_HUB_OFFLINE']=os.environ['TRANSFORMERS_OFFLINE']='1'
    import torch
    torch.set_num_threads(4)
    if args.phase=='run':
        for phase in ('generate','score'):
            subprocess.run([sys.executable,'-m','training.balanced_preference_rollouts','--output',str(args.output),'--phase',phase],check=True)
    elif args.phase=='generate':generate(args.output,args.limit)
    else:score_all(args.output)


if __name__=='__main__':main()
