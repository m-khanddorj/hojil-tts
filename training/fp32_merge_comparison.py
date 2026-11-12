"""Controlled half-strength FP32 LoRA merge versus the accepted endpoint blend.

An inference comparison, with no optimizer or model promotion. CPU synthesis
and GPU ASR scoring are separate phases to avoid competing training jobs.
"""
import argparse
import os
from pathlib import Path
import shutil

from .engine import digest,read,write
from .policy import external
from .preference_pilot import runtime
from .targeted_voice import aggregate,frozen_digest
from .voice_strength_search import goal_screen


def merge_fp32(model,adapter,strength):
    import torch
    import math
    if not math.isfinite(strength) or not 0<=strength<=1:raise ValueError('Invalid strength')
    names=sorted(k.removesuffix('.lora_A.weight') for k in adapter if k.endswith('.lora_A.weight'))
    expected={n+suffix for n in names for suffix in ('.lora_A.weight','.lora_B.weight')}
    if not names or set(adapter)!=expected:raise ValueError('Incomplete or extra adapter tensors')
    state=model.state_dict()
    # Validate everything before mutating any parameter.
    for name in names:
        a,b=adapter[name+'.lora_A.weight'],adapter[name+'.lora_B.weight'];key='talker.'+name+'.weight'
        if (a.dtype!=torch.float32 or b.dtype!=torch.float32 or a.ndim!=2 or b.ndim!=2
                or not torch.isfinite(a).all() or not torch.isfinite(b).all() or a.shape[0]!=b.shape[1]
                or key not in state or state[key].dtype!=torch.float32 or state[key].shape!=(b.shape[0],a.shape[1])):
            raise ValueError('Invalid FP32 merge tensor')
    with torch.no_grad():
        for name in names:
            if strength==0:continue
            a,b=adapter[name+'.lora_A.weight'],adapter[name+'.lora_B.weight']
            weight=state['talker.'+name+'.weight']
            weight.add_(b.to(weight.device)@a.to(weight.device),alpha=strength)
    model.requires_grad_(False)
    return names


def sources():
    return {n:digest(Path(__file__).parent/n) for n in ('fp32_merge_comparison.py','engine.py',
        'policy.py','preference_pilot.py','preference_rollouts.py','targeted_voice.py','evaluate.py',
        'voice_strength_search.py','../scripts/score_latest_audio.py')}


def prepare(parent,training_run,output):
    p=read(parent/'protocol.json');best=read(training_run/'best.json');meta=read(training_run/'run.json')
    adapter=training_run/best['weights'];merged=training_run/'merged-model/model.safetensors'
    if (digest(adapter)!=best['weights_sha256'] or digest(merged)!=p['adapted_sha256']
            or meta['config']['method']!='lora' or meta['config']['rank']!=16):raise ValueError('Source checkpoint mismatch')
    baseline_path=parent/'eval-000000/results.json';baseline=read(baseline_path)
    expected={(c['case'],s) for c in p['cases'] for s in p['seeds']}
    if len(baseline['records'])!=26 or {(r['case'],r['seed']) for r in baseline['records']}!=expected:raise ValueError('Baseline incomplete')
    for r in baseline['records']:
        if digest(baseline_path.parent/r['wav'])!=r['audio_sha256']:raise ValueError('Baseline waveform changed')
    output=external(output.resolve());output.mkdir(exist_ok=False)
    protocol=dict(config=p['config'],parent=str(parent),parent_protocol_sha256=digest(parent/'protocol.json'),
        baseline=str(baseline_path),baseline_sha256=digest(baseline_path),
        training_run=str(training_run),best_sha256=digest(training_run/'best.json'),
        training_run_sha256=digest(training_run/'run.json'),adapter=str(adapter),adapter_sha256=digest(adapter),
        base_sha256=p['base_sha256'],adapted_sha256=p['adapted_sha256'],strength=.5,
        source=sources(),runtime=runtime(),cases=p['cases'],seeds=p['seeds'],generation=p['generation'],decoding=p['decoding'],
        expected_control_sha256=p['waveform_baseline_sha256'],
        intervention='Base FP32 values plus 0.5 times B_fp32 @ A_fp32, alpha/rank=1; original speaker encoder and codec',
        comparison='Accepted FP32 interpolation of base and BF16 exported epoch-2 endpoint; exact same prompts/seeds/backend',
        kind='inference_precision_comparison_no_optimizer',goal_achieved=False,global_model_selection_changed=False)
    write(output/'protocol.json',protocol);shutil.copy2(Path(__file__),output/'source.py')
    write(output/'status.json',dict(status='prepared',optimizer_updates=0))


def verify(run):
    p=read(run/'protocol.json')
    if p['source']!=sources() or p['runtime']!=runtime() or p['strength']!=.5:raise ValueError('Registered implementation changed')
    for path,sha in ((Path(p['parent'])/'protocol.json',p['parent_protocol_sha256']),
        (Path(p['training_run'])/'best.json',p['best_sha256']),(Path(p['training_run'])/'run.json',p['training_run_sha256']),
        (p['adapter'],p['adapter_sha256']),(p['baseline'],p['baseline_sha256'])):
        if digest(path)!=sha:raise ValueError('Comparison provenance changed')
    return p


def load(p,legacy=False):
    import torch
    from qwen_tts import Qwen3TTSModel
    from safetensors import safe_open
    base=Path(p['config']['base']);adapted=Path(p['config']['adapted'])
    if digest(base/'model.safetensors')!=p['base_sha256'] or digest(adapted/'model.safetensors')!=p['adapted_sha256']:
        raise ValueError('Source weights changed')
    wrapper=Qwen3TTSModel.from_pretrained(str(base),device_map='cpu',torch_dtype=torch.float32,attn_implementation='sdpa',local_files_only=True)
    wrapper.model.requires_grad_(False).eval()
    adapter=torch.load(p['adapter'],map_location='cpu',weights_only=True)
    if len(adapter)!=466:raise ValueError('Expected 233 LoRA modules')
    if legacy:
        state=wrapper.model.state_dict()
        with safe_open(adapted/'model.safetensors',framework='pt',device='cpu') as donor,torch.no_grad():
            for key in list(adapter):
                if key.endswith('.lora_A.weight'):
                    name='talker.'+key.removesuffix('.lora_A.weight')+'.weight'
                    state[name].copy_(state[name].lerp(donor.get_tensor(name).float(),.5))
    else:
        names=merge_fp32(wrapper.model,adapter,p['strength'])
        if len(names)!=233:raise ValueError('Unexpected adapted module count')
    return wrapper


def control(run):
    import numpy as np
    import torch
    import soundfile as sf
    p=verify(run);wrapper=load(p,legacy=True)
    case=next(c for c in p['cases'] if c['case']=='custom-001')
    if digest(case['reference'])!=case['reference_sha256']:raise ValueError('Reference changed')
    with torch.inference_mode():
        prompt=wrapper.create_voice_clone_prompt(ref_audio=case['reference'],ref_text=case['reference_text'],x_vector_only_mode=False)
        torch.manual_seed(2026);np.random.seed(2026)
        waves,sr=wrapper.generate_voice_clone(text=case['text'],language='Auto',voice_clone_prompt=prompt,non_streaming_mode=True,**p['generation'])
    path=run/'legacy-control.wav';sf.write(path,waves[0],sr,subtype='PCM_16')
    if digest(path)!=p['expected_control_sha256']:raise ValueError('Legacy construction does not reproduce accepted audio')
    write(run/'control.json',dict(exact_waveform_parity=True,audio_sha256=digest(path),protocol_sha256=digest(run/'protocol.json')))
    print('Legacy halfway control reproduced exactly',flush=True)


def generate(run):
    import torch
    import numpy as np
    import soundfile as sf
    from .preference_rollouts import capture_generation
    p=verify(run);receipt=read(run/'control.json')
    if (not receipt['exact_waveform_parity'] or receipt['protocol_sha256']!=digest(run/'protocol.json')
            or digest(run/'legacy-control.wav')!=receipt['audio_sha256']):raise ValueError('Control verification missing')
    wrapper=load(p);frozen=frozen_digest(wrapper.model);prompts={};records=[]
    write(run/'model.json',dict(frozen_backbone_sha256=frozen,optimizer_updates=0,forward_dtype='float32',source_adapter_sha256=p['adapter_sha256']))
    for seed in p['seeds']:
        for c in p['cases']:
            ref=c['reference']
            if digest(ref)!=c['reference_sha256']:raise ValueError('Reference changed')
            if ref not in prompts:
                with torch.inference_mode():prompts[ref]=wrapper.create_voice_clone_prompt(ref_audio=ref,ref_text=c['reference_text'],x_vector_only_mode=False)
            wav=run/f"{seed}-{c['case']}.wav";meta=wav.with_suffix('.json')
            if meta.exists():
                r=read(meta)
                if (r['protocol_sha256']!=digest(run/'protocol.json') or r['frozen_backbone_sha256']!=frozen
                        or digest(wav)!=r['audio_sha256'] or any(r[k]!=v for k,v in c.items())):raise ValueError('Speech cache changed')
            else:
                torch.manual_seed(seed);np.random.seed(seed)
                with torch.inference_mode(),capture_generation(wrapper.model) as captured:
                    waves,sr=wrapper.generate_voice_clone(text=c['text'],language='Auto',voice_clone_prompt=prompts[ref],non_streaming_mode=True,**p['generation'])
                if sr!=24000 or not len(waves[0]) or not np.isfinite(waves[0]).all() or len(waves[0])/sr>30:raise ValueError('Invalid speech')
                sf.write(wav,waves[0],sr,subtype='PCM_16');wave,sr=sf.read(wav,dtype='float32')
                with torch.inference_mode():
                    vector=wrapper.model.extract_speaker_embedding(audio=wave,sr=sr).flatten()
                    cosine=float(torch.nn.functional.cosine_similarity(vector[None],prompts[ref][0].ref_spk_embedding.flatten()[None]))
                r=dict(**c,seed=seed,wav=wav.name,audio_sha256=digest(wav),cosine=cosine,seconds=len(wave)/sr,
                    terminated=captured['terminated'],frozen_backbone_sha256=frozen,protocol_sha256=digest(run/'protocol.json'))
                write(meta,r)
            records.append(r);write(run/'status.json',dict(status='generating',completed=len(records),total=26,optimizer_updates=0))
            print(f'FP32 half-merge speech {len(records)}/26',flush=True)
    if frozen_digest(wrapper.model)!=frozen:raise ValueError('Inference weights changed')
    write(run/'generated.json',dict(records=records,frozen_backbone_verified=True))
    write(run/'status.json',dict(status='generated_waiting_for_asr',recordings=26,optimizer_updates=0))


def score_all(run):
    import torch,torchaudio
    import soundfile as sf
    from transformers import WhisperForConditionalGeneration,WhisperProcessor
    from scripts.score_latest_audio import score
    from .evaluate import ASR_REV
    p=verify(run);generated=read(run/'generated.json');records=generated['records']
    if not generated['frozen_backbone_verified'] or len(records)!=26:raise ValueError('Generation incomplete')
    asr=Path(p['config']['asr']);manifest=read(asr/'download-manifest.json')
    if manifest['repo']!='BuzzASR/mongolian' or manifest['revision']!=ASR_REV:raise ValueError('ASR mismatch')
    for name,item in manifest['files'].items():
        if digest(asr/name)!=item['sha256']:raise ValueError('ASR changed')
    processor=WhisperProcessor.from_pretrained(asr,local_files_only=True)
    model=WhisperForConditionalGeneration.from_pretrained(asr,torch_dtype=torch.float32,local_files_only=True).to('cuda').eval()
    for r in records:
        if digest(run/r['wav'])!=r['audio_sha256'] or r['protocol_sha256']!=digest(run/'protocol.json'):raise ValueError('Speech changed')
        path=run/f"score-{r['seed']}-{r['case']}.json"
        if path.exists():
            scored=read(path)
            if any(scored[k]!=v for k,v in r.items()) or scored['asr_revision']!=ASR_REV or scored['decoding']!=p['decoding']:raise ValueError('ASR cache mismatch')
        else:
            wave,sr=sf.read(run/r['wav'],dtype='float32');wave=torchaudio.functional.resample(torch.from_numpy(wave),sr,16000).numpy()
            batch=processor(wave,sampling_rate=16000,return_tensors='pt',return_attention_mask=True).to('cuda')
            with torch.inference_mode():tokens=model.generate(batch.input_features,attention_mask=batch.attention_mask,**p['decoding'])
            hypothesis=processor.batch_decode(tokens,skip_special_tokens=True)[0].strip()
            scored=dict(r,hypothesis=hypothesis,score=score(r['text'],hypothesis,True),asr_revision=ASR_REV,decoding=p['decoding'])
            write(path,scored)
        r.update(scored)
    summary=aggregate(records);truncated=sum(not r['terminated'] for r in records)
    write(run/'results.json',dict(records=records,summary=summary,baseline=read(p['baseline'])['summary'],
        truncated=truncated,goal_screen=goal_screen(summary) and truncated==0,goal_achieved=False,optimizer_updates=0,
        global_model_selection_changed=False,kind=p['kind']))
    write(run/'status.json',dict(status='complete',optimizer_updates=0,goal_achieved=False));print(summary,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent',type=Path);parser.add_argument('--training-run',type=Path)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--phase',choices=['prepare','control','generate','score'],required=True)
    args=parser.parse_args()
    if args.phase=='prepare':prepare(args.parent,args.training_run,args.output);return
    p=verify(args.output);cfg=p['config']
    os.environ['NUMBA_CACHE_DIR']=cfg['numba_cache'];os.environ['TEMP']=os.environ['TMP']=cfg['tmp']
    os.environ['HF_HUB_OFFLINE']=os.environ['TRANSFORMERS_OFFLINE']='1'
    import torch
    torch.set_num_threads(4)
    {'control':control,'generate':generate,'score':score_all}[args.phase](args.output)


if __name__=='__main__':main()
