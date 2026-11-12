"""Reference-codec weight routing on audited centered speaker checkpoint 16."""
import argparse
import os
from pathlib import Path
import shutil

from .engine import digest,read,write,save
from .policy import external
from .preference_pilot import runtime
from .targeted_voice import aggregate
from . import centered_speaker_training as initial_trainer
from .centered_speaker_training import frozen_digest
from .reference_codec_routing import ReferenceCodecRouting,weight_modules
from .voice_strength_search import goal_screen


def sources():
    return dict(initial_trainer.source_hashes(),**{n:digest(Path(__file__).parent/n)
        for n in ('reference_codec_comparison.py','reference_codec_routing.py','preference_rollouts.py')})


def prepare(initial_run,step,output):
    if step!=16:raise ValueError('This comparison is planned from centered update 16')
    initial=initial_trainer.verify(initial_run)
    state=initial_trainer.read_checkpoint(initial_run,step,initial)
    baseline_path=initial_run/f'eval-{step:06d}/results.json'
    audit_path=initial_run/f'verification-{step:06d}.json'
    baseline=read(baseline_path);audit=read(audit_path)
    expected={(c['case'],s) for c in initial['cases'] for s in initial['seeds']}
    if (audit['status']!='verified' or audit['step']!=step or audit['recordings']!=26
            or not audit['frozen_original_adapter_and_coordinates_verified']
            or audit['summary']!=baseline['summary']
            or baseline['summary']!=aggregate(baseline['records'])
            or baseline['checkpoint_sha256']!=digest(initial_trainer.checkpoint(initial_run,step))
            or len(baseline['records'])!=26 or {(r['case'],r['seed']) for r in baseline['records']}!=expected):
        raise ValueError('Initial speaker policy is not independently verified')
    for r in baseline['records']:
        if digest(baseline_path.parent/r['wav'])!=r['audio_sha256']:raise ValueError('Baseline waveform changed')
    control=next(r for r in baseline['records'] if r['case']=='custom-001' and r['seed']==2026)
    output=external(output.resolve());output.mkdir(exist_ok=False)
    protocol=dict(config=initial['config'],parent=initial['parent'],parent_protocol_sha256=initial['parent_protocol_sha256'],
        initial_run=str(initial_run),initial_protocol_sha256=digest(initial_run/'protocol.json'),step=step,
        initial_checkpoint_sha256=digest(initial_trainer.checkpoint(initial_run,step)),
        initial_audit=str(audit_path),initial_audit_sha256=digest(audit_path),
        baseline=str(baseline_path),baseline_sha256=digest(baseline_path),
        base_sha256=initial['base_sha256'],adapted_sha256=initial['adapted_sha256'],strength=.5,
        source=sources(),runtime=runtime(),cases=initial['cases'],seeds=initial['seeds'],
        generation=initial['generation'],decoding=initial['decoding'],
        expected_control_sha256=control['audio_sha256'],
        intervention='Halfway main matrices only on reference codec/BOS rows; full text/control/projection processing and cached generated frames; centered update-16 speaker conditioning unchanged',
        caveat='Earlier text/control hidden states must exactly match an independent full-weight prefill on every protected generation. Generated frames still attend modified reference-audio context; quality is unproven.',
        kind='reference_codec_weight_routing_no_optimizer',goal_achieved=False,global_model_selection_changed=False)
    write(output/'protocol.json',protocol);shutil.copy2(Path(__file__),output/'source.py')
    write(output/'status.json',dict(status='prepared',optimizer_updates=0))


def verify(run):
    p=read(run/'protocol.json')
    if p['source']!=sources() or p['runtime']!=runtime() or p['strength']!=.5 or p['step']!=16:
        raise ValueError('Registered routing comparison changed')
    for path,sha in ((Path(p['initial_run'])/'protocol.json',p['initial_protocol_sha256']),
        (initial_trainer.checkpoint(Path(p['initial_run']),p['step']),p['initial_checkpoint_sha256']),
        (p['initial_audit'],p['initial_audit_sha256']),(p['baseline'],p['baseline_sha256'])):
        if digest(path)!=sha:raise ValueError('Routing comparison provenance changed')
    initial_trainer.verify(Path(p['initial_run']))
    return p


def tensor_digest(tensors):
    import hashlib
    value=hashlib.sha256()
    for name,tensor in sorted(tensors.items()):
        value.update(name.encode());value.update(str((tuple(tensor.shape),tensor.dtype)).encode())
        value.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return value.hexdigest()


def model_identity(wrapper):
    router=wrapper.model.reference_codec_routing
    return dict(frozen_backbone_sha256=frozen_digest(wrapper.model),
        speaker_adapter_sha256=tensor_digest(wrapper.model.speaker_preference_adapter.state_dict()),
        alternative_weights_sha256=tensor_digest(router.weights),
        matrices=len(router.weights),main_matrices=len(router.original_linears))


def load(p,legacy=False):
    import torch
    from safetensors import safe_open
    initial=initial_trainer.verify(Path(p['initial_run']))
    state=initial_trainer.read_checkpoint(Path(p['initial_run']),p['step'],initial)
    wrapper=initial_trainer.load(initial)
    initial_trainer.restore_adapter(wrapper.model,state['adapter']);wrapper.model.eval()
    if frozen_digest(wrapper.model)!=state['frozen_backbone_sha256']:raise ValueError('Initial acoustic identity differs')
    base=Path(p['config']['base'])/'model.safetensors'
    if digest(base)!=p['base_sha256']:raise ValueError('Original base weights changed')
    modules=weight_modules(wrapper.model)
    if len(modules)!=196:raise ValueError('Expected exactly 196 main matrices')
    alternate={}
    with safe_open(base,framework='pt',device='cpu') as original,torch.no_grad():
        for name,module in modules.items():
            alternate[name+'.weight']=original.get_tensor(name+'.weight').float().lerp(module.weight.detach(),p['strength'])
    ReferenceCodecRouting(wrapper.model,alternate,protected=not legacy)
    return wrapper


def control(run):
    import numpy as np
    import torch
    import soundfile as sf
    p=verify(run);wrapper=load(p,legacy=True);before=model_identity(wrapper)
    case=next(c for c in p['cases'] if c['case']=='custom-001')
    if digest(case['reference'])!=case['reference_sha256']:raise ValueError('Reference changed')
    with torch.inference_mode():
        prompt=wrapper.create_voice_clone_prompt(ref_audio=case['reference'],ref_text=case['reference_text'],x_vector_only_mode=False)
        torch.manual_seed(2026);np.random.seed(2026)
        waves,sr=wrapper.generate_voice_clone(text=case['text'],language='Auto',voice_clone_prompt=prompt,non_streaming_mode=True,**p['generation'])
    path=run/'routing-control.wav';sf.write(path,waves[0],sr,subtype='PCM_16')
    if digest(path)!=p['expected_control_sha256']:raise ValueError('No-op routing does not reproduce centered update 16')
    if model_identity(wrapper)!=before:raise ValueError('Control state changed')
    write(run/'control.json',dict(exact_waveform_parity=True,audio_sha256=digest(path),protocol_sha256=digest(run/'protocol.json'),model=before,routing=wrapper.model.reference_codec_routing.last_trace))
    print('Centered update-16 control reproduced exactly',flush=True)


def generate(run):
    import torch
    import numpy as np
    import soundfile as sf
    from .preference_rollouts import capture_generation
    p=verify(run);receipt=read(run/'control.json')
    if (not receipt['exact_waveform_parity'] or receipt['protocol_sha256']!=digest(run/'protocol.json')
            or digest(run/'routing-control.wav')!=receipt['audio_sha256']):raise ValueError('Control verification missing')
    wrapper=load(p);identity=model_identity(wrapper);frozen=identity['frozen_backbone_sha256'];prompts={};records=[]
    if identity!=receipt['model']:raise ValueError('Candidate model differs from verified no-op control')
    write(run/'model.json',dict(identity,optimizer_updates=0,forward_dtype='float32',initial_checkpoint_sha256=p['initial_checkpoint_sha256']))
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
                        or r['model_identity']!=identity
                        or digest(run/r['trajectory'])!=r['trajectory_sha256']
                        or digest(wav)!=r['audio_sha256'] or any(r[k]!=v for k,v in c.items())):raise ValueError('Speech cache changed')
            else:
                torch.manual_seed(seed);np.random.seed(seed)
                with torch.inference_mode(),capture_generation(wrapper.model) as captured:
                    waves,sr=wrapper.generate_voice_clone(text=c['text'],language='Auto',voice_clone_prompt=prompts[ref],non_streaming_mode=True,**p['generation'])
                trace=wrapper.model.reference_codec_routing.last_trace
                if trace['decode_steps']!=len(captured['codes']) or not trace['text_prefix_exact']:raise ValueError('Codec routing or full text-prefix verification failed')
                trajectory=wav.with_suffix('.pt')
                save(trajectory,dict(codes=captured['codes'],first_sequence=captured['first_sequence'],
                     routing=trace,protocol_sha256=digest(run/'protocol.json')))
                if sr!=24000 or not len(waves[0]) or not np.isfinite(waves[0]).all() or len(waves[0])/sr>30:raise ValueError('Invalid speech')
                sf.write(wav,waves[0],sr,subtype='PCM_16');wave,sr=sf.read(wav,dtype='float32')
                with torch.inference_mode():
                    vector=wrapper.model.extract_speaker_embedding(audio=wave,sr=sr).flatten()
                    cosine=float(torch.nn.functional.cosine_similarity(vector[None],prompts[ref][0].ref_spk_embedding.flatten()[None]))
                r=dict(**c,seed=seed,wav=wav.name,audio_sha256=digest(wav),cosine=cosine,seconds=len(wave)/sr,
                    terminated=captured['terminated'],frames=len(captured['codes']),routing=trace,
                    trajectory=trajectory.name,trajectory_sha256=digest(trajectory),model_identity=identity,
                    frozen_backbone_sha256=frozen,protocol_sha256=digest(run/'protocol.json'))
                write(meta,r)
            records.append(r);write(run/'status.json',dict(status='generating',completed=len(records),total=26,optimizer_updates=0))
            print(f'Reference-codec speech {len(records)}/26',flush=True)
    if model_identity(wrapper)!=identity:raise ValueError('Inference weights or routing matrices changed')
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
    parser.add_argument('--initial-run',type=Path);parser.add_argument('--step',type=int)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--phase',choices=['prepare','control','generate','score'],required=True)
    args=parser.parse_args()
    if args.phase=='prepare':prepare(args.initial_run,args.step,args.output);return
    p=verify(args.output);cfg=p['config']
    os.environ['NUMBA_CACHE_DIR']=cfg['numba_cache'];os.environ['TEMP']=os.environ['TMP']=cfg['tmp']
    os.environ['HF_HUB_OFFLINE']=os.environ['TRANSFORMERS_OFFLINE']='1'
    import torch
    torch.set_num_threads(4)
    {'control':control,'generate':generate,'score':score_all}[args.phase](args.output)


if __name__=='__main__':main()
