"""Centered speaker-dependent residual with frozen original update-8 adapter."""
import argparse
from collections import defaultdict
import gc
import hashlib
import html
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .engine import check_optimizer,digest,ids,read,save,write
from .policy import external,fingerprint,learning_rate
from . import refreshed_speaker_rollouts as collection
from . import speaker_preference_adapter as adapter_api
from .refreshed_speaker_rollouts import load_policy as load_full
from .targeted_voice import aggregate
from .preference_objective import deterministic_training,dpo_loss
from .voice_preferences import joint_pairs
from .voice_strength_search import goal_screen


DEFAULTS=dict(collection.PLANNED_FIT,seed=11017,updates=24,checkpoints=[8,16,24],pairs_per_update=16)


def frozen_digest(model):
    """Hash only acoustic parameters; frozen adapter components have a separate check."""
    value=hashlib.sha256()
    for name,p in model.named_parameters():
        if name.startswith('speaker_preference_adapter.'):
            continue
        if p.requires_grad:raise ValueError('Acoustic parameter is trainable')
        value.update(name.encode())
        value.update(str((tuple(p.shape),p.dtype)).encode())
        value.update(p.detach().cpu().contiguous().numpy().tobytes())
    return value.hexdigest()


def sampled_pair(pairs,index,seed):
    """Equal speaker visits, cycling cases and then pairs within each speaker."""
    groups=defaultdict(lambda:defaultdict(list))
    for pair in pairs:
        if pair['preference_split']=='train':groups[pair['speaker_id']][pair['case']].append(pair)
    if not groups or index<0:raise ValueError('Invalid fitting schedule')
    speakers=sorted(groups,key=lambda s:fingerprint([seed,s]))
    speaker=speakers[index%len(speakers)];visit=index//len(speakers)
    cases=sorted(groups[speaker],key=lambda c:fingerprint([seed,speaker,c]))
    candidates=sorted(groups[speaker][cases[visit%len(cases)]],key=lambda p:fingerprint([p['chosen'],p['rejected']]))
    return candidates[(visit//len(cases))%len(candidates)]


def source_hashes():
    names=('centered_speaker_training.py','centered_speaker_adapter.py','refreshed_speaker_training.py','refreshed_speaker_rollouts.py','speaker_preference_training.py','speaker_preference_adapter.py',
        'balanced_preference_rollouts.py','speaker_reference_adaptation.py','preference_objective.py',
        'preference_pilot.py','targeted_voice.py','voice_preferences.py','voice_strength_search.py',
        'engine.py','policy.py','sequence.py','evaluate.py','../scripts/score_latest_audio.py','../data/pipeline.py')
    return {n:digest(Path(__file__).parent/n) for n in names}


def audit_pool(pool):
    """Recheck all saved candidates, EOS, edits, pairs and reference conditioning."""
    import torch
    import soundfile as sf
    import numpy as np
    from scripts.score_latest_audio import score
    from .evaluate import ASR_REV
    p=collection.verify(pool)
    if read(pool/'status.json')['status']!='complete':raise ValueError('Collection is still incomplete')
    result=read(pool/'results.json');records=result['records'];cases={c['case']:c for c in p['cases']}
    expected={(c['case'],d) for c in p['cases'] for d in range(p['draws'])}
    if len(records)!=len(expected) or {(r['case'],r['draw']) for r in records}!=expected:
        raise ValueError('Incomplete or duplicate pool')
    wrapper=load_full(p);backbone=frozen_digest(wrapper.model)
    if backbone!=read(pool/'generation-verification.json')['frozen_backbone_sha256']:
        raise ValueError('Generated policy mismatch')
    generation=read(pool/'generation-verification.json')
    if not generation['speaker_adapter_unchanged'] or collection.adapter_digest(wrapper.model)!=generation['initial_adapter_sha256']:
        raise ValueError('Generated speaker adapter differs')
    references={};reference_codes={};by_case=defaultdict(list);first={};errors=[]
    def embed(path):
        wave,sr=sf.read(path,dtype='float32')
        if sr!=24000 or wave.ndim!=1 or not len(wave) or not np.isfinite(wave).all():raise ValueError('Invalid audio')
        with torch.inference_mode():return wrapper.model.extract_speaker_embedding(audio=wave,sr=sr).flatten()
    for r in records:
        case=cases[r['case']]
        if any(r[k]!=v for k,v in case.items()) or r['protocol_sha256']!=digest(pool/'protocol.json'):
            raise ValueError('Candidate conditioning changed')
        if r['partition']!='train' or r['origin']!='model_generated' or r['asr_revision']!=ASR_REV or r['decoding']!=p['decoding']:
            raise ValueError('Candidate provenance changed')
        if digest(r['reference'])!=r['reference_sha256']:raise ValueError('Reference audio changed')
        for field,sha in (('wav','audio_sha256'),('trajectory','trajectory_sha256')):
            if digest(pool/r[field])!=r[sha]:raise ValueError('Candidate artifact changed')
        if score(r['text'],r['hypothesis'],True)!=r['score']:raise ValueError('Candidate edit counts differ')
        t=torch.load(pool/r['trajectory'],map_location='cpu',weights_only=True)
        if (t['protocol_sha256']!=r['protocol_sha256'] or t['draw']!=r['draw'] or t['batch_seed']!=r['batch_seed']
                or t['terminated']!=r['terminated'] or t['codes'].shape!=(r['frames'],16)):
            raise ValueError('Trajectory metadata mismatch')
        stops=(t['first_sequence'][0]==wrapper.model.config.talker_config.codec_eos_token_id).nonzero().flatten()
        if bool(len(stops))!=t['terminated']:raise ValueError('EOS status mismatch')
        if t['terminated']:
            end=int(stops[0])
            if end!=len(t['codes']) or not torch.equal(t['first_sequence'][0,:end],t['codes'][:,0]):
                raise ValueError('Codec trajectory differs from sampled sequence')
            by_case[r['case']].append(r)
        if r['case'] not in first:
            first[r['case']]=t
            if not torch.equal(t['input_ids'],ids(wrapper,r['text'])) or not torch.equal(t['ref_ids'],ids(wrapper,r['reference_text'],True)):
                raise ValueError('Trajectory transcript tokens differ from registered text')
        elif any(not torch.equal(t[k],first[r['case']][k]) for k in ('ref_xvec','ref_codes','input_ids','ref_ids')):
            raise ValueError('Candidates have unequal conditioning')
        ref=r['reference']
        if ref not in references:
            with torch.inference_mode():prompt=wrapper.create_voice_clone_prompt(ref_audio=ref,ref_text=r['reference_text'],x_vector_only_mode=False)
            references[ref]=prompt[0].ref_spk_embedding.flatten()
            reference_codes[ref]=prompt[0].ref_code
        if not torch.equal(t['ref_codes'],reference_codes[ref]):raise ValueError('Trajectory reference codecs differ from audio')
        torch.testing.assert_close(t['ref_xvec'].flatten(),references[ref],atol=2e-5,rtol=2e-5)
        cosine=float(torch.nn.functional.cosine_similarity(embed(pool/r['wav'])[None],references[ref][None]))
        errors.append(abs(cosine-r['cosine']))
        if errors[-1]>2e-6:raise ValueError('Original-reference cosine differs')
    pairs=[]
    for case in p['cases']:
        pairs.extend(dict(pair,preference_split=case['preference_split']) for pair in joint_pairs(by_case[case['case']]))
    if pairs!=result['pairs']:raise ValueError('Joint preference pairs differ')
    coverage={split:dict(speakers=len({q['speaker_id'] for q in pairs if q['preference_split']==split}),
        cases=len({q['case'] for q in pairs if q['preference_split']==split}),
        pairs=sum(q['preference_split']==split for q in pairs),
        pronunciation_pairs=sum(q['preference_split']==split and q['character_error_reduction']>0 for q in pairs))
        for split in ('train','validation')}
    if coverage['train']['speakers']<32 or coverage['train']['cases']<64 or coverage['validation']['speakers']<8 or coverage['validation']['cases']<12:
        raise ValueError(f'Insufficient broad preference coverage: {coverage}')
    del wrapper;gc.collect()
    return p,result,dict(status='verified',records=len(records),coverage=coverage,
        max_cosine_recomputation_error=max(errors),audio_trajectory_hashes_verified=True,
        edit_counts_pairs_recomputed=True,frozen_backbone_sha256=backbone,
        pool_results_sha256=digest(pool/'results.json'))


def prepare(pool,run,registration):
    registered=read(registration)
    if (registered['source']!=source_hashes() or registered['training']!=DEFAULTS
            or registered['runtime']!=collection.runtime()
            or registered['pool_protocol_sha256']!=digest(pool/'protocol.json')):
        raise ValueError('Preregistered fitting settings/source changed')
    p=collection.verify(pool);result=read(pool/'results.json')
    evidence=registered['pool_audit']
    if digest(evidence['path'])!=evidence['sha256']:raise ValueError('Verified pool audit changed')
    audit=read(evidence['path'])
    if (audit['status']!='verified' or audit['records']!=508
            or audit['pool_results_sha256']!=digest(pool/'results.json')
            or not audit['audio_trajectory_hashes_verified'] or not audit['edit_counts_pairs_recomputed']):
        raise ValueError('Pool audit is incomplete or does not match the data')
    coordinates=registered['coordinates']
    read_coordinates(coordinates,pool)
    parent=read(Path(p['parent'])/'protocol.json')
    run=external(run.resolve());run.mkdir(exist_ok=False)
    protocol=dict(pool=str(pool),pool_protocol_sha256=digest(pool/'protocol.json'),
        pool_results_sha256=digest(pool/'results.json'),parent=p['parent'],parent_protocol_sha256=p['parent_protocol_sha256'],
        config=p['config'],initial_policy=p['initial_policy'],training=dict(DEFAULTS),source=source_hashes(),runtime=collection.runtime(),
        registration=str(registration),registration_sha256=digest(registration),
        coordinates=coordinates,pool_audit=evidence,
        base_sha256=p['base_sha256'],adapted_sha256=p['adapted_sha256'],pairs=result['pairs'],
        cases=parent['cases'],generation=parent['generation'],decoding=parent['decoding'],seeds=parent['seeds'],
        halfway_results=str(Path(p['parent'])/'eval-000000/results.json'),
        halfway_results_sha256=digest(Path(p['parent'])/'eval-000000/results.json'),
        reference_policy='Audited update 8, frozen acoustic weights AND original speaker adapter; fresh centered residual and AdamW moments; fitting-only PCA coordinates frozen',
        stopping='Goal screen pass -> confirmation required; two consecutive checkpoints with any group having worse CER and no voice gain, or worst clip cosine drop >0.005 -> stop; two consecutive preference-validation loss rises >0.001 while fitting loss falls >0.001 -> stop',
        goal_achieved=False,global_model_selection_changed=False)
    write(run/'protocol.json',protocol);write(run/'pool-verification.json',audit)
    write(run/'status.json',dict(status='prepared',goal_achieved=False))
    (run/'source').mkdir()
    for name in protocol['source']:shutil.copy2(Path(__file__).parent/name,run/'source'/Path(name).name)


def verify(run):
    p=read(run/'protocol.json')
    collection.verify_initial(p['initial_policy'])
    if p['source']!=source_hashes() or p['runtime']!=collection.runtime():raise ValueError('Training source/runtime changed')
    if digest(p['registration'])!=p['registration_sha256']:raise ValueError('Fit registration changed')
    registered=read(p['registration'])
    if (p['training']!=DEFAULTS or p['coordinates']!=registered['coordinates']
            or p['pool_audit']!=registered['pool_audit']
            or digest(p['pool_audit']['path'])!=p['pool_audit']['sha256']):
        raise ValueError('Centered fit registration differs')
    read_coordinates(p['coordinates'],Path(p['pool']))
    if digest(Path(p['pool'])/'protocol.json')!=p['pool_protocol_sha256'] or digest(Path(p['pool'])/'results.json')!=p['pool_results_sha256']:
        raise ValueError('Preference data changed')
    if digest(Path(p['parent'])/'protocol.json')!=p['parent_protocol_sha256']:raise ValueError('Evaluation parent changed')
    if digest(p['halfway_results'])!=p['halfway_results_sha256']:raise ValueError('Accepted halfway comparison changed')
    return p


def read_coordinates(spec,pool):
    import torch
    if digest(spec['path'])!=spec['sha256']:raise ValueError('Fitted coordinates changed')
    saved=torch.load(spec['path'],map_location='cpu',weights_only=True)
    if (saved['implementation_sha256']!=digest(Path(__file__).with_name('centered_speaker_adapter.py'))
            or saved['pool_protocol_sha256']!=digest(pool/'protocol.json')
            or saved['pool_results_sha256']!=digest(pool/'results.json')):
        raise ValueError('Coordinate source or pool changed')
    source=read(pool/'protocol.json')
    expected=[(c['case'],c['speaker_id']) for c in source['cases'] if c['preference_split']=='train']
    actual=[(c['case'],c['speaker_id']) for c in saved['references']]
    if (actual!=expected or len(actual)!=95 or len({s for _,s in actual})!=48
            or saved['coordinates']['rank']!=16 or saved['coordinates']['speakers']!=48):
        raise ValueError('Coordinate fitting population differs')
    for row in saved['references']:
        if digest(row['trajectory'])!=row['sha256']:raise ValueError('Coordinate reference changed')
    return saved['coordinates']


def load(p):
    import torch
    from .centered_speaker_adapter import CenteredSpeakerAdapter
    wrapper=load_full(p)
    coordinates=read_coordinates(p['coordinates'],Path(p['pool']))
    wrapper.model.speaker_preference_adapter=CenteredSpeakerAdapter(wrapper.model.speaker_preference_adapter,coordinates)
    torch.manual_seed(p['training']['seed'])
    return wrapper


def restore_adapter(model,state):
    adapter=model.speaker_preference_adapter
    fixed=adapter.frozen_state()
    adapter.load_state_dict(state,strict=True)
    adapter.verify_frozen(fixed)


def checkpoint(run,step):return run/f'checkpoint-{step:06d}.pt'


def read_checkpoint(run,step,p):
    import torch
    path=checkpoint(run,step);receipt=read(path.with_suffix('.json'))
    if digest(path)!=receipt['sha256']:raise ValueError('Checkpoint changed')
    state=torch.load(path,map_location='cpu',weights_only=True)
    if state['step']!=step or state['protocol_sha256']!=digest(run/'protocol.json'):raise ValueError('Checkpoint provenance mismatch')
    return state


def metrics(model,pairs,trajectories,references,beta):
    import torch
    model.eval()
    with torch.no_grad():values={k:float(adapter_api.log_probability(model,t)[0]) for k,t in trajectories.items()}
    result={}
    for split in ('train','validation'):
        groups=defaultdict(lambda:defaultdict(list))
        for pair in pairs:
            if pair['preference_split']!=split:continue
            a,b=pair['chosen'],pair['rejected']
            loss,margin=dpo_loss(torch.tensor(values[a],dtype=torch.float64),torch.tensor(values[b],dtype=torch.float64),references[a],references[b],beta)
            groups[pair['speaker_id']][pair['case']].append((float(loss),float(margin),1 if margin>0 else .5 if margin==0 else 0))
        def average(column):
            return sum(sum(sum(v[column] for v in pairs)/len(pairs) for pairs in cases.values())/len(cases) for cases in groups.values())/len(groups)
        result[split]=dict(speakers=len(groups),cases=sum(len(g) for g in groups.values()),
            loss=average(0),margin=average(1),accuracy=average(2))
    result['validation_minus_train_loss']=result['validation']['loss']-result['train']['loss']
    result['mean_abs_logratio_per_token']=sum(abs(values[k]-references[k])/(16*len(t['codes'])+1) for k,t in trajectories.items())/len(trajectories)
    device=next(model.speaker_preference_adapter.parameters()).device
    with torch.no_grad():
        movement=[]
        for t in trajectories.values():
            x=t['ref_xvec'].to(device).flatten();y=model.speaker_preference_adapter(x)
            movement.append(float((y-x).norm()/x.norm()))
    result['speaker_input_relative_displacement']=dict(mean=sum(movement)/len(movement),maximum=max(movement))
    if max(movement)>model.speaker_preference_adapter.radius_fraction+1e-6:raise ValueError('Speaker trust region violated')
    return result


def train(run,until):
    import torch
    p=verify(run);cfg=p['training']
    if until not in [0]+cfg['checkpoints']:raise ValueError('Unregistered checkpoint')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    wrapper=load(p);model=wrapper.model;frozen=frozen_digest(model)
    if frozen!=read(run/'pool-verification.json')['frozen_backbone_sha256']:raise ValueError('Initial policy mismatch')
    model.talker.to('cuda');model.speaker_preference_adapter.to('cuda')
    torch.cuda.set_per_process_memory_fraction(0.84)
    params=[v for v in model.speaker_preference_adapter.parameters() if v.requires_grad]
    fixed_adapter=model.speaker_preference_adapter.frozen_state()
    optimizer=torch.optim.AdamW(params,lr=cfg['peak_lr'],weight_decay=cfg['weight_decay'],fused=True)
    initial={k:v.detach().cpu().clone() for k,v in model.speaker_preference_adapter.state_dict().items()}
    latest=0
    if (run/'latest.json').exists():
        receipt=read(run/'latest.json');latest=receipt['step'];saved=read_checkpoint(run,latest,p)
        if digest(checkpoint(run,latest))!=receipt['sha256'] or saved['frozen_backbone_sha256']!=frozen:raise ValueError('Resume checkpoint differs')
        if latest>=until:raise ValueError('Checkpoint already trained')
        restore_adapter(model,saved['adapter']);optimizer.load_state_dict(saved['optimizer'])
        torch.set_rng_state(saved['rng_cpu']);torch.cuda.set_rng_state(saved['rng_cuda'])
    elif until!=0:raise ValueError('Initialize first')
    records={r['sample_id']:r for r in read(Path(p['pool'])/'results.json')['records']}
    needed={q[side] for q in p['pairs'] for side in ('chosen','rejected')};trajectories={}
    for k in sorted(needed):
        r=records[k];path=Path(p['pool'])/r['trajectory']
        if digest(path)!=r['trajectory_sha256']:raise ValueError('Trajectory changed')
        trajectories[k]=torch.load(path,map_location='cpu',weights_only=True)
    reference_path=run/'reference-logps.json';model.eval()
    if reference_path.exists():
        if digest(reference_path)!=read(run/'reference-logps.sha256.json')['sha256']:raise ValueError('Reference cache changed')
        cached=read(reference_path)
        if cached['protocol_sha256']!=digest(run/'protocol.json') or set(cached['values'])!=needed:raise ValueError('Reference provenance differs')
        references=cached['values']
    else:
        if until!=0:raise ValueError('Missing frozen reference likelihoods')
        with torch.no_grad():references={k:float(adapter_api.log_probability(model,t)[0]) for k,t in trajectories.items()}
        if not all(math.isfinite(x) for x in references.values()):raise ValueError('Nonfinite reference likelihood')
        write(reference_path,dict(protocol_sha256=digest(run/'protocol.json'),values=references,backend='CUDA FP32 SDPA, no TF32/autocast'))
        write(run/'reference-logps.sha256.json',dict(sha256=digest(reference_path)))
    for decoder in (model.talker.model,model.talker.code_predictor.model):
        decoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    deterministic_training(model)
    for step in range(latest+1,until+1):
        start=time.monotonic();optimizer.zero_grad(set_to_none=True)
        lr=learning_rate(step,cfg['updates'],cfg['peak_lr'],cfg['end_lr'],cfg['warmup_updates'])
        for group in optimizer.param_groups:group['lr']=lr
        events=[]
        for offset in range(cfg['pairs_per_update']):
            pair=sampled_pair(p['pairs'],(step-1)*cfg['pairs_per_update']+offset,cfg['seed'])
            a,b=pair['chosen'],pair['rejected']
            event=adapter_api.backward_pair(model,trajectories[a],trajectories[b],(references[a],references[b]),cfg['beta'],1/cfg['pairs_per_update'])
            events.append(dict(event,chosen=a,rejected=b,speaker_id=pair['speaker_id'],case=pair['case']))
        norm=torch.nn.utils.clip_grad_norm_(params,cfg['gradient_clip'],error_if_nonfinite=True)
        optimizer.step();check_optimizer(optimizer)
        event=dict(step=step,lr=lr,gradient_norm=float(norm),seconds=time.monotonic()-start,
            mean_dpo_loss=sum(e['loss'] for e in events)/len(events),pairs=events,gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated())
        write(run/f'train-{step:06d}.json',event);print({k:v for k,v in event.items() if k!='pairs'},flush=True)
    measured=metrics(model,p['pairs'],trajectories,references,cfg['beta'])
    if frozen_digest(model)!=frozen:raise ValueError('Frozen model changed')
    model.speaker_preference_adapter.verify_frozen(fixed_adapter)
    state={k:v.detach().cpu() for k,v in model.speaker_preference_adapter.state_dict().items()}
    change=sum((state[k]-initial[k]).double().square().sum().item() for k in state)**.5
    path=checkpoint(run,until)
    save(path,dict(step=until,protocol_sha256=digest(run/'protocol.json'),adapter=state,optimizer=optimizer.state_dict(),
        rng_cpu=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state(),frozen_backbone_sha256=frozen,losses=measured,adapter_l2_change=change))
    write(path.with_suffix('.json'),dict(step=until,sha256=digest(path),losses=measured,adapter_l2_change=change,frozen_backbone_verified=True))
    write(run/'latest.json',dict(step=until,sha256=digest(path)))
    print(dict(step=until,metrics=measured),flush=True)


def speech(run,step):
    import numpy as np
    import soundfile as sf
    import torch,torchaudio
    from transformers import WhisperForConditionalGeneration,WhisperProcessor
    from scripts.score_latest_audio import score
    from .evaluate import ASR_REV
    from .preference_rollouts import capture_generation
    p=verify(run);saved=read_checkpoint(run,step,p);wrapper=load(p)
    restore_adapter(wrapper.model,saved['adapter']);wrapper.model.eval()
    if frozen_digest(wrapper.model)!=saved['frozen_backbone_sha256']:raise ValueError('Evaluation backbone differs')
    out=run/f'eval-{step:06d}';out.mkdir(exist_ok=True);records=[];prompts={}
    checkpoint_sha=digest(checkpoint(run,step))
    for seed in p['seeds']:
        for case in p['cases']:
            ref=case['reference']
            if digest(ref)!=case['reference_sha256']:raise ValueError('Evaluation reference changed')
            if ref not in prompts:
                with torch.inference_mode():prompts[ref]=wrapper.create_voice_clone_prompt(ref_audio=ref,ref_text=case['reference_text'],x_vector_only_mode=False)
            wav=out/f"{seed}-{case['case']}.wav";meta=wav.with_suffix('.json')
            if meta.exists():
                r=read(meta)
                if (r['protocol_sha256']!=digest(run/'protocol.json') or r['checkpoint_sha256']!=checkpoint_sha
                        or digest(wav)!=r['audio_sha256'] or any(r[k]!=v for k,v in case.items())):
                    raise ValueError('Evaluation cache mismatch')
            else:
                torch.manual_seed(seed);np.random.seed(seed)
                with torch.inference_mode(),capture_generation(wrapper.model) as captured:
                    waves,sr=wrapper.generate_voice_clone(text=case['text'],language='Auto',voice_clone_prompt=prompts[ref],non_streaming_mode=True,**p['generation'])
                if sr!=24000 or not len(waves[0]) or not np.isfinite(waves[0]).all() or len(waves[0])/sr>30:raise ValueError('Invalid speech')
                sf.write(wav,waves[0],sr,subtype='PCM_16');wave,sr=sf.read(wav,dtype='float32')
                original=prompts[ref][0].ref_spk_embedding.flatten()
                with torch.inference_mode():
                    vector=wrapper.model.extract_speaker_embedding(audio=wave,sr=sr).flatten()
                    cosine=float(torch.nn.functional.cosine_similarity(vector[None],original[None]))
                    adjusted=wrapper.model.speaker_preference_adapter(original)
                    displacement=float((adjusted-original).norm()/original.norm())
                r=dict(**case,seed=seed,wav=wav.name,audio_sha256=digest(wav),cosine=cosine,
                    seconds=len(wave)/sr,terminated=captured['terminated'],speaker_input_relative_displacement=displacement,
                    checkpoint_sha256=checkpoint_sha,protocol_sha256=digest(run/'protocol.json'))
                write(meta,r)
            records.append(r);print(f'Speech update {step}: {len(records)}/26',flush=True)
    del wrapper,prompts,saved;gc.collect()
    asr=Path(p['config']['asr']);manifest=read(asr/'download-manifest.json')
    if manifest['repo']!='BuzzASR/mongolian' or manifest['revision']!=ASR_REV:raise ValueError('ASR mismatch')
    for name,item in manifest['files'].items():
        if digest(asr/name)!=item['sha256']:raise ValueError('ASR changed')
    processor=WhisperProcessor.from_pretrained(asr,local_files_only=True)
    model=WhisperForConditionalGeneration.from_pretrained(asr,torch_dtype=torch.float32,local_files_only=True).to('cuda').eval()
    for r in records:
        path=out/f"score-{r['seed']}-{r['case']}.json"
        if path.exists():
            scored=read(path)
            if any(scored[k]!=v for k,v in r.items()) or scored['decoding']!=p['decoding'] or scored['asr_revision']!=ASR_REV:raise ValueError('ASR cache mismatch')
        else:
            wave,sr=sf.read(out/r['wav'],dtype='float32');wave=torchaudio.functional.resample(torch.from_numpy(wave),sr,16000).numpy()
            batch=processor(wave,sampling_rate=16000,return_tensors='pt',return_attention_mask=True).to('cuda')
            with torch.inference_mode():tokens=model.generate(batch.input_features,attention_mask=batch.attention_mask,**p['decoding'])
            hypothesis=processor.batch_decode(tokens,skip_special_tokens=True)[0].strip()
            scored=dict(r,hypothesis=hypothesis,score=score(r['text'],hypothesis,True),asr_revision=ASR_REV,decoding=p['decoding'])
            write(path,scored)
        r.update(scored)
    if step==0:
        previous=read(Path(p['initial_policy']['run'])/f"eval-{p['initial_policy']['step']:06d}/results.json")
        expected={(r['case'],r['seed']):r['audio_sha256'] for r in previous['records']}
        if any(expected.get((r['case'],r['seed']))!=r['audio_sha256'] for r in records):
            raise ValueError('Stage-zero audio must reproduce the initial update-8 policy')
    report=dict(step=step,records=records,summary=aggregate(records),truncated=sum(not r['terminated'] for r in records),checkpoint_sha256=checkpoint_sha)
    write(out/'results.json',report);print(report['summary'],flush=True)


def paired_screen(baseline,current):
    a={(r['case'],r['seed']):r for r in baseline['records']}
    b={(r['case'],r['seed']):r for r in current['records']}
    if len(a)!=26 or len(b)!=26 or len(current['records'])!=26 or set(a)!=set(b):raise ValueError('Incomplete paired comparison')
    for key in a:
        if any(a[key][k]!=b[key][k] for k in ('text','reference_sha256','speaker_id')):raise ValueError('Changed paired conditions')
    old,new=baseline['summary'],current['summary']
    if set(old)!=set(new) or set(new)!={'development','custom'}:raise ValueError('Both groups required')
    worst=min(b[k]['cosine']-a[k]['cosine'] for k in a)
    regressed_groups=[g for g in old if new[g]['pooled_cer']>old[g]['pooled_cer']
        and new[g]['speaker_macro_cosine']<=old[g]['speaker_macro_cosine']+1e-7]
    return dict(goal_screen=goal_screen(new) and current['truncated']==0,worst_clip_cosine_change=worst,
        harmful=bool(regressed_groups) or worst<-.005,regressed_groups=regressed_groups,
        group_voice_gain={g:new[g]['speaker_macro_cosine']-old[g]['speaker_macro_cosine'] for g in old})


def render(run,reports,status):
    page='<!doctype html><html lang="en"><meta charset="utf-8"><title>Speaker preference training</title><style>body{font:16px system-ui;background:#111923;color:#e5edf8;max-width:1150px;margin:36px auto;padding:0 20px}td,th{padding:10px;text-align:left;border-bottom:1px solid #394657}a{color:#9bd1ff}audio{width:100%;max-width:650px}</style><h1>Centered speaker-conditioning preferences</h1><p>'+html.escape(status)+'</p><p>Full epoch-2 acoustic weights are frozen. A centered speaker-dependent residual is fitted on generated-speech preferences; the original update-8 adapter and fitting-only PCA coordinates stay frozen. Every row uses all fixed texts and both seeds; no best-of-N selection. Voice scores use original reference audio.</p>'
    page+='<table><tr><th>Candidate</th><th>Dev CER/WER</th><th>Dev cosine</th><th>Custom CER/WER</th><th>Custom cosine</th></tr>'
    p=read(run/'protocol.json');half=read(p['halfway_results'])['summary']
    rows=[('Accepted halfway voice',half)]+[(f'Additional update {r["step"]}',r['summary']) for r in reports]
    for label,s in rows:
        d,c=s['development'],s['custom']
        page+=f'<tr><td>{label}</td><td>{d["pooled_cer"]:.2%}/{d["pooled_wer"]:.2%}</td><td>{d["speaker_macro_cosine"]:.6f}</td><td>{c["pooled_cer"]:.2%}/{c["pooled_wer"]:.2%}</td><td>{c["speaker_macro_cosine"]:.6f}</td></tr>'
    page+='</table><p>Required: CER below 3% and internal cosine above 0.984 in both groups. A screening pass requires the registered broader confirmation panel; no automatic promotion.</p><h2>Fitting and validation</h2><table><tr><th>Update</th><th>Fitting loss/accuracy</th><th>Validation loss/accuracy</th><th>Loss gap</th><th>Input displacement mean/max</th></tr>'
    for r in reports:
        m=r['metrics'];a,b=m['train'],m['validation'];x=m['speaker_input_relative_displacement']
        page+=f'<tr><td>{r["step"]}</td><td>{a["loss"]:.5f}/{a["accuracy"]:.1%}</td><td>{b["loss"]:.5f}/{b["accuracy"]:.1%}</td><td>{m["validation_minus_train_loss"]:+.5f}</td><td>{x["mean"]:.3%}/{x["maximum"]:.3%}</td></tr>'
    page+='</table><p>Preference validation speakers were excluded from adapter fitting but appeared in original supervised training. Lower fitting loss is not speech-quality evidence.</p>'
    from urllib.parse import quote
    ref=next(c for c in p['cases'] if c['group']=='custom')
    url=quote(os.path.relpath(ref['reference'],run).replace('\\','/'),safe='/.:')
    page+=f'<h2>Original reference</h2><audio controls preload="none" src="{url}"></audio>'
    for item in reports:
        report=read(run/f'eval-{item["step"]:06d}/results.json')
        page+=f'<details><summary>Update {item["step"]}: custom-voice samples</summary>'
        for r in report['records']:
            if r['group']=='custom':page+=f'<p>{html.escape(r["text"])} — seed {r["seed"]}, CER {r["score"]["cer"]:.2%}, cosine {r["cosine"]:.6f}</p><audio controls preload="none" src="eval-{item["step"]:06d}/{r["wav"]}"></audio>'
        page+='</details>'
    page+='<p><a href="protocol.json">Protocol</a> · <a href="pool-verification.json">Data audit</a> · <a href="results.json">All results</a></p></html>'
    (run/'report.html').write_text(page,encoding='utf-8')


def run_all(run):
    p=verify(run);reports=[];harmful=0;overfit=0
    for step in [0]+p['training']['checkpoints']:
        for phase in ('train','speech'):
            target=checkpoint(run,step) if phase=='train' else run/f'eval-{step:06d}/results.json'
            if target.exists():continue
            write(run/'status.json',dict(status='running',phase=phase,step=step,updated=time.time()))
            render(run,reports,f'{phase} update {step}')
            try:
                with (run/f'{phase}-{step}.log').open('ab') as log:
                    subprocess.run([sys.executable,'-m','training.centered_speaker_training','--output',str(run),'--phase',phase,'--step',str(step)],stdout=log,stderr=subprocess.STDOUT,check=True)
            except BaseException as error:
                write(run/'status.json',dict(status='failed',phase=phase,step=step,error=repr(error)));raise
        state=read_checkpoint(run,step,p);report=read(run/f'eval-{step:06d}/results.json')
        if report['checkpoint_sha256']!=digest(checkpoint(run,step)):raise ValueError('Speech checkpoint mismatch')
        if step==0:baseline=report
        screen=paired_screen(baseline,report)
        item=dict(step=step,summary=report['summary'],metrics=state['losses'],adapter_l2_change=state['adapter_l2_change'],screen=screen)
        if reports:
            previous=reports[-1]['metrics'];current=item['metrics']
            overfit=overfit+1 if current['validation']['loss']>previous['validation']['loss']+.001 and current['train']['loss']<previous['train']['loss']-.001 else 0
            harmful=harmful+1 if screen['harmful'] else 0
        reports.append(item)
        reason='needs_broader_confirmation' if screen['goal_screen'] else 'speech_regression' if harmful>=2 else 'preference_overfit' if overfit>=2 else None
        write(run/'results.json',dict(checkpoints=reports,goal_achieved=False,global_model_selection_changed=False,stopping_reason=reason))
        write(run/'status.json',dict(status='stopped' if reason else 'evaluated',step=step,reason=reason,updated=time.time()))
        render(run,reports,reason or f'Evaluated update {step}')
        if reason:return
    write(run/'status.json',dict(status='complete',goal_achieved=False,updated=time.time()))
    render(run,reports,'Centered fit complete; goal still requires verification')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pool',type=Path);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--registration',type=Path)
    parser.add_argument('--phase',choices=['prepare','train','speech','run'],required=True);parser.add_argument('--step',type=int)
    args=parser.parse_args()
    # Set cache environment before importing Qwen/librosa in pool auditing.
    config=read(args.pool/'protocol.json')['config'] if args.phase=='prepare' else read(args.output/'protocol.json')['config']
    os.environ['NUMBA_CACHE_DIR']=config['numba_cache'];os.environ['TEMP']=os.environ['TMP']=config['tmp']
    os.environ['HF_HUB_OFFLINE']=os.environ['TRANSFORMERS_OFFLINE']='1'
    import torch
    torch.set_num_threads(4)
    if args.phase=='prepare':
        if args.registration is None:raise ValueError('--registration required before fitting')
        prepare(args.pool,args.output,args.registration)
    elif args.phase=='train':train(args.output,args.step)
    elif args.phase=='speech':speech(args.output,args.step)
    else:run_all(args.output)


if __name__=='__main__':main()
