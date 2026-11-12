"""Bounded joint-pronunciation/voice DPO pilot on the accepted halfway model.

All training examples are generated rollouts from the original train partition.
Every saved update checkpoint receives the existing two-seed, two-group speech
evaluation. Model promotion requires separate confirmation after a screen pass.
"""
import argparse
from collections import defaultdict
import copy
import html
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

from .engine import digest, read, save, write
from .policy import external, fingerprint
from . import targeted_voice as pilot
from . import reference_protected as protected
from .preference_objective import log_probability, dpo_loss, backward_pair, deterministic_training
from .voice_preferences import joint_pairs
from .voice_strength_search import goal_screen
from data.pipeline import text_key


def sources():
    return {name:digest(Path(__file__).parent/name) for name in (
        'preference_pilot.py','preference_objective.py','voice_preferences.py',
        'preference_rollouts.py','voice_strength_search.py','../scripts/score_latest_audio.py')}


def runtime():
    import importlib.metadata
    import importlib.util
    package=Path(next(iter(importlib.util.find_spec('qwen_tts').submodule_search_locations)))
    files=('core/models/modeling_qwen3_tts.py','core/models/configuration_qwen3_tts.py',
        'inference/qwen3_tts_model.py')
    return dict(python=sys.version,versions={name:importlib.metadata.version(name)
        for name in ('torch','transformers','peft','qwen-tts','numpy','soundfile')},
        qwen_source={name:digest(package/name) for name in files})


def verify_pool(pool):
    import torch
    import numpy as np
    import soundfile as sf
    from .preference_rollouts import verify
    from scripts.score_latest_audio import score
    p, parent = verify(pool)
    if read(pool/'status.json')['status'] != 'complete':
        raise ValueError('Preference collection is not complete')
    result = read(pool/'results.json')
    records = result['records']
    expected = {(c['case'],s) for c in p['cases'] for s in p['seeds']}
    if len(records) != len(expected) or {(r['case'],r['seed']) for r in records} != expected:
        raise ValueError('Incomplete or duplicate rollout pool')
    by_case = defaultdict(list)
    cases = {c['case']:c for c in p['cases']}
    for r in records:
        if r['partition'] != 'train' or r['origin'] != 'model_generated':
            raise ValueError('Only generated training data allowed')
        if r['protocol_sha256'] != digest(pool/'protocol.json'):
            raise ValueError('Rollout protocol mismatch')
        if any(r[key] != value for key,value in cases[r['case']].items()):
            raise ValueError('Rollout conditions differ from the registered case')
        if digest(r['reference']) != r['reference_sha256']:
            raise ValueError('Rollout reference changed')
        for field, sha in (('wav','audio_sha256'),('trajectory','trajectory_sha256')):
            if digest(pool/r[field]) != r[sha]:
                raise ValueError('Rollout file changed')
        wav, sr = sf.read(pool/r['wav'], dtype='float32')
        if sr != 24000 or not len(wav) or not np.isfinite(wav).all():
            raise ValueError('Invalid rollout audio')
        if score(r['text'], r['hypothesis'], True) != r['score']:
            raise ValueError('Inconsistent ASR score')
        t = torch.load(pool/r['trajectory'], map_location='cpu', weights_only=True)
        if (t['protocol_sha256'] != r['protocol_sha256'] or t['terminated'] != r['terminated']
                or len(t['codes']) != r['frames'] or t['codes'].shape[1] != 16):
            raise ValueError('Trajectory metadata mismatch')
        if t['terminated']:
            if not torch.equal(t['first_sequence'][0,-len(t['codes'])-1:-1], t['codes'][:,0]):
                raise ValueError('Sampled codec mismatch')
            by_case[r['case']].append(r)
    recomputed = []
    for case in p['cases']:
        recomputed.extend(dict(pair,preference_split=case['preference_split'])
            for pair in joint_pairs(by_case[case['case']]))
    if recomputed != result['pairs']:
        raise ValueError('Preference pairs disagree with registered filter')
    return p, parent, result


def fitting_pairs(pool_protocol, pairs):
    """Exclude transcript overlap in addition to the registered speaker split."""
    cases = {c['case']:c for c in pool_protocol['cases']}
    val = [c for c in cases.values() if c['preference_split']=='validation']
    train = [c for c in cases.values() if c['preference_split']=='train']
    if {c['speaker_id'] for c in val} & {c['speaker_id'] for c in train}:
        raise ValueError('Preference validation shares training speakers')
    val_texts = {text_key(c[key]) for c in val for key in ('text','reference_text')}
    excluded = sorted(c['case'] for c in train
        if any(text_key(c[k]) in val_texts for k in ('text','reference_text')))
    selected = [p for p in pairs if p['case'] not in excluded]
    return selected, excluded


def prepare(pool, output):
    pool_p, parent, result = verify_pool(pool)
    pairs, excluded = fitting_pairs(pool_p, result['pairs'])
    counts = {split:len({p['case'] for p in pairs if p['preference_split']==split})
        for split in ('train','validation')}
    if counts['train'] < 12 or counts['validation'] < 4:
        raise ValueError(f'Insufficient preference coverage for a pilot: {counts}')
    output = external(output.resolve())
    output.mkdir(exist_ok=False)
    write(output/'pool-verification.json',dict(status='verified',records=len(result['records']),
        all_audio_and_trajectory_hashes_verified=True,asr_edit_counts_recomputed=True,
        pairs_recomputed=len(result['pairs']),selected_pairs=len(pairs),
        excluded_overlap_cases=excluded,case_counts=counts,
        pool_results_sha256=digest(pool/'results.json')))
    p = copy.deepcopy(parent)
    p['config'].update(output=str(output), checkpoints=[4,8,16,24], updates=24,
        peak_lr=5e-5, end_lr=1e-5, warmup_updates=2, clips_per_update=4,
        beta=0.1, weight_decay=0.0, preference_seed=4671, voice_patience=2)
    p.update(parent=pool_p['parent'], parent_protocol_sha256=pool_p['parent_protocol_sha256'],
        pool=str(pool), pool_protocol_sha256=digest(pool/'protocol.json'),
        pool_results_sha256=digest(pool/'results.json'), preference_source=sources(), runtime=runtime(),
        pairs=pairs, excluded_overlap_cases=excluded,
        selection_scope='Joint-goal development/custom screening. Preference-validation speakers are held out only from this preference fit, not from original SFT.',
        training_objective='Standard DPO, beta 0.1, summed raw model log probability over all 16 codebooks and EOS; no CE auxiliary; FP32 without autocast or TF32',
        batching='Four distinct cases per update, uniform deterministic case cycling; one rotating pair per case, no all-pairs multiplicity weighting',
        stopping='Two successive voice-guard failures, or three successive preference-validation loss increases >0.001 with training loss decreases >0.001; stop for broader confirmation on a joint-goal screen pass')
    write(output/'protocol.json', p)
    (output/'source').mkdir()
    for name in p['preference_source']:
        shutil.copy2(Path(__file__).parent/name, output/'source'/Path(name).name)
    source = Path(p['parent'])/'eval-000000/results.json'
    baseline = read(source)
    baseline['reused_baseline'] = dict(source=str(source), source_sha256=digest(source),
        explanation='Exact accepted halfway baseline; original per-record provenance retained')
    (output/'eval-000000').mkdir()
    write(output/'eval-000000/results.json', baseline)
    # Audio copies are presentation assets, retaining original verified hashes.
    for r in baseline['records']:
        wav = source.parent/r['wav']
        if digest(wav) != r['audio_sha256']:
            raise ValueError('Baseline audio changed')
        shutil.copy2(wav, output/'eval-000000'/r['wav'])
    write(output/'status.json', dict(status='prepared', preference_case_counts=counts,
        excluded_overlap_cases=excluded, goal_achieved=False))
    return p


def verify(run):
    p = read(run/'protocol.json')
    if p['preference_source'] != sources():
        raise ValueError('Preference pilot implementation changed')
    if p['runtime'] != runtime():
        raise ValueError('Preference pilot runtime changed')
    if (digest(Path(p['parent'])/'protocol.json') != p['parent_protocol_sha256']
            or digest(Path(p['pool'])/'protocol.json') != p['pool_protocol_sha256']
            or digest(Path(p['pool'])/'results.json') != p['pool_results_sha256']):
        raise ValueError('Preference pilot input changed')
    return p


def load_model(p):
    protected.configure_helpers()
    return protected.load_model(read(Path(p['parent'])/'protocol.json'))


def metrics(model, pairs, trajectories, references, beta):
    import torch
    result = {}
    # Aggregate by case: cases with more valid pair combinations get no extra weight.
    with torch.no_grad():
        values = {key:log_probability(model,t)[0].item() for key,t in trajectories.items()}
    for split in ('train','validation'):
        groups = defaultdict(list)
        for pair in pairs:
            if pair['preference_split'] != split:
                continue
            a,b=pair['chosen'],pair['rejected']
            loss,margin=dpo_loss(torch.tensor(values[a],dtype=torch.float64),
                torch.tensor(values[b],dtype=torch.float64),references[a],references[b],beta)
            groups[pair['case']].append((loss.item(),margin.item()))
        result[split] = dict(cases=len(groups), pairs=sum(map(len,groups.values())),
            loss=sum(sum(x[0] for x in g)/len(g) for g in groups.values())/len(groups),
            margin=sum(sum(x[1] for x in g)/len(g) for g in groups.values())/len(groups),
            accuracy=sum(sum(1 if x[1]>0 else 0.5 if x[1]==0 else 0 for x in g)/len(g) for g in groups.values())/len(groups))
    result['validation_minus_train_loss']=result['validation']['loss']-result['train']['loss']
    result['mean_abs_logratio_per_token']=sum(abs(values[k]-references[k])/(16*len(t['codes'])+1)
        for k,t in trajectories.items())/len(trajectories)
    return result


def train(run, until):
    import torch
    from peft import get_peft_model_state_dict
    p=verify(run); cfg=p['config']
    if until not in [0]+cfg['checkpoints']:
        raise ValueError('Unregistered checkpoint')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    wrapper=load_model(p); model=wrapper.model
    parent_zero=Path(p['parent'])/'checkpoint-000000.pt'
    if digest(parent_zero)!=read(parent_zero.with_suffix('.json'))['sha256']:
        raise ValueError('Halfway initialization changed')
    initial=torch.load(parent_zero,map_location='cpu',weights_only=True)
    pilot.load_adapter(model.talker,initial['adapter'])
    freeze=pilot.frozen_digest(model)
    if freeze!=initial['frozen_backbone_sha256']:
        raise ValueError('Frozen backbone differs')
    model.talker.to('cuda')
    free,total=torch.cuda.mem_get_info()
    torch.cuda.set_per_process_memory_fraction(min(0.90,(free+torch.cuda.memory_allocated()-256*2**20)/total))
    if any(p.dtype!=torch.float32 for p in model.talker.parameters()):
        raise ValueError('Preference training requires FP32 storage')
    params=[v for v in model.talker.parameters() if v.requires_grad]
    optimizer=torch.optim.AdamW(params,lr=cfg['peak_lr'],weight_decay=cfg['weight_decay'],fused=True)
    latest=0
    if (run/'latest.json').exists():
        receipt=read(run/'latest.json'); latest=receipt['step']
        path=pilot.checkpoint_path(run,latest)
        if digest(path)!=receipt['sha256']:
            raise ValueError('Training checkpoint changed')
        saved=torch.load(path,map_location='cpu',weights_only=True)
        if saved['protocol_sha256']!=digest(run/'protocol.json'):
            raise ValueError('Training checkpoint protocol mismatch')
        pilot.load_adapter(model.talker,saved['adapter']);optimizer.load_state_dict(saved['optimizer'])
        torch.set_rng_state(saved['rng_cpu']);torch.cuda.set_rng_state(saved['rng_cuda'])
        if latest>=until:
            raise ValueError('Checkpoint already trained')
    elif until!=0:
        raise ValueError('Initialize checkpoint zero first')
    pool=Path(p['pool']);records={r['sample_id']:r for r in read(pool/'results.json')['records']}
    needed={pair[side] for pair in p['pairs'] for side in ('chosen','rejected')}
    trajectories={}
    for key in sorted(needed):
        record=records[key];path=pool/record['trajectory']
        if digest(path)!=record['trajectory_sha256']:
            raise ValueError('Training trajectory changed')
        trajectories[key]=torch.load(path,map_location='cpu',weights_only=True)
    model.talker.eval()
    reference_path=run/'reference-logps.json'
    if reference_path.exists():
        reference_receipt=read(run/'reference-logps.sha256.json')
        if digest(reference_path)!=reference_receipt['sha256']:
            raise ValueError('Reference likelihood cache changed')
        cache=read(reference_path)
        if cache['protocol_sha256']!=digest(run/'protocol.json') or set(cache['values'])!=needed:
            raise ValueError('Reference likelihood provenance mismatch')
        references=cache['values']
    else:
        if until!=0:
            raise ValueError('Missing frozen reference likelihoods')
        references={}
        with torch.no_grad():
            for key,t in trajectories.items():
                references[key]=log_probability(model,t)[0].item()
        if not all(math.isfinite(x) for x in references.values()):
            raise ValueError('Nonfinite reference likelihood')
        write(reference_path,dict(protocol_sha256=digest(run/'protocol.json'),values=references,
            backend='CUDA FP32 SDPA, no autocast/TF32', source_checkpoint_sha256=digest(parent_zero)))
        write(run/'reference-logps.sha256.json',dict(sha256=digest(reference_path)))
    for decoder in (model.talker.model,model.talker.code_predictor.model):
        decoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    deterministic_training(model)
    by_case=defaultdict(list)
    for pair in p['pairs']:
        if pair['preference_split']=='train':
            by_case[pair['case']].append(pair)
    cases=sorted(by_case);random.Random(cfg['preference_seed']).shuffle(cases)
    for case in cases:
        by_case[case].sort(key=lambda pair:fingerprint([pair['chosen'],pair['rejected']]))
    for step in range(latest+1,until+1):
        started=time.monotonic();optimizer.zero_grad(set_to_none=True)
        from .policy import learning_rate
        lr=learning_rate(step,cfg['updates'],cfg['peak_lr'],cfg['end_lr'],cfg['warmup_updates'])
        for group in optimizer.param_groups:group['lr']=lr
        events=[]
        for offset in range(cfg['clips_per_update']):
            index=(step-1)*cfg['clips_per_update']+offset
            case=cases[index%len(cases)]
            pair=by_case[case][(index//len(cases))%len(by_case[case])]
            a,b=pair['chosen'],pair['rejected']
            event=backward_pair(model,trajectories[a],trajectories[b],(references[a],references[b]),
                cfg['beta'],1/cfg['clips_per_update'])
            events.append(dict(event,case=case,chosen=a,rejected=b))
        norm=torch.nn.utils.clip_grad_norm_(params,1.0,error_if_nonfinite=True)
        optimizer.step();pilot.check_optimizer(optimizer)
        event=dict(step=step,lr=lr,gradient_norm=float(norm),seconds=time.monotonic()-started,
            mean_dpo_loss=sum(e['loss'] for e in events)/len(events),pairs=events,
            gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated())
        write(run/f'train-{step:06d}.json',event)
        print({k:v for k,v in event.items() if k!='pairs'},flush=True)
    measured=metrics(model,p['pairs'],trajectories,references,cfg['beta'])
    if pilot.frozen_digest(model)!=freeze:
        raise ValueError('Frozen model modified')
    state={k:v.detach().cpu() for k,v in get_peft_model_state_dict(model.talker).items()}
    if len(state)!=4 or any(not k.startswith('text_projection.') for k in state):
        raise ValueError('Unexpected preference adapter coverage')
    change=sum((state[k]-initial['adapter'][k]).double().square().sum().item() for k in state)**0.5
    path=pilot.checkpoint_path(run,until)
    save(path,dict(step=until,protocol_sha256=digest(run/'protocol.json'),adapter=state,
        optimizer=optimizer.state_dict(),rng_cpu=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state(),
        frozen_backbone_sha256=freeze,losses=measured,adapter_l2_change=change))
    write(path.with_suffix('.json'),dict(step=until,sha256=digest(path),frozen_backbone_verified=True,
        losses=measured,adapter_l2_change=change))
    write(run/'latest.json',dict(step=until,sha256=digest(path)))
    print(dict(step=until,metrics=measured,adapter_l2_change=change),flush=True)


def evaluate(run, step):
    p=verify(run)
    # Reuse historical evaluator unchanged; parent loading preserves exact blend.
    pilot.load_model=lambda ignored:load_model(p)
    original=pilot.load_adapter
    pilot.load_adapter=lambda talker,state:protected.audited_load_adapter(talker,state,run/f'routing-{step}.json')
    try:
        pilot.speech(p,step)
    finally:
        pilot.load_adapter=original


def voice_guard(baseline, report):
    a={(r['case'],r['seed']):r for r in baseline['records']}
    b={(r['case'],r['seed']):r for r in report['records']}
    if (len(a)!=26 or set(a)!=set(b) or len(report['records'])!=26
            or set(report['summary'])!={'development','custom'}):
        raise ValueError('Incomplete voice comparison')
    worst=min(b[k]['cosine']-a[k]['cosine'] for k in a)
    return dict(passed=all(s['speaker_macro_cosine']>0.984 for s in report['summary'].values())
        and worst>=-0.005,worst_clip_cosine_change=worst)


def render(run, reports, status):
    from urllib.parse import quote
    page = '''<!doctype html><html lang="en"><meta charset="utf-8">
<title>Joint pronunciation and voice training</title><style>
body{font:16px system-ui;background:#101923;color:#e9f0f8;max-width:1200px;margin:36px auto;padding:0 20px}
table{border-collapse:collapse;width:100%;margin-bottom:28px}td,th{padding:10px;border-bottom:1px solid #354759;text-align:left}
a{color:#92caff}audio{width:100%;max-width:650px}.note{color:#b8c8d8}details{margin:16px 0}
</style><h1>Joint pronunciation and voice training</h1>'''
    if status.lower().startswith(('train','evaluate')):
        page += '<meta http-equiv="refresh" content="60">'
    page += '<p>'+html.escape(status)+'</p><p>Target: CER &lt;3% and internal cosine &gt;0.984 in <strong>both groups</strong>. Two seeds per text; no best-of-N selection. A passing screen needs broader confirmation and listening.</p>'
    page += '<table><tr><th>Update</th><th>Dev CER / WER</th><th>Dev cosine</th><th>Reference CER / WER</th><th>Reference cosine</th><th>Voice guard</th><th>Goal screen</th></tr>'
    for r in reports:
        d,c=r['summary']['development'],r['summary']['custom']
        page += (f'<tr><td>{r["step"]}</td><td>{d["pooled_cer"]:.2%} / {d["pooled_wer"]:.2%}</td>'
            f'<td>{d["speaker_macro_cosine"]:.6f}</td><td>{c["pooled_cer"]:.2%} / {c["pooled_wer"]:.2%}</td>'
            f'<td>{c["speaker_macro_cosine"]:.6f}</td><td>{"Pass" if r["voice_guard"]["passed"] else "Fail"}</td>'
            f'<td>{"Needs confirmation" if r["goal_screen"] else "Not met"}</td></tr>')
    page += '</table><h2>Preference learning and overfit checks</h2><p class="note">Validation speakers were held out from preference fitting, but participated in original SFT. Preference accuracy uses 50% for ties. Lower preference loss alone does not establish better speech.</p><table><tr><th>Update</th><th>Train loss / accuracy</th><th>Validation loss / accuracy</th><th>Loss gap</th><th>Mean absolute log-ratio / token</th><th>Adapter change L2</th></tr>'
    for r in reports:
        m=r['preference_metrics'];a,b=m['train'],m['validation']
        page += (f'<tr><td>{r["step"]}</td><td>{a["loss"]:.5f} / {a["accuracy"]:.1%}</td>'
            f'<td>{b["loss"]:.5f} / {b["accuracy"]:.1%}</td><td>{m["validation_minus_train_loss"]:+.5f}</td>'
            f'<td>{m["mean_abs_logratio_per_token"]:.6f}</td><td>{r["adapter_l2_change"]:.6f}</td></tr>')
    page += '</table><p class="note">The log-ratio statistic is a drift diagnostic, not a measured KL divergence. Voice cosine is an internal encoder score, not a percentage of perceptual identity.</p>'
    baseline=read(run/'eval-000000/results.json')
    reference=next(r for r in baseline['records'] if r['group']=='custom')
    url=quote(os.path.relpath(reference['reference'],run).replace('\\','/'),safe='/.:')
    page += '<h2>Reference voice</h2><audio controls preload="none" src="'+url+'"></audio>'
    for r in reports:
        result=read(run/f'eval-{r["step"]:06d}/results.json')
        page += f'<details><summary>Update {r["step"]}: per-seed scores and reference-voice examples</summary><table><tr><th>Group</th><th>Seed</th><th>CER</th><th>WER</th></tr>'
        for group,scores in r['summary'].items():
            for seed,s in scores['per_seed'].items():
                page += f'<tr><td>{html.escape(group)}</td><td>{seed}</td><td>{s["pooled_cer"]:.2%}</td><td>{s["pooled_wer"]:.2%}</td></tr>'
        page += '</table>'
        for row in result['records']:
            if row['group']=='custom':
                page += f'<p>{html.escape(row["text"])} — seed {row["seed"]}, CER {row["score"]["cer"]:.2%}, cosine {row["cosine"]:.6f}</p><audio controls preload="none" src="eval-{r["step"]:06d}/{row["wav"]}"></audio>'
        page += '</details>'
    page += '<p><a href="protocol.json">Registered protocol</a> · <a href="results.json">Results</a> · <a href="pool-verification.json">Training-data verification</a></p></html>'
    (run/'report.html').write_text(page,encoding='utf-8')


def run_all(run):
    p=verify(run)
    baseline=read(run/'eval-000000/results.json')
    reports=[];voice_failures=0;overfit=0
    for step in [0]+p['config']['checkpoints']:
        for phase in ('train','evaluate') if step else ('train',):
            target=pilot.checkpoint_path(run,step) if phase=='train' else run/f'eval-{step:06d}/results.json'
            if target.exists():continue
            write(run/'status.json',dict(status='running',phase=phase,step=step,updated=time.time()))
            render(run,reports,f'{phase.capitalize()} update {step}')
            with (run/f'{phase}-{step}.log').open('ab') as log:
                subprocess.run([sys.executable,'-m','training.preference_pilot','--output',str(run),
                    '--phase',phase,'--step',str(step)],stdout=log,stderr=subprocess.STDOUT,check=True)
        checkpoint=read(pilot.checkpoint_path(run,step).with_suffix('.json'))
        if digest(pilot.checkpoint_path(run,step))!=checkpoint['sha256']:
            raise ValueError('Completed checkpoint mismatch')
        report=baseline if step==0 else read(run/f'eval-{step:06d}/results.json')
        guard=voice_guard(baseline,report)
        passed=goal_screen(report['summary'])
        item=dict(step=step,summary=report['summary'],preference_metrics=checkpoint['losses'],
            adapter_l2_change=checkpoint['adapter_l2_change'],voice_guard=guard,goal_screen=passed)
        if reports:
            previous=reports[-1]['preference_metrics']
            current=item['preference_metrics']
            overfit=overfit+1 if (current['validation']['loss']>previous['validation']['loss']+0.001
                and current['train']['loss']<previous['train']['loss']-0.001) else 0
        reports.append(item)
        voice_failures=0 if guard['passed'] else voice_failures+1
        reason='needs_broader_confirmation' if passed else 'voice_guard' if voice_failures>=2 else 'preference_overfit' if overfit>=3 else None
        write(run/'results.json',dict(checkpoints=reports,goal_achieved=False,
            global_model_selection_changed=False,stopping_reason=reason))
        write(run/'status.json',dict(status='stopped' if reason else 'evaluated',step=step,
            reason=reason,voice_failures=voice_failures,preference_overfit_count=overfit,updated=time.time()))
        render(run,reports,reason or f'Evaluated update {step}')
        if reason:return
    write(run/'status.json',dict(status='complete',step=reports[-1]['step'],goal_achieved=False,updated=time.time()))
    render(run,reports,'Bounded pilot complete; goal remains unverified')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pool',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--phase',choices=['prepare','run','train','evaluate'],default='run')
    parser.add_argument('--step',type=int)
    args=parser.parse_args()
    if not (args.output/'protocol.json').exists():
        if args.pool is None:raise ValueError('--pool required for preparation')
        prepare(args.pool,args.output)
    if args.phase=='prepare':return
    p=verify(args.output);cfg=p['config']
    os.environ['NUMBA_CACHE_DIR']=cfg['numba_cache']
    os.environ['TEMP']=os.environ['TMP']=cfg['tmp']
    os.environ['HF_HUB_OFFLINE']=os.environ['TRANSFORMERS_OFFLINE']='1'
    import torch
    torch.set_num_threads(4)
    if args.phase=='train':train(args.output,args.step)
    elif args.phase=='evaluate':evaluate(args.output,args.step)
    else:run_all(args.output)


if __name__=='__main__':
    main()
