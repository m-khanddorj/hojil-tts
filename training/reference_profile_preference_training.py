"""Frozen-backbone, reference-specific DPO with an immutable initial policy."""
import argparse
import math
import time
from collections import Counter
from pathlib import Path
import torch
from .engine import read,write,save,digest,ids
from .policy import external
from . import reference_profile_preference_collection as collection
from . import reference_profile_preference_objective as objective
from . import reference_voice_profile as profiles
from .preference_objective import deterministic_training
from .targeted_voice import aggregate
from .voice_preferences import joint_pairs

SETTINGS=dict(beta=.1,learning_rate=.0005,weight_decay=0.,gradient_clip=1.,updates=16,
    checkpoints=[0,4,8,16],maximum_input_fraction=.01,optimizer='AdamW',seed=61571,
    betas=[.9,.999],epsilon=1e-8,reference_log_probability_tolerance=.002)


def sources():
    result=collection.sources()
    for name in ('reference_profile_preference_training.py','reference_profile_preference_objective.py',
                 'reference_profile_preference_evaluation.py','codec_policy_replay.py'):
        result[name]=digest(Path(__file__).with_name(name))
    for name in ('test_training_reference_profile_preference_training.py',
                 'test_training_reference_profile_preference_objective.py',
                 'test_training_reference_profile_preference_evaluation.py'):
        result[name]=digest(Path(__file__).parent.parent/'tests'/name)
    return result


def pool_payload(p,report,audit,pair_data):
    records=collection.preference_records(p,report)
    if (audit['status']!='verified' or audit['recordings']!=64 or audit['truncated'] or report['truncated']
            or not audit['all_weights_unchanged'] or audit['maximum_cosine_recomputation_error']>2e-6
            or audit['summary']!=report['summary'] or aggregate(records)!=report['summary']):
        raise ValueError('Complete independently audited training pool required')
    pairs=[]
    for case in p['cases']:
        pairs.extend(joint_pairs([r for r in records if r['case']==case['case']],p['settings']['min_cosine_margin']))
    if (not pairs or pair_data['records']!=records or pair_data['pairs']!=pairs
            or pair_data['pair_count']!=len(pairs)
            or pair_data['cases_with_pairs']!=len({r['case'] for r in pairs}) or pair_data['total_cases']!=16):
        raise ValueError('Nonempty complete joint preferences must recompute exactly')
    return records,pairs


def pool_evidence(pool):
    status_path=pool.parent/'reference-profile-preferences-status.json';status=read(status_path)
    if status['status']!='complete' or status['reason']!='preferences_verified':
        raise ValueError('Preference collection must complete before training')
    p=collection.verify(pool);report=read(pool/'baseline/results.json');audit=read(pool/'verification.json');data=read(pool/'pairs.json')
    records,pairs=pool_payload(p,report,audit,data)
    if (data['results_sha256']!=digest(pool/'baseline/results.json') or data['audit_sha256']!=digest(pool/'verification.json')
            or status['pair_count']!=len(pairs) or status['cases_with_pairs']!=data['cases_with_pairs']):
        raise ValueError('Preference data provenance differs')
    for r in records:
        for name,sha in (('wav','audio_sha256'),('trajectory','trajectory_sha256')):
            if digest(pool/'baseline'/r[name])!=r[sha]:raise ValueError('Training artifact changed')
    paths=[status_path,pool/'protocol.json',pool/'baseline/results.json',pool/'verification.json',pool/'pairs.json',pool/'control-verification.json']
    if not read(paths[-1])['verified']:raise ValueError('Native initial-profile control missing')
    return p,records,pairs,{str(path):digest(path) for path in paths}


def pool_statistics(records,pairs):
    by_id={r['sample_id']:r for r in records};counts=Counter(p['case'] for p in pairs);weights=[1/(len(counts)*counts[p['case']]) for p in pairs]
    chosen=[by_id[p['chosen']] for p in pairs];rejected=[by_id[p['rejected']] for p in pairs]
    def distribution(values):
        return dict(minimum=min(values),maximum=max(values),mean=sum(values)/len(values))
    corrected=[p['character_error_reduction']>0 for p in pairs]
    return dict(recordings=64,cases=16,cases_with_pairs=len(counts),pairs=len(pairs),
        pronunciation_improving_pairs=sum(corrected),voice_only_pairs=len(pairs)-sum(corrected),
        pronunciation_improving_case_balanced_mass=sum(w for w,yes in zip(weights,corrected) if yes),
        voice_only_case_balanced_mass=sum(w for w,yes in zip(weights,corrected) if not yes),
        chosen_cer=distribution([r['score']['cer'] for r in chosen]),chosen_cosine=distribution([r['cosine'] for r in chosen]),
        chosen_pair_endpoints_passing_both=sum(r['score']['cer']<.03 and r['cosine']>.984 for r in chosen),
        unique_chosen_recordings=len({p['chosen'] for p in pairs}),
        unique_chosen_passing_both=len({r['sample_id'] for r in chosen if r['score']['cer']<.03 and r['cosine']>.984}),
        chosen_minus_rejected_seconds=distribution([a['seconds']-b['seconds'] for a,b in zip(chosen,rejected)]),
        interpretation='Training-only relative labels; absolute quality is diagnostic, not an additional pair filter')


def prepare(pool,run,driver):
    p,records,pairs,evidence=pool_evidence(pool)
    tests={}
    for name,count in [('reference-profile-preference-training-tests.log',3),('reference-profile-preference-objective-tests.log',3)]:
        path=run.parent/name;log=path.read_text(encoding='utf-8-sig')
        if f'Ran {count} tests' not in log or '\nOK\n' not in log or 'skipped' in log.lower():raise ValueError('Trainer tests missing')
        tests[str(path)]=digest(path)
    run=external(run.resolve());run.mkdir(exist_ok=False)
    write(run/'protocol.json',dict(pool=str(pool),previous=p['previous'],pool_evidence_hashes=evidence,
        source=sources(),runtime=collection.direct.initial.collection.runtime(),numeric_config=collection.numeric_config(),
        settings=SETTINGS,checkpoints=SETTINGS['checkpoints'],profile_reference=p['profile_reference'],
        generation=p['generation'],decoding=p['decoding'],config=p['config'],cases=p['cases'],
        training_sample_ids=[r['sample_id'] for r in records],training_pairs=pairs,tests=tests,
        driver=str(driver),driver_sha256=digest(driver),pool_statistics=pool_statistics(records,pairs),
        objective='Case-balanced raw sequence DPO, all16 codebooks plus EOS, fixed zero-profile reference policy, unique-trajectory replay',
        scope='Only the1024D reference profile trains; acoustic/attention/shared-speaker weights remain frozen',
        validation_used_for_gradients=False,goal_achieved=False,global_model_selection_changed=False))
    state=dict(step=0,profile=dict(reference_sha256=p['profile_reference']['reference_sha256'],
        radius_fraction=p['settings']['radius_fraction'],raw=torch.zeros(1024,dtype=torch.float32)),
        optimizer=None,protocol_sha256=digest(run/'protocol.json'),reference_cache_sha256=None)
    save_checkpoint(run,state)


def verify(run):
    p=read(run/'protocol.json');c,records,pairs,evidence=pool_evidence(Path(p['pool']))
    if (p['source']!=sources() or p['runtime']!=collection.direct.initial.collection.runtime()
            or p['numeric_config']!=collection.numeric_config() or p['settings']!=SETTINGS
            or p['checkpoints']!=SETTINGS['checkpoints'] or p['pool_evidence_hashes']!=evidence
            or p['training_sample_ids']!=[r['sample_id'] for r in records] or p['training_pairs']!=pairs
            or p['previous']!=c['previous'] or p['profile_reference']!=c['profile_reference']
            or p['driver_sha256']!=digest(p['driver'])):raise ValueError('Registered training execution changed')
    for key in ('generation','decoding','config','cases'):
        if p[key]!=c[key]:raise ValueError('Training conditioning or generation identity changed')
    for path,sha in p['tests'].items():
        if digest(path)!=sha:raise ValueError('Registered training test changed')
    return p


def validate_optimizer(state,step):
    if step==0:
        if state is not None:raise ValueError('Initial checkpoint must have no optimizer state')
        return
    if not isinstance(state,dict) or set(state)!= {'state','param_groups'} or len(state['param_groups'])!=1:
        raise ValueError('Unexpected profile optimizer coverage')
    group=state['param_groups'][0]
    if (group['params']!=[0] or set(state['state'])!={0} or group['lr']!=SETTINGS['learning_rate']
            or group['weight_decay']!=0 or tuple(group['betas'])!=tuple(SETTINGS['betas'])
            or group['eps']!=SETTINGS['epsilon'] or group['amsgrad'] or group['maximize']):
        raise ValueError('Optimizer configuration changed')
    values=state['state'][0]
    if set(values)!={'step','exp_avg','exp_avg_sq'} or float(values['step'])!=step:
        raise ValueError('Optimizer step or moment fields differ')
    for key in ('exp_avg','exp_avg_sq'):
        value=values[key]
        if value.dtype!=torch.float32 or value.shape!=(1024,) or not torch.isfinite(value).all():raise ValueError('Invalid FP32 profile moments')
    if (values['exp_avg_sq']<0).any():raise ValueError('Invalid second moment')


def checkpoint_path(run,step):return run/f'checkpoint-{step:06d}.pt'


def save_checkpoint(run,state):
    path=checkpoint_path(run,state['step']);receipt=path.with_suffix('.json')
    if path.exists() or receipt.exists():raise ValueError('Do not overwrite a checkpoint')
    save(path,state);write(receipt,dict(step=state['step'],sha256=digest(path),protocol_sha256=state['protocol_sha256']))


def read_checkpoint(run,step,protocol=None):
    p=verify(run) if protocol is None else protocol
    if step not in p['checkpoints']:raise ValueError('Unregistered checkpoint')
    path=checkpoint_path(run,step);receipt=read(path.with_suffix('.json'));identity=digest(run/'protocol.json')
    if receipt!=dict(step=step,sha256=digest(path),protocol_sha256=identity):raise ValueError('Checkpoint receipt differs')
    state=torch.load(path,map_location='cpu',weights_only=True);value=state['profile']['raw']
    if (state['step']!=step or state['protocol_sha256']!=identity
            or state['profile']['reference_sha256']!=p['profile_reference']['reference_sha256']
            or state['profile']['radius_fraction']!=.05 or value.dtype!=torch.float32 or value.shape!=(1024,)
            or not torch.isfinite(value).all()):raise ValueError('Profile checkpoint identity differs')
    fraction=float(.05*value.norm()/torch.sqrt(1+value.square().sum()))
    if fraction>p['settings']['maximum_input_fraction']+1e-7 or (step==0 and torch.count_nonzero(value)):
        raise ValueError('Profile checkpoint trust region differs')
    validate_optimizer(state['optimizer'],step)
    if step==0:
        if state['reference_cache_sha256'] is not None:raise ValueError('Initial checkpoint must precede reference cache')
    elif state['reference_cache_sha256']!=digest(run/'references.json'):raise ValueError('Reference policy cache changed')
    return state


def load(run,step,device='cpu'):
    p=verify(run);checkpoint=read_checkpoint(run,step,p)
    _,wrapper,profile,state=collection.load(Path(p['pool']),device)
    profiles.restore(profile,checkpoint['profile']);wrapper.model.eval()
    return p,wrapper,profile,state,checkpoint


def unchanged(model,profile,state,expected):
    actual=profiles.snapshot(profile)
    if (profile._active or actual['reference_sha256']!=expected['reference_sha256']
            or actual['radius_fraction']!=expected['radius_fraction'] or not torch.equal(actual['raw'],expected['raw'])):
        raise ValueError('Profile changed during read-only operation')
    with torch.no_grad():profile.raw.zero_()
    try:collection.unchanged(model,profile,state)
    finally:profiles.restore(profile,expected)


def configure_training(model):
    for decoder in (model.talker.model,model.talker.code_predictor.model):
        decoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    deterministic_training(model)


def load_trajectories(p,wrapper,records):
    pool=Path(p['pool'])/'baseline';result={};reference=p['profile_reference'];cached_text={}
    with torch.no_grad():prompt=wrapper.create_voice_clone_prompt(ref_audio=reference['reference'],ref_text=reference['reference_text'],x_vector_only_mode=False)[0]
    ref_vector=prompt.ref_spk_embedding.detach().cpu().flatten();ref_codes=prompt.ref_code.detach().cpu()
    ref_ids=ids(wrapper,reference['reference_text'],True).cpu();eos=wrapper.model.config.talker_config.codec_eos_token_id
    for r in records:
        trajectory=collection.direct.trajectory(pool,r);codes=trajectory['codes'];sequence=trajectory['first_sequence']
        if r['case'] not in cached_text:cached_text[r['case']]=ids(wrapper,r['text']).cpu()
        if (trajectory['reference_sha256']!=r['reference_sha256'] or trajectory['terminated']!=r['terminated']
                or not torch.equal(trajectory['input_ids'],cached_text[r['case']])
                or not torch.equal(trajectory['ref_ids'],ref_ids) or not torch.equal(trajectory['ref_codes'],ref_codes)
                or not torch.equal(trajectory['ref_xvec'].flatten(),ref_vector)
                or codes.ndim!=2 or codes.shape[1]!=16 or not len(codes) or not bool(sequence[0,-1]==eos)
                or not torch.equal(sequence[0,-(len(codes)+1):-1],codes[:,0])):
            raise ValueError('Native training text/reference/EOS conditioning differs')
        result[r['sample_id']]=trajectory
    return result


def reference_cache(run,p,records):
    cache=read(run/'references.json')
    if (cache['protocol_sha256']!=digest(run/'protocol.json') or cache['checkpoint_sha256']!=digest(checkpoint_path(run,0))
            or not cache['zero_profile'] or not cache['all_original_weights_unchanged']
            or not cache['native_conditioning_verified'] or cache['training_configuration']!='deterministic_gradient_checkpointing'
            or set(cache['records'])!={r['sample_id'] for r in records}):raise ValueError('Fixed reference policy cache differs')
    for r in records:
        value=cache['records'][r['sample_id']]
        if (value['trajectory_sha256']!=r['trajectory_sha256'] or not math.isfinite(value['log_probability'])
                or value['token_count']<=0):raise ValueError('Invalid reference likelihood')
    return cache


def acoustic_mass_probe(model,profile,trajectories):
    from .codec_policy_replay import policy_outputs
    key=sorted(trajectories)[0]
    with profile.use_reference(trajectories[key]['reference_sha256']),torch.no_grad():
        result=policy_outputs(model,trajectories[key]);mass=result['main_acoustic_mass']
        return dict(sample_id=key,mean=float(mass.mean()),minimum=float(mass.min()),frames=len(mass),
            interpretation='Fixed training-trajectory raw main-code acoustic mass; not generated-speech quality')


def references(run):
    if (run/'references-started.json').exists() or (run/'references.json').exists():raise ValueError('No implicit reference-cache restart')
    p=verify(run);write(run/'references-started.json',dict(protocol_sha256=digest(run/'protocol.json'),started=time.time()))
    p,wrapper,profile,state,checkpoint=load(run,0,'cuda');configure_training(wrapper.model)
    _,records,_,_=pool_evidence(Path(p['pool']));trajectories=load_trajectories(p,wrapper,records);values={}
    with torch.no_grad():
        for i,r in enumerate(records):
            value,count=objective.log_probability(wrapper.model,profile,trajectories[r['sample_id']])
            if not torch.isfinite(value) or count<=0:raise ValueError('Invalid native reference likelihood')
            values[r['sample_id']]=dict(log_probability=float(value),token_count=count,frames=len(trajectories[r['sample_id']]['codes']),
                trajectory_sha256=r['trajectory_sha256'])
            print(f'references: {i+1}/64',flush=True)
    unchanged(wrapper.model,profile,state,checkpoint['profile'])
    write(run/'references.json',dict(protocol_sha256=digest(run/'protocol.json'),checkpoint_sha256=digest(checkpoint_path(run,0)),
        records=values,zero_profile=True,all_original_weights_unchanged=True,native_conditioning_verified=True,
        training_configuration='deterministic_gradient_checkpointing',acoustic_mass_probe=acoustic_mass_probe(wrapper.model,profile,trajectories)))


def require_prior_evaluation(run,p,start):
    from . import reference_profile_preference_evaluation as evaluator
    evaluator.verify(run);history=[]
    for step in p['checkpoints'][:p['checkpoints'].index(start)+1]:
        receipt=read(run/f'verification-{step:06d}.json')
        if (receipt['status']!='verified' or receipt['recordings']!=50 or receipt['step']!=step
                or not receipt['all_weights_unchanged'] or not receipt['inactive_development_exact_parity']
                or receipt['checkpoint_sha256']!=digest(checkpoint_path(run,step))
                or receipt['evaluation_protocol_sha256']!=digest(run/'evaluation-protocol.json')
                or (step==0 and not receipt['all_baseline_waveforms_exact'])):
            raise ValueError('Complete unchanged prior checkpoint evaluation is required')
        for kind,count in (('probes',24),('screen',26)):
            dest=evaluator.directory(run,step,kind);path=dest/'verification.json';part=read(path)
            if (digest(path)!=receipt['partition_audit_sha256'][kind] or part['status']!='verified'
                    or part['recordings']!=count or part['checkpoint_sha256']!=receipt['checkpoint_sha256']
                    or part['results_sha256']!=digest(dest/'results.json')):
                raise ValueError('Prior checkpoint partition audit changed')
        history.append(receipt)
    if evaluator.progression(history)['stop']:raise ValueError('Registered speech stopping criterion already met')


def length_diagnostics(receipt,references,trajectories):
    values=receipt['log_probabilities'];keys=sorted(values)
    def correlation(a,b):
        x=torch.tensor(a,dtype=torch.float64);y=torch.tensor(b,dtype=torch.float64);x=x-x.mean();y=y-y.mean()
        denominator=x.norm()*y.norm()
        return None if denominator==0 else float(x.dot(y)/denominator)
    ratios=[values[k]-references[k] for k in keys]
    return dict(logratio_token_count_correlation=correlation(ratios,[receipt['token_counts'][k] for k in keys]),
        logratio_frame_count_correlation=correlation(ratios,[len(trajectories[k]['codes']) for k in keys]))


def train(run,stop_step):
    p=verify(run);checkpoints=p['checkpoints']
    if stop_step not in checkpoints[1:]:raise ValueError('Training must end at the next registered checkpoint')
    start=checkpoints[checkpoints.index(stop_step)-1];marker=run/f'train-{start:06d}-{stop_step:06d}-started.json'
    if marker.exists() or checkpoint_path(run,stop_step).exists() or any((run/f'update-{i:06d}.json').exists() for i in range(start+1,stop_step+1)):
        raise ValueError('No overwrite or implicit training restart')
    require_prior_evaluation(run,p,start)
    p,wrapper,profile,state,checkpoint=load(run,start,'cuda');model=wrapper.model;configure_training(model)
    _,records,pairs,_=pool_evidence(Path(p['pool']));cache=reference_cache(run,p,records)
    trajectories=load_trajectories(p,wrapper,records);refs={k:v['log_probability'] for k,v in cache['records'].items()}
    optimizer=torch.optim.AdamW([profile.raw],lr=SETTINGS['learning_rate'],betas=tuple(SETTINGS['betas']),eps=SETTINGS['epsilon'],weight_decay=0.)
    if checkpoint['optimizer'] is not None:optimizer.load_state_dict(checkpoint['optimizer'])
    if sum(v.numel() for v in model.parameters() if v.requires_grad)!=1024:raise ValueError('Only the profile may train')
    if start==0:
        maximum=0.;initial_values={}
        with torch.no_grad():
            for r in records:
                value,count=objective.log_probability(model,profile,trajectories[r['sample_id']])
                initial_values[r['sample_id']]=value
                maximum=max(maximum,abs(float(value)-refs[r['sample_id']]))
                if count!=cache['records'][r['sample_id']]['token_count']:raise ValueError('Reference token coverage differs')
        if maximum>SETTINGS['reference_log_probability_tolerance']:raise ValueError('Initial policy differs from frozen reference')
        initial_loss,_,initial_margins=objective.objective(initial_values,refs,pairs,SETTINGS['beta'])
        margin_error=max(abs(r['margin']) for r in initial_margins)
        if margin_error>2*SETTINGS['beta']*SETTINGS['reference_log_probability_tolerance']+1e-10:
            raise ValueError('Initial DPO margin differs from zero')
        write(run/'initial-reference-verification.json',dict(verified=True,recordings=64,maximum_log_probability_error=maximum,
            loss=float(initial_loss),maximum_absolute_margin=margin_error))
    write(marker,dict(start=start,stop=stop_step,protocol_sha256=digest(run/'protocol.json'),started=time.time()))
    torch.manual_seed(SETTINGS['seed']+start)
    for step in range(start+1,stop_step+1):
        began=time.monotonic();model.zero_grad(set_to_none=True);torch.cuda.reset_peak_memory_stats()
        receipt=objective.backward_batch(model,profile,trajectories,refs,pairs,SETTINGS['beta'])
        gradient=float(profile.raw.grad.norm())
        if not math.isfinite(gradient) or gradient==0:raise ValueError('Empty or nonfinite preference gradient')
        if any(v.grad is not None for v in model.parameters() if v is not profile.raw):raise ValueError('Frozen model received a gradient')
        torch.nn.utils.clip_grad_norm_([profile.raw],SETTINGS['gradient_clip'],error_if_nonfinite=True);optimizer.step()
        projection=objective.project_input_radius(profile,SETTINGS['maximum_input_fraction']);model.zero_grad(set_to_none=True)
        current=profiles.snapshot(profile);unchanged(model,profile,state,current)
        receipt.update(step=step,gradient_norm_before_clipping=gradient,projection=projection,seconds=time.monotonic()-began,
            raw_parameter_norm=float(profile.raw.detach().norm()),peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),reference_cache_sha256=digest(run/'references.json'),
            all_original_weights_unchanged=True,interpretation='Pre-update fitting preference loss; not generated speech quality',
            **length_diagnostics(receipt,refs,trajectories))
        write(run/f'update-{step:06d}.json',receipt);print({k:receipt[k] for k in ('step','loss','weighted_preference_accuracy','gradient_norm_before_clipping','projection')},flush=True)
    with torch.no_grad():
        values={};counts={}
        for r in records:values[r['sample_id']],counts[r['sample_id']]=objective.log_probability(model,profile,trajectories[r['sample_id']])
        loss,_,margins=objective.objective(values,refs,pairs,SETTINGS['beta'])
    write(run/f'checkpoint-metrics-{stop_step:06d}.json',dict(step=stop_step,loss=float(loss),margins=margins,
        weighted_preference_accuracy=sum(r['weight']*(r['margin']>0) for r in margins),
        fitting_mean_sequence_nll=sum(-float(values[k])/counts[k] for k in values)/len(values),
        fitting_pooled_token_nll=-sum(float(v) for v in values.values())/sum(counts.values()),
        recordings=64,goal_achieved=False,all_original_weights_unchanged=True,
        acoustic_mass_probe=acoustic_mass_probe(model,profile,trajectories)))
    current=profiles.snapshot(profile);unchanged(model,profile,state,current);verify(run)
    optimizer_state=optimizer.state_dict()
    for value in optimizer_state['state'].values():
        for key,tensor in value.items():
            if isinstance(tensor,torch.Tensor):value[key]=tensor.detach().cpu()
    validate_optimizer(optimizer_state,stop_step)
    save_checkpoint(run,dict(step=stop_step,profile=current,optimizer=optimizer_state,
        protocol_sha256=digest(run/'protocol.json'),reference_cache_sha256=digest(run/'references.json')))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=['prepare','references','train'])
    parser.add_argument('--run',type=Path,required=True);parser.add_argument('--pool',type=Path);parser.add_argument('--driver',type=Path)
    parser.add_argument('--stop-step',type=int);args=parser.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if args.phase=='prepare':prepare(args.pool,args.run,args.driver)
    elif args.phase=='references':references(args.run)
    else:train(args.run,args.stop_step)


if __name__=='__main__':main()
