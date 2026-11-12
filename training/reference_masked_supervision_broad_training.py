"""Data-only 256-speaker comparison; immutable v2 numerical/optimizer helpers.

Registration and every model phase are explicit. Old calibration assets never
enter fitting batches. No base-K/V transfer or automatic model promotion.
"""
import argparse
import gc
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn.functional as F

from . import reference_masked_supervision_training_v2 as old
from . import reference_masked_supervision_policy_v2 as policy
from . import reference_masked_supervision_broad_features as fresh
from . import reference_masked_supervision_features as original
from .reference_masked_supervision_plan import SETTINGS

require,read,digest=policy.require,policy.read,policy.digest
CHECKPOINTS,NUMERIC,NATIVE_PERTURBATION=old.CHECKPOINTS,old.NUMERIC,old.NATIVE_PERTURBATION
# These helpers contain no dependency on a particular fitting manifest.
write_new,configure,numeric_config=old.write_new,old.configure,old.numeric_config
pair,normalized_objective,observation,summarize=old.pair,old.normalized_objective,old.observation,old.summarize
backward_batch,make_optimizer,optimizer_check=old.backward_batch,old.make_optimizer,old.optimizer_check
rng_state,restore_rng=old.rng_state,old.restore_rng
checkpoint_path,save_checkpoint,read_checkpoint=old.checkpoint_path,old.save_checkpoint,old.read_checkpoint
memory_frontier,native_probes=old.memory_frontier,old.native_probes
perturbed_snapshot,perturb_native_probe=old.perturbed_snapshot,old.perturb_native_probe
parity_registration,hardware,validate_memory=old.parity_registration,old.hardware,old.validate_memory
FEATURE_SHA='01247c5a5176b0938d57414c4fe248a24531ba83e0acea3b045127083065963e'
TEST_LOGS=(('reference-masked-supervision-broad-training-tests-v2.log',8),
           ('reference-masked-supervision-broad-speech-tests-v2.log',5))


def sources():
    paths=dict(old.sources());folder=Path(__file__).parent
    for name in ('reference_masked_supervision_broad_training.py','reference_masked_supervision_broad_speech.py',
                 'reference_masked_supervision_broad_features.py','reference_masked_supervision_broad_data.py',
                 'reference_masked_supervision_speech_v2.py',
                 '../tests/test_reference_masked_supervision_broad_training.py',
                 '../tests/test_reference_masked_supervision_broad_speech.py'):
        path=(folder/name).resolve();paths[str(path)]=digest(path)
    return paths


def test_evidence(root):
    tests={}
    for name,count in TEST_LOGS:
        path=Path(root)/name;value=path.read_text(encoding='utf-8-sig')
        require(f'Ran {count} tests' in value and value.strip().endswith('OK') and 'skipped' not in value.lower()
            and 'FAILED' not in value,'Complete focused tests required')
        tests[str(path.resolve())]=digest(path)
    return tests


def plan_data(path):
    path=Path(path);root=path.parent.parent
    dp=read(path.parent/'protocol.json')
    require(digest(path.parent/'protocol.json')==fresh.DATA_SHA and digest(path.parent/'audit.json')==fresh.AUDIT_SHA,
            'Audited broader data identity differs')
    for name,sha in dp['manifests'].items():require(digest(path.parent/name)==sha,'Broader manifest changed')
    schedule=read(path)
    rows=[json.loads(s) for s in (path.parent/'selected.jsonl').read_text(encoding='utf-8-sig').splitlines() if s.strip()]
    selected={r['id']:r for r in rows}
    validate_schedule(schedule,selected)
    comparator=Path(schedule['comparator']['train']);p=read(comparator/'protocol.json')
    require(digest(comparator/'protocol.json')==schedule['comparator']['protocol_sha256']
        and p['settings']==schedule['settings']==SETTINGS and p['policy']==schedule['comparator']['frozen_policy'],
        'Data-only comparator differs')
    return schedule,selected,p['calibration']


def validate_schedule(schedule,selected):
    require(schedule['settings']==SETTINGS and len(selected)==256 and len({r['speaker_id'] for r in selected.values()})==256,
        'Exactly256 fitting pairs/speakers and unchanged recipe required')
    visits=[]
    for number,entry in enumerate(schedule['updates'],1):
        require(entry['update']==number and len(entry['pair_ids'])==8 and entry['pair_weight']==.125 and entry['mode_weight']==.5
            and entry['modes']==(['full','half'] if number%2 else ['half','full']), 'Fixed32-update order differs')
        require(all(i in selected and selected[i]['partition']=='fitting' for i in entry['pair_ids']), 'Non-fitting update')
        visits.extend(entry['pair_ids'])
    require(len(schedule['updates'])==32 and len(visits)==len(set(visits))==256 and set(visits)==set(selected),
        'Every scheduled fitting pair must occur once')
    probes=schedule['fitting_probe_pair_ids']
    require(len(probes)==len(set(probes))==16 and set(probes)<=set(visits),'Probe must be actual fitting subset')


def exposure(schedule,step):
    require(step in CHECKPOINTS,'Unregistered exposure checkpoint')
    visits=[i for row in schedule['updates'][:step] for i in row['pair_ids']]
    counts={i:visits.count(i) for i in schedule['fitting_probe_pair_ids']}
    return dict(pair_visits=counts,mode_visits={i:dict(full=n,half=n) for i,n in counts.items()},
        seen_pairs=sum(n>0 for n in counts.values()),unseen_pairs=sum(n==0 for n in counts.values()),
        comparable_to_old_fitting_probe_loss=False,matched_cross_run_teacher_panel='unchanged32 calibration cases',
        label='Fixed scheduled fitting probes; exposure is checkpoint-specific, not all-seen training loss')


def calibration_assets(schedule):
    source=schedule['calibration_feature_source'];run=Path(source['run'])
    for filename,key in (('protocol.json','protocol_sha256'),('verification.json','audit_sha256'),('extraction.json','extraction_sha256')):
        require(digest(run/filename)==source[key],'Original calibration feature evidence changed')
    p=original.verify_complete(run);audit=read(run/'verification.json');extraction=read(run/'extraction.json')
    selected={r['id']:r for r in p['records'] if r['id'] in source['ids']}
    require(len(selected)==56 and set(selected)==set(source['ids']),'Exact oldcal56 required')
    assets={}
    for identity,descriptor in selected.items():
        value,receipt=original.read_asset(run,descriptor,source['protocol_sha256'])
        require(receipt['feature_sha256']==audit['feature_sha256'][identity]
            and digest(run/'assets'/original.asset_key(identity)/'receipt.json')==extraction['asset_receipts'][identity],
            'Original calibration asset changed')
        assets[identity]=value
    return assets


def load_assets(p):
    schedule=read(p['plan']);fp,fitting=fresh.load_features(p['features']);calibration=calibration_assets(schedule)
    require(len(fitting)==512 and not set(fitting)&set(calibration)
        and set(fitting)==set(schedule['fitting_feature_ids']), 'Fitting/calibration feature boundary differs')
    return fp,{**fitting,**calibration}


def parent_evidence(features):
    features=Path(features);root=features.parent
    require(digest(features/'protocol.json')==FEATURE_SHA,'Registered fresh512 stage changed')
    fresh.verify_complete(features,FEATURE_SHA)
    status_path=root/'reference-masked-supervision-broad-features-v1-status.json'
    closure_path=root/'reference-masked-supervision-broad-features-v1-process-closed.json'
    status=read(status_path);fp=read(features/'protocol.json')
    require(status['status']=='complete' and status['protocol_sha256']==FEATURE_SHA
        and status['audit_sha256']==digest(features/'verification.json'), 'Fresh512 extraction incomplete')
    require(set(status['logs'])=={'extract','audit'},'Both feature phases required')
    evidence={}
    independent=root/'broad-features-independent-audit-v1.json'
    require(digest(independent)=='a62a09e9132afdc348d89454c0e6d5e9e4a7ca5d92076aeda5c698cc983f7865',
        'Independent fresh512 saved-tensor audit changed')
    evidence[str(independent.resolve())]=digest(independent)
    for row in status['logs'].values():
        require(row['exit_code']==0 and digest(row['path'])==row['sha256'],'Feature phase failed or changed')
        evidence[row['path']]=row['sha256']
    fresh.closed(read(closure_path),45998,dict(protocol_sha256=FEATURE_SHA,driver_sha256=digest(fp['driver']),
        terminal_status_sha256=digest(status_path),audit_sha256=digest(features/'verification.json')))
    for path in (features/'protocol.json',features/'verification.json',features/'extraction.json',status_path,closure_path):
        evidence[str(path.resolve())]=digest(path)
    return evidence


def comparator(schedule):
    from . import reference_masked_supervision_speech_v2 as speech
    path=Path(schedule['comparator']['train']);p,e=speech.verify(path)
    require(digest(checkpoint_path(path,0))==schedule['comparator']['checkpoint0_sha256']
        and digest(path/'speech-protocol.json')==schedule['comparator']['speech_protocol_sha256'], 'Comparator0 changed')
    old.validate_native_proof(path,p)
    baseline=speech.verified_receipt(path,p,e,0);state=old.read_checkpoint(path,0)
    require(state['adapter_movement']['maximum']==0 and state['optimizer'] is None and p['policy']['kind']=='centered16',
        'Exact untrained centered16 comparator required')
    return p,e,state,baseline


def restore_zero(core,state):
    require(state['step']==0 and state['optimizer'] is None and state['adapter_movement']['maximum']==0,'Old checkpoint0 required')
    policy.restore(core,state['adapter']);restore_rng(state['rng'],include_cuda=False)
    require(policy.tensor_digest(policy.snapshot(core).items())==policy.tensor_digest(state['adapter'].items()),'Checkpoint0 tensor mismatch')


def load(run,step,device='cpu',training=False):
    p=verify(run);state=read_checkpoint(run,step);wrapper,core=policy.load_base(p['policy'],p['settings'])
    fresh.assert_native_components(wrapper,p['features'])
    require(policy.frozen_identity(wrapper.model,core)==state['frozen_identity']==p['frozen_identity'],'Frozen policy differs')
    policy.restore(core,state['adapter'])
    require(policy.movement(core)==state['adapter_movement'],'Checkpoint movement differs')
    if training:policy.move_for_training(wrapper,core,device)
    else:require(device=='cpu','Native speech requires CPU FP32');wrapper.model.eval()
    restore_rng(state['rng'],include_cuda=training)
    return p,wrapper,core,state


def preflight_frontiers(fitting,calibration,assets):
    # Calibration is forward-only: never pass its targets to backward.
    return memory_frontier(list(fitting.values()),assets)+memory_frontier(calibration,assets)


def validate_preflight(memory,frontier):
    require(memory['device']=='cuda' and memory['hardware']['total_memory']>0,'Actual CUDA preflight required')
    expected={(r['pair_id'],mode):r for r in frontier for mode in ('full','half')}
    require(len(memory['measurements'])==len(expected)
        and {(r['pair_id'],r['mode']) for r in memory['measurements']}==set(expected),'Exact frontier coverage required')
    for r in memory['measurements']:
        item=expected[r['pair_id'],r['mode']];backward=item['partition']=='fitting';T=item['target_frames']
        require(r['backward_performed'] is backward and r['partition']==item['partition']
            and r['frames']==T and r['main_tokens']==T+1 and r['residual_tokens']==15*T
            and all(math.isfinite(r[k]) and r[k]>=0 for k in ('objective','main_nll','residual_nll'))
            and math.isfinite(r['seconds']) and r['seconds']>0
            and 0<r['peak_allocated_bytes']<=r['peak_reserved_bytes']<=memory['hardware']['total_memory'],
            'Incomplete/nonfinite frontier measurement')
        require((math.isfinite(r['gradient_norm']) and r['gradient_norm']>0) if backward else r['gradient_norm'] is None,
            'Fitting gradient or calibration no-gradient proof differs')


def preflight(plan,features,receipt_path):
    configure();receipt_path=Path(receipt_path);require(not receipt_path.exists(),'No preflight overwrite')
    parents=parent_evidence(features);schedule,fitting,calibration=plan_data(plan)
    p0,_,state,_=comparator(schedule);_,assets=load_assets(dict(plan=str(plan),features=str(features)))
    frontier=preflight_frontiers(fitting,calibration,assets)
    write_new(receipt_path.with_suffix('.started.json'),dict(plan_sha256=digest(plan),frontier=frontier,source=sources()))
    wrapper,core=policy.load_base(p0['policy'],SETTINGS);fresh.assert_native_components(wrapper,features)
    restore_zero(core,state);frozen=policy.frozen_identity(wrapper.model,core)
    require(frozen==state['frozen_identity']==p0['frozen_identity'],'Preflight frozen policy differs')
    initial=policy.snapshot(core);before_rng=rng_state();policy.move_for_training(wrapper,core,'cuda')
    indexed={r['id']:r for r in list(fitting.values())+calibration};measurements=[]
    try:
        for item in frontier:
            value=pair(indexed[item['pair_id']],assets);backward=item['partition']=='fitting'
            for mode in ('full','half'):
                wrapper.model.zero_grad(set_to_none=True);torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
                started=time.monotonic()
                with torch.set_grad_enabled(backward),core.replay(value['input_ids'],value['codes'],value['reference'],mode=='half',terminated=True) as seq:
                    output=core.outputs(seq);loss=normalized_objective(output,len(value['codes']))
                    if backward:loss.backward()
                    row=observation(output,len(value['codes']),value['id'],mode)
                    if backward:
                        grads=[v.grad for v in core.parameters()]
                        require(all(v is not None and torch.isfinite(v).all() for v in grads),'Preflight gradients invalid')
                        grad=math.sqrt(sum(float(v.detach().square().sum()) for v in grads));del grads
                    else:
                        require(all(v.grad is None for v in wrapper.model.parameters()),'Calibration accumulated gradients');grad=None
                    del output,loss
                del seq
                row.update(partition=item['partition'],backward_performed=backward,gradient_norm=grad,
                    seconds=time.monotonic()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved())
                measurements.append(row);gc.collect()
    finally:
        wrapper.model.zero_grad(set_to_none=True);restore_rng(before_rng)
    require(policy.tensor_digest(policy.snapshot(core).items())==policy.tensor_digest(initial.items())
        and policy.frozen_identity(wrapper.model,core)==frozen,'Preflight changed weights')
    receipt=dict(status='verified',plan_sha256=digest(plan),feature_protocol_sha256=digest(Path(features)/'protocol.json'),
        features_audit_sha256=digest(Path(features)/'verification.json'),source=sources(),parent_evidence=parents,
        numeric_config=numeric_config(),frontier=frontier,measurements=measurements,device='cuda',hardware=hardware(),
        frozen_identity=frozen,adapter_snapshot_sha256=policy.tensor_digest(initial.items()),optimizer_updates=0,
        all_parameters_restored=True,calibration_backward_calls=0,checkpoint0_sha256=schedule['comparator']['checkpoint0_sha256'])
    validate_preflight(receipt,frontier);write_new(receipt_path,receipt)


def prepare(plan,features,run,driver,memory_receipt):
    from .preference_pilot import runtime
    from .policy import external
    configure();parents=parent_evidence(features);schedule,fitting,calibration=plan_data(plan)
    p0,e0,state,baseline=comparator(schedule);_,assets=load_assets(dict(plan=str(plan),features=str(features)))
    memory=read(memory_receipt);frontier=preflight_frontiers(fitting,calibration,assets);validate_preflight(memory,frontier)
    require(memory['source']==sources() and memory['status']=='verified' and memory['plan_sha256']==digest(plan)
        and memory['feature_protocol_sha256']==FEATURE_SHA and memory['features_audit_sha256']==digest(Path(features)/'verification.json')
        and memory['parent_evidence']==parents and memory['numeric_config']==NUMERIC and memory['frontier']==frontier
        and memory['frozen_identity']==state['frozen_identity'] and memory['checkpoint0_sha256']==schedule['comparator']['checkpoint0_sha256']
        and memory['adapter_snapshot_sha256']==policy.tensor_digest(state['adapter'].items())
        and memory['optimizer_updates']==memory['calibration_backward_calls']==0 and memory['all_parameters_restored'] is True,
        'Bound unchanged-policy preflight required')
    run=external(Path(run).resolve());require(not run.exists(),'No implicit restart')
    tests=test_evidence(run.parent)
    probes=native_probes(schedule,fitting,assets)
    old_records=read(Path(schedule['comparator']['train'])/'eval-000000/results.json')['records']
    controls=[next(r for r in old_records if r['group']==group) for group in ('development','custom')]
    p=dict(plan=str(Path(plan).resolve()),plan_sha256=digest(plan),features=str(Path(features).resolve()),
        feature_protocol_sha256=FEATURE_SHA,feature_audit_sha256=digest(Path(features)/'verification.json'),
        calibration_features=schedule['calibration_feature_source'],comparator=schedule['comparator'],
        policy=p0['policy'],settings=SETTINGS,checkpoints=CHECKPOINTS,source=sources(),runtime=runtime(),numeric_config=NUMERIC,
        memory_receipt=str(Path(memory_receipt).resolve()),memory_receipt_sha256=digest(memory_receipt),
        training_device='cuda',training_hardware=memory['hardware'],frozen_identity=state['frozen_identity'],
        fitting_rows=fitting,calibration=calibration,custom_reference=p0['custom_reference'],reference_sha256=p0['reference_sha256'],
        config=p0['config'],generation=p0['generation'],decoding=p0['decoding'],native_probes=probes,
        native_probe_pair_id=schedule['fitting_probe_pair_ids'][0],native_control_count=4*len(probes),
        native_perturbation=NATIVE_PERTURBATION,numerical_parity=parity_registration(p0['config']),
        parent_evidence=parents,tests=tests,driver=str(Path(driver).resolve()),driver_sha256=digest(driver),
        objective=p0['objective'],fitting_change_only=True,base_kv_intervention=False,training_updates=0,
        baseline_controls=controls,
        goal_achieved=False,global_model_selection_changed=False)
    run.mkdir();write_new(run/'protocol.json',p);return p


def verify(run):
    from .preference_pilot import runtime
    run=Path(run);p=read(run/'protocol.json');schedule,fitting,calibration=plan_data(p['plan'])
    require(p['source']==sources() and p['runtime']==runtime() and p['numeric_config']==numeric_config()==NUMERIC
        and p['settings']==SETTINGS and p['checkpoints']==CHECKPOINTS and digest(p['driver'])==p['driver_sha256']
        and p['plan_sha256']==digest(p['plan']) and p['fitting_rows']==fitting and p['calibration']==calibration
        and p['parent_evidence']==parent_evidence(p['features']) and p['feature_protocol_sha256']==FEATURE_SHA
        and p['feature_audit_sha256']==digest(Path(p['features'])/'verification.json')
        and p['calibration_features']==schedule['calibration_feature_source'] and p['comparator']==schedule['comparator'],
        'Broader run source/runtime/data changed')
    p0,_,state,_=comparator(schedule)
    old_records=read(Path(schedule['comparator']['train'])/'eval-000000/results.json')['records']
    require(p['baseline_controls']==[next(r for r in old_records if r['group']==g) for g in ('development','custom')],
        'Representative shared80 controls differ')
    require(all(p[k]==p0[k] for k in ('policy','settings','config','generation','decoding','numeric_config','calibration',
        'custom_reference','reference_sha256','objective','frozen_identity')) and p['training_hardware']==p0['training_hardware']
        and p['training_device']=='cuda' and p['fitting_change_only'] is True and p['base_kv_intervention'] is False,
        'Only fitting data may differ from comparator')
    for path,sha in p['tests'].items():require(digest(path)==sha,'Test evidence changed')
    require(digest(p['memory_receipt'])==p['memory_receipt_sha256'],'Memory receipt changed')
    _,assets=load_assets(p);frontier=preflight_frontiers(fitting,calibration,assets);memory=read(p['memory_receipt'])
    validate_preflight(memory,frontier)
    require(memory['frontier']==frontier and memory['source']==sources() and memory['hardware']==p['training_hardware']
        and memory['frozen_identity']==p['frozen_identity'] and memory['parent_evidence']==p['parent_evidence']
        and memory['plan_sha256']==p['plan_sha256'] and memory['features_audit_sha256']==p['feature_audit_sha256']
        and memory['adapter_snapshot_sha256']==policy.tensor_digest(state['adapter'].items())
        and memory['optimizer_updates']==memory['calibration_backward_calls']==0 and memory['all_parameters_restored'] is True,
        'Memory proof binding differs')
    require(p['native_probes']==native_probes(schedule,fitting,assets) and p['native_control_count']==4*len(p['native_probes'])
        and p['native_perturbation']==NATIVE_PERTURBATION and p['numerical_parity']==parity_registration(p['config']),
        'Native parity scope changed')
    return p


def initialize(run):
    run=Path(run);p=verify(run);schedule=read(p['plan']);_,_,state,_=comparator(schedule)
    require(hardware()==p['training_hardware'],'Initialization CUDA hardware differs')
    write_new(run/'initialize-started.json',dict(protocol_sha256=digest(run/'protocol.json')))
    wrapper,core=policy.load_base(p['policy'],SETTINGS);fresh.assert_native_components(wrapper,p['features']);restore_zero(core,state)
    require(policy.frozen_identity(wrapper.model,core)==p['frozen_identity'],'Zero frozen identity differs')
    # Saving before CUDA diagnostics preserves exactly the comparator zero RNG.
    restore_rng(state['rng']);save_checkpoint(run,0,core,None,p['frozen_identity'])
    require(same_rng(read_checkpoint(run,0)['rng'],state['rng']),'Saved zero RNG differs')
    policy.move_for_training(wrapper,core,'cuda');_,assets=load_assets(p);save_teacher(run,0,wrapper,core,p,assets)


def same_rng(left,right,include_cuda=True):
    return (left['python']==right['python'] and left['numpy']==right['numpy'] and torch.equal(left['torch'],right['torch'])
        and (not include_cuda or (len(left['cuda'])==len(right['cuda']) and all(torch.equal(a,b) for a,b in zip(left['cuda'],right['cuda'])))))


def teacher_metrics(wrapper,core,p,assets):
    schedule,_,_=plan_data(p['plan']);result={}
    decoders=(wrapper.model.talker.model,wrapper.model.talker.code_predictor.model)
    modes=[d.training for d in decoders]
    for d in decoders:d.eval()
    try:
        with torch.no_grad():
            for split,rows in (('fitting',[p['fitting_rows'][i] for i in schedule['fitting_probe_pair_ids']]),('calibration',p['calibration'])):
                observations=[]
                for row in rows:
                    value=pair(row,assets)
                    for mode in ('full','half'):
                        with core.replay(value['input_ids'],value['codes'],value['reference'],mode=='half',terminated=True) as seq:
                            output=core.outputs(seq);frames=len(value['codes']);measurement=observation(output,frames,value['id'],mode)
                            logits=output['residual_logits']
                            nll=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),seq.codes[:,1:].reshape(-1),reduction='none').reshape(frames,15)
                            measurement['residual_codebook_nll']=[float(x) for x in nll.sum(0)]
                            del output,logits,nll
                        del seq
                        observations.append(measurement)
                result[split]=dict(summary=summarize(observations),examples=observations)
    finally:
        for decoder,mode in zip(decoders,modes):decoder.train(mode)
    result['calibration_minus_fitting_objective']=result['calibration']['summary']['objective']-result['fitting']['summary']['objective']
    return result

def save_teacher(run,step,wrapper,core,p,assets):
    require(p['training_device']=='cuda' and hardware()==p['training_hardware']
        and all(v.device.type=='cuda' and v.dtype==torch.float32 for v in core.parameters())
        and next(wrapper.model.talker.parameters()).device.type=='cuda',
        'Teacher diagnostics require the registered CUDA FP32 training backend at every checkpoint')
    require(numeric_config()==p['numeric_config']==NUMERIC,'Teacher numeric configuration differs')
    value=teacher_metrics(wrapper,core,p,assets)
    value['fitting_probe_exposure']=exposure(read(p['plan']),step)
    require(policy.frozen_identity(wrapper.model,core)==p['frozen_identity'],'Teacher diagnostics changed frozen state')
    write_new(Path(run)/f'teacher-{step:06d}.json',dict(step=step,protocol_sha256=digest(Path(run)/'protocol.json'),
        checkpoint_sha256=digest(checkpoint_path(run,step)),measurements=value,frozen_identity=p['frozen_identity'],
        backend=dict(device=p['training_device'],hardware=p['training_hardware'],dtype='float32',
                     numeric_config=numeric_config(),decoder_training=False,grad_enabled=False)))

def native_controls(run):
    import soundfile as sf
    
    run=Path(run);p,wrapper,core,state=load(run,0)
    write_new(run/'native-controls-started.json',dict(protocol_sha256=digest(run/'protocol.json')))
    records=[]
    for case in p['baseline_controls']:
        require(digest(case['reference'])==case['reference_sha256'],'Historical reference changed')
        with torch.inference_mode():
            prompt=wrapper.create_voice_clone_prompt(ref_audio=case['reference'],ref_text=case['reference_text'],x_vector_only_mode=False)
            torch.manual_seed(case['seed']);np.random.seed(case['seed'])
            with core.native(case['reference_sha256']==p['reference_sha256']):
                waves,sr=wrapper.generate_voice_clone(text=case['text'],language='Auto',voice_clone_prompt=prompt,non_streaming_mode=True,**p['generation'])
        path=run/f"native-zero-{case['group']}.wav";require(not path.exists(),'Native control exists')
        sf.write(path,waves[0],sr,subtype='PCM_16')
        require(digest(path)==case['audio_sha256'],'Zero-update historical waveform mismatch')
        records.append(dict(case=case['case'],seed=case['seed'],group=case['group'],wav=path.name,audio_sha256=digest(path)))
    _,assets=load_assets(p)
    initial=policy.snapshot(core);initial_sha=policy.tensor_digest(initial.items());proof=[];perturbed_sha=None
    try:
        for nonzero in (False,True):
            if nonzero:
                perturbed_sha=perturb_native_probe(core)
            for probe in p['native_probes']:
                value=pair(p['fitting_rows'][probe['pair_id']],assets)
                for half in (False,True):
                    print(f"Native parity {len(proof)+1}/{p['native_control_count']} starting: pair={value['id']} half={half} nonzero={nonzero}",flush=True)
                    result=policy.verify_replay_pair(wrapper.model,core,value['input_ids'],value['codes'],value['reference'],p['generation'],half)
                    result.update(pair_id=value['id'],nonzero_adapter=nonzero,adapter_movement=policy.movement(core),
                        adapter_snapshot_sha256=perturbed_sha if nonzero else initial_sha)
                    require((result['adapter_movement']['maximum']>0)==nonzero,'Native perturbation control differs')
                    proof.append(result)
                    metrics=result['logit_metrics']['residual_logits']
                    print(f"Native parity {len(proof)}/{p['native_control_count']} passed: max_logit={metrics['maximum_absolute_logit_error']:.8g} "
                        f"max_chosen_logp={metrics['maximum_chosen_log_probability_error']:.8g} "
                        f"max_KL={max(metrics['maximum_forward_kl'],metrics['maximum_reverse_kl']):.8g}",flush=True)
    finally:
        policy.restore(core,initial);wrapper.model.zero_grad(set_to_none=True);restore_rng(state['rng'],include_cuda=False)
    require(policy.snapshot(core).keys()==initial.keys() and all(torch.equal(v,policy.snapshot(core)[n]) for n,v in initial.items())
        and policy.frozen_identity(wrapper.model,core)==p['frozen_identity'],'Native proof did not restore exact original state')
    write_new(run/'native-verification.json',dict(status='verified',protocol_sha256=digest(run/'protocol.json'),
        checkpoint_sha256=digest(checkpoint_path(run,0)),features_audit_sha256=p['feature_audit_sha256'],
        historical_waveform_controls=records,forced_trajectory=proof,native_probes=p['native_probes'],
        native_perturbation=NATIVE_PERTURBATION,initial_adapter_sha256=initial_sha,perturbed_adapter_sha256=perturbed_sha,
        frozen_identity=p['frozen_identity'],all_new_parameters_restored=True,optimizer_updates=0,
        checkpoint0_cpu_rng_restored=same_rng(rng_state(),state['rng'],include_cuda=False),
        cuda_rng_not_used_in_native_controls=True))

def validate_native_proof(run,p):
    from .reference_masked_supervision_parity import validate_metrics
    run=Path(run);proof=read(run/'native-verification.json')
    require(proof['status']=='verified' and proof['protocol_sha256']==digest(run/'protocol.json')
        and proof['checkpoint_sha256']==digest(checkpoint_path(run,0)) and proof['features_audit_sha256']==p['feature_audit_sha256']
        and proof['all_new_parameters_restored'] and proof['checkpoint0_cpu_rng_restored'] is True
        and proof['cuda_rng_not_used_in_native_controls'] is True and proof['optimizer_updates']==0 and proof['frozen_identity']==p['frozen_identity']
        and proof['native_probes']==p['native_probes'] and proof['native_perturbation']==p['native_perturbation']==NATIVE_PERTURBATION
        and len(proof['forced_trajectory'])==4*len(p['native_probes'])
        and {(r['pair_id'],r['half_reference'],r['nonzero_adapter']) for r in proof['forced_trajectory']}
            =={(v['pair_id'],a,b) for v in p['native_probes'] for a in (False,True) for b in (False,True)},
        'Complete registered zero/native replay proof required')
    zero=read_checkpoint(run,0)
    require(proof['initial_adapter_sha256']==policy.tensor_digest(zero['adapter'].items())
        and proof['perturbed_adapter_sha256']==policy.tensor_digest(perturbed_snapshot(zero['adapter']).items())
        and proof['perturbed_adapter_sha256']!=proof['initial_adapter_sha256'],
        'Native proof adapter identity differs')
    require(len(proof['historical_waveform_controls'])==2,'Both deployment branches need zero control')
    expected={(r['case'],r['seed'],r['group']):r for r in p['baseline_controls']}
    require({(r['case'],r['seed'],r['group']) for r in proof['historical_waveform_controls']}==set(expected),'Native control case scope differs')
    for row in proof['historical_waveform_controls']:
        require(digest(run/row['wav'])==row['audio_sha256']==expected[row['case'],row['seed'],row['group']]['audio_sha256'],
                'Native control WAV changed')
    for row in proof['forced_trajectory']:
        frames=next(v['target_frames'] for v in p['native_probes'] if v['pair_id']==row['pair_id'])
        require(row['frames']==frames and row['token_count']==16*frames+1 and row['prefix_and_consumed_frames_exact']
            and math.isfinite(row['adapter_movement']['maximum']) and row['adapter_movement']['maximum']>=0
            and ((row['adapter_movement']['maximum']>0)==row['nonzero_adapter'])
            and row['adapter_snapshot_sha256']==proof['perturbed_adapter_sha256' if row['nonzero_adapter'] else 'initial_adapter_sha256']
            and set(row['logit_metrics'])=={'main_logits','residual_logits'},'Native replay proof incomplete')
        for metrics in row['logit_metrics'].values():validate_metrics(metrics)
        require(row['logit_metrics']['main_logits']['shape']==[frames+1,p['numerical_parity']['main_vocabulary']]
            and row['logit_metrics']['residual_logits']['shape']==[frames,15,p['numerical_parity']['residual_vocabulary']],
            'Native parity vocabulary/frame distribution coverage differs')
    return proof

def permitted(run,p,previous):
    from .reference_masked_supervision_evaluation import progression
    from .reference_masked_supervision_broad_speech import verify as verify_speech,verified_receipt
    run=Path(run);validate_native_proof(run,p)
    checked,e=verify_speech(run);require(checked==p,'Speech/training registration differs')
    history=[]
    for step in CHECKPOINTS[:CHECKPOINTS.index(previous)+1]:
        audit=verified_receipt(run,p,e,step)
        require(audit['adapter_movement']==read_checkpoint(run,step)['adapter_movement'],'Audited adapter movement differs')
        history.append(audit)
    decision=progression(history)
    require(not decision['stop'],'Calibration progression forbids further updates: '+decision['reason'])
    return history

def train(run,stop_step,device='cuda'):
    
    run=Path(run);p=verify(run)
    require(device==p['training_device']=='cuda' and hardware()==p['training_hardware'],
            'Fitting hardware differs from verified CUDA memory preflight')
    require(stop_step in CHECKPOINTS[1:],'Unregistered training stop step')
    previous=CHECKPOINTS[CHECKPOINTS.index(stop_step)-1];permitted(run,p,previous)
    require(not checkpoint_path(run,stop_step).exists(),'Target checkpoint already exists')
    write_new(run/f'segment-{previous:06d}-{stop_step:06d}-started.json',dict(protocol_sha256=digest(run/'protocol.json'),
        previous_checkpoint_sha256=digest(checkpoint_path(run,previous)),native_proof_sha256=digest(run/'native-verification.json')))
    p,wrapper,core,state=load(run,previous,device,training=True)
    _,assets=load_assets(p);schedule=read(p['plan'])['updates'];optimizer=make_optimizer(core)
    if state['optimizer'] is not None:optimizer.load_state_dict(state['optimizer'])
    optimizer_check(optimizer,previous);restore_rng(state['rng'])
    try:
        for step in range(previous+1,stop_step+1):
            entry=schedule[step-1]
            require(entry['update']==step and len(entry['pair_ids'])==8 and entry['pair_weight']==.125 and entry['mode_weight']==.5,'Scheduled batch differs')
            rows=[p['fitting_rows'][i] for i in entry['pair_ids']]
            require(len({r['speaker_id'] for r in rows})==8 and all(r['partition']=='fitting' for r in rows),'Fitting speaker/partition boundary differs')
            items=[pair(r,assets) for r in rows];optimizer.zero_grad(set_to_none=True)
            if device=='cuda':torch.cuda.reset_peak_memory_stats()
            started=time.monotonic();measurements=backward_batch(core,items,entry['modes'])
            before=float(torch.nn.utils.clip_grad_norm_(core.parameters(),1.,error_if_nonfinite=True))
            after=math.sqrt(sum(float(p.grad.detach().square().sum()) for p in core.parameters()))
            optimizer.step();optimizer_check(optimizer,step)
            require(all(torch.isfinite(v).all() for v in core.parameters()),'Optimizer produced nonfinite parameters')
            frozen=policy.frozen_identity(wrapper.model,core)
            require(frozen==p['frozen_identity'],'Frozen parameters changed during fitting')
            receipt=dict(step=step,protocol_sha256=digest(run/'protocol.json'),schedule=entry,measurements=measurements,
                gradient_norm_before_clip=before,gradient_norm_after_clip=after,gradient_clipped=before>1.,learning_rate=optimizer.param_groups[0]['lr'],
                adapter_movement=policy.movement(core),frozen_identity=frozen,seconds=time.monotonic()-started,
                peak_allocated_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else None,
                peak_reserved_bytes=torch.cuda.max_memory_reserved() if device=='cuda' else None,oom_retries=0)
            write_new(run/f'update-{step:06d}.json',receipt)
            print(f'Masked supervision update {step}/{stop_step}: objective={measurements["summary"]["objective"]:.6f}',flush=True)
            del items,measurements;gc.collect()
        optimizer.zero_grad(set_to_none=True)
        save_checkpoint(run,stop_step,core,optimizer,p['frozen_identity'])
        save_teacher(run,stop_step,wrapper,core,p,assets)
    except BaseException as exc:
        wrapper.model.zero_grad(set_to_none=True)
        write_new(run/f'segment-{previous:06d}-{stop_step:06d}-failure.json',dict(error=type(exc).__name__,message=str(exc),
            oom=isinstance(exc,torch.cuda.OutOfMemoryError),automatic_retry=False,goal_achieved=False))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('preflight','prepare','initialize','native-controls','train'))
    for name in ('run','plan','features','driver','memory-receipt'):parser.add_argument('--'+name,type=Path)
    parser.add_argument('--stop-step',type=int);args=parser.parse_args();configure()
    if args.phase=='preflight':preflight(args.plan,args.features,args.memory_receipt)
    elif args.phase=='prepare':prepare(args.plan,args.features,args.run,args.driver,args.memory_receipt)
    elif args.phase=='initialize':initialize(args.run)
    elif args.phase=='native-controls':native_controls(args.run)
    else:train(args.run,args.stop_step)


if __name__=='__main__':main()

