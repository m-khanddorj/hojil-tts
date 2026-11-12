"""V2 bounded masked supervision; expanded independent native parity probes.

Speech generation/scoring/selection is a separate evaluator. This runner only
consumes its complete hash-bound audits and the immutable pre-quality plan.
"""
import argparse
from collections import defaultdict
import gc
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from . import reference_masked_supervision_policy_v2 as policy
from .reference_masked_supervision import NativeReference
from .reference_masked_supervision_plan import SETTINGS,build_plan

require,read,digest=policy.require,policy.read,policy.digest
PLAN_SHA='a4cefa5a77ca97696048df57a31f023c5a0441f88a626811e46edd13470c0c34'
CHECKPOINTS=[0,8,16,32]
NUMERIC=dict(cpu_threads=4,matmul_tf32=False,cudnn_tf32=False)
NATIVE_PERTURBATION=dict(seed=SETTINGS['seed']+1,distribution='clipped_standard_normal',
    standard_deviation=1e-4,clip_standard_deviations=3.,maximum_absolute_B=3e-4,unchanged_A=True)


def write_new(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')


def sources():
    folder=Path(__file__).parent
    names=['reference_masked_supervision_training_v2.py','reference_masked_supervision_policy_v2.py',
        'reference_masked_supervision.py','reference_masked_supervision_plan.py',
        'reference_masked_supervision_features.py','reference_masked_supervision_evaluation.py',
        'reference_masked_supervision_parity.py','../tests/test_reference_masked_supervision_parity.py',
        '../tests/test_reference_masked_supervision_training_v2.py','../tests/test_reference_masked_supervision.py']
    return {str((folder/name).resolve()):digest(folder/name) for name in names}


def numeric_config():
    return dict(cpu_threads=torch.get_num_threads(),matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                cudnn_tf32=torch.backends.cudnn.allow_tf32)


def configure():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    os.environ['HF_HUB_OFFLINE']=os.environ['TRANSFORMERS_OFFLINE']='1'


def plan_data(path):
    path=Path(path);require(digest(path)==PLAN_SHA,'Pinned pre-quality plan changed')
    p=read(path);data=Path(p['data_protocol']).parent
    for name,sha in p['sources'].items():require(digest(name)==sha,'Plan source changed')
    require(digest(data/'protocol.json')==p['data_protocol_sha256']
        and digest(data/'audit.json')==p['data_audit_sha256'],'Paired data registration changed')
    values={}
    for name,sha in p['manifest_sha256'].items():
        require(digest(data/name)==sha,'Real pair manifest changed')
        values[name]=[json.loads(line) for line in (data/name).read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    expected=build_plan(values['train.jsonl'],values['calibration.jsonl'])
    require(all(p[k]==v for k,v in expected.items()) and p['settings']==SETTINGS,'Deterministic schedule changed')
    indexed={r['id']:r for r in values['train.jsonl']}
    selected={i:indexed[i] for i in p['selected_fitting_pair_ids']}
    require(len(selected)==204 and len(p['real_feature_ids'])==323,'Pinned pair/feature coverage differs')
    return p,selected,values['calibration.jsonl']


def pair(row,features):
    require(row['id']!=row['ref_id'],'Reference and target must differ')
    target,ref=features[row['id']],features[row['ref_id']]
    require(target['id']==row['id'] and ref['id']==row['ref_id']
        and target['audio_sha256']==row['audio_sha256'] and ref['audio_sha256']==row['ref_audio_sha256']
        and target['protocol_sha256']==ref['protocol_sha256'],'Native pair feature provenance differs')
    reference=NativeReference(ref['ref_ids'],ref['codes'],ref['speaker'],ref['audio_sha256'],ref['protocol_sha256'])
    return dict(id=row['id'],speaker_id=row['speaker_id'],input_ids=target['input_ids'],codes=target['codes'],
                reference=reference,partition=row['partition'])


def normalized_objective(output,frames):
    require(frames>0 and output['token_count']==16*frames+1,'Complete codec/EOS denominator required')
    return output['main_nll']/(frames+1)+.3*output['residual_nll']/(15*frames)


def observation(output,frames,pair_id,mode):
    loss=normalized_objective(output,frames)
    require(torch.isfinite(loss) and torch.isfinite(output['main_nll']) and torch.isfinite(output['residual_nll']),
            'Nonfinite teacher objective')
    return dict(pair_id=pair_id,mode=mode,frames=frames,main_tokens=frames+1,residual_tokens=15*frames,
        main_nll=float(output['main_nll'].detach()),residual_nll=float(output['residual_nll'].detach()),
        objective=float(loss.detach()))


def summarize(rows):
    require(rows,'Empty teacher measurements')
    result={}
    for mode in ('full','half'):
        group=[r for r in rows if r['mode']==mode]
        require(group and len({r['pair_id'] for r in group})==len(group),'Missing or duplicate mode probe')
        main=sum(r['main_nll'] for r in group);residual=sum(r['residual_nll'] for r in group)
        main_tokens=sum(r['main_tokens'] for r in group);residual_tokens=sum(r['residual_tokens'] for r in group)
        item=dict(examples=len(group),objective=sum(r['objective'] for r in group)/len(group),
            main_mean_ce=sum(r['main_nll']/r['main_tokens'] for r in group)/len(group),
            residual_mean_ce=sum(r['residual_nll']/r['residual_tokens'] for r in group)/len(group),
            main_nll_sum=main,residual_nll_sum=residual,main_tokens=main_tokens,residual_tokens=residual_tokens,
            pooled_main_ce=main/main_tokens,pooled_residual_ce=residual/residual_tokens)
        if all('residual_codebook_nll' in r for r in group):
            item['pooled_residual_codebook_ce']=[sum(r['residual_codebook_nll'][i] for r in group)/sum(r['frames'] for r in group) for i in range(15)]
        result[mode]=item
    require(result['full']['examples']==result['half']['examples'],'Mode weights must be equal')
    return dict(objective=(result['full']['objective']+result['half']['objective'])/2,modes=result)


def backward_batch(core,pairs,modes):
    require(pairs and modes in (['full','half'],['half','full'])
        and len({p['id'] for p in pairs})==len(pairs),'Invalid pair batch or mode order')
    require(all(p['partition']=='fitting' for p in pairs),'Calibration data cannot enter backward')
    rows=[]
    for item in pairs:
        require(item['partition']=='fitting','Calibration data cannot enter backward')
        for mode in modes:
            with core.replay(item['input_ids'],item['codes'],item['reference'],mode=='half',terminated=True) as seq:
                output=core.outputs(seq);frames=len(item['codes'])
                row=observation(output,frames,item['id'],mode)
                logits=output['residual_logits'].detach()
                nll=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),seq.codes[:,1:].reshape(-1),reduction='none').reshape(frames,15)
                row['residual_codebook_nll']=[float(v) for v in nll.sum(0)]
                (normalized_objective(output,frames)/(2*len(pairs))).backward()
                del output,nll,logits
            del seq
            rows.append(row)
    return dict(summary=summarize(rows),examples=rows)


def make_optimizer(core):
    return torch.optim.AdamW(core.parameters(),lr=SETTINGS['learning_rate'],weight_decay=0.,foreach=False)


def optimizer_check(optimizer,step):
    require(len(optimizer.param_groups)==1,'Unexpected optimizer group count')
    group=optimizer.param_groups[0]
    require(group['lr']==SETTINGS['learning_rate'] and group['weight_decay']==0
        and tuple(group['betas'])==(.9,.999) and group['eps']==1e-8 and not group['amsgrad'],'Optimizer recipe differs')
    if step==0:require(not optimizer.state,'Fresh optimizer required');return
    require(len(optimizer.state)==len(group['params']),'Optimizer moment coverage differs')
    for parameter in group['params']:
        state=optimizer.state[parameter]
        require(set(state)=={'step','exp_avg','exp_avg_sq'} and float(state['step'])==step,'Optimizer update counter differs')
        for key in ('exp_avg','exp_avg_sq'):
            value=state[key]
            require(value.dtype==torch.float32 and value.shape==parameter.shape and torch.isfinite(value).all(),
                    'Invalid FP32 optimizer moments')


def rng_state():
    value=np.random.get_state()
    return dict(python=random.getstate(),numpy=dict(name=value[0],keys=value[1].tolist(),position=value[2],
        gaussian=value[3],cached=value[4]),torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [])


def restore_rng(value,include_cuda=True):
    random.setstate(value['python']);n=value['numpy']
    np.random.set_state((n['name'],np.array(n['keys'],dtype=np.uint32),n['position'],n['gaussian'],n['cached']))
    torch.set_rng_state(value['torch'])
    if include_cuda and value['cuda']:
        require(torch.cuda.is_available(),'Checkpoint CUDA RNG requires CUDA');torch.cuda.set_rng_state_all(value['cuda'])


def checkpoint_path(run,step):return Path(run)/f'checkpoint-{step:06d}.pt'


def save_checkpoint(run,step,core,optimizer,frozen):
    path=checkpoint_path(run,step)
    require(step in CHECKPOINTS and not path.exists() and not path.with_suffix('.json').exists(),'Checkpoint exists or step unregistered')
    if optimizer is not None:optimizer_check(optimizer,step)
    else:require(step==0,'Trained checkpoint requires optimizer')
    value=dict(step=step,adapter=policy.snapshot(core),optimizer=None if optimizer is None else optimizer.state_dict(),
        rng=rng_state(),protocol_sha256=digest(Path(run)/'protocol.json'),frozen_identity=frozen,adapter_movement=policy.movement(core))
    # FP32 moments are explicitly serialized on CPU; reload maps onto parameters.
    if value['optimizer'] is not None:
        value['optimizer']={**value['optimizer'],'state':{k:{n:v.detach().cpu() if isinstance(v,torch.Tensor) else v
                            for n,v in row.items()} for k,row in value['optimizer']['state'].items()}}
    with path.open('xb') as stream:torch.save(value,stream)
    write_new(path.with_suffix('.json'),dict(sha256=digest(path),step=step,protocol_sha256=value['protocol_sha256'],
        frozen_identity=frozen,adapter_movement=value['adapter_movement']))
    return value


def read_checkpoint(run,step):
    path=checkpoint_path(run,step);receipt=read(path.with_suffix('.json'))
    require(step in CHECKPOINTS and receipt['step']==step and receipt['sha256']==digest(path)
        and receipt['protocol_sha256']==digest(Path(run)/'protocol.json'),'Checkpoint hash/protocol mismatch')
    state=torch.load(path,map_location='cpu',weights_only=True)
    require(state['step']==step and state['protocol_sha256']==receipt['protocol_sha256']
        and state['frozen_identity']==receipt['frozen_identity'] and state['adapter_movement']==receipt['adapter_movement'],
        'Checkpoint receipt differs')
    require((step==0)==(state['optimizer'] is None),'Optimizer checkpoint boundary differs')
    require(state['adapter'] and all(v.dtype==torch.float32 and torch.isfinite(v).all() for v in state['adapter'].values()),'Invalid adapter checkpoint')
    return state


def memory_frontier(rows,features):
    coordinates={}
    for row in rows:
        value=pair(row,features);r=value['reference'];T=len(value['codes'])
        P=8+(r.ref_ids.shape[1]-5)+(value['input_ids'].shape[1]-8)+len(r.ref_codes)+2
        entry=dict(pair_id=row['id'],partition=row['partition'],prefix_positions=P,target_frames=T,total_positions=P+T)
        key=P+T,T
        if key not in coordinates or row['id']<coordinates[key]['pair_id']:coordinates[key]=entry
    return sorted([v for key,v in coordinates.items() if not any(other[0]>=key[0] and other[1]>=key[1]
        and other!=key for other in coordinates)],key=lambda r:(r['total_positions'],r['target_frames'],r['pair_id']))


def native_probes(schedule,fitting,assets):
    """Only fitting pairs: original fixed probe plus the fitting (N,T) frontier."""
    require(fitting and all(r['partition']=='fitting' for r in fitting.values()),'Native parity probes must be fitting pairs')
    original=schedule['fitting_probe_pair_ids'][0]
    values=memory_frontier(list(fitting.values()),assets)+memory_frontier([fitting[original]],assets)
    indexed={r['pair_id']:r for r in values}
    return [indexed[key] for key in sorted(indexed)]


def perturbed_snapshot(initial):
    """Reproducible bounded B-only probe state, independently hash-verifiable."""
    generator=torch.Generator(device='cpu').manual_seed(NATIVE_PERTURBATION['seed'])
    result={key:value.detach().cpu().clone() for key,value in initial.items()}
    require(result and all(key.endswith(('.A','.B')) and value.dtype==torch.float32 for key,value in result.items()),
        'Native perturbation parameter coverage differs')
    for key,value in result.items():
        if key.endswith('.B'):
            value.copy_(torch.randn(value.shape,generator=generator).clamp(-3.,3.)*1e-4)
            require(float(value.abs().max())<=NATIVE_PERTURBATION['maximum_absolute_B']+1e-10,
                'Native perturbation bound differs')
    return result


def perturb_native_probe(core):
    """Temporary native control only; never used to initialize fitting."""
    require(all(v.device.type=='cpu' for v in core.parameters()),'Native perturbation requires CPU FP32')
    state=perturbed_snapshot(policy.snapshot(core));policy.restore(core,state)
    return policy.tensor_digest(state.items())


def parity_registration(config):
    from .reference_masked_supervision_parity import VERSION,LIMITS
    talker=read(Path(config['base'])/'config.json')['talker_config']
    require(talker['num_code_groups']==16,'Native parity requires16 codebooks')
    return dict(version=VERSION,limits=dict(LIMITS),
        main_vocabulary=talker['vocab_size'],residual_vocabulary=talker['code_predictor_config']['vocab_size'])


def prior_failure_evidence(root):
    """Preserve v1's failed zero-update attempt and completed read-only diagnosis."""
    root=Path(root);old=root/'reference-masked-supervision-v1'
    status_path=root/'reference-masked-supervision-v1-status.json';status=read(status_path)
    inputs_path=root/'reference-masked-supervision-v1-driver-inputs.json';inputs=read(inputs_path)
    protocol=read(old/'protocol.json');diagnosis_path=root/'masked-native-replay-diagnosis-v1.json'
    diagnosis=read(diagnosis_path);script=root/'diagnose-masked-native-replay-v1.py'
    require(status['status']=='failed' and 'native-controls' in status['error']
        and status['goal_achieved'] is False and status['global_model_selection_changed'] is False
        and status['driver_inputs_sha256']==digest(inputs_path),'Preserved failed v1 driver evidence required')
    require(digest(diagnosis_path)=='96af10a4551829788234ea52e6e91fa0673b7fd6267b774fd27573d07d7cd87d'
        and diagnosis['protocol_sha256']==digest(old/'protocol.json')
        and diagnosis['checkpoint_sha256']==digest(checkpoint_path(old,0))
        and diagnosis['diagnostic_source_sha256']==digest(script)
        and diagnosis['all_parameters_unchanged'] is True and diagnosis['optimizer_updates']==0
        and diagnosis['goal_achieved'] is False and len(diagnosis['modes'])==2
        and {r['half_reference'] for r in diagnosis['modes']}=={False,True}
        and all(r['native_embeddings_exact'] for r in diagnosis['modes']),
        'Completed unchanged-policy numerical diagnosis required')
    require(not list(old.glob('update-*.json')) and not (old/'native-verification.json').exists()
        and {v.name for v in old.glob('checkpoint-*.pt')}=={'checkpoint-000000.pt'},
        'V1 must remain an untrained failed native-control attempt')
    evidence=dict(inputs['files'])
    evidence.update(protocol['source']);evidence.update(protocol['tests'])
    for path,sha in evidence.items():require(digest(path)==sha,'Preserved v1 source/input evidence changed')
    for item in status['logs'].values():
        require(digest(item['path'])==item['sha256'],'Preserved v1 phase log changed')
        evidence[item['path']]=item['sha256']
    for path in (status_path,inputs_path,old/'protocol.json',checkpoint_path(old,0),checkpoint_path(old,0).with_suffix('.json'),
            old/'native-controls-started.json',diagnosis_path,script,root/'masked-native-replay-diagnosis-v1.log',
            root/'reference-masked-supervision-v1-native-controls.log'):
        evidence[str(path.resolve())]=digest(path)
    return evidence


def hardware():
    value=torch.cuda.get_device_properties(torch.cuda.current_device())
    return dict(name=value.name,total_memory=value.total_memory,compute_capability=[value.major,value.minor])


def validate_memory(memory,frontier):
    require(memory['device']=='cuda' and isinstance(memory['hardware'],dict)
        and memory['hardware']['total_memory']>0,'Production CUDA memory preflight required')
    expected={(r['pair_id'],mode):r for r in frontier for mode in ('full','half')}
    rows=memory['measurements']
    require(len(rows)==len(expected) and {(r['pair_id'],r['mode']) for r in rows}==set(expected),
            'Exact frontier/mode measurement coverage required')
    for row in rows:
        frames=expected[row['pair_id'],row['mode']]['target_frames']
        require(row['frames']==frames and row['main_tokens']==frames+1 and row['residual_tokens']==15*frames
            and all(math.isfinite(row[k]) and row[k]>=0 for k in ('objective','main_nll','residual_nll'))
            and math.isfinite(row['gradient_norm']) and row['gradient_norm']>0
            and math.isfinite(row['seconds']) and row['seconds']>0
            and 0<row['peak_allocated_bytes']<=row['peak_reserved_bytes']<=memory['hardware']['total_memory'],
            'Nonfinite or incomplete CUDA memory measurement')


def preflight(plan,features,policy_kind,policy_run,receipt_path,device='cuda'):
    from . import reference_masked_supervision_features as feature_api
    configure();receipt_path=Path(receipt_path)
    require(not receipt_path.exists(),'Memory preflight already recorded')
    schedule,fitting,calibration=plan_data(plan);fp,assets=feature_api.load_features(features)
    require(Path(fp['plan']).resolve()==Path(plan).resolve(),'Feature plan differs')
    descriptor=policy.baseline_evidence(policy_kind,policy_run)
    frontier=memory_frontier(list(fitting.values())+calibration,assets)
    write_new(receipt_path.with_suffix('.started.json'),dict(plan_sha256=PLAN_SHA,policy=descriptor,frontier=frontier))
    wrapper,core=policy.load_base(descriptor,SETTINGS);feature_api.assert_native_components(wrapper,features)
    frozen=policy.frozen_identity(wrapper.model,core);initial=policy.snapshot(core)
    policy.move_for_training(wrapper,core,device);indexed={r['id']:r for r in list(fitting.values())+calibration}
    measurements=[]
    for item in frontier:
        value=pair(indexed[item['pair_id']],assets)
        for mode in ('full','half'):
            wrapper.model.zero_grad(set_to_none=True)
            if torch.device(device).type=='cuda':torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
            started=time.monotonic()
            with core.replay(value['input_ids'],value['codes'],value['reference'],mode=='half',terminated=True) as seq:
                output=core.outputs(seq);loss=normalized_objective(output,len(value['codes']))
                require(torch.isfinite(loss),'Nonfinite preflight objective');loss.backward()
                grads=[p.grad for p in core.parameters()]
                require(all(g is not None and torch.isfinite(g).all() for g in grads),'Preflight gradient coverage differs')
                row=observation(output,len(value['codes']),value['id'],mode)
                row['gradient_norm']=math.sqrt(sum(float(g.detach().square().sum()) for g in grads))
                del output,loss,grads
            del seq
            row.update(seconds=time.monotonic()-started,
                peak_allocated_bytes=torch.cuda.max_memory_allocated() if torch.device(device).type=='cuda' else None,
                peak_reserved_bytes=torch.cuda.max_memory_reserved() if torch.device(device).type=='cuda' else None)
            measurements.append(row);gc.collect()
    wrapper.model.zero_grad(set_to_none=True)
    require(all(torch.equal(v,policy.snapshot(core)[n]) for n,v in initial.items())
        and policy.frozen_identity(wrapper.model,core)==frozen,'Preflight mutated a policy parameter')
    policy.restore(core,initial)
    write_new(receipt_path,dict(status='verified',plan_sha256=PLAN_SHA,feature_protocol_sha256=digest(Path(features)/'protocol.json'),
        features_audit_sha256=digest(Path(features)/'verification.json'),policy=descriptor,source=sources(),numeric_config=numeric_config(),
        frontier=frontier,measurements=measurements,device=str(device),hardware=hardware() if device=='cuda' else None,frozen_identity=frozen,
        adapter_snapshot_sha256=policy.tensor_digest(initial.items()),optimizer_updates=0,all_parameters_restored=True))


def prepare(plan,features,policy_kind,policy_run,run,driver,memory_receipt):
    from . import reference_masked_supervision_features as feature_api
    from .policy import external
    from .preference_pilot import runtime
    require(numeric_config()==NUMERIC,'Registered CPU/TF32 numeric settings required')
    schedule,fitting,calibration=plan_data(plan);fp,assets=feature_api.load_features(features)
    require(Path(fp['plan']).resolve()==Path(plan).resolve(),'Feature plan differs')
    descriptor=policy.baseline_evidence(policy_kind,policy_run);memory=read(memory_receipt)
    frontier=memory_frontier(list(fitting.values())+calibration,assets);validate_memory(memory,frontier)
    require(memory['status']=='verified' and memory['plan_sha256']==PLAN_SHA
        and memory['feature_protocol_sha256']==digest(Path(features)/'protocol.json')
        and memory['features_audit_sha256']==digest(Path(features)/'verification.json')
        and memory['source']==sources() and memory['numeric_config']==NUMERIC and memory['policy']==descriptor
        and memory['frontier']==frontier
        and len(memory['measurements'])==2*len(memory['frontier']) and memory['optimizer_updates']==0
        and memory['all_parameters_restored'],'Complete unchanged-policy memory preflight required')
    run=external(Path(run).resolve());require(not run.exists(),'Refuse run overwrite')
    tests={}
    for name,count in (('reference-masked-supervision-tests.log',8),('reference-masked-supervision-training-v2-tests.log',11),
            ('reference-masked-supervision-parity-tests.log',11)):
        path=run.parent/name;value=path.read_text(encoding='utf-8-sig')
        require(f'Ran {count} tests' in value and '\nOK\n' in value and 'skipped' not in value.lower(),'Required tiny tests missing')
        tests[str(path)]=digest(path)
    custom=next(r for r in descriptor['controls'] if r['group']=='custom');probes=native_probes(schedule,fitting,assets)
    protocol=dict(plan=str(Path(plan).resolve()),plan_sha256=PLAN_SHA,features=str(Path(features).resolve()),
        feature_protocol_sha256=digest(Path(features)/'protocol.json'),feature_audit_sha256=digest(Path(features)/'verification.json'),
        policy=descriptor,settings=dict(SETTINGS),checkpoints=CHECKPOINTS,source=sources(),runtime=runtime(),numeric_config=NUMERIC,
        memory_receipt=str(Path(memory_receipt).resolve()),memory_receipt_sha256=digest(memory_receipt),
        training_device='cuda',training_hardware=memory['hardware'],
        frozen_identity=memory['frozen_identity'],fitting_rows=fitting,calibration=calibration,
        custom_reference={k:custom[k] for k in ('reference','reference_sha256','reference_text','speaker_id')},
        reference_sha256=descriptor['reference_sha256'],config=descriptor['config'],generation=descriptor['generation'],decoding=descriptor['decoding'],
        native_probe_pair_id=schedule['fitting_probe_pair_ids'][0],native_probes=probes,native_control_count=4*len(probes),
        native_perturbation=NATIVE_PERTURBATION,numerical_parity=parity_registration(descriptor['config']),
        previous_failure_evidence=prior_failure_evidence(run.parent),
        tests=tests,driver=str(Path(driver).resolve()),driver_sha256=digest(driver),
        objective='equal8pairs/equal2modes: main_nll/(T+1)+0.3*residual_nll/(15*T)',
        training_updates=0,goal_achieved=False,global_model_selection_changed=False)
    run.mkdir();write_new(run/'protocol.json',protocol)
    return protocol


def verify(run):
    from . import reference_masked_supervision_features as feature_api
    from .preference_pilot import runtime
    run=Path(run);p=read(run/'protocol.json')
    require(p['source']==sources() and p['runtime']==runtime() and p['numeric_config']==numeric_config()==NUMERIC
        and p['settings']==SETTINGS and p['checkpoints']==CHECKPOINTS and digest(p['driver'])==p['driver_sha256'],
        'Masked training source/runtime/recipe changed')
    schedule,fitting,calibration=plan_data(p['plan'])
    require(p['plan_sha256']==PLAN_SHA and p['fitting_rows']==fitting and p['calibration']==calibration,'Pair scope changed')
    for path,sha in p['tests'].items():require(digest(path)==sha,'Test evidence changed')
    require(digest(p['memory_receipt'])==p['memory_receipt_sha256']
        and digest(Path(p['features'])/'protocol.json')==p['feature_protocol_sha256']
        and digest(Path(p['features'])/'verification.json')==p['feature_audit_sha256'],'Preparation evidence changed')
    memory=read(p['memory_receipt']);validate_memory(memory,memory['frontier'])
    require(p['training_device']==memory['device']=='cuda' and p['training_hardware']==memory['hardware'],
            'Training device differs from memory proof')
    _,assets=feature_api.load_features(p['features'])
    require(p['policy']==policy.baseline_evidence(p['policy']['kind'],p['policy']['run']),'Selected baseline changed')
    require(p['native_probe_pair_id']==schedule['fitting_probe_pair_ids'][0],'Native probe selection changed')
    require(p['native_probes']==native_probes(schedule,fitting,assets) and p['native_perturbation']==NATIVE_PERTURBATION,
        'Registered fitting-only native probe scope or perturbation changed')
    require(p['numerical_parity']==parity_registration(p['config']),'Numerical parity registration changed')
    require(p['native_control_count']==4*len(p['native_probes'])
        and p['previous_failure_evidence']==prior_failure_evidence(run.parent),'V2 scope or preceding failure evidence changed')
    return p


def load(run,step,device='cpu',training=False):
    from . import reference_masked_supervision_features as feature_api
    p=verify(run);state=read_checkpoint(run,step);wrapper,core=policy.load_base(p['policy'],p['settings'])
    feature_api.assert_native_components(wrapper,p['features'])
    require(policy.frozen_identity(wrapper.model,core)==p['frozen_identity']==state['frozen_identity'],'Frozen original identity differs')
    policy.restore(core,state['adapter'])
    require(policy.movement(core)==state['adapter_movement'],'Checkpoint movement differs')
    if training:policy.move_for_training(wrapper,core,device)
    else:require(device=='cpu','Native speech evaluation is CPU FP32 only');wrapper.model.eval()
    restore_rng(state['rng'],include_cuda=training)
    return p,wrapper,core,state


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
    require(policy.frozen_identity(wrapper.model,core)==p['frozen_identity'],'Teacher diagnostics changed frozen state')
    write_new(Path(run)/f'teacher-{step:06d}.json',dict(step=step,protocol_sha256=digest(Path(run)/'protocol.json'),
        checkpoint_sha256=digest(checkpoint_path(run,step)),measurements=value,frozen_identity=p['frozen_identity'],
        backend=dict(device=p['training_device'],hardware=p['training_hardware'],dtype='float32',
                     numeric_config=numeric_config(),decoder_training=False,grad_enabled=False)))


def initialize(run):
    from . import reference_masked_supervision_features as feature_api
    run=Path(run);p=verify(run)
    require(p['training_device']=='cuda' and hardware()==p['training_hardware'],
            'Initialization diagnostics hardware differs from verified CUDA memory preflight')
    write_new(run/'initialize-started.json',dict(protocol_sha256=digest(run/'protocol.json')))
    wrapper,core=policy.load_base(p['policy'],p['settings']);feature_api.assert_native_components(wrapper,p['features'])
    require(policy.frozen_identity(wrapper.model,core)==p['frozen_identity'] and policy.movement(core)['maximum']==0,'Invalid zero initialization')
    random.seed(SETTINGS['seed']);np.random.seed(SETTINGS['seed'])
    save_checkpoint(run,0,core,None,p['frozen_identity'])
    policy.move_for_training(wrapper,core,p['training_device'])
    _,assets=feature_api.load_features(p['features']);save_teacher(run,0,wrapper,core,p,assets)


def native_controls(run):
    import soundfile as sf
    from . import reference_masked_supervision_features as feature_api
    run=Path(run);p,wrapper,core,state=load(run,0)
    write_new(run/'native-controls-started.json',dict(protocol_sha256=digest(run/'protocol.json')))
    records=[]
    for case in p['policy']['controls']:
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
    _,assets=feature_api.load_features(p['features'])
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
        policy.restore(core,initial);wrapper.model.zero_grad(set_to_none=True)
    require(policy.snapshot(core).keys()==initial.keys() and all(torch.equal(v,policy.snapshot(core)[n]) for n,v in initial.items())
        and policy.frozen_identity(wrapper.model,core)==p['frozen_identity'],'Native proof did not restore exact original state')
    write_new(run/'native-verification.json',dict(status='verified',protocol_sha256=digest(run/'protocol.json'),
        checkpoint_sha256=digest(checkpoint_path(run,0)),features_audit_sha256=p['feature_audit_sha256'],
        historical_waveform_controls=records,forced_trajectory=proof,native_probes=p['native_probes'],
        native_perturbation=NATIVE_PERTURBATION,initial_adapter_sha256=initial_sha,perturbed_adapter_sha256=perturbed_sha,
        frozen_identity=p['frozen_identity'],all_new_parameters_restored=True,optimizer_updates=0))


def validate_native_proof(run,p):
    from .reference_masked_supervision_parity import validate_metrics
    run=Path(run);proof=read(run/'native-verification.json')
    require(proof['status']=='verified' and proof['protocol_sha256']==digest(run/'protocol.json')
        and proof['checkpoint_sha256']==digest(checkpoint_path(run,0)) and proof['features_audit_sha256']==p['feature_audit_sha256']
        and proof['all_new_parameters_restored'] and proof['optimizer_updates']==0 and proof['frozen_identity']==p['frozen_identity']
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
    expected={(r['case'],r['seed'],r['group']):r for r in p['policy']['controls']}
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
    from .reference_masked_supervision_speech_v2 import verify as verify_speech,verified_receipt
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
    from . import reference_masked_supervision_features as feature_api
    run=Path(run);p=verify(run)
    require(device==p['training_device']=='cuda' and hardware()==p['training_hardware'],
            'Fitting hardware differs from verified CUDA memory preflight')
    require(stop_step in CHECKPOINTS[1:],'Unregistered training stop step')
    previous=CHECKPOINTS[CHECKPOINTS.index(stop_step)-1];permitted(run,p,previous)
    require(not checkpoint_path(run,stop_step).exists(),'Target checkpoint already exists')
    write_new(run/f'segment-{previous:06d}-{stop_step:06d}-started.json',dict(protocol_sha256=digest(run/'protocol.json'),
        previous_checkpoint_sha256=digest(checkpoint_path(run,previous)),native_proof_sha256=digest(run/'native-verification.json')))
    p,wrapper,core,state=load(run,previous,device,training=True)
    _,assets=feature_api.load_features(p['features']);schedule=read(p['plan'])['updates'];optimizer=make_optimizer(core)
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
    parser.add_argument('--phase',required=True,choices=['preflight','prepare','initialize','native-controls','train'])
    for key in ('run','plan','features','policy-run','driver','memory-receipt'):parser.add_argument('--'+key,type=Path)
    parser.add_argument('--policy-kind',choices=['centered16','attention4']);parser.add_argument('--stop-step',type=int)
    parser.add_argument('--device',default='cuda',choices=['cpu','cuda']);args=parser.parse_args();configure()
    if args.phase=='preflight':preflight(args.plan,args.features,args.policy_kind,args.policy_run,args.memory_receipt,args.device)
    elif args.phase=='prepare':prepare(args.plan,args.features,args.policy_kind,args.policy_run,args.run,args.driver,args.memory_receipt)
    elif args.phase=='initialize':initialize(args.run)
    elif args.phase=='native-controls':native_controls(args.run)
    else:train(args.run,args.stop_step,args.device)


if __name__=='__main__':main()
