"""Conditional no-update projected-reference-K/V intervention diagnostic.

Donors are captured from teacher0's actual native prefill, before normalization
and rotary position encoding. No target recording or future token enters them.
This fitting-only diagnostic cannot select/promote a policy or establish quality.
"""
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import math
from pathlib import Path

from . import reference_masked_supervision_routing_ablation as ablation

read,write,digest,require = ablation.read,ablation.write,ablation.digest,ablation.require
ABLATION_SHA='faaca2f70c9b47dd3a3373c5de0605a8a6d2bd1efe0cf2c0f952cba50c024ee9'
FLAGS=dict(diagnostic_only=True,optimizer_updates=0,goal_achieved=False,global_model_selection_changed=False,
    policy_selected=False,target_audio_used_by_donor=False,target_codes_used_by_donor=False)
SETTINGS=dict(seed=2026,modes=['full','half'],arms=['teacher0','self_graft0','student32','graft32'],
    pairs=2,recordings=16,comparison_recordings=12,self_graft_controls=4,
    selection='first fixed fitting probe, then longest-prefix other fixed probe; ties retain plan order',
    donor='actual first native prefill projected K/V; protected positions only; teacher0',
    location='after routed main k_proj/v_proj forward, before native k_norm/RoPE/attention/cache',
    dummy_prefix_check=[0,1],device='cpu',dtype='float32',cpu_threads=4,
    backend='unchanged registered CPU FP32 SDPA; no backend experiment',
    serial_pinned_asr=True,no_fitting=True,first_frame_state_not_frozen=True)


def failed_parent(run):
    run=Path(run);p,e=ablation.verify(run,ABLATION_SHA)
    status_path=run.parent/(run.name+'-status.json');closure_path=run.parent/(run.name+'-process-closed.json')
    status,closure=read(status_path),read(closure_path);result=read(run/'results.json')
    driver=run.parent/'run-masked-supervision-routing-ablation-v1.py'
    bindings=dict(protocol_sha256=ABLATION_SHA,driver_sha256=digest(driver),
        terminal_status_sha256=digest(status_path),results_sha256=digest(run/'results.json'))
    require(status['status']=='complete' and status['protocol_sha256']==ABLATION_SHA
        and status['results_sha256']==bindings['results_sha256']
        and all(status.get(k)==v for k,v in ablation.FLAGS.items()),'Complete unpromoted ablation required')
    require(closure.get('status')=='closed' and type(closure.get('exit_code')) is int and closure['exit_code']==0
        and type(closure.get('original_session_id')) is int and closure['original_session_id']==23429
        and all(closure.get(k)==v for k,v in bindings.items()),'Authoritative original23429 exit0 closure required')
    require(result['status']=='verified' and result['protocol_sha256']==ABLATION_SHA
        and result['new_recordings']==128 and result['reused_recordings']==160
        and set(result['arms'])=={'0','32'} and all(result.get(k)==v for k,v in ablation.FLAGS.items()),
        'Both complete ablation arms required')
    phases={f'{phase}-{step:06d}' for step in (0,32) for phase in ('generate','score','audit')}|{'finalize'}
    require(set(status['logs'])==phases,'Incomplete ablation phase history')
    hashes=dict(e['evidence']);paths=[run/'protocol.json',run/'results.json',status_path,closure_path,driver]
    for item in status['logs'].values():
        require(type(item['exit_code']) is int and item['exit_code']==0 and digest(item['path'])==item['sha256'],
            'Ablation phase failed or changed')
        paths.append(Path(item['path']))
    comparisons={}
    for step in (0,32):
        audit,new=ablation.audited_arm(run,p,e,step)
        old=read(Path(e['train'])/f'eval-{step:06d}/results.json')['records']
        comparisons[str(step)]=ablation.combined(e['original_cases'],old,new)
        require(all(result['arms'][str(step)].get(k)==v for k,v in comparisons[str(step)].items()),
            'Ablation result differs from native evidence')
        dest=ablation.directory(run,step)
        paths.extend((run/f'verification-{step:06d}.json',dest/'generated.json',dest/'results.json'))
        for row in new:paths.extend((dest/row['wav'],dest/row['trajectory']))
    require(comparisons['32']['calibration_thresholds_met'] is False,
        'Qualified checkpoint32 must go to original114 before this diagnostic')
    for path in paths:hashes[str(path.resolve())]=digest(path)
    return p,e,hashes


def selected_probes(fixed,fitting,shapes):
    require(len(fixed)==len(set(fixed))==16,'Exact existing fixed16 fitting probes required')
    candidates=[]
    for identity in fixed:
        row=fitting[identity]
        require(row['id']==identity and row['partition']=='fitting' and row['ref_id']!=identity,
            'Only distinct fitting target/reference pairs permitted')
        nt=shapes[identity]['input_ids']['shape'][1]-8
        nr=shapes[row['ref_id']]['ref_ids']['shape'][1]-5
        frames=shapes[row['ref_id']]['codes']['shape'][0]
        require(min(nt,nr,frames)>0,'Nonempty native prefix required')
        # No target-code shape or target-audio feature is consulted.
        candidates.append(dict(pair_id=identity,reference_id=row['ref_id'],text=row['text'],
            reference=row['ref_audio'],reference_sha256=row['ref_audio_sha256'],reference_text=row['ref_text'],
            speaker_id=row['speaker_id'],partition='fitting',target_input_ids=shapes[identity]['input_ids'],
            layout=dict(target_start=8+nr,target_stop=8+nr+nt,prefix_length=8+nr+nt+frames+2)))
    return [candidates[0],max(candidates[1:],key=lambda r:r['layout']['prefix_length'])]


def panel(p):
    plan=read(p['plan']);feature=Path(p['features']);extraction=read(feature/'extraction.json');shapes={};hashes={}
    fixed=plan['fitting_probe_pair_ids'];ids=set(fixed)|{p['fitting_rows'][i]['ref_id'] for i in fixed}
    for identity in sorted(ids):
        path=feature/'assets'/hashlib.sha256(identity.encode()).hexdigest()/'receipt.json'
        require(digest(path)==extraction['asset_receipts'][identity],'Native feature metadata changed')
        receipt=read(path);require(receipt['id']==identity,'Feature metadata identity differs')
        shapes[identity]=receipt['tensors'];hashes[str(path.resolve())]=digest(path)
    return selected_probes(fixed,p['fitting_rows'],shapes),hashes


def prepare(run,out,driver,test_log,note):
    run,out=Path(run).resolve(),Path(out).resolve()
    require(out.parent==run.parent and out.name=='masked-supervision-kv-probe-v1' and not out.exists(),
        'New external diagnostic directory required')
    p,e,hashes=failed_parent(run);cases,metadata=panel(p);hashes.update(metadata)
    log=Path(test_log).read_text(encoding='utf-8-sig')
    require(any(x.startswith('Ran 10 tests in ') for x in log.splitlines()) and log.strip().endswith('OK')
        and 'skipped' not in log.lower() and 'FAILED' not in log,'Ten passing focused tests without skips required')
    for path in (Path(__file__),Path(__file__).parents[1]/'tests/test_reference_masked_supervision_kv_probe.py',
                 Path(driver),Path(test_log),Path(note)):
        hashes[str(path.resolve())]=digest(path)
    protocol=dict(ablation=str(run),ablation_protocol_sha256=ABLATION_SHA,train=e['train'],cases=cases,
        evidence=hashes,settings=SETTINGS,config=p['config'],features=p['features'],runtime=p['runtime'],
        numeric_config=p['numeric_config'],generation=p['generation'],decoding=p['decoding'],
        asr_manifest=e['asr_manifest'],asr_manifest_sha256=e['asr_manifest_sha256'],
        main_layers=read(Path(p['config']['base'])/'config.json')['talker_config']['num_hidden_layers'],
        checkpoint_sha256={str(s):digest(Path(e['train'])/f'checkpoint-{s:06d}.pt') for s in (0,32)},**FLAGS)
    out.mkdir();write(out/'protocol.json',protocol);return protocol


def verify(out,protocol_sha):
    out=Path(out);require(digest(out/'protocol.json')==protocol_sha,'Registered KV diagnostic protocol changed')
    e=read(out/'protocol.json');p,parent,hashes=failed_parent(e['ablation']);cases,metadata=panel(p);hashes.update(metadata)
    require(e['settings']==SETTINGS and e['ablation_protocol_sha256']==ABLATION_SHA and e['cases']==cases
        and e['train']==parent['train'] and all(e.get(k)==v for k,v in FLAGS.items())
        and all(e[k]==p[k] for k in ('config','features','runtime','numeric_config','generation','decoding'))
        and e['asr_manifest']==parent['asr_manifest'] and e['asr_manifest_sha256']==parent['asr_manifest_sha256']
        and e['main_layers']==read(Path(p['config']['base'])/'config.json')['talker_config']['num_hidden_layers']
        and e['checkpoint_sha256']=={str(s):digest(Path(e['train'])/f'checkpoint-{s:06d}.pt') for s in (0,32)}
        and all(e['evidence'].get(k)==v for k,v in hashes.items()),'Diagnostic scope/parent changed')
    for path,sha in e['evidence'].items():require(digest(path)==sha,'Bound diagnostic evidence changed')
    return p,e


def tensor_sha(value):
    from .reference_masked_supervision_features import tensor_sha256
    return tensor_sha256(value)


def equal(a,b):
    import torch
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and a.dtype==b.dtype and torch.equal(a.cpu(),b.cpu())
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(equal(v,b[k]) for k,v in a.items())
    if isinstance(a,(list,tuple)):return type(a) is type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


class ProjectionBank:
    """Temporary hooks on OUTER routed projections only; no norm/RoPE patches."""
    def __init__(self,core,layout,binding,donor=None):
        self.core,self.layout,self.binding,self.donor=core,layout,binding,donor
        self.modules={f'{i}.{kind}':getattr(layer.self_attn,kind+'_proj')
            for i,layer in enumerate(core.model.talker.model.layers) for kind in ('k','v')}
        self.values={};self.checks={};self.calls={k:0 for k in self.modules};self.active=False;self.handles=[]
        require(self.modules and len({id(m) for m in self.modules.values()})==len(self.modules),'Unique main K/V projections required')
        require(binding['layout']==asdict(layout) and binding['positions']==list(range(layout.prefix_length)),
            'Explicit same-layout donor binding required')
        if donor is not None:
            require(donor['binding']==binding and set(donor['values'])==set(self.modules)
                and donor['prefix_positions']==binding['positions'] and donor['generated_rows_added_to_donor']==0
                and donor['protected_positions']==[i for i in binding['positions']
                    if not layout.target_start<=i<layout.target_stop], 'Donor context/layer/position coverage differs')

    def hook(self,key):
        def apply(module,args,output):
            import torch
            require(self.active and not torch.is_grad_enabled() and self.core.scope is not None
                and self.core.scope['kind']=='native' and self.core.scope['layout']==self.layout
                and self.core.scope['half_reference']==self.binding['half_reference'],
                'Projected KV outside its immutable inference context')
            positions=self.core.positions
            require(positions is not None and output.dtype==torch.float32 and output.device.type=='cpu'
                and output.ndim==3 and output.shape[:2]==(1,len(positions)) and torch.isfinite(output).all(),
                'Finite single-example CPU FP32 projected KV required')
            self.calls[key]+=1
            if int(positions[0])>=self.layout.prefix_length:
                require(key in self.values and len(positions)==1 and int(positions[0])==self.layout.prefix_length+self.calls[key]-2,
                    'Unexpected generated KV position')
                return output  # exact original object; never stored as donor
            require(key not in self.values and torch.equal(positions.cpu(),torch.arange(self.layout.prefix_length)),
                'Exactly one full native prefill capture per main projection required')
            allowed,protected=self.layout.masks(positions)
            require(torch.equal(allowed,~protected),'Prefix row partition differs')
            self.values[key]=output[:,protected,:].detach().cpu().clone()
            self.checks[key]=dict(allowed_before_sha256=tensor_sha(output[:,allowed,:]),
                allowed_after_sha256=tensor_sha(output[:,allowed,:]),protected_after_sha256=tensor_sha(self.values[key]))
            if self.donor is None:return output
            value=self.donor['values'][key]
            require(value.dtype==torch.float32 and value.shape==self.values[key].shape and torch.isfinite(value).all(),
                'Invalid protected donor tensor')
            replacement=output.clone();replacement[:,protected,:]=value
            require(torch.equal(replacement[:,allowed,:],output[:,allowed,:])
                and torch.equal(replacement[:,protected,:],value),'Immediate target KV changed or donor not exact')
            self.checks[key].update(allowed_after_sha256=tensor_sha(replacement[:,allowed,:]),
                protected_after_sha256=tensor_sha(replacement[:,protected,:]))
            return replacement
        return apply

    def __enter__(self):
        require(not self.active and not self.handles,'No nested hook bank')
        require(all(not m._forward_hooks for m in self.modules.values()),'Projection already has output hooks')
        self.active=True
        try:
            for key,module in self.modules.items():self.handles.append(module.register_forward_hook(self.hook(key)))
        except BaseException:self.__exit__(None,None,None);raise
        return self

    def __exit__(self,*args):
        for handle in self.handles:handle.remove()
        self.handles=[];self.active=False

    def bank(self,frames):
        require(set(self.values)==set(self.modules) and all(n==frames+1 for n in self.calls.values()),
            'Incomplete main projection/prefill/decode coverage')
        return dict(binding=self.binding,values=self.values,prefix_positions=list(range(self.layout.prefix_length)),
            protected_positions=[i for i in range(self.layout.prefix_length)
                if not self.layout.target_start<=i<self.layout.target_stop],calls=self.calls,checks=self.checks,
            grafted=self.donor is not None,generated_rows_added_to_donor=0)


def drift(teacher,student):
    import torch
    require(teacher['binding']==student['binding'] and teacher['values'].keys()==student['values'].keys(),
        'Drift must compare identical teacher/reference contexts')
    rows={}
    for name,a in teacher['values'].items():
        b=student['values'][name];require(a.shape==b.shape and torch.isfinite(a).all() and torch.isfinite(b).all(),'Invalid KV comparison')
        a,b=a.double(),b.double();delta=b-a;an=float(a.norm());bn=float(b.norm())
        rows[name]=dict(max_abs=float(delta.abs().max()),mean_abs=float(delta.abs().mean()),l2=float(delta.norm()),
            teacher_l2=an,student_l2=bn,relative_l2=float(delta.norm())/an if an else None,
            cosine=float((a*b).sum())/(an*bn) if an and bn else None)
    return rows


@contextmanager
def preserve_state(core,policy,trainer):
    import copy
    state=policy.snapshot(core);rng=trainer.rng_state();parameters=core.parameters()
    grads=[None if p.grad is None else p.grad.detach().clone() for p in parameters]
    frozen=policy.frozen_identity(core.model,core)
    talker=core.model.talker;had_rope=hasattr(talker,'rope_deltas')
    rope=copy.deepcopy(getattr(talker,'rope_deltas',None))
    try:yield
    finally:
        policy.restore(core,state)
        for p,g in zip(parameters,grads):p.grad=g
        trainer.restore_rng(rng)
        if had_rope:talker.rope_deltas=rope
        elif hasattr(talker,'rope_deltas'):del talker.rope_deltas
        require(equal(policy.snapshot(core),state) and equal(trainer.rng_state(),rng)
            and all(equal(p.grad,g) for p,g in zip(parameters,grads))
            and equal(getattr(talker,'rope_deltas',None),rope)
            and policy.frozen_identity(core.model,core)==frozen and core.scope is None,
            'Diagnostic state/RNG/frozen identity not restored')


def dummy_prefix(core,input_ids,reference,half,layout):
    import torch
    reference.validate(core.model.config.talker_config.hidden_size,16)
    prefixes=[]
    for constant in (0,1):
        with core._context('donor-prefix',half,layout),torch.inference_mode():
            fake=torch.full((1,16),constant,dtype=torch.long)
            sequence=core._build(input_ids,fake,reference)
            prefixes.append(sequence.embeddings[:layout.prefix_length].detach().cpu().clone())
            del sequence,fake
    require(equal(prefixes[0],prefixes[1]),'Prefix illegally depends on dummy target codec')
    return prefixes[0]


@contextmanager
def native_observation(model):
    """Observe main inputs and all main/residual head logits without alteration."""
    import torch
    data=dict(inputs=[],contexts=[],logits={})
    handles=[]
    def capture(module,args,kwargs):
        require(not args and kwargs.get('inputs_embeds') is not None,'Unexpected decoder call')
        data['inputs'].append(kwargs['inputs_embeds'].detach().cpu().clone())
        context={k:None if kwargs.get(k) is None else kwargs[k].detach().cpu().clone()
                 for k in ('position_ids','cache_position','attention_mask')}
        cache=kwargs.get('past_key_values');context['past_length']=0 if cache is None else cache.get_seq_length()
        data['contexts'].append(context)
    def logits(name):
        def capture(module,args,value):
            require(value.dtype==torch.float32 and torch.isfinite(value).all(),'Nonfinite native logits')
            data['logits'][name].append(value.detach().cpu().clone())
        return capture
    try:
        handles.append(model.talker.model.register_forward_pre_hook(capture,with_kwargs=True))
        heads={'main':model.talker.codec_head,**{f'residual{i}':m for i,m in enumerate(model.talker.code_predictor.lm_head)}}
        require(len(heads)==16,'Native sixteen codec heads required')
        for name,module in heads.items():
            data['logits'][name]=[];handles.append(module.register_forward_hook(logits(name)))
        yield data
    finally:
        for handle in handles:handle.remove()


def identity_control(teacher,repeat):
    require(teacher['audio_sha256']==repeat['audio_sha256'],'Self-graft WAV identity failed')
    for key in ('codes','first_sequence','inputs','contexts','logits'):
        require(equal(teacher[key],repeat[key]),'Self-graft native identity failed: '+key)
    require(equal(teacher['bank']['values'],repeat['bank']['values']),'Self-graft incoming protected KV differs')
    return dict(wav_exact=True,codes_exact=True,all_main_residual_logits_exact=True,
        native_inputs_contexts_exact=True,all_protected_incoming_kv_exact=True)


def one_clip(out,label,wrapper,core,state,case,prompt,reference,half,binding,donor,expected_prefix):
    import random,time
    import numpy as np
    import soundfile as sf
    import torch
    from .engine import ids,save
    from .reference_masked_supervision import Layout
    from .reference_masked_supervision_speech_v2 import unchanged
    from .preference_rollouts import capture_generation
    layout=Layout(**case['layout']);prefix=dummy_prefix(core,ids(wrapper,case['text']),reference,half,layout)
    require(expected_prefix is None or equal(prefix,expected_prefix),'Frozen native input prefix changed between checkpoints')
    random.seed(2026);np.random.seed(2026);torch.manual_seed(2026)
    started=time.monotonic()
    with ProjectionBank(core,layout,binding,donor) as hook,core.native(half),torch.inference_mode(), \
            native_observation(wrapper.model) as observed,capture_generation(wrapper.model) as captured:
        waves,sr=wrapper.generate_voice_clone(text=case['text'],language='Auto',voice_clone_prompt=prompt,
            non_streaming_mode=True,**binding['generation'])
    elapsed=time.monotonic()-started
    codes=captured['codes'];seq=captured['first_sequence'];frames=len(codes)
    require(codes.dtype==seq.dtype==torch.long and codes.shape==(frames,16) and frames>0
        and bool(((codes>=0)&(codes<2048)).all()) and captured['terminated'] is True
        and seq.shape==(1,frames+1) and torch.equal(seq[0,:-1],codes[:,0])
        and int(seq[0,-1])==wrapper.model.config.talker_config.codec_eos_token_id,'Complete native all16/EOS trajectory required')
    require(sr==24000 and len(waves)==1 and len(waves[0])>0 and np.isfinite(waves[0]).all()
        and len(waves[0])/sr<=30,'Invalid diagnostic audio')
    require(len(observed['inputs'])==frames+1 and len(observed['logits']['main'])==frames+1
        and all(len(v)==frames for k,v in observed['logits'].items() if k!='main')
        and equal(observed['inputs'][0][0],prefix),'Native input/logit coverage or dummy prefix differs')
    require(observed['contexts'][0]['past_length']==0
        and all(c['past_length']==layout.prefix_length+i-1 for i,c in enumerate(observed['contexts']) if i),
        'Native cache positions differ')
    bank=hook.bank(frames)
    if donor is not None:
        require(bank['prefix_positions']==donor['prefix_positions'] and bank['protected_positions']==donor['protected_positions']
            and all(bank['checks'][k]['protected_after_sha256']==tensor_sha(donor['values'][k])
                    for k in bank['values']),'Protected donor insertion differs')
    path=Path(out)/(label+'.wav');require(not path.exists(),'No audio overwrite');sf.write(path,waves[0],sr,subtype='PCM_16')
    wave,sr=sf.read(path,dtype='float32')
    with torch.inference_mode():
        vector=wrapper.model.extract_speaker_embedding(audio=wave,sr=sr).flatten()
        cosine=float(torch.nn.functional.cosine_similarity(vector[None],reference.speaker[None]))
    require(math.isfinite(cosine),'Nonfinite original-reference diagnostic cosine')
    artifact=dict(codes=codes,first_sequence=seq,inputs=observed['inputs'],contexts=observed['contexts'],
        logits=observed['logits'],bank=bank,prefix=prefix,ref_ids=reference.ref_ids,ref_codes=reference.ref_codes,
        original_ref_xvec=reference.speaker,input_ids=ids(wrapper,case['text']).cpu(),audio_sha256=digest(path),
        eos_token_id=wrapper.model.config.talker_config.codec_eos_token_id)
    trajectory=path.with_suffix('.pt');require(not trajectory.exists(),'No tensor artifact overwrite');save(trajectory,artifact)
    unchanged(wrapper.model,core,state)
    return artifact,dict(label=label,pair_id=case['pair_id'],case=case['pair_id'],group='fitting',
        text=case['text'],reference=case['reference'],reference_sha256=case['reference_sha256'],
        reference_text=case['reference_text'],speaker_id=case['speaker_id'],cosine=cosine,
        half_reference=half,seed=2026,frames=frames,terminated=True,
        seconds=len(waves[0])/sr,elapsed_seconds=elapsed,wav=path.name,audio_sha256=digest(path),
        trajectory=trajectory.name,trajectory_sha256=digest(trajectory),projected_kv_hooks=len(bank['values']),
        native_inputs_exact=True,dummy_prefix_invariant=True,immediate_allowed_kv_unchanged=True,
        generated_rows_added_to_donor=0,grafted=donor is not None)


def run(out,protocol_sha):
    import torch
    from . import reference_masked_supervision_training_v2 as trainer
    from . import reference_masked_supervision_policy_v2 as policy
    from . import reference_masked_supervision_features as features
    from .reference_masked_supervision import Layout,NativeReference
    from .engine import ids,save
    out=Path(out);p,e=verify(out,protocol_sha)
    require(not (out/'started.json').exists(),'No native diagnostic restart')
    write(out/'started.json',dict(protocol_sha256=protocol_sha))
    ablation.native_verify(Path(e['ablation']),ABLATION_SHA,0)
    _,wrapper,core,zero=trainer.load(e['train'],0,device='cpu',training=False)
    student=trainer.read_checkpoint(e['train'],32)
    features.assert_native_components(wrapper,p['features'])
    hook_snapshot={name:(tuple(m._forward_hooks),tuple(m._forward_pre_hooks)) for name,m in wrapper.model.named_modules()}
    records=[];controls=[];drifts=[];proof_paths=[]
    with preserve_state(core,policy,trainer):
        for index,case in enumerate(e['cases']):
            # Load only legitimate reference audio features, never target codec/audio tensors.
            asset=features.load_feature(p['features'],case['reference_id'])
            with torch.inference_mode():
                prompt=wrapper.create_voice_clone_prompt(ref_audio=case['reference'],ref_text=case['reference_text'],x_vector_only_mode=False)
            require(digest(case['reference'])==case['reference_sha256'] and asset['audio_sha256']==case['reference_sha256']
                and equal(prompt[0].ref_code.cpu(),asset['codes'])
                and equal(prompt[0].ref_spk_embedding.flatten().cpu(),asset['speaker'])
                and equal(ids(wrapper,case['reference_text'],True).cpu(),asset['ref_ids'])
                and tensor_sha(ids(wrapper,case['text']).cpu())==case['target_input_ids']['sha256'],
                'Fresh legitimate reference/text provenance differs')
            reference=NativeReference(asset['ref_ids'],asset['codes'],asset['speaker'],case['reference_sha256'],asset['protocol_sha256'])
            for mode in SETTINGS['modes']:
                half=mode=='half';tag=f'pair{index}-{mode}'
                binding=dict(pair_id=case['pair_id'],text=case['text'],reference_sha256=case['reference_sha256'],
                    reference_text=case['reference_text'],ref_ids_sha256=tensor_sha(reference.ref_ids),
                    ref_codes_sha256=tensor_sha(reference.ref_codes),original_xvec_sha256=tensor_sha(reference.speaker),
                    half_reference=half,teacher_checkpoint_sha256=e['checkpoint_sha256']['0'],
                    student_checkpoint_sha256=e['checkpoint_sha256']['32'],layout=case['layout'],
                    positions=list(range(case['layout']['prefix_length'])),numeric_config=e['numeric_config'],
                    backend=SETTINGS['backend'],generation=e['generation'])
                policy.restore(core,zero['adapter']);print(tag+' teacher0',flush=True)
                teacher,row=one_clip(out,tag+'-teacher0',wrapper,core,zero,case,prompt,reference,half,binding,None,None)
                records.append(dict(row,arm='teacher0',checkpoint_sha256=e['checkpoint_sha256']['0']))
                donor=teacher['bank'];print(tag+' self_graft0 control',flush=True)
                repeat,row=one_clip(out,tag+'-self_graft0',wrapper,core,zero,case,prompt,reference,half,binding,donor,teacher['prefix'])
                control=dict(pair_id=case['pair_id'],mode=mode,**identity_control(teacher,repeat));controls.append(control)
                records.append(dict(row,arm='self_graft0',checkpoint_sha256=e['checkpoint_sha256']['0']));del repeat
                policy.restore(core,student['adapter']);print(tag+' student32 baseline',flush=True)
                plain,row=one_clip(out,tag+'-student32',wrapper,core,student,case,prompt,reference,half,binding,None,teacher['prefix'])
                require(equal(teacher['contexts'][0],plain['contexts'][0]),'Teacher/student native prefill positions/mask differ')
                records.append(dict(row,arm='student32',checkpoint_sha256=e['checkpoint_sha256']['32']))
                report=dict(pair_id=case['pair_id'],mode=mode,layers=drift(donor,plain['bank']))
                drift_path=out/(tag+'-drift-before-graft.json');write(drift_path,report);proof_paths.append(drift_path);drifts.append(report)
                del plain
                print(tag+' graft32 intervention (identity controls passed; drift saved)',flush=True)
                graft,row=one_clip(out,tag+'-graft32',wrapper,core,student,case,prompt,reference,half,binding,donor,teacher['prefix'])
                require(equal(teacher['contexts'][0],graft['contexts'][0]),'Graft native prefill positions/mask differ')
                records.append(dict(row,arm='graft32',checkpoint_sha256=e['checkpoint_sha256']['32']))
                del graft,teacher,donor
    verify(out,protocol_sha)
    require(len(records)==16 and len(controls)==len(drifts)==4,'Incomplete fixed diagnostic')
    hooks_restored=hook_snapshot=={name:(tuple(m._forward_hooks),tuple(m._forward_pre_hooks)) for name,m in wrapper.model.named_modules()}
    require(hooks_restored,'Native observation hooks leaked')
    write(out/'generated.json',dict(protocol_sha256=protocol_sha,records=records,controls=controls,drifts=drifts,
        frozen_identity=policy.frozen_identity(wrapper.model,core),all_adapter_rng_grad_state_restored=True,
        all_hooks_removed=hooks_restored,
        evidence={str(path.resolve()):digest(path) for path in proof_paths},**FLAGS))


def audit(out,protocol_sha):
    """Independent saved-tensor/drift and freshly extracted original-voice audit."""
    import torch
    import soundfile as sf
    import numpy as np
    from . import reference_masked_supervision_training_v2 as trainer
    from . import reference_masked_supervision_features as features
    from .reference_masked_supervision_speech_v2 import unchanged
    from .reference_masked_supervision import Layout,NativeReference
    from .engine import ids
    out=Path(out);p,e=verify(out,protocol_sha);generated=read(out/'generated.json')
    ablation.native_verify(Path(e['ablation']),ABLATION_SHA,0)
    _,wrapper,core,state=trainer.load(e['train'],0,device='cpu',training=False)
    features.assert_native_components(wrapper,p['features']);maximum_cosine_error=0.
    config=wrapper.model.config.talker_config
    rows=generated['records'];expected={(c['pair_id'],mode,arm) for c in e['cases']
        for mode in SETTINGS['modes'] for arm in SETTINGS['arms']}
    require(len(rows)==16 and {(r['pair_id'],'half' if r['half_reference'] else 'full',r['arm']) for r in rows}==expected
        and generated['protocol_sha256']==protocol_sha and generated['all_adapter_rng_grad_state_restored'] is True
        and generated['all_hooks_removed'] is True and generated['frozen_identity']==p['frozen_identity']
        and all(generated.get(k)==v for k,v in FLAGS.items()),'Incomplete diagnostic or restoration proof')
    for path,sha in generated['evidence'].items():require(digest(path)==sha,'Pre-intervention drift evidence changed')
    indexed={(r['pair_id'],'half' if r['half_reference'] else 'full',r['arm']):r for r in rows}
    controls=[];drifts=[];artifacts={}
    for case in e['cases']:
        asset=features.load_feature(p['features'],case['reference_id'])
        with torch.inference_mode():
            prompt=wrapper.create_voice_clone_prompt(ref_audio=case['reference'],ref_text=case['reference_text'],x_vector_only_mode=False)
        require(equal(prompt[0].ref_code.cpu(),asset['codes']) and equal(prompt[0].ref_spk_embedding.flatten().cpu(),asset['speaker']),
            'Independent fresh reference extraction differs')
        for mode in SETTINGS['modes']:
            layout=Layout(**case['layout'])
            reference=NativeReference(asset['ref_ids'],asset['codes'],asset['speaker'],case['reference_sha256'],asset['protocol_sha256'])
            rebuilt_prefix=dummy_prefix(core,ids(wrapper,case['text']),reference,mode=='half',layout)
            values={}
            for arm in SETTINGS['arms']:
                r=indexed[case['pair_id'],mode,arm];wav=out/r['wav'];path=out/r['trajectory']
                require(digest(wav)==r['audio_sha256'] and digest(path)==r['trajectory_sha256']
                    and r['seed']==2026 and r['terminated'] is True and r['generated_rows_added_to_donor']==0
                    and r['grafted']==(arm in ('self_graft0','graft32'))
                    and r['checkpoint_sha256']==e['checkpoint_sha256']['0' if arm in ('teacher0','self_graft0') else '32'],
                    'Diagnostic recording identity differs')
                t=torch.load(path,map_location='cpu',weights_only=True);bank=t['bank'];frames=r['frames']
                require(t['audio_sha256']==r['audio_sha256'] and bank['binding']['pair_id']==case['pair_id']
                    and bank['binding']['text']==case['text'] and bank['binding']['reference_sha256']==case['reference_sha256']
                    and bank['binding']['half_reference']==(mode=='half') and bank['binding']['layout']==case['layout']
                    and bank['binding']['teacher_checkpoint_sha256']==e['checkpoint_sha256']['0']
                    and bank['binding']['student_checkpoint_sha256']==e['checkpoint_sha256']['32']
                    and bank['binding']['numeric_config']==e['numeric_config'] and bank['binding']['generation']==e['generation'],
                    'Saved donor context changed')
                require(t['codes'].dtype==torch.long and t['codes'].shape==(frames,16)
                    and t['first_sequence'].shape==(1,frames+1) and torch.equal(t['first_sequence'][0,:-1],t['codes'][:,0])
                    and int(t['first_sequence'][0,-1])==t['eos_token_id']==config.codec_eos_token_id
                    and bool(((t['codes']>=0)&(t['codes']<2048)).all())
                    and equal(t['ref_codes'],asset['codes']) and equal(t['ref_ids'],asset['ref_ids'])
                    and equal(t['original_ref_xvec'],asset['speaker'])
                    and equal(t['input_ids'],ids(wrapper,case['text']).cpu()) and equal(t['prefix'],rebuilt_prefix)
                    and len(t['inputs'])==len(t['contexts'])==frames+1 and equal(t['inputs'][0][0],t['prefix'])
                    and all(v==frames+1 for v in bank['calls'].values())
                    and len(bank['values'])==r['projected_kv_hooks']==2*e['main_layers'] and bank['generated_rows_added_to_donor']==0
                    and set(bank['values'])==set(bank['checks'])==set(bank['calls'])==
                        {f'{i}.{k}' for i in range(e['main_layers']) for k in ('k','v')}
                    and bank['prefix_positions']==list(range(layout.prefix_length))
                    and bank['protected_positions']==[i for i in range(layout.prefix_length) if not layout.target_start<=i<layout.target_stop]
                    and all(v['allowed_before_sha256']==v['allowed_after_sha256'] for v in bank['checks'].values()),
                    'Saved native/projection coverage differs')
                require(all(torch.isfinite(v).all() for v in bank['values'].values())
                    and all(torch.isfinite(v).all() for group in t['logits'].values() for v in group), 'Nonfinite saved diagnostic tensors')
                wave,sr=sf.read(wav,dtype='float32')
                require(sr==24000 and wave.ndim==1 and len(wave)>0 and np.isfinite(wave).all()
                    and r['seconds']==len(wave)/sr,'Saved WAV differs')
                with torch.inference_mode():
                    vector=wrapper.model.extract_speaker_embedding(audio=wave,sr=sr).flatten()
                    cosine=float(torch.nn.functional.cosine_similarity(vector[None],asset['speaker'][None]))
                require(math.isfinite(cosine) and math.isfinite(r['cosine']),'Nonfinite audited cosine')
                maximum_cosine_error=max(maximum_cosine_error,abs(cosine-r['cosine']))
                values[arm]=t;artifacts[str(wav.resolve())]=digest(wav);artifacts[str(path.resolve())]=digest(path)
            teacher=values['teacher0'];controls.append(dict(pair_id=case['pair_id'],mode=mode,
                **identity_control(teacher,values['self_graft0'])))
            for arm in ('self_graft0','graft32'):
                bank=values[arm]['bank']
                require(bank['binding']==teacher['bank']['binding'] and bank['grafted'] is True
                    and all(bank['checks'][key]['protected_after_sha256']==tensor_sha(value)
                        for key,value in teacher['bank']['values'].items()), 'Saved exact protected injection differs')
            drifts.append(dict(pair_id=case['pair_id'],mode=mode,layers=drift(teacher['bank'],values['student32']['bank'])))
    require(controls==generated['controls'] and drifts==generated['drifts'],'Independent control/drift recomputation differs')
    require(maximum_cosine_error<=2e-6,'Independent original-reference cosine differs')
    unchanged(wrapper.model,core,state)
    verify(out,protocol_sha)
    write(out/'structural-verification.json',dict(status='verified',recordings=16,comparison_recordings=12,self_graft_controls=4,
        protocol_sha256=protocol_sha,generated_sha256=digest(out/'generated.json'),controls=controls,drifts=drifts,
        artifacts=artifacts,all_adapter_rng_grad_state_restored=True,all_hooks_removed=True,
        original_reference_cosines_recomputed=16,max_cosine_recomputation_error=maximum_cosine_error,
        interpretation='Projected protected K/V intervention only; queries/hidden state/first frame and target/generated states are not frozen; fitting-only, no quality inference',**FLAGS))


def structural_evidence(out,protocol_sha,e):
    out=Path(out);receipt=read(out/'structural-verification.json');generated=read(out/'generated.json')
    require(receipt['status']=='verified' and receipt['recordings']==16 and receipt['self_graft_controls']==4
        and receipt['comparison_recordings']==12 and receipt['protocol_sha256']==protocol_sha
        and receipt['generated_sha256']==digest(out/'generated.json')
        and receipt['all_adapter_rng_grad_state_restored'] is True and receipt['all_hooks_removed'] is True
        and receipt['original_reference_cosines_recomputed']==16
        and math.isfinite(receipt['max_cosine_recomputation_error']) and 0<=receipt['max_cosine_recomputation_error']<=2e-6
        and all(receipt.get(k)==v for k,v in FLAGS.items()),'Complete structural and original-voice audit required')
    for path,sha in receipt['artifacts'].items():require(digest(path)==sha,'Audited diagnostic artifact changed')
    rows=generated['records'];cases={c['pair_id']:c for c in e['cases']}
    expected={(i,mode,arm) for i in cases for mode in SETTINGS['modes'] for arm in SETTINGS['arms']}
    require(len(rows)==16 and {(r['pair_id'],'half' if r['half_reference'] else 'full',r['arm']) for r in rows}==expected,
        'Exact sixteen diagnostic recordings required')
    for r in rows:
        c=cases[r['pair_id']]
        require(all(r[k]==c[k] for k in ('text','reference','reference_sha256','reference_text','speaker_id'))
            and r['seed']==2026 and r['group']=='fitting' and r['terminated'] is True
            and math.isfinite(r['cosine']) and -1<=r['cosine']<=1
            and digest(out/r['wav'])==r['audio_sha256'] and digest(out/r['trajectory'])==r['trajectory_sha256'],
            'Diagnostic metadata or waveform changed')
    return receipt,rows


def score(out,protocol_sha):
    import soundfile as sf
    import torch,torchaudio
    from transformers import WhisperForConditionalGeneration,WhisperProcessor
    from .evaluate import ASR_REV
    from scripts.score_latest_audio import score as score_text
    out=Path(out);p,e=verify(out,protocol_sha);_,rows=structural_evidence(out,protocol_sha,e)
    require(not (out/'scored.json').exists(),'No ASR score overwrite')
    manifest=read(e['asr_manifest']);asr=Path(e['config']['asr'])
    require(digest(e['asr_manifest'])==e['asr_manifest_sha256'] and manifest['repo']=='BuzzASR/mongolian'
        and manifest['revision']==ASR_REV,'Original pinned recognizer required')
    for name,item in manifest['files'].items():require(digest(asr/name)==item['sha256'],'ASR asset changed')
    processor=WhisperProcessor.from_pretrained(asr,local_files_only=True)
    model=WhisperForConditionalGeneration.from_pretrained(asr,torch_dtype=torch.float32,local_files_only=True).to('cuda').eval()
    for r in rows:
        wave,sr=sf.read(out/r['wav'],dtype='float32');wave=torchaudio.functional.resample(torch.from_numpy(wave),sr,16000).numpy()
        batch=processor(wave,sampling_rate=16000,return_tensors='pt',return_attention_mask=True).to('cuda')
        with torch.inference_mode():tokens=model.generate(batch.input_features,attention_mask=batch.attention_mask,**e['decoding'])
        hypothesis=processor.batch_decode(tokens,skip_special_tokens=True)[0].strip()
        r.update(hypothesis=hypothesis,score=score_text(r['text'],hypothesis,True),asr_revision=ASR_REV,decoding=e['decoding'])
    write(out/'scored.json',dict(records=rows,protocol_sha256=protocol_sha,
        structural_verification_sha256=digest(out/'structural-verification.json')))


def scalar_summary(original,scored,decoding):
    from .evaluate import ASR_REV
    from .targeted_voice import aggregate
    from scripts.score_latest_audio import score as score_text
    require(len(original)==len(scored)==16 and len({r['label'] for r in original})==16,'Complete diagnostic score scope required')
    indexed={r['label']:r for r in scored};require(len(indexed)==16 and set(indexed)=={r['label'] for r in original},'Scored coverage differs')
    for r in original:
        s=indexed[r['label']]
        require(set(s)==set(r)|{'hypothesis','score','asr_revision','decoding'} and all(s[k]==v for k,v in r.items())
            and s['asr_revision']==ASR_REV and s['decoding']==decoding
            and s['score']==score_text(r['text'],s['hypothesis'],True),'Independent edit score or native metadata differs')
    summary={}
    for arm in SETTINGS['arms']:
        summary[arm]={}
        for mode in SETTINGS['modes']:
            rows=[r for r in scored if r['arm']==arm and r['half_reference']==(mode=='half')]
            require(len(rows)==2 and len({r['pair_id'] for r in rows})==2,'Each arm/route requires two fitting probes')
            summary[arm][mode]=aggregate(rows)['fitting']
    return summary


def scalar_audit(out,protocol_sha):
    out=Path(out);p,e=verify(out,protocol_sha);structural,original=structural_evidence(out,protocol_sha,e)
    scored=read(out/'scored.json')
    require(scored['protocol_sha256']==protocol_sha and scored['structural_verification_sha256']==digest(out/'structural-verification.json'),
        'Scoring provenance differs')
    summary=scalar_summary(original,scored['records'],e['decoding'])
    write(out/'results.json',dict(status='verified',recordings=16,comparison_recordings=12,self_graft_controls=4,
        protocol_sha256=protocol_sha,structural_verification_sha256=digest(out/'structural-verification.json'),
        scored_sha256=digest(out/'scored.json'),generated_sha256=digest(out/'generated.json'),
        summary=summary,controls=structural['controls'],drifts=structural['drifts'],
        interpretation='Two fitting probes, one seed, two routes; descriptive ASR/voice and projected-KV metrics, no generalization or causal-quality claim',**FLAGS))


def main():
    import argparse,os
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('prepare','run','audit','score','scalar-audit'));parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--ablation',type=Path);parser.add_argument('--driver',type=Path)
    parser.add_argument('--test-log',type=Path);parser.add_argument('--note',type=Path);parser.add_argument('--protocol-sha')
    args=parser.parse_args()
    if args.phase=='prepare':
        require(all((args.ablation,args.driver,args.test_log,args.note)),'Registration inputs required')
        prepare(args.ablation,args.out,args.driver,args.test_log,args.note);return
    require(args.protocol_sha is not None,'Explicit registered protocol hash required')
    p,e=verify(args.out,args.protocol_sha)
    os.environ['TEMP']=os.environ['TMP']=p['config']['tmp'];os.environ['NUMBA_CACHE_DIR']=p['config']['numba_cache']
    os.environ['HF_HUB_OFFLINE']=os.environ['TRANSFORMERS_OFFLINE']='1'
    if args.phase=='scalar-audit':scalar_audit(args.out,args.protocol_sha);return
    import torch
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    {'run':run,'audit':audit,'score':score}[args.phase](args.out,args.protocol_sha)


if __name__=='__main__':main()
