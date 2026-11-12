"""Conditional original-base protected-K/V calibration screen; never trains.

The donor's native embeddings intentionally differ from the student's. Only
legitimate text/reference IDs, codec features, positions and compatible native
normalization/rotary interfaces are shared. Passing80 requires NEW114 evidence.
"""
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path

from . import reference_masked_supervision_kv_probe as old

read,write,digest,require,equal,tensor_sha = old.read,old.write,old.digest,old.require,old.equal,old.tensor_sha
PARENT_SHA='8e4878d8f82b9c0b3feee353609e617fb84c50fb719b8fb15178cc2ce155a58a'
FLAGS=dict(optimizer_updates=0,goal_achieved=False,global_model_selection_changed=False,
    policy_selected=False,target_audio_used_by_donor=False,target_codes_used_by_donor=False)
SETTINGS=dict(recordings=80,development=64,custom=16,contexts=40,seeds=[2026,2027],
    checkpoint=0,donor='original native base full-reference',student='centered16 checkpoint0; full development, half custom',
    intervention='all main protected-prefix k_proj/v_proj outputs before native norm/RoPE/cache',
    device='cpu',dtype='float32',cpu_threads=4,control_recordings=8,
    prefix_only='abort after actual first native main decoder forward; never sample a target frame',
    calibration_gate=dict(cer_strictly_below=.03,original_reference_speaker_macro_cosine_strictly_above=.984,
        truncated=0),no_automatic_training=True,requires_new_original114=True)


def no_effect(rows):
    require(len(rows)==16,'Complete sixteen parent clips required')
    indexed={(r['pair_id'],r['half_reference'],r['arm']):r for r in rows}
    contexts={(r['pair_id'],r['half_reference']) for r in rows}
    require(len(indexed)==16 and len(contexts)==4,'Parent scope differs')
    for pair,half in contexts:
        for a,b in [('teacher0','self_graft0'),('student32','graft32')]:
            x,y=(indexed[pair,half,arm] for arm in (a,b))
            require(x['audio_sha256']==y['audio_sha256'] and x['cosine']==y['cosine']
                and x['frames']==y['frames'] and x['terminated'] is y['terminated'] is True,
                'Teacher0 intervention has an output effect; root must reconsider priority')


def closure_ok(closure,bindings):
    require(closure.get('status')=='closed' and type(closure.get('exit_code')) is int
        and closure['exit_code']==0 and closure.get('original_session_id')==35473
        and all(closure.get(k)==v for k,v in bindings.items()),'Authoritative original35473 exit0 required')


def parent(run):
    run=Path(run);p,e=old.verify(run,PARENT_SHA)
    receipt,rows=old.structural_evidence(run,PARENT_SHA,e);no_effect(rows)
    scored=read(run/'scored.json');result=read(run/'results.json')
    summary=old.scalar_summary(rows,scored['records'],e['decoding'])
    require(result['status']=='verified' and result['summary']==summary
        and result['protocol_sha256']==PARENT_SHA
        and result['scored_sha256']==digest(run/'scored.json')
        and result['generated_sha256']==digest(run/'generated.json')
        and result['structural_verification_sha256']==digest(run/'structural-verification.json')
        and scored['structural_verification_sha256']==digest(run/'structural-verification.json')
        and scored['protocol_sha256']==PARENT_SHA and all(result.get(k)==v for k,v in old.FLAGS.items()),
        'Complete parent scalar audit required')
    status_path=run.parent/(run.name+'-status.json');closure_path=run.parent/(run.name+'-process-closed.json')
    driver=run.parent/'run-masked-supervision-kv-probe-v1.py';status=read(status_path)
    bindings=dict(protocol_sha256=PARENT_SHA,driver_sha256=digest(driver),
        terminal_status_sha256=digest(status_path),results_sha256=digest(run/'results.json'))
    closure_ok(read(closure_path),bindings)
    require(status['status']=='complete' and status['protocol_sha256']==PARENT_SHA
        and status['results_sha256']==bindings['results_sha256']
        and set(status['logs'])=={'run','audit','score','scalar-audit'},'Parent phases incomplete')
    hashes=dict(e['evidence']);hashes.update(receipt['artifacts'])
    for path in [run/n for n in ('protocol.json','generated.json','scored.json','results.json','structural-verification.json')]+[driver,status_path,closure_path]:
        hashes[str(path.resolve())]=digest(path)
    for item in status['logs'].values():
        require(item['exit_code']==0 and digest(item['path'])==item['sha256'],'Parent phase changed')
        hashes[item['path']]=item['sha256']
    # Independent tensor inspection is deferred to native controls, not prepare.
    return p,e,hashes


def contexts(cases):
    require(len(cases)==80 and len({(r['case'],r['seed']) for r in cases})==80,'Exact80 required')
    require({g:sum(r['group']==g for r in cases) for g in ('development','custom')}=={'development':64,'custom':16},'Group scope differs')
    result=[]
    for name in dict.fromkeys(r['case'] for r in cases):
        pair=[r for r in cases if r['case']==name]
        require(len(pair)==2 and {r['seed'] for r in pair}=={2026,2027},'Both original seeds required')
        a,b=({k:v for k,v in r.items() if k!='seed'} for r in pair)
        require(a==b and a['half_reference']==(a['group']=='custom'),'Student route or context differs')
        result.append(a)
    require(len(result)==40 and len({r['speaker_id'] for r in result if r['group']=='development'})==16
        and len({r['speaker_id'] for r in result if r['group']=='custom'})==1,'Speaker coverage differs')
    return result


def eligible(summary,truncated):
    require(set(summary)=={'development','custom'} and type(truncated) is int and truncated>=0,'Invalid quality scope')
    for group,count,speakers in [('development',64,16),('custom',16,1)]:
        row=summary[group]
        require(row['recordings']==count and row['speakers']==speakers
            and all(math.isfinite(row[k]) for k in ('pooled_cer','pooled_wer','speaker_macro_cosine'))
            and row['pooled_cer']>=0 and row['pooled_wer']>=0 and -1<=row['speaker_macro_cosine']<=1,'Invalid group metrics')
    return truncated==0 and all(r['pooled_cer']<.03 and r['speaker_macro_cosine']>.984 for r in summary.values())


def prepare(run,out,driver,test_log,note):
    run,out=Path(run).resolve(),Path(out).resolve();p,e,hashes=parent(run)
    require(not out.exists() and out.parent==run.parent and out.name=='masked-supervision-base-kv-v1','New external screen directory required')
    cases=read(Path(e['train'])/'speech-protocol.json')['cases'];unique=contexts(cases)
    log=Path(test_log).read_text(encoding='utf-8-sig')
    require('Ran 11 tests in ' in log and log.strip().endswith('OK') and 'skipped' not in log.lower() and 'FAILED' not in log,'Eleven passing tests without skips required')
    for path in (Path(__file__),Path(__file__).parents[1]/'tests/test_reference_masked_supervision_base_kv.py',Path(driver),Path(test_log),Path(note)):
        hashes[str(path.resolve())]=digest(path)
    control_cases=[next(c for c in unique if c['group']==g) for g in ('development','custom')]
    protocol=dict(parent=str(run),parent_protocol_sha256=PARENT_SHA,train=e['train'],settings=SETTINGS,
        cases=cases,contexts=unique,control_cases=control_cases,evidence=hashes,
        **{k:e[k] for k in ('config','features','runtime','numeric_config','generation','decoding','asr_manifest','asr_manifest_sha256')},
        checkpoint_sha256=e['checkpoint_sha256']['0'],
        base_model_files=read(Path(p['features'])/'protocol.json')['model_files'],**FLAGS)
    out.mkdir();write(out/'protocol.json',protocol);return protocol


def verify(out,sha):
    out=Path(out);require(digest(out/'protocol.json')==sha,'Registered screen protocol changed')
    e=read(out/'protocol.json');p,previous,hashes=parent(e['parent'])
    cases=read(Path(e['train'])/'speech-protocol.json')['cases'];unique=contexts(cases)
    require(e['settings']==SETTINGS and e['cases']==cases and e['contexts']==unique
        and e['control_cases']==[next(c for c in unique if c['group']==g) for g in ('development','custom')]
        and e['train']==previous['train'] and e['parent_protocol_sha256']==PARENT_SHA
        and e['checkpoint_sha256']==previous['checkpoint_sha256']['0']
        and e['base_model_files']==read(Path(p['features'])/'protocol.json')['model_files']
        and all(e[k]==previous[k] for k in ('config','features','runtime','numeric_config','generation','decoding','asr_manifest','asr_manifest_sha256'))
        and all(e.get(k)==v for k,v in FLAGS.items()) and all(e['evidence'].get(k)==v for k,v in hashes.items()),'Screen context/parent changed')
    for path,value in e['evidence'].items():require(digest(path)==value,'Bound screen evidence changed')
    return p,e


class PrefixComplete(Exception):pass


class Bank:
    """Native positions observed independently; no model/core monkeypatching."""
    def __init__(self,model,layout,binding,donor=None,prefix_only=False):
        self.model,self.layout,self.binding,self.donor,self.prefix_only=model,layout,binding,donor,prefix_only
        self.modules={f'{i}.{k}':getattr(layer.self_attn,k+'_proj') for i,layer in enumerate(model.talker.model.layers) for k in ('k','v')}
        self.values={};self.checks={};self.calls={k:0 for k in self.modules};self.handles=[];self.inputs=[];self.contexts=[];self.positions=None
        require(len(set(map(id,self.modules.values())))==len(self.modules)>0,'Distinct outer projections required')
        if donor is not None:require(donor['binding']==binding and set(donor['values'])==set(self.modules),'Donor context differs')

    def before(self,module,args,kwargs):
        import torch
        require(not args and not torch.is_grad_enabled() and not module.training,'Native inference only')
        x=kwargs['inputs_embeds'];pos=kwargs['cache_position'];P=self.layout.prefix_length
        require(x.dtype==torch.float32 and x.device.type=='cpu' and x.ndim==3 and x.shape[0]==1
            and torch.isfinite(x).all() and pos is not None,'Native CPUFP32 inputs required')
        expected=torch.arange(P) if not self.inputs else torch.tensor([P+len(self.inputs)-1])
        require(torch.equal(pos.cpu(),expected) and x.shape[1]==len(pos),'Unexpected prefill/decode positions')
        self.positions=pos;self.inputs.append(x.detach().cpu().clone())
        cache=kwargs.get('past_key_values')
        self.contexts.append(dict(position_ids=None if kwargs.get('position_ids') is None else kwargs['position_ids'].detach().cpu().clone(),
            cache_position=pos.detach().cpu().clone(),attention_mask=None if kwargs.get('attention_mask') is None else kwargs['attention_mask'].detach().cpu().clone(),
            past_length=0 if cache is None else cache.get_seq_length()))

    def hook(self,key):
        def apply(module,args,value):
            import torch
            require(self.positions is not None and not torch.is_grad_enabled() and value.dtype==torch.float32
                and value.device.type=='cpu' and value.ndim==3 and value.shape[:2]==(1,len(self.positions))
                and torch.isfinite(value).all(),'Invalid native projected K/V')
            self.calls[key]+=1
            if int(self.positions[0])>=self.layout.prefix_length:
                require(key in self.values,'Decode before native prefill');return value
            require(key not in self.values,'Duplicate native prefix')
            allowed,protected=self.layout.masks(self.positions)
            self.values[key]=value[:,protected].detach().cpu().clone()
            outgoing=value
            if self.donor is not None:
                source=self.donor['values'][key]
                require(source.dtype==value.dtype and source.shape==self.values[key].shape and torch.isfinite(source).all(),'Invalid donor tensor')
                outgoing=value.clone();outgoing[:,protected]=source
            require(torch.equal(outgoing[:,allowed],value[:,allowed]),'Immediate target K/V changed')
            self.checks[key]=dict(allowed_before_sha256=tensor_sha(value[:,allowed]),allowed_after_sha256=tensor_sha(outgoing[:,allowed]),
                protected_after_sha256=tensor_sha(outgoing[:,protected]))
            return outgoing
        return apply

    def after(self,module,args,kwargs,output):
        if self.prefix_only:raise PrefixComplete()

    def __enter__(self):
        require(not self.handles and all(not m._forward_hooks for m in self.modules.values()),'No nested/projected hooks allowed')
        try:
            self.handles.append(self.model.talker.model.register_forward_pre_hook(self.before,with_kwargs=True))
            for k,m in self.modules.items():self.handles.append(m.register_forward_hook(self.hook(k)))
            if self.prefix_only:self.handles.append(self.model.talker.model.register_forward_hook(self.after,with_kwargs=True))
        except BaseException:self.__exit__(None,None,None);raise
        return self

    def __exit__(self,*args):
        for h in self.handles:h.remove()
        self.handles=[];self.positions=None

    def receipt(self):
        require(set(self.values)==set(self.modules) and all(v==len(self.inputs) for v in self.calls.values()),'Incomplete all-layer native coverage')
        return dict(binding=self.binding,values=self.values,checks=self.checks,calls=self.calls,
            layout=asdict(self.layout),prefix_positions=list(range(self.layout.prefix_length)),
            protected_positions=[i for i in range(self.layout.prefix_length) if not self.layout.target_start<=i<self.layout.target_stop],
            generated_rows_added_to_donor=0)


def interface(wrapper):
    """Norm weights/config and actual rotary implementation; projections differ."""
    from .reference_masked_supervision_policy_v2 import tensor_digest
    model=wrapper.model;decoder=model.talker.model;cfg=model.config.talker_config
    keys=('hidden_size','num_hidden_layers','num_attention_heads','num_key_value_heads','head_dim',
          'rope_theta','rope_scaling','max_position_embeddings','rms_norm_eps','num_code_groups',
          'codec_eos_token_id','codec_bos_id','codec_pad_id','codec_nothink_id','codec_think_bos_id','codec_think_eos_id')
    norms=[]
    for i,layer in enumerate(decoder.layers):
        for name in ('q_norm','k_norm'):
            module=getattr(layer.self_attn,name)
            norms.extend((f'{i}.{name}.{k}',v) for k,v in module.state_dict().items())
    rotary=decoder.rotary_emb
    return dict(config={k:getattr(cfg,k,None) for k in keys},norm_sha256=tensor_digest(norms),
        rotary_class=type(rotary).__module__+'.'+type(rotary).__qualname__,
        rotary_buffers_sha256=tensor_digest(list(rotary.named_buffers())),
        attention_classes=sorted({type(x.self_attn).__module__+'.'+type(x.self_attn).__qualname__ for x in decoder.layers}),
        attention_implementation=cfg._attn_implementation)


def load_base(e):
    import torch
    from qwen_tts import Qwen3TTSModel
    from . import reference_masked_supervision_features as features
    wrapper=Qwen3TTSModel.from_pretrained(e['config']['base'],device_map='cpu',torch_dtype=torch.float32,
        attn_implementation='sdpa',local_files_only=True)
    wrapper.model.eval().requires_grad_(False)
    wrapper.model.speech_tokenizer.model.eval().requires_grad_(False)
    require(all(v.device.type=='cpu' and (not v.is_floating_point() or v.dtype==torch.float32)
        for module in (wrapper.model,wrapper.model.speech_tokenizer.model) for v in module.state_dict().values()),
        'Native base/codec CPUFP32 required')
    require(not hasattr(wrapper.model,'speaker_preference_adapter'),'Original base must have no speaker adapter')
    features.assert_native_components(wrapper,e['features']);return wrapper


def reference_assets(wrapper,e):
    """Verify once per process/model and load ONLY legitimate reference assets."""
    from . import reference_masked_supervision_features as features
    cache=getattr(wrapper,'_base_kv_reference_assets',None)
    if cache is not None:return cache
    run=Path(e['features']);p=features.verify_complete(run);audit=read(run/'verification.json')
    extraction=read(run/'extraction.json');protocol_sha=digest(run/'protocol.json')
    wanted={c['feature_id'] for c in e['contexts']};descriptors={d['id']:d for d in p['records']};cache={}
    require(wanted<=set(descriptors),'Reference asset outside registered features')
    for identity in sorted(wanted):
        value,receipt=features.read_asset(run,descriptors[identity],protocol_sha)
        require(receipt['feature_sha256']==audit['feature_sha256'][identity]
            and digest(run/'assets'/features.asset_key(identity)/'receipt.json')==extraction['asset_receipts'][identity],
            'Bound original-reference asset changed')
        cache[identity]=value
    wrapper._base_kv_reference_assets=cache
    return cache


def fresh(wrapper,c,e):
    import torch
    from .engine import ids
    from .reference_masked_supervision import NativeReference,Layout
    asset=reference_assets(wrapper,e)[c['feature_id']]
    require(digest(c['reference'])==c['reference_sha256']==asset['audio_sha256'],'Original reference changed')
    with torch.inference_mode():prompt=wrapper.create_voice_clone_prompt(ref_audio=c['reference'],ref_text=c['reference_text'],x_vector_only_mode=False)
    ref=NativeReference(asset['ref_ids'],asset['codes'],asset['speaker'],c['reference_sha256'],asset['protocol_sha256'])
    target=ids(wrapper,c['text']).cpu()
    require(equal(ids(wrapper,c['reference_text'],True).cpu(),ref.ref_ids)
        and equal(prompt[0].ref_code.cpu(),ref.ref_codes) and equal(prompt[0].ref_spk_embedding.flatten().cpu(),ref.speaker),'Fresh original reference features differ')
    a=8+ref.ref_ids.shape[1]-5;b=a+target.shape[1]-8;layout=Layout(a,b,b+len(ref.ref_codes)+2)
    binding=dict(case=c,text=c['text'],input_ids_sha256=tensor_sha(target),ref_ids_sha256=tensor_sha(ref.ref_ids),
        ref_codes_sha256=tensor_sha(ref.ref_codes),original_xvec_sha256=tensor_sha(ref.speaker),layout=asdict(layout),
        donor_base=e['config']['base'],base_model_files_sha256=hashlib.sha256(json.dumps(e['base_model_files'],sort_keys=True).encode()).hexdigest(),
        student_checkpoint_sha256=e['checkpoint_sha256'],
        donor_full_reference=True,numeric_config=e['numeric_config'],generation=e['generation'],seeds=[2026,2027])
    return prompt,ref,target,layout,binding


@contextmanager
def state_guard(wrapper,core=None,state=None):
    import copy
    from . import reference_masked_supervision_training_v2 as trainer
    from .reference_masked_supervision_policy_v2 import tensor_digest
    from .reference_masked_supervision_speech_v2 import unchanged
    model=wrapper.model;rng=trainer.rng_state();weights=tensor_digest(model.state_dict().items())
    codec=model.speech_tokenizer.model;codec_sha=tensor_digest(codec.state_dict().items())
    talker=model.talker;had=hasattr(talker,'rope_deltas');rope=copy.deepcopy(getattr(talker,'rope_deltas',None))
    hooks={n:(tuple(m._forward_hooks),tuple(m._forward_pre_hooks)) for n,m in model.named_modules()}
    try:yield
    finally:
        trainer.restore_rng(rng)
        if had:talker.rope_deltas=rope
        elif hasattr(talker,'rope_deltas'):del talker.rope_deltas
        require(weights==tensor_digest(model.state_dict().items()) and codec_sha==tensor_digest(codec.state_dict().items())
            and equal(trainer.rng_state(),rng)
            and all(v.grad is None for v in model.parameters())
            and hooks=={n:(tuple(m._forward_hooks),tuple(m._forward_pre_hooks)) for n,m in model.named_modules()},'Model/RNG/gradient/hooks changed')
        if core is not None:unchanged(model,core,state)


def observe(wrapper,c,e,seed,core=None,donor=None,prefix_only=False,logits=False):
    import random
    from contextlib import nullcontext
    import numpy as np
    import torch
    from .preference_rollouts import capture_generation
    prompt,ref,target,layout,binding=fresh(wrapper,c,e)
    torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    native=core.native(c['half_reference']) if core is not None else nullcontext()
    observer=old.native_observation(wrapper.model) if logits else nullcontext({})
    with Bank(wrapper.model,layout,binding,donor,prefix_only) as bank,native,torch.inference_mode(),observer as seen:
        if prefix_only:
            stopped=False
            try:wrapper.generate_voice_clone(text=c['text'],language='Auto',voice_clone_prompt=prompt,non_streaming_mode=True,**e['generation'])
            except PrefixComplete:stopped=True
            require(stopped and len(bank.inputs)==1,'Donor must stop after only actual native prefill')
            waves,sr,capture=None,None,None
        else:
            with capture_generation(wrapper.model) as capture:
                waves,sr=wrapper.generate_voice_clone(text=c['text'],language='Auto',voice_clone_prompt=prompt,non_streaming_mode=True,**e['generation'])
    artifact=dict(bank=bank.receipt(),inputs=bank.inputs,contexts=bank.contexts,input_ids=target,ref_ids=ref.ref_ids,
        ref_codes=ref.ref_codes,original_ref_xvec=ref.speaker,logits=seen.get('logits',{}))
    if capture is not None:
        codes=capture['codes'];seq=capture['first_sequence'];eos=wrapper.model.config.talker_config.codec_eos_token_id
        require(capture['terminated'] is True and len(codes)>0 and codes.dtype==seq.dtype==torch.long
            and codes.shape==(len(codes),16) and bool(((codes>=0)&(codes<2048)).all())
            and seq.shape==(1,len(codes)+1) and torch.equal(seq[0,:-1],codes[:,0]) and int(seq[0,-1])==eos
            and len(bank.inputs)==len(codes)+1,'Incomplete native all16/EOS trajectory')
        require(sr==24000 and len(waves)==1 and 0<len(waves[0])<=sr*30 and np.isfinite(waves[0]).all(),'Invalid waveform')
        artifact.update(codes=codes,first_sequence=seq,terminated=True,eos_token_id=eos)
    return artifact,waves,sr


def exact_control(a,b):
    require(a['audio_sha256']==b['audio_sha256'] and all(equal(a[k],b[k]) for k in
        ('codes','first_sequence','inputs','contexts','logits')) and equal(a['bank']['values'],b['bank']['values']),
        'Native self-graft/sham changed waveform, codes, logits, inputs or incoming K/V')


def save_clip(out,label,t,waves,sr):
    import soundfile as sf
    from .engine import save
    path=Path(out)/(label+'.wav');require(not path.exists() and not path.with_suffix('.pt').exists(),'No clip overwrite')
    sf.write(path,waves[0],sr,subtype='PCM_16');t['audio_sha256']=digest(path);save(path.with_suffix('.pt'),t)
    return dict(wav=path.name,audio_sha256=digest(path),trajectory=path.with_suffix('.pt').name,trajectory_sha256=digest(path.with_suffix('.pt')))


def controls_base(out,sha):
    import torch
    from .engine import save
    out=Path(out);p,e=verify(out,sha);require(not (out/'base-controls-started.json').exists(),'No control restart')
    write(out/'base-controls-started.json',dict(protocol_sha256=sha));wrapper=load_base(e);identity=interface(wrapper);records=[]
    # Prove parent code identity, independently of equal waveform hashes.
    parent_run=Path(e['parent']);old_rows=read(parent_run/'generated.json')['records']
    for row in old_rows:
        if row['arm']!='student32':continue
        other=next(r for r in old_rows if r['arm']=='graft32' and r['pair_id']==row['pair_id'] and r['half_reference']==row['half_reference'])
        a=torch.load(parent_run/row['trajectory'],weights_only=True,map_location='cpu');b=torch.load(parent_run/other['trajectory'],weights_only=True,map_location='cpu')
        require(equal(a['codes'],b['codes']) and equal(a['first_sequence'],b['first_sequence']),'Parent codec intervention effect requires reconsideration')
    with state_guard(wrapper):
        for i,c in enumerate(e['control_cases']):
            a,w,sr=observe(wrapper,c,e,2026,logits=True);ra=save_clip(out,f'control-base-{i}',a,w,sr)
            b,w,sr=observe(wrapper,c,e,2026,donor=a['bank'],logits=True);rb=save_clip(out,f'control-base-self-{i}',b,w,sr);exact_control(a,b)
            for seed in (2026,2027):
                short,_,_=observe(wrapper,c,e,seed,prefix_only=True)
                require(equal(short['bank']['values'],a['bank']['values']) and equal(short['inputs'][0],a['inputs'][0])
                    and equal(short['contexts'][0],a['contexts'][0]),'Prefix-only/seed donor differs from actual native full generation')
            records.append(dict(case=c['case'],baseline=ra,self_graft=rb,exact=True,prefix_only_exact=True,both_seeds_exact=True))
    write(out/'base-controls.json',dict(protocol_sha256=sha,interface=identity,records=records,all_state_restored=True))


def controls_student(out,sha):
    from . import reference_masked_supervision_training_v2 as trainer
    from . import reference_masked_supervision_features as features
    out=Path(out);p,e=verify(out,sha);require(not (out/'controls.json').exists(),'No control overwrite')
    base=read(out/'base-controls.json');require(base['protocol_sha256']==sha,'Base controls changed')
    _,wrapper,core,state=trainer.load(e['train'],0,device='cpu',training=False);features.assert_native_components(wrapper,e['features'])
    require(interface(wrapper)==base['interface'],'Base/student native norm/RoPE/GQA/config interfaces differ')
    historical=read(Path(e['train'])/'eval-000000/results.json')['records'];records=[]
    with state_guard(wrapper,core,state):
        for i,c in enumerate(e['control_cases']):
            a,w,sr=observe(wrapper,c,e,2026,core=core,logits=True);ra=save_clip(out,f'control-student-{i}',a,w,sr)
            b,w,sr=observe(wrapper,c,e,2026,core=core,donor=a['bank'],logits=True);rb=save_clip(out,f'control-student-sham-{i}',b,w,sr);exact_control(a,b)
            previous=next(r for r in historical if r['case']==c['case'] and r['seed']==2026)
            require(a['audio_sha256']==previous['audio_sha256'],'Student sham baseline differs from historical centered0')
            records.append(dict(case=c['case'],baseline=ra,sham=rb,exact=True,historical_wave_exact=True))
    write(out/'controls.json',dict(status='verified',protocol_sha256=sha,base_controls_sha256=digest(out/'base-controls.json'),
        interface=base['interface'],records=records,control_recordings=8,all_state_restored=True))


def bound_controls(out,sha):
    result=read(out/'controls.json');base=read(out/'base-controls.json')
    expected={c['case'] for c in read(out/'protocol.json')['control_cases']}
    require(result['status']=='verified' and result['protocol_sha256']==base['protocol_sha256']==sha
        and result['base_controls_sha256']==digest(out/'base-controls.json') and result['control_recordings']==8
        and result['all_state_restored'] is base['all_state_restored'] is True
        and len(result['records'])==len(base['records'])==2
        and {r['case'] for r in result['records']}=={r['case'] for r in base['records']}==expected
        and all(r['historical_wave_exact'] is True for r in result['records'])
        and all(r['prefix_only_exact'] is r['both_seeds_exact'] is True for r in base['records']),
        'Complete controls required')
    for r in result['records']+base['records']:
        require(r['exact'] is True,'Control identity failed')
        for label in ('baseline','self_graft','sham'):
            if label not in r:continue
            a=r[label];require(digest(out/a['wav'])==a['audio_sha256'] and digest(out/a['trajectory'])==a['trajectory_sha256'],'Control artifact changed')
    return result


def donor_path(out,c):return out/'donors'/(hashlib.sha256(c['case'].encode()).hexdigest()+'.pt')


def donors(out,sha):
    from .engine import save
    out=Path(out);p,e=verify(out,sha);controls=bound_controls(out,sha)
    dest=out/'donors';require(not dest.exists(),'No donor-cache restart');dest.mkdir()
    wrapper=load_base(e);require(interface(wrapper)==controls['interface'],'Original base interface changed');records=[]
    with state_guard(wrapper):
        for i,c in enumerate(e['contexts']):
            a,_,_=observe(wrapper,c,e,2026,prefix_only=True);b,_,_=observe(wrapper,c,e,2027,prefix_only=True)
            require(equal(a,b),'Donor prefix depends on sampling seed')
            path=donor_path(out,c);save(path,a);records.append(dict(case=c['case'],path=str(path),sha256=digest(path),both_seeds_exact=True))
            print(f'base prefix {i+1}/40; both seeds exact; no target frame sampled',flush=True)
    write(out/'donors.json',dict(protocol_sha256=sha,controls_sha256=digest(out/'controls.json'),records=records,contexts=40,
        donor_future_frames=0,all_state_restored=True,interface=controls['interface']))


def bound_donors(out,sha,e):
    controls=bound_controls(out,sha);r=read(out/'donors.json')
    require(r['protocol_sha256']==sha and r['controls_sha256']==digest(out/'controls.json')
        and r['interface']==controls['interface'] and r['contexts']==40 and r['donor_future_frames']==0
        and r['all_state_restored'] is True and len(r['records'])==40
        and {x['case'] for x in r['records']}=={c['case'] for c in e['contexts']},'Complete base donor coverage required')
    for row in r['records']:
        c=next(c for c in e['contexts'] if c['case']==row['case'])
        require(Path(row['path'])==donor_path(out,c) and digest(row['path'])==row['sha256'] and row['both_seeds_exact'] is True,'Donor cache changed')
    return r


def generate(out,sha):
    import soundfile as sf
    import torch
    from . import reference_masked_supervision_training_v2 as trainer
    from . import reference_masked_supervision_features as features
    out=Path(out);p,e=verify(out,sha);donors_receipt=bound_donors(out,sha,e)
    require(not (out/'generation-started.json').exists(),'No screen restart');write(out/'generation-started.json',dict(protocol_sha256=sha))
    _,wrapper,core,state=trainer.load(e['train'],0,device='cpu',training=False);features.assert_native_components(wrapper,e['features'])
    require(interface(wrapper)==donors_receipt['interface'],'Student native interface changed');rows=[]
    with state_guard(wrapper,core,state):
        for cseed in e['cases']:
            c={k:v for k,v in cseed.items() if k!='seed'};seed=cseed['seed'];d=torch.load(donor_path(out,c),map_location='cpu',weights_only=True)
            t,w,sr=observe(wrapper,c,e,seed,core=core,donor=d['bank']);saved=save_clip(out,f"{seed}-{c['case']}",t,w,sr)
            require(equal(t['contexts'][0],d['contexts'][0]),'Base/student native positions/mask differ')
            wave,sr=sf.read(out/saved['wav'],dtype='float32')
            with torch.inference_mode():
                vector=wrapper.model.extract_speaker_embedding(audio=wave,sr=sr).flatten()
                cosine=float(torch.nn.functional.cosine_similarity(vector[None],t['original_ref_xvec'][None]))
            require(math.isfinite(cosine),'Nonfinite original-reference similarity')
            row=dict(cseed,**saved,step=0,cosine=cosine,seconds=len(wave)/sr,frames=len(t['codes']),terminated=True,
                protocol_sha256=sha,checkpoint_sha256=e['checkpoint_sha256'],donor_sha256=digest(donor_path(out,c)))
            write(out/(saved['wav']+'.json'),row);rows.append(row);print(f'base-K/V calibration {len(rows)}/80',flush=True)
    write(out/'generated.json',dict(protocol_sha256=sha,records=rows,donors_sha256=digest(out/'donors.json'),all_state_restored=True))


def bound_generation(out,sha,e):
    bound_donors(out,sha,e);g=read(out/'generated.json');rows=g['records'];expected={(c['case'],c['seed']):c for c in e['cases']}
    require(g['protocol_sha256']==sha and g['donors_sha256']==digest(out/'donors.json') and g['all_state_restored'] is True
        and len(rows)==80 and {(r['case'],r['seed']) for r in rows}==set(expected),'Complete80 native screen required')
    for r in rows:
        c=expected[r['case'],r['seed']]
        require(all(r.get(k)==v for k,v in c.items()) and r['step']==0 and r['protocol_sha256']==sha
            and r['checkpoint_sha256']==e['checkpoint_sha256'] and r['terminated'] is True
            and math.isfinite(r['cosine']) and -1<=r['cosine']<=1
            and r['donor_sha256']==digest(donor_path(out,c)) and digest(r['reference'])==r['reference_sha256']
            and digest(out/r['wav'])==r['audio_sha256'] and digest(out/r['trajectory'])==r['trajectory_sha256'],'Native metadata/artifact differs')
    return rows


def audit_donors(out,sha):
    import torch
    out=Path(out);p,e=verify(out,sha);bound_donors(out,sha,e);require(not (out/'donor-audit.json').exists(),'No donor-audit overwrite')
    wrapper=load_base(e)
    with state_guard(wrapper):
        for r in read(out/'base-controls.json')['records']:
            a=torch.load(out/r['baseline']['trajectory'],map_location='cpu',weights_only=True)
            b=torch.load(out/r['self_graft']['trajectory'],map_location='cpu',weights_only=True);exact_control(a,b)
        for c in e['contexts']:
            saved=torch.load(donor_path(out,c),map_location='cpu',weights_only=True)
            fresh_t,_,_=observe(wrapper,c,e,2026,prefix_only=True)
            require(equal(saved,fresh_t),'Independent original-base native donor reconstruction differs')
    write(out/'donor-audit.json',dict(status='verified',protocol_sha256=sha,contexts=40,
        donors_sha256=digest(out/'donors.json'),all_state_restored=True))


def audit(out,sha):
    import numpy as np
    import soundfile as sf
    import torch
    from . import reference_masked_supervision_training_v2 as trainer
    from . import reference_masked_supervision_features as features
    out=Path(out);p,e=verify(out,sha);rows=bound_generation(out,sha,e);da=read(out/'donor-audit.json')
    require(da==dict(status='verified',protocol_sha256=sha,contexts=40,donors_sha256=digest(out/'donors.json'),all_state_restored=True)
        and not (out/'native-verification.json').exists(),'Fresh complete donor audit required')
    _,wrapper,core,state=trainer.load(e['train'],0,device='cpu',training=False);features.assert_native_components(wrapper,e['features'])
    maximum=0.;artifacts={};require(interface(wrapper)==read(out/'controls.json')['interface'],'Audit native interface differs')
    with state_guard(wrapper,core,state):
        for r in read(out/'controls.json')['records']:
            a=torch.load(out/r['baseline']['trajectory'],map_location='cpu',weights_only=True)
            b=torch.load(out/r['sham']['trajectory'],map_location='cpu',weights_only=True);exact_control(a,b)
        for r in rows:
            c={k:v for k,v in next(c for c in e['cases'] if c['case']==r['case'] and c['seed']==r['seed']).items() if k!='seed'}
            _,ref,target,layout,binding=fresh(wrapper,c,e)
            t=torch.load(out/r['trajectory'],map_location='cpu',weights_only=True);d=torch.load(donor_path(out,c),map_location='cpu',weights_only=True)
            codes=t['codes'];seq=t['first_sequence'];eos=wrapper.model.config.talker_config.codec_eos_token_id
            require(t['terminated'] is True and codes.dtype==seq.dtype==torch.long and codes.shape==(r['frames'],16) and len(codes)>0
                and bool(((codes>=0)&(codes<2048)).all()) and seq.shape==(1,len(codes)+1)
                and torch.equal(seq[0,:-1],codes[:,0]) and int(seq[0,-1])==eos==t['eos_token_id']
                and t['audio_sha256']==r['audio_sha256'],'Native all16/EOS evidence differs')
            require(equal(t['input_ids'],target) and equal(t['ref_ids'],ref.ref_ids) and equal(t['ref_codes'],ref.ref_codes)
                and equal(t['original_ref_xvec'],ref.speaker) and equal(d['input_ids'],target)
                and equal(d['ref_ids'],ref.ref_ids) and equal(d['ref_codes'],ref.ref_codes)
                and equal(d['original_ref_xvec'],ref.speaker) and equal(d['contexts'][0],t['contexts'][0]),'Base/student legitimate reference identities or positions differ')
            with core._context('conditioning-audit',c['half_reference'],layout),torch.inference_mode():rebuilt=core._build(target,codes,ref)
            actual=torch.cat(t['inputs'],dim=1)[0]
            require(equal(rebuilt.embeddings,actual) and torch.isfinite(actual).all(),'Student native prefix/suffix reconstruction differs')
            bank=t['bank'];donor=d['bank'];keys={f'{i}.{k}' for i in range(len(wrapper.model.talker.model.layers)) for k in ('k','v')}
            require(bank['binding']==donor['binding']==binding and set(bank['values'])==set(bank['checks'])==set(bank['calls'])==keys
                and bank['layout']==donor['layout']==asdict(layout) and bank['prefix_positions']==list(range(layout.prefix_length))
                and bank['protected_positions']==donor['protected_positions']
                and bank['generated_rows_added_to_donor']==donor['generated_rows_added_to_donor']==0
                and len(t['inputs'])==len(t['contexts'])==len(codes)+1 and all(n==len(codes)+1 for n in bank['calls'].values()),'Projected-K/V coverage differs')
            for k in keys:
                check=bank['checks'][k]
                require(check['allowed_before_sha256']==check['allowed_after_sha256']
                    and check['protected_after_sha256']==tensor_sha(donor['values'][k])
                    and torch.isfinite(bank['values'][k]).all(),'Protected graft/allowed K/V evidence differs')
            for i,ctx in enumerate(t['contexts']):
                expected=torch.arange(layout.prefix_length) if i==0 else torch.tensor([layout.prefix_length+i-1])
                require(equal(ctx['cache_position'],expected) and ctx['past_length']==(0 if i==0 else layout.prefix_length+i-1),'Native cache alignment differs')
            wave,sr=sf.read(out/r['wav'],dtype='float32')
            require(sr==24000 and wave.ndim==1 and np.isfinite(wave).all() and r['seconds']==len(wave)/sr<=30,'Waveform changed')
            with torch.inference_mode():
                v=wrapper.model.extract_speaker_embedding(audio=wave,sr=sr).flatten();cos=float(torch.nn.functional.cosine_similarity(v[None],ref.speaker[None]))
            require(math.isfinite(cos),'Nonfinite independent cosine');maximum=max(maximum,abs(cos-r['cosine']))
            for path in (out/r['wav'],out/r['trajectory'],Path(r['reference']),donor_path(out,c)):artifacts[str(path)]=digest(path)
    require(maximum<=2e-6,'Independent reference cosine differs');verify(out,sha)
    write(out/'native-verification.json',dict(status='verified',protocol_sha256=sha,recordings=80,contexts=40,truncated=0,
        generated_sha256=digest(out/'generated.json'),donor_audit_sha256=digest(out/'donor-audit.json'),
        controls_sha256=digest(out/'controls.json'),max_cosine_recomputation_error=maximum,
        frozen_weights_verified=True,native_inputs_verified=80,all_hooks_removed=True,all_state_restored=True,
        artifacts=artifacts,**FLAGS))


def native_evidence(out,sha,e):
    rows=bound_generation(out,sha,e);a=read(out/'native-verification.json')
    require(a['status']=='verified' and a['protocol_sha256']==sha and a['recordings']==a['native_inputs_verified']==80
        and a['contexts']==40 and a['truncated']==0 and a['generated_sha256']==digest(out/'generated.json')
        and a['donor_audit_sha256']==digest(out/'donor-audit.json') and a['controls_sha256']==digest(out/'controls.json')
        and math.isfinite(a['max_cosine_recomputation_error']) and 0<=a['max_cosine_recomputation_error']<=2e-6
        and all(a[k] is True for k in ('frozen_weights_verified','all_hooks_removed','all_state_restored'))
        and all(a.get(k)==v for k,v in FLAGS.items()),'Complete native80 audit required')
    for path,value in a['artifacts'].items():require(digest(path)==value,'Audited artifact changed')
    return rows


def score(out,sha):
    import soundfile as sf
    import torch,torchaudio
    from transformers import WhisperForConditionalGeneration,WhisperProcessor
    from .evaluate import ASR_REV
    from scripts.score_latest_audio import score as edits
    out=Path(out);p,e=verify(out,sha);rows=native_evidence(out,sha,e);require(not (out/'scored.json').exists(),'No score overwrite')
    manifest=read(e['asr_manifest']);asr=Path(e['config']['asr'])
    require(digest(e['asr_manifest'])==e['asr_manifest_sha256'] and manifest['repo']=='BuzzASR/mongolian' and manifest['revision']==ASR_REV,'Original ASR required')
    for name,r in manifest['files'].items():require(digest(asr/name)==r['sha256'],'ASR changed')
    processor=WhisperProcessor.from_pretrained(asr,local_files_only=True)
    model=WhisperForConditionalGeneration.from_pretrained(asr,torch_dtype=torch.float32,local_files_only=True).to('cuda').eval()
    for r in rows:
        wave,sr=sf.read(out/r['wav'],dtype='float32');wave=torchaudio.functional.resample(torch.from_numpy(wave),sr,16000).numpy()
        batch=processor(wave,sampling_rate=16000,return_tensors='pt',return_attention_mask=True).to('cuda')
        with torch.inference_mode():tokens=model.generate(batch.input_features,attention_mask=batch.attention_mask,**e['decoding'])
        hypothesis=processor.batch_decode(tokens,skip_special_tokens=True)[0].strip()
        r.update(hypothesis=hypothesis,score=edits(r['text'],hypothesis,True),asr_revision=ASR_REV,decoding=e['decoding'])
    write(out/'scored.json',dict(protocol_sha256=sha,native_verification_sha256=digest(out/'native-verification.json'),records=rows))


def scalar_audit(out,sha):
    from .reference_masked_supervision_evaluation import validate_records
    from .evaluate import ASR_REV
    out=Path(out);p,e=verify(out,sha);original=native_evidence(out,sha,e);scored=read(out/'scored.json')
    require(scored['protocol_sha256']==sha and scored['native_verification_sha256']==digest(out/'native-verification.json')
        and not (out/'results.json').exists(),'Scoring provenance/overwrite error')
    rows=scored['records'];summary=validate_records(e['cases'],rows);by={(r['case'],r['seed']):r for r in rows}
    for r in original:
        s=by[r['case'],r['seed']]
        require(set(s)==set(r)|{'hypothesis','score','asr_revision','decoding'} and all(s[k]==v for k,v in r.items())
            and s['asr_revision']==ASR_REV and s['decoding']==e['decoding'],'Scoring changed native record')
    baseline=read(Path(e['train'])/'eval-000000/results.json');baseline_summary=validate_records(e['cases'],baseline['records'])
    qualified=eligible(summary,0)
    write(out/'results.json',dict(status='verified',protocol_sha256=sha,recordings=80,summary=summary,truncated=0,
        historical_centered0_summary=baseline_summary,historical_results_sha256=digest(Path(e['train'])/'eval-000000/results.json'),
        native_verification_sha256=digest(out/'native-verification.json'),scored_sha256=digest(out/'scored.json'),
        generated_sha256=digest(out/'generated.json'),calibration_qualified=qualified,
        next_stage='new_original114_confirmation_after_authoritative_process_closure' if qualified else 'failed_screen_no_automatic_training',**FLAGS))


def main():
    import argparse,os
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=('prepare','controls-base','controls-student','donors','generate','audit-donors','audit','score','scalar-audit'))
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--parent',type=Path);parser.add_argument('--driver',type=Path)
    parser.add_argument('--test-log',type=Path);parser.add_argument('--note',type=Path);parser.add_argument('--protocol-sha')
    args=parser.parse_args()
    if args.phase=='prepare':prepare(args.parent,args.out,args.driver,args.test_log,args.note);return
    require(args.protocol_sha,'Explicit registered hash required');p,e=verify(args.out,args.protocol_sha)
    os.environ['TEMP']=os.environ['TMP']=e['config']['tmp'];os.environ['NUMBA_CACHE_DIR']=e['config']['numba_cache']
    os.environ['HF_HUB_OFFLINE']=os.environ['TRANSFORMERS_OFFLINE']='1'
    if args.phase=='scalar-audit':scalar_audit(args.out,args.protocol_sha);return
    import torch
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    globals()[args.phase.replace('-','_')](args.out,args.protocol_sha)


if __name__=='__main__':main()
