"""No-training attention4/speaker8 routing comparison on the original full panel.

Baseline114 is inactive routing, control26 contains all custom texts with half
PAD, and candidate114 changes only custom recurrent PAD to full. Each stage is
generated on CPU, scored on GPU, and independently audited on CPU.
"""
import argparse
import math
import os
from pathlib import Path
import numpy as np
import torch
from .engine import digest,ids,read,save,write
from .policy import external
from . import middle_attention_training as initial
from . import middle_attention_adapter as adapters
from . import reference_profile_collection as previous
from . import reference_prefix_generation_pad_evaluation as shared
from .reference_prefix_generation_pad import ReferencePrefixGenerationPad,tensor_sha256
from .reference_prefix_comparison import tensor_digest
from .confirm_joint_candidate import validate_panel
from .targeted_voice import aggregate
from .voice_strength_search import goal_screen

STAGES=('baseline','control','candidate')
COUNTS=dict(baseline=114,control=26,candidate=114)
DISABLED='0'*64
FAILED_AUDIT_SHA='4b743778ea3bb71afb277867721f0db480a60904580349ca1fb29bf759dc377d'
NUMERIC=shared.NUMERIC
require=shared.require


def sources():
    result=shared.sources()
    for name in ('reference_prefix_lora_routing.py','reference_prefix_lora_pad_evaluation.py'):
        result[name]=digest(Path(__file__).with_name(name))
    for name in ('test_reference_prefix_lora_routing.py','test_training_reference_prefix_lora_pad_evaluation.py'):
        result[name]=digest(Path(__file__).parent.parent/'tests'/name)
    return result


def cases_for(p,stage):
    require(stage in STAGES,'Unknown comparison stage')
    return [c for c in p['cases'] if stage!='control' or c['group']=='custom']


def validate_scope(panel,screen,reference_sha):
    validate_panel(panel)
    cases=[dict(c,seed=s) for s in panel['seeds'] for c in panel['cases']]
    expanded={(c['case'],c['seed']):c for c in cases}
    require(len(screen)==26 and len({(c['case'],c['seed']) for c in screen})==26,'Original screen incomplete')
    for c in screen:
        full=expanded.get((c['case'],c['seed']))
        require(full is not None and all(full.get(k)==v for k,v in c.items()),'Historical screen metadata changed')
    require(reference_sha!=DISABLED and all(c['reference_sha256']!=DISABLED and
        ((c['reference_sha256']==reference_sha)==(c['group']=='custom')) for c in cases),'Reference dispatch scope changed')
    return cases


def validate_records(cases,records):
    expected={(c['case'],c['seed']):c for c in cases}
    require(len(expected)==len(cases)==len(records) and {(r['case'],r['seed']) for r in records}==set(expected),
        'Incomplete or duplicate stage recordings')
    for r in records:
        require(all(r.get(k)==v for k,v in expected[r['case'],r['seed']].items()),'Case metadata changed')
        require(math.isfinite(r['cosine']) and -1<=r['cosine']<=1 and type(r['terminated']) is bool,'Invalid voice/termination metric')
    return aggregate(records)


def trigger(root):
    path=root/'prefix-pad-goal-audit.json';require(digest(path)==FAILED_AUDIT_SHA,'Prior failure trigger changed')
    result=read(path);native_path=root/'reference-prefix-generation-pad-confirmation/verification.json'
    status_path=root/'reference-prefix-generation-pad-confirmation-status.json';status=read(status_path);native=read(native_path)
    require(result['status']=='verified' and result['recordings']==114 and result['goal_criteria_met'] is False
        and status['status']=='complete' and status['confirmation_pass'] is False
        and native['status']=='verified' and native['recordings']==114 and native['confirmation_pass'] is False
        and result['evidence']['native_audit_sha256']==digest(native_path)
        and result['evidence']['terminal_status_sha256']==digest(status_path),'Completed failed114 experiment required')
    return {str(f):digest(f) for f in (path,native_path,status_path)}


def historical(initial_run,previous_run):
    p=initial.verify(initial_run);old=previous.verify(previous_run)
    direct=previous.direct.verify(Path(old['initial_run']))
    require(Path(direct['candidate']).resolve()==initial_run.resolve(),'Historical attention policy differs')
    report_path=initial_run/'eval-000004/results.json';audit_path=initial_run/'verification-000004.json'
    report=read(report_path);audit=read(audit_path)
    screen=[dict(c,seed=s) for s in p['seeds'] for c in p['cases']]
    summary=validate_records(screen,report['records'])
    require(audit['status']=='verified' and audit['step']==4 and audit['recordings']==26
        and audit['acoustic_weights_verified'] and audit['frozen_speaker_verified'] and audit['middle_attention_coverage_verified']
        and audit['all_artifact_hashes_verified'] and audit['asr_edit_counts_and_aggregates_recomputed']
        and audit['cosines_reextracted_from_audio'] and math.isfinite(audit['max_cosine_recomputation_error'])
        and 0<=audit['max_cosine_recomputation_error']<=2e-6
        and audit['summary']==report['summary']==summary and report['checkpoint_sha256']==digest(initial.checkpoint(initial_run,4)),
        'Verified attention4/speaker8 screen required')
    auditor=initial_run.parent/'verify-middle-attention-checkpoint.py'
    require(digest(auditor)==audit['source_sha256'],'Historical audit implementation changed')
    panel_path=Path(old['panel']);panel=read(panel_path)
    require(digest(panel_path)==old['panel_sha256'] and all(panel[k]==p[k] for k in ('seeds','generation','decoding')),
        'Original114 panel identity/settings changed')
    target=old['profile_reference']['reference_sha256'];cases=validate_scope(panel,screen,target)
    for r in report['records']:
        require(digest(report_path.parent/r['wav'])==r['audio_sha256'] and digest(r['reference'])==r['reference_sha256'],
            'Historical baseline waveform/reference changed')
    paths=[initial_run/'protocol.json',initial.checkpoint(initial_run,4),initial.checkpoint(initial_run,4).with_suffix('.json'),
        report_path,audit_path,auditor,previous_run/'protocol.json',Path(old['initial_run'])/'protocol.json',panel_path,
        initial_run.parent/'reference-prefix-half-routing/model.json',Path(p['config']['asr'])/'download-manifest.json']
    return p,cases,screen,target,report['records'],{str(f):digest(f) for f in paths}


def prepare(initial_run,previous_run,out,driver):
    hashes=trigger(out.parent)
    p,cases,screen,target,_,bound=historical(initial_run,previous_run);hashes.update(bound)
    require(previous.numeric_config()==NUMERIC,'CPU FP32 numerical settings required')
    logs={}
    for name,count in (('reference-prefix-lora-routing-tests.log',6),('reference-prefix-lora-pad-evaluation-tests.log',5)):
        path=out.parent/name;content=path.read_text(encoding='utf-8-sig')
        require(f'Ran {count} tests' in content and '\nOK\n' in content and 'skipped' not in content.lower(),'New CPU tests must pass')
        logs[str(path)]=digest(path)
    protocol=dict(initial=str(initial_run),previous=str(previous_run),step=4,evidence=hashes,source=sources(),
        runtime=initial.collection.runtime(),numeric_config=NUMERIC,tests=logs,driver=str(driver),driver_sha256=digest(driver),
        cases=cases,screen_cases=screen,reference_sha256=target,disabled_reference_sha256=DISABLED,
        stages=list(STAGES),recordings=COUNTS,config=p['config'],generation=p['generation'],decoding=p['decoding'],
        base_sha256=p['base_sha256'],adapted_sha256=p['adapted_sha256'],
        baseline_policy='Audited attention4 plus speaker8, unmerged original LoRA arithmetic retained',
        halfway_policy='198 original base-to-acoustic halfway matrices; outer attention LoRA bypassed only on protected reference rows',
        intervention='Exact custom reference only: half reference/control prefix, full target/generated decoder rows; candidate recurrent PAD full',
        scope='Original114 baseline, all26custom half-PAD control, original114candidate;88development exact baseline reproduction',
        optimizer_updates=0,goal_achieved=False,global_model_selection_changed=False)
    out=external(out.resolve());out.mkdir(exist_ok=False);write(out/'protocol.json',protocol)


def verify(out):
    p=read(out/'protocol.json')
    require(p['source']==sources() and p['runtime']==initial.collection.runtime() and p['numeric_config']==previous.numeric_config()==NUMERIC
        and p['stages']==list(STAGES) and p['recordings']==COUNTS and p['step']==4 and p['disabled_reference_sha256']==DISABLED
        and digest(p['driver'])==p['driver_sha256'],'Comparison registration changed')
    for path,sha in dict(p['tests'],**p['evidence']).items():require(digest(path)==sha,'Registered evidence changed')
    old,cases,screen,target,rows,_=historical(Path(p['initial']),Path(p['previous']))
    require(p['cases']==cases and p['screen_cases']==screen and p['reference_sha256']==target
        and all(p[k]==old[k] for k in ('config','generation','decoding','base_sha256','adapted_sha256')),'Registered conditions changed')
    return p,old,rows


def identity(model,router):
    outer={name+'.'+key:getattr(m,key) for name,m in adapters.modules(model).items() for key in ('lora_A','lora_B')}
    require(len(outer)==32,'Expected16rank4outerLoRA projections')
    return dict(frozen_backbone_sha256=adapters.frozen_digest(model),speaker_adapter_sha256=adapters.speaker_digest(model),
        attention_adapter_sha256=tensor_digest(outer),alternative_weights_sha256=tensor_digest(router.weights),
        matrices=len(router.weights),main_matrices=len(router.original_linears),outer_lora_modules=16)


def load(p,old,stage):
    from safetensors import safe_open
    from .reference_prefix_lora_routing import ReferencePrefixLoRARouting,weight_modules,base_linear
    require(stage in STAGES,'Unknown stage')
    state=initial.read_checkpoint(Path(p['initial']),4,old);wrapper=initial.load(old)
    adapters.restore(wrapper.model,state['adapter']);wrapper.model.requires_grad_(False);wrapper.model.eval()
    require(adapters.frozen_digest(wrapper.model)==state['frozen_backbone_sha256']
        and adapters.speaker_digest(wrapper.model)==state['frozen_speaker_sha256'],'Frozen acoustic/speaker state differs')
    require(all(v.device.type=='cpu' and v.dtype==torch.float32 for v in wrapper.model.parameters())
        and wrapper.model.talker.config._attn_implementation=='sdpa','CPU FP32 SDPA required')
    base=Path(p['config']['base'])/'model.safetensors';require(digest(base)==p['base_sha256'],'Original base weights changed')
    modules=weight_modules(wrapper.model);require(len(modules)==198,'Expected198canonicalroute matrices');alternate={}
    with safe_open(base,framework='pt',device='cpu') as original,torch.no_grad():
        for name,module in modules.items():
            alternate[name+'.weight']=original.get_tensor(name+'.weight').float().lerp(base_linear(module).weight.detach(),.5)
    router=ReferencePrefixLoRARouting(wrapper.model,alternate)
    model_identity=identity(wrapper.model,router)
    historic=read(Path(p['initial']).parent/'reference-prefix-half-routing/model.json')
    require(model_identity['alternative_weights_sha256']==historic['alternative_weights_sha256']
        and model_identity['attention_adapter_sha256']==tensor_digest(state['adapter'])
        and model_identity['matrices']==198 and model_identity['main_matrices']==196,'Historical halfway/attention matrix identity differs')
    target=DISABLED if stage=='baseline' else p['reference_sha256']
    patch=ReferencePrefixGenerationPad(router,target,full_generation_pad=stage=='candidate')
    return wrapper,patch,model_identity


def unchanged(wrapper,patch,expected):
    require(identity(wrapper.model,patch.router)==expected and not patch.router.active
        and all(not v.requires_grad and v.grad is None for v in wrapper.model.parameters()),'Frozen policy changed')
    patch.remove()


def completed(out,p,stage):
    directory=out/stage;receipt=read(directory/'verification.json');report=read(directory/'results.json')
    summary=validate_records(cases_for(p,stage),report['records']);error=receipt['maximum_cosine_recomputation_error']
    require(receipt['status']=='verified' and receipt['stage']==stage and receipt['recordings']==COUNTS[stage]
        and receipt['all_weights_unchanged'] and receipt['all_recomputed_cosines_finite']
        and receipt['native_conditioning_verified'] and receipt['pad_traces_verified']==COUNTS[stage]
        and receipt['historical_screen_reproduced']==(26 if stage=='baseline' else 0)
        and receipt['inactive_development_exact']==(88 if stage=='candidate' else 0)
        and receipt['custom_prefill_and_first_frame_verified']==(26 if stage=='candidate' else 0)
        and receipt['optimizer_updates']==0 and receipt['goal_achieved'] is False and receipt['global_model_selection_changed'] is False
        and math.isfinite(error) and 0<=error<=2e-6 and receipt['summary']==report['summary']==summary
        and receipt['protocol_sha256']==digest(out/'protocol.json') and receipt['results_sha256']==digest(directory/'results.json')
        and receipt['truncated']==report['truncated']==sum(not r['terminated'] for r in report['records']), 'Prior complete stage audit required')
    require(receipt['prior_audit_sha256']=={name:digest(out/name/'verification.json') for name in STAGES[:STAGES.index(stage)]},
        'Prior-stage audit bindings differ')
    for r in report['records']:
        require(digest(directory/r['wav'])==r['audio_sha256'] and digest(directory/r['trajectory'])==r['trajectory_sha256'],
            'Prior stage speech changed')
    return report['records'],receipt


def priors(out,p,stage):
    return {name:completed(out,p,name) for name in STAGES[:STAGES.index(stage)]}


def generate(out,stage):
    import soundfile as sf
    from .preference_rollouts import capture_generation
    p,old,historical_rows=verify(out);prior=priors(out,p,stage);wrapper,patch,model_identity=load(p,old,stage)
    require(all(receipt['model_identity']==model_identity for _,receipt in prior.values()),'Prior policy identity differs')
    historical_map={(r['case'],r['seed']):r for r in historical_rows}
    baseline_map={} if 'baseline' not in prior else {(r['case'],r['seed']):r for r in prior['baseline'][0]}
    dest=out/stage;dest.mkdir(exist_ok=False);records=[];prompts={}
    for c in cases_for(p,stage):
        require(digest(c['reference'])==c['reference_sha256'],'Original reference changed')
        if c['reference'] not in prompts:
            with torch.inference_mode():prompts[c['reference']]=wrapper.create_voice_clone_prompt(ref_audio=c['reference'],ref_text=c['reference_text'],x_vector_only_mode=False)
        prompt=prompts[c['reference']];torch.manual_seed(c['seed']);np.random.seed(c['seed'])
        with patch.use_reference(c['reference_sha256']),torch.inference_mode(),capture_generation(wrapper.model) as captured:
            waves,sr=wrapper.generate_voice_clone(text=c['text'],language='Auto',voice_clone_prompt=prompt,non_streaming_mode=True,**p['generation'])
        require(sr==24000 and len(waves[0])>0 and np.isfinite(waves[0]).all() and len(waves[0])/sr<=30,'Invalid generated speech')
        path=dest/f"{c['seed']}-{c['case']}.wav";sf.write(path,waves[0],sr,subtype='PCM_16');wave,sr=sf.read(path,dtype='float32')
        key=c['case'],c['seed']
        if stage=='baseline' and key in historical_map:
            require(digest(path)==historical_map[key]['audio_sha256'],'Historical baseline waveform parity failed')
        if stage=='candidate' and c['group']=='development':
            require(digest(path)==baseline_map[key]['audio_sha256'],'Inactive development waveform parity failed')
        with torch.inference_mode():
            ref=prompt[0].ref_spk_embedding.flatten();vector=wrapper.model.extract_speaker_embedding(audio=wave,sr=sr).flatten()
            cosine=float(torch.nn.functional.cosine_similarity(vector[None],ref[None]))
        require(math.isfinite(cosine),'Nonfinite generated cosine')
        trajectory=path.with_suffix('.pt');trace=patch.last_trace;routing=patch.router.last_trace
        save(trajectory,dict(codes=captured['codes'],first_sequence=captured['first_sequence'],terminated=captured['terminated'],
            ref_codes=prompt[0].ref_code.cpu(),ref_xvec=ref.cpu(),input_ids=ids(wrapper,c['text']).cpu(),ref_ids=ids(wrapper,c['reference_text'],True).cpu(),
            reference_sha256=c['reference_sha256'],pad_trace=trace,routing=routing))
        row=dict(c,wav=path.name,audio_sha256=digest(path),trajectory=trajectory.name,trajectory_sha256=digest(trajectory),cosine=cosine,
            seconds=len(wave)/sr,frames=len(captured['codes']),terminated=captured['terminated'],stage=stage,pad_trace=trace,routing=routing,
            model_identity=model_identity,protocol_sha256=digest(out/'protocol.json'))
        write(path.with_suffix('.json'),row);records.append(row);print(f'{stage}: {len(records)}/{COUNTS[stage]}',flush=True)
    unchanged(wrapper,patch,model_identity)
    write(dest/'generated.json',dict(records=records,all_weights_unchanged=True,model_identity=model_identity,protocol_sha256=digest(out/'protocol.json')))


def score(out,stage):
    import soundfile as sf
    import torchaudio
    from transformers import WhisperForConditionalGeneration,WhisperProcessor
    from .evaluate import ASR_REV
    from scripts.score_latest_audio import score as score_text
    p,_,_=verify(out);priors(out,p,stage);dest=out/stage;require(not (dest/'results.json').exists(),'No scored speech overwrite')
    generated=read(dest/'generated.json');require(generated['all_weights_unchanged'] and generated['protocol_sha256']==digest(out/'protocol.json'),'Generation receipt changed')
    asr=Path(p['config']['asr']);manifest=read(asr/'download-manifest.json')
    require(manifest['repo']=='BuzzASR/mongolian' and manifest['revision']==ASR_REV,'ASR identity differs')
    for name,item in manifest['files'].items():require(digest(asr/name)==item['sha256'],'ASR weights changed')
    processor=WhisperProcessor.from_pretrained(asr,local_files_only=True)
    model=WhisperForConditionalGeneration.from_pretrained(asr,torch_dtype=torch.float32,local_files_only=True).to('cuda').eval()
    records=generated['records']
    for r in records:
        require(digest(dest/r['wav'])==r['audio_sha256'],'Generated audio changed')
        wave,sr=sf.read(dest/r['wav'],dtype='float32');wave=torchaudio.functional.resample(torch.from_numpy(wave),sr,16000).numpy()
        batch=processor(wave,sampling_rate=16000,return_tensors='pt',return_attention_mask=True).to('cuda')
        with torch.inference_mode():tokens=model.generate(batch.input_features,attention_mask=batch.attention_mask,**p['decoding'])
        hypothesis=processor.batch_decode(tokens,skip_special_tokens=True)[0].strip()
        r.update(hypothesis=hypothesis,score=score_text(r['text'],hypothesis,True),decoding=p['decoding'],asr_revision=ASR_REV)
    write(dest/'results.json',dict(records=records,summary=validate_records(cases_for(p,stage),records),truncated=sum(not r['terminated'] for r in records)))


def score_parity(row,old):
    require(row['audio_sha256']==old['audio_sha256'] and row['score']==old['score']
        and math.isfinite(row['cosine']) and math.isfinite(old['cosine']) and abs(row['cosine']-old['cosine'])<=2e-6,
        'Required exact waveform/score parity failed')


def cosine_error(actual,stored):
    require(math.isfinite(actual) and math.isfinite(stored),'Nonfinite independently recomputed cosine')
    return abs(actual-stored)


def audit(out,stage):
    import soundfile as sf
    from .evaluate import ASR_REV
    from scripts.score_latest_audio import score as score_text
    p,old,historical_rows=verify(out);prior=priors(out,p,stage);dest=out/stage
    require(not (dest/'verification.json').exists(),'No audit overwrite')
    wrapper,patch,model_identity=load(p,old,stage);report=read(dest/'results.json');records=report['records']
    summary=validate_records(cases_for(p,stage),records);refs={};maximum=0.;historical_checked=0;dev_checked=0;custom_paired=0
    history={(r['case'],r['seed']):r for r in historical_rows}
    baseline={} if 'baseline' not in prior else {(r['case'],r['seed']):r for r in prior['baseline'][0]}
    control={} if 'control' not in prior else {(r['case'],r['seed']):r for r in prior['control'][0]}
    require(all(receipt['model_identity']==model_identity for _,receipt in prior.values()),'Policy identity changed between stages')
    special,full,half=patch._pads()
    for r in records:
        require(r['stage']==stage and r['protocol_sha256']==digest(out/'protocol.json') and r['model_identity']==model_identity
            and r['asr_revision']==ASR_REV and r['decoding']==p['decoding'] and digest(dest/r['wav'])==r['audio_sha256']
            and digest(dest/r['trajectory'])==r['trajectory_sha256'] and digest(r['reference'])==r['reference_sha256']
            and r['score']==score_text(r['text'],r['hypothesis'],True),'Native speech provenance changed')
        t=torch.load(dest/r['trajectory'],map_location='cpu',weights_only=True);codes=t['codes'];seq=t['first_sequence'][0];eos=wrapper.model.config.talker_config.codec_eos_token_id
        require(codes.shape==(r['frames'],16) and len(codes)>0 and len(seq)==len(codes)+1 and torch.equal(seq[:-1],codes[:,0])
            and bool(seq[-1]==eos)==r['terminated']==t['terminated'] and not bool((seq[:-1]==eos).any())
            and t['reference_sha256']==r['reference_sha256'],'Native codec/EOS sequence differs')
        require(torch.equal(t['input_ids'],ids(wrapper,r['text']).cpu()) and torch.equal(t['ref_ids'],ids(wrapper,r['reference_text'],True).cpu()),'Native text IDs differ')
        if r['reference'] not in refs:
            with torch.inference_mode():
                prompt=wrapper.create_voice_clone_prompt(ref_audio=r['reference'],ref_text=r['reference_text'],x_vector_only_mode=False)
                ref,sr=sf.read(r['reference'],dtype='float32');vector=wrapper.model.extract_speaker_embedding(audio=ref,sr=sr).flatten()
            refs[r['reference']]=(prompt[0],vector)
        prompt,ref=refs[r['reference']]
        require(torch.equal(t['ref_codes'],prompt.ref_code.cpu()) and torch.equal(t['ref_xvec'],prompt.ref_spk_embedding.flatten().cpu()),'Native original reference conditioning differs')
        trace_p=dict(p,reference_sha256=DISABLED if stage=='baseline' else p['reference_sha256'])
        shared.validate_trace(r,t,trace_p,'full-pad' if stage=='candidate' else 'control')
        require(r['pad_trace']['special_token_ids_sha256']==tensor_sha256(special) and r['pad_trace']['full_pad_sha256']==tensor_sha256(full)
            and r['pad_trace']['half_pad_sha256']==tensor_sha256(half),'Independent PAD reconstruction differs')
        key=r['case'],r['seed']
        if stage=='baseline' and key in history:score_parity(r,history[key]);historical_checked+=1
        if stage=='candidate':
            source='control' if r['group']=='custom' else 'baseline';b=(control if source=='control' else baseline)[key]
            bt=torch.load(out/source/b['trajectory'],map_location='cpu',weights_only=True)
            shared.paired_prefill(r,t,b,bt)
            if source=='control':custom_paired+=1
            else:
                score_parity(r,b);require(torch.equal(codes,bt['codes']) and torch.equal(t['first_sequence'],bt['first_sequence']),
                    'Inactive development codec trajectory differs');dev_checked+=1
        wave,sr=sf.read(dest/r['wav'],dtype='float32');require(sr==24000 and wave.ndim==1 and len(wave)>0 and np.isfinite(wave).all(),'Invalid audited waveform')
        with torch.inference_mode():
            vector=wrapper.model.extract_speaker_embedding(audio=wave,sr=sr).flatten()
            actual=float(torch.nn.functional.cosine_similarity(vector[None],ref[None]))
        maximum=max(maximum,cosine_error(actual,r['cosine']))
    require(historical_checked==(26 if stage=='baseline' else 0) and dev_checked==(88 if stage=='candidate' else 0)
        and custom_paired==(26 if stage=='candidate' else 0),'Stage parity coverage incomplete')
    truncated=sum(not r['terminated'] for r in records)
    require(maximum<=2e-6 and summary==report['summary'] and truncated==report['truncated'],'Independent stage scores differ')
    unchanged(wrapper,patch,model_identity)
    write(dest/'verification.json',dict(status='verified',stage=stage,recordings=len(records),summary=summary,truncated=truncated,
        confirmation_pass=stage=='candidate' and goal_screen(summary) and truncated==0,goal_achieved=False,global_model_selection_changed=False,
        optimizer_updates=0,all_weights_unchanged=True,model_identity=model_identity,native_conditioning_verified=True,
        pad_traces_verified=len(records),historical_screen_reproduced=historical_checked,inactive_development_exact=dev_checked,
        custom_prefill_and_first_frame_verified=custom_paired,all_recomputed_cosines_finite=True,maximum_cosine_recomputation_error=maximum,
        results_sha256=digest(dest/'results.json'),protocol_sha256=digest(out/'protocol.json'),
        prior_audit_sha256={name:digest(out/name/'verification.json') for name in prior}))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=['prepare','generate','score','audit'])
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--initial',type=Path);parser.add_argument('--previous',type=Path)
    parser.add_argument('--driver',type=Path);parser.add_argument('--stage',choices=STAGES);args=parser.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if args.phase=='prepare':prepare(args.initial,args.previous,args.out,args.driver);return
    p=read(args.out/'protocol.json');os.environ['NUMBA_CACHE_DIR']=p['config']['numba_cache'];os.environ['TEMP']=os.environ['TMP']=p['config']['tmp']
    os.environ['HF_HUB_OFFLINE']=os.environ['TRANSFORMERS_OFFLINE']='1'
    {'generate':generate,'score':score,'audit':audit}[args.phase](args.out,args.stage)


if __name__=='__main__':main()
