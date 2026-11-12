"""Memory-separated, fitting-text-only proposals for one reference profile.

The codec path supplies a biased straight-through direction. Real generated
ASR and voice scores must select a candidate; fixed-token loss is not quality.
"""
import argparse,gc,time
from pathlib import Path
import torch
from .engine import read,write,save,digest
from .policy import external
from . import reference_profile_collection as collection
from . import reference_voice_profile as profiles
from . import direct_voice_training as direct
from . import middle_attention_adapter as attention
from .codec_policy_replay import policy_outputs,backward_probability_gradient,combine_reference_probabilities
from .direct_voice_objective import checkpointed_waveform
from .profile_descent import common_direction,raw_for_displacement
from .preference_objective import deterministic_training

SETTINGS=dict(input_fractions=[.005,.0025,.00125,.000625],anchor_tolerance=1e-6,
    maximum_actual_fitting_trials=4,maximum_clip_cosine_drop=.005,maximum_added_character_errors_per_clip=1)


def fitting_records(p,report):
    expected={(c['case'],s) for c in p['cases'] if c['group']=='profile-fit' for s in p['settings']['seeds']}
    records=[r for r in report['records'] if r['group']=='profile-fit']
    if (len(expected)!=16 or len(records)!=16 or {(r['case'],r['seed']) for r in records}!=expected
            or any(not r['terminated'] for r in records)):raise ValueError('Incomplete/terminated fitting coverage')
    return records


def sources():
    result=collection.sources()
    for name in ('reference_profile_optimization.py','profile_descent.py'):
        result[name]=digest(Path(__file__).with_name(name))
    for name in ('test_training_profile_descent.py','test_training_reference_profile_optimization.py'):
        result[name]=digest(Path(__file__).parent.parent/'tests'/name)
    return result


def prepare(data,out,driver):
    p=collection.verify(data);audit=read(data/'verification.json');report=read(data/'baseline/results.json')
    if audit['status']!='verified' or audit['truncated'] or audit['summary']!=report['summary']:
        raise ValueError('Complete independently audited baseline required')
    records=fitting_records(p,report)
    tests={}
    for name in ('profile-descent-tests.log','reference-profile-optimization-tests.log'):
        path=out.parent/name;log=path.read_text(encoding='utf-8-sig')
        if 'Ran 2 tests' not in log or '\nOK\n' not in log or 'skipped' in log.lower():raise ValueError('Optimization tests missing')
        tests[str(path)]=digest(path)
    out=external(out.resolve());out.mkdir(exist_ok=False)
    write(out/'protocol.json',dict(data=str(data),data_protocol_sha256=digest(data/'protocol.json'),
        data_results_sha256=digest(data/'baseline/results.json'),data_audit_sha256=digest(data/'verification.json'),
        source=sources(),runtime=direct.initial.collection.runtime(),numeric_config=collection.numeric_config(),
        settings=SETTINGS,fitting_conditions=[dict(case=r['case'],seed=r['seed']) for r in records],tests=tests,
        driver=str(driver),driver_sha256=digest(driver),
        objective='Mean generated speaker-loss gradient and mean all-codebook-plus-EOS token-NLL gradient on16 fitting trajectories; normalized common descent, then exact input-displacement candidates',
        limitations='Biased local straight-through surrogate, not GRPO or an unbiased policy gradient; no reference reconstruction or validation-target fitting',
        selection='First feasible candidate in descending input-fraction order that passes matched16-fitting-speech CER/nonregression and voice-improvement gate; reserved8 texts reported only after fitting selection, then complete26 CPU screen',
        validation_used_for_gradients=False,goal_achieved=False,global_model_selection_changed=False))


def verify(out):
    p=read(out/'protocol.json');data=Path(p['data']);c=collection.verify(data)
    if (p['source']!=sources() or p['runtime']!=direct.initial.collection.runtime()
            or p['numeric_config']!=collection.numeric_config() or p['settings']!=SETTINGS
            or digest(p['driver'])!=p['driver_sha256']):raise ValueError('Registered optimization execution changed')
    for path,key in [(data/'protocol.json','data_protocol_sha256'),(data/'baseline/results.json','data_results_sha256'),
                     (data/'verification.json','data_audit_sha256')]:
        if digest(path)!=p[key]:raise ValueError('Registered optimization data changed')
    for path,expected in p['tests'].items():
        if digest(path)!=expected:raise ValueError('Registered test changed')
    if digest(c['profile_reference']['reference'])!=c['profile_reference']['reference_sha256']:
        raise ValueError('Original reference changed')
    records=fitting_records(c,read(data/'baseline/results.json'))
    if p['fitting_conditions']!=[dict(case=r['case'],seed=r['seed']) for r in records]:raise ValueError('Fitting partition changed')
    return p,c,records


def posteriors(out):
    p,c,records=verify(out);data=Path(p['data']);_,wrapper,profile,state=collection.load(data,'cuda');receipts=[]
    for r in records:
        trajectory=direct.trajectory(data/'baseline',r)
        if trajectory['reference_sha256']!=c['profile_reference']['reference_sha256']:raise ValueError('Wrong profile trajectory')
        with profile.use_reference(r['reference_sha256']),torch.no_grad():v=policy_outputs(wrapper.model,trajectory)
        path=out/f"posterior-{r['seed']}-{r['case']}.pt";save(path,v['probabilities'].cpu())
        receipts.append(dict(case=r['case'],seed=r['seed'],path=path.name,sha256=digest(path),
            raw_log_probability=float(v['log_probability']),token_count=v['token_count'],
            main_acoustic_mass_min=float(v['main_acoustic_mass'].min())))
    collection.unchanged(wrapper.model,profile,state)
    write(out/'posteriors.json',dict(records=receipts,data_results_sha256=p['data_results_sha256'],zero_profile=True))


def codec(out):
    from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram
    from .targeted_voice import frozen_digest
    p,c,records=verify(out);data=Path(p['data']);cached=read(out/'posteriors.json')
    if not cached['zero_profile'] or cached['data_results_sha256']!=p['data_results_sha256']:raise ValueError('Stale policy posteriors')
    entries={(r['case'],r['seed']):r for r in cached['records']}
    decoder,speaker=direct.load_codec(direct.verify(Path(c['initial_run'])))
    before=frozen_digest(decoder),frozen_digest(speaker);receipts=[]
    for r in records:
        item=entries[r['case'],r['seed']];path=out/item['path']
        if digest(path)!=item['sha256']:raise ValueError('Posterior changed')
        trajectory=direct.trajectory(data/'baseline',r)
        gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();started=time.monotonic()
        probabilities=torch.load(path,map_location='cpu',weights_only=True).to('cuda').requires_grad_(True)
        codes=torch.cat([trajectory['ref_codes'],trajectory['codes']]).T[None].to('cuda')
        with torch.no_grad():native=decoder.chunked_decode(codes)
        wave=checkpointed_waveform(decoder,codes,combine_reference_probabilities(trajectory['ref_codes'],probabilities))
        if not torch.equal(native,wave.detach()):raise ValueError('Native checkpointed hard waveform mismatch')
        cut=int(len(trajectory['ref_codes'])/codes.shape[-1]*wave.shape[-1])
        mel=mel_spectrogram(wave[0,0,cut:].reshape(1,-1),n_fft=1024,num_mels=128,sampling_rate=24000,
            hop_size=256,win_size=1024,fmin=0,fmax=12000).transpose(1,2)
        vector=speaker(mel).flatten();target=trajectory['ref_xvec'].to('cuda').flatten()
        loss=1-torch.nn.functional.cosine_similarity(vector[None],target[None]).mean()
        gradient,=torch.autograd.grad(loss,probabilities)
        if not torch.isfinite(gradient).all() or gradient.norm()==0:raise ValueError('Invalid probability gradient')
        target_path=out/f"gradient-{r['seed']}-{r['case']}.pt";save(target_path,gradient.cpu())
        receipts.append(dict(case=r['case'],seed=r['seed'],path=target_path.name,sha256=digest(target_path),
            loss=float(loss.detach()),gradient_norm=float(gradient.norm()),seconds=time.monotonic()-started,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            frames=codes.shape[-1],reference_probabilities_constant=True))
        print(receipts[-1],flush=True)
        del probabilities,codes,native,wave,mel,vector,target,loss,gradient
    if (frozen_digest(decoder),frozen_digest(speaker))!=before or any(v.grad is not None for m in (decoder,speaker) for v in m.parameters()):
        raise ValueError('Frozen codec/scorer changed or received a gradient')
    write(out/'gradients.json',dict(records=receipts,posteriors_sha256=digest(out/'posteriors.json'),frozen_weights_unchanged=True))


def mean_nll(model,profile,data,records,backward=False):
    values=[]
    for r in records:
        with profile.use_reference(r['reference_sha256']):
            logp,tokens=attention.log_probability(model,direct.trajectory(data/'baseline',r))
            loss=-logp/tokens;values.append(float(loss.detach()))
            if backward:(loss/len(records)).backward()
    return sum(values)/len(values)


def proposals(out):
    p,c,records=verify(out);data=Path(p['data']);_,wrapper,profile,state=collection.load(data,'cuda');model=wrapper.model
    for decoder in (model.talker.model,model.talker.code_predictor.model):
        decoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    deterministic_training(model)
    cached=read(out/'posteriors.json');grads=read(out/'gradients.json')
    if (cached['data_results_sha256']!=p['data_results_sha256'] or not cached['zero_profile']
            or grads['posteriors_sha256']!=digest(out/'posteriors.json') or not grads['frozen_weights_unchanged']):
        raise ValueError('Gradient provenance changed')
    posts={(r['case'],r['seed']):r for r in cached['records']};gradients={(r['case'],r['seed']):r for r in grads['records']}
    errors=[];model.zero_grad(set_to_none=True)
    for r in records:
        a,b=posts[r['case'],r['seed']],gradients[r['case'],r['seed']]
        if digest(out/a['path'])!=a['sha256'] or digest(out/b['path'])!=b['sha256']:raise ValueError('Gradient cache changed')
        probabilities=torch.load(out/a['path'],map_location='cpu',weights_only=True)
        gradient=torch.load(out/b['path'],map_location='cpu',weights_only=True)
        with profile.use_reference(r['reference_sha256']):
            receipt=backward_probability_gradient(model,direct.trajectory(data/'baseline',r),probabilities,gradient/len(records))
        errors.append(receipt['maximum_probability_replay_error'])
    voice=profile.raw.grad.detach().clone();model.zero_grad(set_to_none=True)
    before=mean_nll(model,profile,data,records,True);anchor=profile.raw.grad.detach().clone();model.zero_grad(set_to_none=True)
    direction,geometry=common_direction(voice,anchor);rows=[]
    reference=direct.trajectory(data/'baseline',records[0])['ref_xvec'].to('cuda')
    with torch.no_grad():baseline_input=profile.base(reference)
    for index,fraction in enumerate(SETTINGS['input_fractions']):
        raw=raw_for_displacement(direction,fraction,profile.radius_fraction)
        with torch.no_grad():
            profile.raw.copy_(raw);after=mean_nll(model,profile,data,records)
            with profile.use_reference(c['profile_reference']['reference_sha256']):
                actual_fraction=float((profile(reference)-baseline_input).norm()/reference.norm())
        if abs(actual_fraction-fraction)>1e-6:raise ValueError('Actual input displacement differs')
        predicted=float(raw.double().dot(voice.double()))
        feasible=after<=before+SETTINGS['anchor_tolerance'] and predicted<0
        row=dict(index=index,input_fraction=fraction,actual_input_fraction=actual_fraction,feasible=feasible,
            anchor_nll_before=before,anchor_nll_after=after,predicted_voice_change=predicted)
        if feasible:
            path=out/f'candidate-{index:02d}.pt';save(path,dict(profile=profiles.snapshot(profile),index=index,
                protocol_sha256=digest(out/'protocol.json')));row.update(path=path.name,sha256=digest(path))
        rows.append(row);write(out/f'proposal-{index:02d}.json',row);print(row,flush=True)
    with torch.no_grad():profile.raw.zero_()
    model.zero_grad(set_to_none=True);collection.unchanged(model,profile,state);verify(out)
    write(out/'results.json',dict(status='profile_proposals_prepared',proposals=rows,geometry=geometry,
        maximum_posterior_replay_error=max(errors),fitting_recordings=len(records),validation_used=False,
        initial_model_restored=True,all_original_weights_unchanged=True,goal_achieved=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=['prepare','posteriors','codec','proposals'])
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--data',type=Path);parser.add_argument('--driver',type=Path)
    args=parser.parse_args();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if args.phase=='prepare':prepare(args.data,args.out,args.driver)
    else:{'posteriors':posteriors,'codec':codec,'proposals':proposals}[args.phase](args.out)


if __name__=='__main__':main()
