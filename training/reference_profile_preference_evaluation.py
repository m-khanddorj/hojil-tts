"""Complete checkpoint speech evaluation for reference-specific preference fitting.

Training probe16 and reserved8 use the collection GPU backend. The registered
development/custom screen26 remains CPU FP32. Every checkpoint audits all50.
"""
import argparse
import math
from pathlib import Path
import numpy as np
import torch
from .engine import read,write,save,digest,ids
from . import reference_profile_preference_training as trainer
from . import reference_profile_collection as previous
from . import reference_profile_preference_collection as collection
from . import direct_voice_training as direct
from .targeted_voice import aggregate
from .voice_strength_search import goal_screen

CONDITION=('case','speaker_id','group','text','reference','reference_sha256','reference_text','seed')


def source_hashes():
    return {name:digest(Path(__file__).parent/name) for name in
        ('reference_profile_preference_evaluation.py','../tests/test_training_reference_profile_preference_evaluation.py')}


def baseline_data(p):
    pool=Path(p['pool']);old=Path(p['previous']);cp=collection.verify(pool);op=previous.verify(old)
    paths=dict(training=pool/'baseline/results.json',reserved=old/'baseline/results.json',
        screen=Path(op['initial_run'])/'eval-000000/results.json')
    first_seed=cp['settings']['seeds'][0]
    if first_seed!=41301:raise ValueError('Registered training probe seed changed')
    fit=[r for r in read(paths['training'])['records'] if r['seed']==first_seed]
    reserved=[r for r in read(paths['reserved'])['records'] if r['group']=='profile-validation']
    screen=read(paths['screen'])['records']
    if (len(fit)!=16 or len({r['case'] for r in fit})!=16 or any(r['group']!='profile-preference-train' for r in fit)
            or len(reserved)!=8 or len({r['case'] for r in reserved})!=4 or len(screen)!=26):
        raise ValueError('Complete fixed16/8/26 baseline required')
    expected={(c['case'],s) for c in op['control_cases'] for s in op['control_seeds']}
    if {(r['case'],r['seed']) for r in screen}!=expected:raise ValueError('Original screen coverage changed')
    audit_paths=[pool/'verification.json',old/'verification.json',Path(op['initial_run'])/'verification-000000.json']
    for path in audit_paths:
        if read(path)['status']!='verified':raise ValueError('Baseline independent audit missing')
    hashes={str(path):digest(path) for path in list(paths.values())+audit_paths}
    records=dict(probes=fit+reserved,screen=screen)
    for kind,rows in records.items():
        for r in rows:
            path=paths['screen'] if kind=='screen' else paths['training' if r['group']=='profile-preference-train' else 'reserved']
            if digest(path.parent/r['wav'])!=r['audio_sha256']:raise ValueError('Baseline waveform changed')
    return records,hashes


def prepare(run):
    p=trainer.verify(run);records,hashes=baseline_data(p);path=run/'evaluation-protocol.json'
    if path.exists():raise ValueError('Evaluation is already registered')
    tests=run.parent/'reference-profile-preference-evaluation-tests.log';log=tests.read_text(encoding='utf-8-sig')
    if 'Ran 4 tests' not in log or '\nOK\n' not in log or 'skipped' in log.lower():raise ValueError('Evaluation tests missing')
    write(path,dict(training_protocol_sha256=digest(run/'protocol.json'),source=source_hashes(),baseline_hashes=hashes,
        cases={kind:[{k:r[k] for k in CONDITION} for r in rows] for kind,rows in records.items()},
        tests=str(tests),tests_sha256=digest(tests),checkpoints=p['checkpoints'],
        numeric_config=previous.numeric_config(),
        scope='Every checkpoint:16 fixed training-probe outputs,8 reserved outputs,26 unchanged two-group screen outputs',
        stopping='Stop for confirmation on a strict joint screen pass; otherwise stop after two consecutive checkpoints are custom-CER/cosine Pareto-dominated by an earlier evaluated checkpoint',
        goal_achieved=False,global_model_selection_changed=False))


def verify(run):
    p=trainer.verify(run);e=read(run/'evaluation-protocol.json');baseline,hashes=baseline_data(p)
    expected={kind:[{k:r[k] for k in CONDITION} for r in rows] for kind,rows in baseline.items()}
    if (e['training_protocol_sha256']!=digest(run/'protocol.json') or e['source']!=source_hashes()
            or e['baseline_hashes']!=hashes or e['cases']!=expected or e['checkpoints']!=p['checkpoints']
            or e['numeric_config']!=previous.numeric_config()
            or previous.numeric_config()!=dict(cpu_threads=4,matmul_tf32=False,cudnn_tf32=False)
            or digest(e['tests'])!=e['tests_sha256']):raise ValueError('Evaluation registration changed')
    return p,e,baseline


def directory(run,step,kind):return run/f'eval-{step:06d}'/kind


def permitted(run,p,step):
    if step not in p['checkpoints']:raise ValueError('Unregistered evaluation checkpoint')
    if step:
        baseline=read(run/'verification-000000.json')
        if baseline['status']!='verified' or baseline['recordings']!=50 or not baseline['all_baseline_waveforms_exact']:
            raise ValueError('All50 baseline controls must verify first')


def validate_records(cases,records):
    expected={(r['case'],r['seed']):r for r in cases}
    if len(records)!=len(expected) or {(r['case'],r['seed']) for r in records}!=set(expected):
        raise ValueError('Incomplete or duplicate checkpoint speech')
    for r in records:
        if any(r[k]!=expected[r['case'],r['seed']][k] for k in CONDITION):raise ValueError('Checkpoint speech condition changed')
        if not math.isfinite(r['cosine']) or not -1<=r['cosine']<=1:raise ValueError('Invalid cosine')
    return aggregate(records)


def paired_changes(before,after):
    old={(r['case'],r['seed']):r for r in before}
    return [dict(case=r['case'],seed=r['seed'],group=r['group'],
        character_errors_before=old[r['case'],r['seed']]['score']['character_errors'],character_errors_after=r['score']['character_errors'],
        word_errors_before=old[r['case'],r['seed']]['score']['word_errors'],word_errors_after=r['score']['word_errors'],
        cosine_before=old[r['case'],r['seed']]['cosine'],cosine_after=r['cosine'],
        cosine_delta=r['cosine']-old[r['case'],r['seed']]['cosine'],terminated=r['terminated']) for r in after]


def generate(run,step,kind):
    import soundfile as sf
    from .preference_rollouts import capture_generation
    p,e,baseline=verify(run);permitted(run,p,step)
    if kind not in ('probes','screen'):raise ValueError('Unknown speech partition')
    _,wrapper,profile,state,checkpoint=trainer.load(run,step,'cpu' if kind=='screen' else 'cuda')
    wrapper.model.eval();identity=digest(run/f'checkpoint-{step:06d}.pt')
    dest=directory(run,step,kind);dest.mkdir(parents=True,exist_ok=False);records=[];prompts={}
    for c in e['cases'][kind]:
        if digest(c['reference'])!=c['reference_sha256']:raise ValueError('Reference changed')
        if c['reference'] not in prompts:
            with torch.inference_mode():prompts[c['reference']]=wrapper.create_voice_clone_prompt(
                ref_audio=c['reference'],ref_text=c['reference_text'],x_vector_only_mode=False)
        prompt=prompts[c['reference']];torch.manual_seed(c['seed']);np.random.seed(c['seed'])
        with profile.use_reference(c['reference_sha256']),torch.inference_mode(),capture_generation(wrapper.model) as captured:
            waves,sr=wrapper.generate_voice_clone(text=c['text'],language='Auto',voice_clone_prompt=prompt,
                non_streaming_mode=True,**p['generation'])
        if sr!=24000 or not len(waves[0]) or not np.isfinite(waves[0]).all() or len(waves[0])/sr>30:raise ValueError('Invalid generated speech')
        path=dest/f"{c['seed']}-{c['case']}.wav";sf.write(path,waves[0],sr,subtype='PCM_16');wave,sr=sf.read(path,dtype='float32')
        with torch.inference_mode():
            vector=wrapper.model.extract_speaker_embedding(audio=wave,sr=sr).flatten();ref=prompt[0].ref_spk_embedding.flatten()
            cosine=float(torch.nn.functional.cosine_similarity(vector[None],ref[None]))
        trajectory=path.with_suffix('.pt');save(trajectory,dict(codes=captured['codes'],first_sequence=captured['first_sequence'],
            terminated=captured['terminated'],ref_codes=prompt[0].ref_code.detach().cpu(),ref_xvec=ref.detach().cpu(),
            input_ids=ids(wrapper,c['text']).cpu(),ref_ids=ids(wrapper,c['reference_text'],True).cpu(),reference_sha256=c['reference_sha256']))
        row=dict(c,wav=path.name,audio_sha256=digest(path),trajectory=trajectory.name,trajectory_sha256=digest(trajectory),
            cosine=cosine,seconds=len(wave)/sr,terminated=captured['terminated'],checkpoint_sha256=identity,
            evaluation_protocol_sha256=digest(run/'evaluation-protocol.json'))
        write(path.with_suffix('.json'),row);records.append(row);print(f'{step}/{kind}: {len(records)}/{len(e["cases"][kind])}',flush=True)
    trainer.unchanged(wrapper.model,profile,state,checkpoint['profile'])
    old={(r['case'],r['seed']):r for r in baseline[kind]}
    checked=[r for r in records if step==0 or r['group']=='development']
    if any(r['audio_sha256']!=old[r['case'],r['seed']]['audio_sha256'] for r in checked):raise ValueError('Required baseline waveform parity failed')
    write(dest/'generated.json',dict(records=records,all_weights_unchanged=True,baseline_waveforms_checked=len(checked)))


def score(run,step):
    import soundfile as sf
    import torchaudio
    from transformers import WhisperForConditionalGeneration,WhisperProcessor
    from .evaluate import ASR_REV
    from scripts.score_latest_audio import score as score_text
    p,e,_=verify(run);permitted(run,p,step);asr=Path(p['config']['asr']);manifest=read(asr/'download-manifest.json')
    if manifest['repo']!='BuzzASR/mongolian' or manifest['revision']!=ASR_REV:raise ValueError('ASR mismatch')
    for name,item in manifest['files'].items():
        if digest(asr/name)!=item['sha256']:raise ValueError('ASR weights changed')
    processor=WhisperProcessor.from_pretrained(asr,local_files_only=True)
    model=WhisperForConditionalGeneration.from_pretrained(asr,torch_dtype=torch.float32,local_files_only=True).to('cuda').eval()
    for kind in ('probes','screen'):
        dest=directory(run,step,kind)
        if (dest/'results.json').exists():raise ValueError('Do not overwrite scored speech')
        generated=read(dest/'generated.json');records=generated['records']
        if not generated['all_weights_unchanged']:raise ValueError('Generation weights changed')
        for r in records:
            if digest(dest/r['wav'])!=r['audio_sha256']:raise ValueError('Audio changed')
            wave,sr=sf.read(dest/r['wav'],dtype='float32');wave=torchaudio.functional.resample(torch.from_numpy(wave),sr,16000).numpy()
            batch=processor(wave,sampling_rate=16000,return_tensors='pt',return_attention_mask=True).to('cuda')
            with torch.inference_mode():tokens=model.generate(batch.input_features,attention_mask=batch.attention_mask,**p['decoding'])
            hypothesis=processor.batch_decode(tokens,skip_special_tokens=True)[0].strip()
            r.update(hypothesis=hypothesis,score=score_text(r['text'],hypothesis,True),decoding=p['decoding'],asr_revision=ASR_REV)
        summary=validate_records(e['cases'][kind],records)
        write(dest/'results.json',dict(records=records,summary=summary,truncated=sum(not r['terminated'] for r in records)))


def audit(run,step,kind):
    import soundfile as sf
    from .evaluate import ASR_REV
    from scripts.score_latest_audio import score as score_text
    p,e,baseline=verify(run);permitted(run,p,step);dest=directory(run,step,kind)
    if (dest/'verification.json').exists():raise ValueError('Do not overwrite independent audit')
    _,wrapper,profile,state,checkpoint=trainer.load(run,step,'cpu' if kind=='screen' else 'cuda')
    report=read(dest/'results.json');records=report['records'];summary=validate_records(e['cases'][kind],records)
    identity=digest(run/f'checkpoint-{step:06d}.pt');old={(r['case'],r['seed']):r for r in baseline[kind]};refs={};error=0.;parity=0
    for r in records:
        if (r['checkpoint_sha256']!=identity or r['evaluation_protocol_sha256']!=digest(run/'evaluation-protocol.json')
                or r['decoding']!=p['decoding'] or r['asr_revision']!=ASR_REV or digest(dest/r['wav'])!=r['audio_sha256']
                or digest(r['reference'])!=r['reference_sha256']):raise ValueError('Checkpoint speech provenance changed')
        if r['score']!=score_text(r['text'],r['hypothesis'],True):raise ValueError('ASR edit counts differ')
        t=direct.trajectory(dest,r);codes=t['codes'];seq=t['first_sequence'];eos=wrapper.model.config.talker_config.codec_eos_token_id
        if (t['reference_sha256']!=r['reference_sha256'] or codes.ndim!=2 or codes.shape[1]!=16
                or bool(seq[0,-1]==eos)!=r['terminated']):raise ValueError('Trajectory coverage differs')
        if r['terminated'] and not torch.equal(seq[0,-(len(codes)+1):-1],codes[:,0]):raise ValueError('Main codes differ')
        if not torch.equal(t['input_ids'],ids(wrapper,r['text']).cpu()) or not torch.equal(t['ref_ids'],ids(wrapper,r['reference_text'],True).cpu()):
            raise ValueError('Trajectory transcript tokens differ')
        wave,sr=sf.read(dest/r['wav'],dtype='float32')
        with torch.inference_mode():
            if r['reference'] not in refs:
                ref,refsr=sf.read(r['reference'],dtype='float32');refs[r['reference']]=wrapper.model.extract_speaker_embedding(audio=ref,sr=refsr).flatten()
            vector=wrapper.model.extract_speaker_embedding(audio=wave,sr=sr).flatten()
            cosine=float(torch.nn.functional.cosine_similarity(vector[None],refs[r['reference']][None]))
        error=max(error,abs(cosine-r['cosine']))
        if step==0 or r['group']=='development':
            b=old[r['case'],r['seed']]
            if r['audio_sha256']!=b['audio_sha256'] or r['score']!=b['score'] or abs(r['cosine']-b['cosine'])>2e-6:
                raise ValueError('Baseline waveform/score control differs')
            parity+=1
    if error>2e-6 or summary!=report['summary'] or report['truncated']!=sum(not r['terminated'] for r in records):raise ValueError('Independent scores differ')
    trainer.unchanged(wrapper.model,profile,state,checkpoint['profile'])
    receipt=dict(status='verified',step=step,kind=kind,recordings=len(records),summary=summary,truncated=report['truncated'],
        maximum_cosine_recomputation_error=error,all_weights_unchanged=True,baseline_waveforms_checked=parity,
        checkpoint_sha256=identity,results_sha256=digest(dest/'results.json'),evaluation_protocol_sha256=digest(run/'evaluation-protocol.json'),
        paired_changes=paired_changes(baseline[kind],records),goal_achieved=False)
    write(dest/'verification.json',receipt)


def finalize(run,step):
    p,e,_=verify(run);permitted(run,p,step);path=run/f'verification-{step:06d}.json'
    if path.exists():raise ValueError('Do not overwrite checkpoint verification')
    receipts={};reports={};records=[]
    for kind,count in (('probes',24),('screen',26)):
        dest=directory(run,step,kind);a=read(dest/'verification.json');report=read(dest/'results.json')
        if (a['status']!='verified' or a['step']!=step or a['recordings']!=count or not a['all_weights_unchanged']
                or a['results_sha256']!=digest(dest/'results.json') or a['summary']!=validate_records(e['cases'][kind],report['records'])
                or a['checkpoint_sha256']!=digest(run/f'checkpoint-{step:06d}.pt')
                or a['baseline_waveforms_checked']!=(count if step==0 else 20 if kind=='screen' else 0)):
            raise ValueError('Complete independent checkpoint audits required')
        receipts[kind]=digest(dest/'verification.json');reports[kind]=report;records.extend(report['records'])
    summary=aggregate(records);screen=reports['screen'];truncated=sum(not r['terminated'] for r in records)
    write(path,dict(status='verified',step=step,recordings=50,summary=summary,screen_summary=screen['summary'],
        truncated=truncated,screen_truncated=screen['truncated'],goal_screen=goal_screen(screen['summary']) and screen['truncated']==0,
        all_weights_unchanged=True,all_baseline_waveforms_exact=step==0,
        inactive_development_exact_parity=True,partition_audit_sha256=receipts,
        checkpoint_sha256=digest(run/f'checkpoint-{step:06d}.pt'),evaluation_protocol_sha256=digest(run/'evaluation-protocol.json'),
        goal_achieved=False,global_model_selection_changed=False))


def progression(history):
    """Registered stopping rule: two consecutive custom Pareto regressions."""
    if not history or [r['step'] for r in history]!=[0,4,8,16][:len(history)]:
        raise ValueError('Complete ordered checkpoint history beginning at zero required')
    warnings=0;decisions=[]
    for i,current in enumerate(history):
        if current['status']!='verified' or current['recordings']!=50 or not current['inactive_development_exact_parity']:
            raise ValueError('Only complete audited checkpoints may drive stopping')
        summary=current['screen_summary'];passed=goal_screen(summary) and current['screen_truncated']==0
        if current['goal_screen']!=passed:raise ValueError('Stored goal screen differs')
        custom=summary['custom'];dominators=[]
        for earlier in history[:i]:
            old=earlier['screen_summary']['custom']
            if (old['pooled_cer']<=custom['pooled_cer'] and old['speaker_macro_cosine']>=custom['speaker_macro_cosine']
                    and (old['pooled_cer']<custom['pooled_cer'] or old['speaker_macro_cosine']>custom['speaker_macro_cosine'])):
                dominators.append(dict(step=earlier['step'],cer=old['pooled_cer'],cosine=old['speaker_macro_cosine']))
        warnings=warnings+1 if dominators else 0
        decisions.append(dict(step=current['step'],custom_cer=custom['pooled_cer'],custom_cosine=custom['speaker_macro_cosine'],
            dominating_earlier_checkpoints=dominators,consecutive_regressions=warnings,goal_screen=passed))
        if passed:return dict(stop=True,reason='needs_broader_confirmation',step=current['step'],decisions=decisions)
        if warnings>=2:return dict(stop=True,reason='consecutive_pareto_regressions',step=current['step'],decisions=decisions)
    return dict(stop=False,reason='continue',step=history[-1]['step'],decisions=decisions)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase');parser.add_argument('--run',type=Path,required=True);parser.add_argument('--step',type=int)
    args=parser.parse_args();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    phase=args.phase
    if phase=='prepare':prepare(args.run)
    elif phase in ('generate-probes','generate-screen'):generate(args.run,args.step,phase.removeprefix('generate-'))
    elif phase=='score':score(args.run,args.step)
    elif phase in ('audit-probes','audit-screen'):audit(args.run,args.step,phase.removeprefix('audit-'))
    elif phase=='finalize':finalize(args.run,args.step)
    else:raise ValueError('Unknown evaluation phase')


if __name__=='__main__':main()
