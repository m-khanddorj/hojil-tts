"""Complete114 confirmation for a terminal, independently audited DPO screen pass.

No fitting, checkpoint search or promotion occurs here. CPU generation and audit
are separated from GPU ASR. Original screening waveforms must reproduce exactly.
"""
import argparse
import math
from pathlib import Path
import numpy as np
import torch
from .engine import read,write,save,digest,ids
from .policy import external
from . import reference_profile_preference_training as trainer
from . import reference_profile_preference_evaluation as evaluation
from . import reference_profile_collection as previous
from . import direct_voice_training as direct
from .confirm_joint_candidate import validate_panel,validate_records
from .targeted_voice import aggregate
from .voice_strength_search import goal_screen


def sources():
    result=trainer.sources()
    for name in ('reference_profile_preference_confirmation.py','confirm_joint_candidate.py','joint_confirmation.py'):
        result[name]=digest(Path(__file__).with_name(name))
    result['test_training_reference_profile_preference_confirmation.py']=digest(Path(__file__).parent.parent/'tests/test_training_reference_profile_preference_confirmation.py')
    return result


def validate_entry(cases,reports,audits,receipt,step):
    """Require all50 independent measurements and an actual strict screen pass."""
    if (receipt['status']!='verified' or receipt['step']!=step or receipt['recordings']!=50
            or not receipt['all_weights_unchanged'] or not receipt['inactive_development_exact_parity']
            or not receipt['goal_screen'] or (step==0 and not receipt['all_baseline_waveforms_exact'])):
        raise ValueError('Complete50 audited checkpoint is required')
    combined=[]
    for kind,count in (('probes',24),('screen',26)):
        report,audit=reports[kind],audits[kind];records=report['records']
        if len(cases[kind])!=count or len(records)!=count:raise ValueError('Complete50 partition coverage is required')
        summary=evaluation.validate_records(cases[kind],records);truncated=sum(not r['terminated'] for r in records)
        error=audit['maximum_cosine_recomputation_error']
        if (audit['status']!='verified' or audit['step']!=step or audit['kind']!=kind or audit['recordings']!=count
                or not audit['all_weights_unchanged'] or not math.isfinite(error) or error>2e-6
                or summary!=report['summary'] or summary!=audit['summary']
                or report['truncated']!=truncated or audit['truncated']!=truncated
                or audit['baseline_waveforms_checked']!=(count if step==0 else 20 if kind=='screen' else 0)):
            raise ValueError('Independent partition audit differs')
        combined.extend(records)
    screen=reports['screen'];screen_summary=screen['summary']
    if (receipt['summary']!=aggregate(combined) or receipt['screen_summary']!=screen_summary
            or receipt['truncated']!=sum(not r['terminated'] for r in combined)
            or receipt['screen_truncated']!=screen['truncated'] or screen['truncated']
            or not goal_screen(screen_summary)):
        raise ValueError('Strict complete26 joint screening thresholds are not met')
    return screen_summary


def confirmation_measurement(panel,records,screen_records):
    result=validate_records(panel,records)
    if len(screen_records)!=26 or len({(r['case'],r['seed']) for r in screen_records})!=26:
        raise ValueError('Complete26 screening parity required')
    current={(r['case'],r['seed']):r for r in records}
    for r in screen_records:
        key=r['case'],r['seed']
        if key not in current or current[key]['audio_sha256']!=r['audio_sha256']:
            raise ValueError('Screening waveform parity failed')
    return dict(result,screening_recordings_reproduced_exactly=26)


def cosine_error(recomputed,stored):
    if not math.isfinite(recomputed) or not math.isfinite(stored):
        raise ValueError('Nonfinite independent speaker measurement')
    return abs(recomputed-stored)


def evidence(run):
    from .evaluate import ASR_REV
    from scripts.score_latest_audio import score as score_text
    p,e,baseline=evaluation.verify(run)
    status_path=run.parent/'reference-profile-preference-training-status.json';status=read(status_path)
    if status['status']!='complete' or status['reason']!='needs_broader_confirmation':
        raise ValueError('DPO driver must finish with a joint screen pass first')
    step=status['step']
    if step not in p['checkpoints']:raise ValueError('Unregistered terminal checkpoint')
    history=[];paths=[status_path,run/'protocol.json',run/'evaluation-protocol.json',run/'progression.json']
    selected_reports=None
    for current in p['checkpoints'][:p['checkpoints'].index(step)+1]:
        final_path=run/f'verification-{current:06d}.json';receipt=read(final_path)
        checkpoint=trainer.checkpoint_path(run,current);trainer.read_checkpoint(run,current,p)
        identity=digest(checkpoint);eval_identity=digest(run/'evaluation-protocol.json')
        if (receipt['status']!='verified' or receipt['recordings']!=50 or receipt['step']!=current
                or receipt['checkpoint_sha256']!=identity or receipt['evaluation_protocol_sha256']!=eval_identity
                or not receipt['all_weights_unchanged'] or not receipt['inactive_development_exact_parity']
                or (current==0 and not receipt['all_baseline_waveforms_exact'])):
            raise ValueError('Checkpoint history lacks complete independent audits')
        reports={};audits={}
        for kind,count in (('probes',24),('screen',26)):
            dest=evaluation.directory(run,current,kind);audit_path=dest/'verification.json';report_path=dest/'results.json'
            part=read(audit_path);report=read(report_path)
            if (digest(audit_path)!=receipt['partition_audit_sha256'][kind] or part['status']!='verified'
                    or part['recordings']!=count or part['checkpoint_sha256']!=identity
                    or part['evaluation_protocol_sha256']!=eval_identity or part['results_sha256']!=digest(report_path)
                    or part['summary']!=evaluation.validate_records(e['cases'][kind],report['records'])):
                raise ValueError('Checkpoint partition evidence changed')
            reports[kind]=report;audits[kind]=part;paths.extend((audit_path,report_path))
            if current==step:
                originals={(r['case'],r['seed']):r for r in baseline[kind]}
                for r in report['records']:
                    if (r['checkpoint_sha256']!=identity or r['evaluation_protocol_sha256']!=eval_identity
                            or r['asr_revision']!=ASR_REV or r['decoding']!=p['decoding']
                            or digest(dest/r['wav'])!=r['audio_sha256'] or digest(dest/r['trajectory'])!=r['trajectory_sha256']
                            or digest(r['reference'])!=r['reference_sha256'] or r['score']!=score_text(r['text'],r['hypothesis'],True)):
                        raise ValueError('Selected checkpoint speech artifact changed')
                    if (current==0 or r['group']=='development') and r['audio_sha256']!=originals[r['case'],r['seed']]['audio_sha256']:
                        raise ValueError('Selected checkpoint baseline waveform changed')
        if current==step:
            validate_entry(e['cases'],reports,audits,receipt,current);selected_reports=reports
        history.append(receipt);paths.extend((final_path,checkpoint,checkpoint.with_suffix('.json')))
    decision=evaluation.progression(history)
    if (not decision['stop'] or decision['reason']!='needs_broader_confirmation' or decision['step']!=step
            or read(run/'progression.json')!=decision):raise ValueError('Terminal checkpoint differs from registered stopping decision')
    old=previous.verify(Path(p['previous']));panel_path=Path(old['panel']);panel=read(panel_path);validate_panel(panel)
    if (panel['generation']!=p['generation'] or panel['decoding']!=p['decoding'] or panel['seeds']!=old['control_seeds']
            or panel['preparation_source_sha256']!=digest(Path(__file__).with_name('joint_confirmation.py'))):
        raise ValueError('Original114 generation settings changed')
    expanded={(case['case'],seed):dict(case,seed=seed) for seed in panel['seeds'] for case in panel['cases']}
    for case in e['cases']['screen']:
        full=expanded.get((case['case'],case['seed']))
        if full is None or any(full.get(k)!=v for k,v in case.items()):raise ValueError('Original screen is not preserved in114 panel')
    paths.append(panel_path)
    return p,panel,step,{str(path):digest(path) for path in paths},selected_reports['screen']['records'],panel_path


def prepare(run,out,driver):
    p,panel,step,hashes,_,panel_path=evidence(run)
    asr_manifest_sha256=digest(Path(p['config']['asr'])/'download-manifest.json')
    tests=out.parent/'reference-profile-preference-confirmation-tests.log';log=tests.read_text(encoding='utf-8-sig')
    if 'Ran 4 tests' not in log or '\nOK\n' not in log or 'skipped' in log.lower():raise ValueError('Confirmation tests missing')
    out=external(out.resolve());out.mkdir(exist_ok=False)
    write(out/'protocol.json',dict(training_run=str(run),step=step,evidence_hashes=hashes,source=sources(),
        runtime=trainer.collection.direct.initial.collection.runtime(),numeric_config=previous.numeric_config(),
        panel=str(panel_path),panel_sha256=digest(panel_path),checkpoint_sha256=digest(trainer.checkpoint_path(run,step)),
        cases=[dict(case,seed=seed) for seed in panel['seeds'] for case in panel['cases']],
        generation=p['generation'],decoding=p['decoding'],config=p['config'],asr_manifest_sha256=asr_manifest_sha256,
        tests=str(tests),tests_sha256=digest(tests),driver=str(driver),driver_sha256=digest(driver),
        scope='Original114 recordings:88development and26custom, bothfixedseeds, originalreferences, all26screenwaveforms exact',
        goal_achieved=False,global_model_selection_changed=False))


def verify(out):
    p=read(out/'protocol.json')
    if digest(Path(p['config']['asr'])/'download-manifest.json')!=p['asr_manifest_sha256']:
        raise ValueError('Registered ASR manifest changed')
    old,panel,step,hashes,screen,panel_path=evidence(Path(p['training_run']))
    if (p['source']!=sources() or p['runtime']!=trainer.collection.direct.initial.collection.runtime()
            or p['numeric_config']!=previous.numeric_config() or previous.numeric_config()!=dict(cpu_threads=4,matmul_tf32=False,cudnn_tf32=False)
            or p['evidence_hashes']!=hashes or p['step']!=step or p['panel']!=str(panel_path) or p['panel_sha256']!=digest(panel_path)
            or p['checkpoint_sha256']!=digest(trainer.checkpoint_path(Path(p['training_run']),step))
            or digest(p['tests'])!=p['tests_sha256'] or digest(p['driver'])!=p['driver_sha256']):
        raise ValueError('Confirmation execution registration changed')
    if p['cases']!=[dict(case,seed=seed) for seed in panel['seeds'] for case in panel['cases']]:raise ValueError('Full114 scope changed')
    if any(p[k]!=old[k] for k in ('generation','decoding','config')):raise ValueError('Screened generation settings changed')
    return p,panel,screen


def load(out):
    p,panel,screen=verify(out)
    _,wrapper,profile,state,checkpoint=trainer.load(Path(p['training_run']),p['step'],'cpu')
    wrapper.model.eval()
    return p,panel,screen,wrapper,profile,state,checkpoint


def generate(out):
    import soundfile as sf
    from .preference_rollouts import capture_generation
    p,panel,screen,wrapper,profile,state,checkpoint=load(out);dest=out/'speech';dest.mkdir(exist_ok=False)
    records=[];prompts={};original={(r['case'],r['seed']):r for r in screen}
    for c in p['cases']:
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
            cosine=cosine,seconds=len(wave)/sr,terminated=captured['terminated'],checkpoint_sha256=p['checkpoint_sha256'],
            protocol_sha256=digest(out/'protocol.json'))
        key=c['case'],c['seed']
        if key in original and row['audio_sha256']!=original[key]['audio_sha256']:raise ValueError('Screening waveform parity failed')
        write(path.with_suffix('.json'),row);records.append(row);print(f'confirmation: {len(records)}/114',flush=True)
    trainer.unchanged(wrapper.model,profile,state,checkpoint['profile'])
    write(dest/'generated.json',dict(records=records,all_weights_unchanged=True,screening_recordings_reproduced_exactly=26))


def score(out):
    import soundfile as sf
    import torchaudio
    from transformers import WhisperForConditionalGeneration,WhisperProcessor
    from .evaluate import ASR_REV
    from scripts.score_latest_audio import score as score_text
    p,panel,screen=verify(out);dest=out/'speech'
    if (dest/'results.json').exists():raise ValueError('Do not overwrite scored confirmation')
    generated=read(dest/'generated.json');records=generated['records']
    if not generated['all_weights_unchanged'] or generated['screening_recordings_reproduced_exactly']!=26:raise ValueError('Generation controls missing')
    asr=Path(p['config']['asr']);manifest=read(asr/'download-manifest.json')
    if manifest['repo']!='BuzzASR/mongolian' or manifest['revision']!=ASR_REV:raise ValueError('ASR mismatch')
    for name,item in manifest['files'].items():
        if digest(asr/name)!=item['sha256']:raise ValueError('ASR weights changed')
    processor=WhisperProcessor.from_pretrained(asr,local_files_only=True)
    model=WhisperForConditionalGeneration.from_pretrained(asr,torch_dtype=torch.float32,local_files_only=True).to('cuda').eval()
    for r in records:
        if digest(dest/r['wav'])!=r['audio_sha256']:raise ValueError('Audio changed')
        wave,sr=sf.read(dest/r['wav'],dtype='float32');wave=torchaudio.functional.resample(torch.from_numpy(wave),sr,16000).numpy()
        batch=processor(wave,sampling_rate=16000,return_tensors='pt',return_attention_mask=True).to('cuda')
        with torch.inference_mode():tokens=model.generate(batch.input_features,attention_mask=batch.attention_mask,**p['decoding'])
        hypothesis=processor.batch_decode(tokens,skip_special_tokens=True)[0].strip()
        r.update(hypothesis=hypothesis,score=score_text(r['text'],hypothesis,True),decoding=p['decoding'],asr_revision=ASR_REV)
    measured=confirmation_measurement(panel,records,screen)
    write(dest/'results.json',dict(records=records,summary=measured['summary'],truncated=measured['truncated']))


def audit(out):
    import soundfile as sf
    from .evaluate import ASR_REV
    from scripts.score_latest_audio import score as score_text
    if (out/'verification.json').exists():raise ValueError('Do not overwrite independent confirmation audit')
    p,panel,screen,wrapper,profile,state,checkpoint=load(out);dest=out/'speech';report=read(dest/'results.json');records=report['records']
    measured=confirmation_measurement(panel,records,screen);expected={(r['case'],r['seed']):r for r in p['cases']};prompts={};error=0.
    for r in records:
        if any(r[k]!=v for k,v in expected[r['case'],r['seed']].items()):raise ValueError('Full confirmation metadata changed')
        if (r['checkpoint_sha256']!=p['checkpoint_sha256'] or r['protocol_sha256']!=digest(out/'protocol.json')
                or r['asr_revision']!=ASR_REV or r['decoding']!=p['decoding'] or digest(dest/r['wav'])!=r['audio_sha256']
                or digest(r['reference'])!=r['reference_sha256'] or r['score']!=score_text(r['text'],r['hypothesis'],True)):
            raise ValueError('Confirmation speech provenance or edit counts changed')
        if r['reference'] not in prompts:
            with torch.inference_mode():prompts[r['reference']]=wrapper.create_voice_clone_prompt(
                ref_audio=r['reference'],ref_text=r['reference_text'],x_vector_only_mode=False)
        prompt=prompts[r['reference']][0]
        t=direct.trajectory(dest,r);codes=t['codes'];seq=t['first_sequence'];eos=wrapper.model.config.talker_config.codec_eos_token_id
        if (t['reference_sha256']!=r['reference_sha256'] or codes.ndim!=2 or codes.shape[1]!=16
                or t['terminated']!=r['terminated'] or bool(seq[0,-1]==eos)!=r['terminated']):raise ValueError('Confirmation trajectory coverage differs')
        if r['terminated'] and not torch.equal(seq[0,-(len(codes)+1):-1],codes[:,0]):raise ValueError('Main sampled codes differ')
        if (not torch.equal(t['input_ids'],ids(wrapper,r['text']).cpu()) or not torch.equal(t['ref_ids'],ids(wrapper,r['reference_text'],True).cpu())
                or not torch.equal(t['ref_codes'],prompt.ref_code.detach().cpu())
                or not torch.equal(t['ref_xvec'].flatten(),prompt.ref_spk_embedding.detach().cpu().flatten())):
            raise ValueError('Native reference/target conditioning differs')
        wave,sr=sf.read(dest/r['wav'],dtype='float32')
        with torch.inference_mode():
            vector=wrapper.model.extract_speaker_embedding(audio=wave,sr=sr).flatten();reference=prompt.ref_spk_embedding.flatten()
            cosine=float(torch.nn.functional.cosine_similarity(vector[None],reference[None]))
        error=max(error,cosine_error(cosine,r['cosine']))
    if error>2e-6 or report['summary']!=measured['summary'] or report['truncated']!=measured['truncated']:
        raise ValueError('Independent confirmation scores differ')
    trainer.unchanged(wrapper.model,profile,state,checkpoint['profile']);verify(out)
    write(out/'verification.json',dict(status='verified',recordings=114,step=p['step'],**measured,
        maximum_cosine_recomputation_error=error,all_weights_unchanged=True,native_conditioning_verified=True,
        checkpoint_sha256=p['checkpoint_sha256'],protocol_sha256=digest(out/'protocol.json'),results_sha256=digest(dest/'results.json'),
        goal_achieved=False,global_model_selection_changed=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=['prepare','generate','score','audit'])
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--run',type=Path);parser.add_argument('--driver',type=Path)
    args=parser.parse_args();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if args.phase=='prepare':prepare(args.run,args.out,args.driver)
    else:{'generate':generate,'score':score,'audit':audit}[args.phase](args.out)


if __name__=='__main__':main()
