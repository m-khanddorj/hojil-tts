"""Complete114 confirmation for a reference profile with an audited joint screen pass."""
import argparse
from pathlib import Path
import numpy as np
import torch
from .engine import read,write,save,digest,ids
from . import reference_profile_optimization as proposal
from . import reference_profile_collection as collection
from . import reference_voice_profile as profiles
from . import direct_voice_training as direct
from .reference_profile_gate import compare
from .reference_profile_mean_selection import select
from . import reference_profile_mean_evaluation as screened
from .confirm_joint_candidate import validate_panel,validate_records
from .targeted_voice import aggregate
from .voice_strength_search import goal_screen

CONDITION=('case','speaker_id','group','text','reference','reference_sha256','reference_text','seed')


def sources():
    result=screened.sources()
    for name in ('reference_profile_confirmation.py','confirm_joint_candidate.py','joint_confirmation.py'):
        result[name]=digest(Path(__file__).with_name(name))
    result['test_training_reference_profile_confirmation.py']=digest(Path(__file__).parent.parent/'tests/test_training_reference_profile_confirmation.py')
    return result


def screen_entry(report,audit,cases):
    records=report['records'];expected={(c['case'],c['seed']):c for c in cases}
    if (len(expected)!=26 or len(records)!=26 or {(r['case'],r['seed']) for r in records}!=set(expected)
            or audit['status']!='verified' or audit['recordings']!=26 or not audit['goal_screen']
            or not audit['all_weights_unchanged'] or not audit['inactive_development_exact_parity']
            or audit['maximum_cosine_recomputation_error']>2e-6):raise ValueError('Complete audited26 screen required')
    for r in records:
        if any(r[k]!=v for k,v in expected[r['case'],r['seed']].items()):raise ValueError('Screen condition changed')
    summary=aggregate(records)
    if (summary!=report['summary'] or summary!=audit['summary'] or report['truncated'] or audit['truncated']
            or not all(r['terminated'] for r in records) or not goal_screen(summary)):
        raise ValueError('Strict joint screening thresholds not met')
    return summary


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


def evidence(prior):
    p,q,c=screened.verify(prior);status_path=prior.parent/'reference-profile-mean-evaluation-status.json';status=read(status_path)
    if status['status']!='complete' or status['reason']!='needs_broader_confirmation':raise ValueError('Screen run must finish and pass first')
    index=read(prior/'selection.json')['index'];screened.permitted(prior,p,index,'screen')
    report_path=prior/f'screen-{index:02d}/results.json';audit_path=prior/f'screen-verification-{index:02d}.json'
    report=read(report_path);screen_entry(report,read(audit_path),p['screen_cases'])
    for r in report['records']:
        if digest(report_path.parent/r['wav'])!=r['audio_sha256']:raise ValueError('Screening audio changed')
    panel_path=Path(c['panel']);panel=read(panel_path);validate_panel(panel)
    if (panel['generation']!=p['generation'] or panel['decoding']!=p['decoding'] or panel['seeds']!=c['control_seeds']
            or panel['preparation_source_sha256']!=digest(Path(__file__).with_name('joint_confirmation.py'))):
        raise ValueError('Original confirmation settings changed')
    all_cases={(case['case'],seed):dict(case,seed=seed) for seed in panel['seeds'] for case in panel['cases']}
    if any(all_cases.get((case['case'],case['seed']))!=case for case in p['screen_cases']):
        raise ValueError('Confirmation must include every screening condition unchanged')
    paths=(status_path,prior/'protocol.json',prior/'selection.json',report_path,audit_path,panel_path)
    hashes={str(path):digest(path) for path in paths}
    return p,q,c,panel,index,hashes


def prepare(prior,out,driver):
    old,q,c,panel,index,hashes=evidence(prior)
    tests=out.parent/'reference-profile-confirmation-tests.log';log=tests.read_text(encoding='utf-8-sig')
    if 'Ran 3 tests' not in log or '\nOK\n' not in log or 'skipped' in log.lower():raise ValueError('Confirmation tests missing')
    out.mkdir(exist_ok=False);p=dict(old)
    p.update(screen_run=str(prior),screen_evidence_hashes=hashes,source=sources(),driver=str(driver),driver_sha256=digest(driver),
        confirmation_tests=str(tests),confirmation_tests_sha256=digest(tests),selected_index=index,
        confirmation_panel=c['panel'],screen_cases=[dict(case,seed=seed) for seed in panel['seeds'] for case in panel['cases']],
        scope='All114 registered recordings:88development and26custom; both seeds; original references; no fitting or selection',
        goal_achieved=False,global_model_selection_changed=False)
    write(out/'protocol.json',p)


def verify(out):
    p=read(out/'protocol.json');old,q,c,panel,index,hashes=evidence(Path(p['screen_run']))
    if (p['source']!=sources() or p['runtime']!=direct.initial.collection.runtime() or p['numeric_config']!=collection.numeric_config()
            or p['screen_evidence_hashes']!=hashes or p['selected_index']!=index or p['confirmation_panel']!=c['panel']
            or digest(p['driver'])!=p['driver_sha256'] or digest(p['confirmation_tests'])!=p['confirmation_tests_sha256']):
        raise ValueError('Confirmation registration changed')
    expected=[dict(case,seed=seed) for seed in panel['seeds'] for case in panel['cases']]
    if p['screen_cases']!=expected:raise ValueError('Full114 scope changed')
    for key in ('proposals','proposal_protocol_sha256','proposal_results_sha256','trials','settings','generation','decoding','config'):
        if p[key]!=old[key]:raise ValueError('Screened candidate identity or generation changed')
    return p,q,c


def load(out,index,device):
    p,q,c=verify(out);_,wrapper,profile,state=collection.load(Path(q['data']),device)
    identity=c['initial_checkpoint_sha256']
    if index is not None:
        rows=[r for r in p['trials'] if r['index']==index]
        if len(rows)!=1:raise ValueError('Unregistered profile')
        row=rows[0];path=Path(p['proposals'])/row['path']
        if digest(path)!=row['sha256']:raise ValueError('Profile changed')
        saved=torch.load(path,map_location='cpu',weights_only=True)
        if saved['index']!=index or saved['protocol_sha256']!=p['proposal_protocol_sha256']:raise ValueError('Profile provenance changed')
        profiles.restore(profile,saved['profile']);identity=row['sha256']
    wrapper.model.eval()
    return p,q,c,wrapper,profile,state,profiles.snapshot(profile),identity


def unchanged(model,profile,state,expected):
    actual=profiles.snapshot(profile)
    if (actual['reference_sha256']!=expected['reference_sha256'] or actual['radius_fraction']!=expected['radius_fraction']
            or not torch.equal(actual['raw'],expected['raw']) or profile._active):raise ValueError('Evaluation profile changed or remained active')
    with torch.no_grad():profile.raw.zero_()
    try:collection.unchanged(model,profile,state)
    finally:profiles.restore(profile,expected)


def directory(out,kind,index):return out/(kind if index is None else f'{kind}-{index:02d}')


def conditions(p,kind):
    if kind=='control':return p['fit_cases']+p['validation_cases']
    return p[kind+'_cases']


def permitted(out,p,index,kind):
    if kind!='screen' or index!=p['selected_index']:raise ValueError('Only the screened profile and full confirmation panel are allowed')


def generate(out,kind,index):
    import soundfile as sf
    from .preference_rollouts import capture_generation
    p,q,c,wrapper,profile,state,expected,identity=load(out,index,'cpu' if kind=='screen' else 'cuda')
    permitted(out,p,index,kind)
    dest=directory(out,kind,index);dest.mkdir(exist_ok=False);records=[];prompts={};cases=conditions(p,kind)
    for condition in cases:
        row=dict(condition);seed=row['seed']
        if digest(row['reference'])!=row['reference_sha256']:raise ValueError('Reference changed')
        if row['reference'] not in prompts:
            with torch.inference_mode():prompts[row['reference']]=wrapper.create_voice_clone_prompt(
                ref_audio=row['reference'],ref_text=row['reference_text'],x_vector_only_mode=False)
        prompt=prompts[row['reference']];torch.manual_seed(seed);np.random.seed(seed)
        with profile.use_reference(row['reference_sha256']),torch.inference_mode(),capture_generation(wrapper.model) as captured:
            waves,sr=wrapper.generate_voice_clone(text=row['text'],language='Auto',voice_clone_prompt=prompt,
                non_streaming_mode=True,**p['generation'])
        if sr!=24000 or not len(waves[0]) or not np.isfinite(waves[0]).all() or len(waves[0])/sr>30:raise ValueError('Invalid generated audio')
        path=dest/f"{seed}-{row['case']}.wav";sf.write(path,waves[0],sr,subtype='PCM_16');wave,sr=sf.read(path,dtype='float32')
        with torch.inference_mode():
            vector=wrapper.model.extract_speaker_embedding(audio=wave,sr=sr).flatten();ref=prompt[0].ref_spk_embedding.flatten()
            cosine=float(torch.nn.functional.cosine_similarity(vector[None],ref[None]))
        trajectory=path.with_suffix('.pt');save(trajectory,dict(codes=captured['codes'],first_sequence=captured['first_sequence'],
            terminated=captured['terminated'],ref_codes=prompt[0].ref_code.detach().cpu(),ref_xvec=ref.detach().cpu(),
            input_ids=ids(wrapper,row['text']).cpu(),ref_ids=ids(wrapper,row['reference_text'],True).cpu(),reference_sha256=row['reference_sha256']))
        record=dict(row,wav=path.name,audio_sha256=digest(path),trajectory=trajectory.name,trajectory_sha256=digest(trajectory),
            cosine=cosine,seconds=len(wave)/sr,terminated=captured['terminated'],candidate_sha256=identity,
            protocol_sha256=digest(out/'protocol.json'))
        write(path.with_suffix('.json'),record);records.append(record);print(f'{dest.name}: {len(records)}/{len(cases)}',flush=True)
    unchanged(wrapper.model,profile,state,expected);write(dest/'generated.json',dict(records=records,all_weights_unchanged=True))

def score(out,kind,index):
    import soundfile as sf
    import torchaudio
    from transformers import WhisperForConditionalGeneration,WhisperProcessor
    from .evaluate import ASR_REV
    from scripts.score_latest_audio import score as score_text
    p,q,c=verify(out);dest=directory(out,kind,index);generated=read(dest/'generated.json');records=generated['records']
    if not generated['all_weights_unchanged']:raise ValueError('Generation changed weights')
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
    write(dest/'results.json',dict(records=records,summary=aggregate(records),truncated=sum(not r['terminated'] for r in records)))


def audit(out,kind,index):
    import soundfile as sf
    from .evaluate import ASR_REV
    from scripts.score_latest_audio import score as score_text
    p,q,c,wrapper,profile,state,expected_profile,identity=load(out,index,'cpu' if kind=='screen' else 'cuda')
    permitted(out,p,index,kind);dest=directory(out,kind,index);report=read(dest/'results.json');records=report['records']
    expected={(r['case'],r['seed']):r for r in conditions(p,kind)}
    if len(records)!=len(expected) or {(r['case'],r['seed']) for r in records}!=set(expected):raise ValueError('Incomplete speech panel')
    refs={};error=0.
    for r in records:
        if any(r[k]!=expected[r['case'],r['seed']][k] for k in CONDITION):raise ValueError('Speech condition changed')
        if (r['candidate_sha256']!=identity or r['protocol_sha256']!=digest(out/'protocol.json')
                or r['decoding']!=p['decoding'] or r['asr_revision']!=ASR_REV or digest(dest/r['wav'])!=r['audio_sha256']
                or digest(r['reference'])!=r['reference_sha256']):raise ValueError('Speech provenance changed')
        if r['score']!=score_text(r['text'],r['hypothesis'],True):raise ValueError('ASR counts changed')
        t=direct.trajectory(dest,r);codes=t['codes'];seq=t['first_sequence'];eos=wrapper.model.config.talker_config.codec_eos_token_id
        if (t['reference_sha256']!=r['reference_sha256'] or codes.ndim!=2 or codes.shape[1]!=16
                or bool(seq[0,-1]==eos)!=r['terminated']):raise ValueError('Trajectory coverage differs')
        if r['terminated'] and not torch.equal(seq[0,-(len(codes)+1):-1],codes[:,0]):raise ValueError('Main codes differ')
        wave,sr=sf.read(dest/r['wav'],dtype='float32')
        with torch.inference_mode():
            if r['reference'] not in refs:
                ref,refsr=sf.read(r['reference'],dtype='float32');refs[r['reference']]=wrapper.model.extract_speaker_embedding(audio=ref,sr=refsr).flatten()
            vector=wrapper.model.extract_speaker_embedding(audio=wave,sr=sr).flatten()
            cosine=float(torch.nn.functional.cosine_similarity(vector[None],refs[r['reference']][None]))
        error=max(error,abs(cosine-r['cosine']))
    if error>2e-6 or report['summary']!=aggregate(records) or report['truncated']!=sum(not r['terminated'] for r in records):
        raise ValueError('Independent scores differ')
    unchanged(wrapper.model,profile,state,expected_profile)
    receipt=dict(status='verified',recordings=len(records),summary=report['summary'],truncated=report['truncated'],
        maximum_cosine_recomputation_error=error,all_weights_unchanged=True,candidate_index=index,goal_achieved=False)
    screen_records=read(Path(p['screen_run'])/f'screen-{index:02d}/results.json')['records']
    result=confirmation_measurement(read(p['confirmation_panel']),records,screen_records)
    receipt.update(result)
    write(out/'verification.json',receipt)
    print(receipt,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=['prepare','generate','score','audit'])
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--prior',type=Path);parser.add_argument('--driver',type=Path)
    args=parser.parse_args();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if args.phase=='prepare':prepare(args.prior,args.out,args.driver)
    else:
        index=read(args.out/'protocol.json')['selected_index']
        {'generate':generate,'score':score,'audit':audit}[args.phase](args.out,'screen',index)


if __name__=='__main__':main()
