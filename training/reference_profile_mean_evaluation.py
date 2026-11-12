"""Separate group-mean follow-up; original per-clip experiment remains unchanged."""
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
from . import reference_profile_evaluation as original
from .targeted_voice import aggregate
from .voice_strength_search import goal_screen

CONDITION=('case','speaker_id','group','text','reference','reference_sha256','reference_text','seed')


def sources():
    result=original.sources()
    for name in ('reference_profile_mean_evaluation.py','reference_profile_mean_selection.py'):
        result[name]=digest(Path(__file__).with_name(name))
    result['test_training_reference_profile_mean_selection.py']=digest(Path(__file__).parent.parent/'tests/test_training_reference_profile_mean_selection.py')
    result['test_training_reference_profile_mean_evaluation.py']=digest(Path(__file__).parent.parent/'tests/test_training_reference_profile_mean_evaluation.py')
    return result


def fitting_evidence(prior):
    p,q,c=original.verify(prior)
    status_path=prior.parent/'reference-profile-evaluation-status.json';status=read(status_path)
    if status['status']!='complete' or status['reason']!='no_fitting_speech_pass':
        raise ValueError('Original four-trial experiment must finish without a fitting pass')
    if (prior/'selection.json').exists():raise ValueError('Original selected candidate takes priority')
    baseline=[r for r in read(Path(q['data'])/'baseline/results.json')['records'] if r['group']=='profile-fit']
    trials=[];hashes={str(status_path):digest(status_path),str(prior/'protocol.json'):digest(prior/'protocol.json'),
        str(prior/'control-verification.json'):digest(prior/'control-verification.json')}
    if not read(prior/'control-verification.json')['verified']:raise ValueError('Exact24 baseline control required')
    for row in p['trials']:
        index=row['index'];gate_path=prior/f'gate-{index:02d}.json';report_path=prior/f'fit-{index:02d}/results.json'
        gate=read(gate_path);report=read(report_path)
        if gate['passed']:raise ValueError('Original gate pass exists')
        recomputed=compare(baseline,report['records'],p['settings'],'profile-fit')
        if {k:v for k,v in gate.items() if k!='audit'}!=recomputed:raise ValueError('Original gate changed')
        for r in report['records']:
            if digest(report_path.parent/r['wav'])!=r['audio_sha256']:raise ValueError('Original fitting audio changed')
        trials.append(dict(index=index,records=report['records'],audit=gate['audit']))
        for path in (gate_path,report_path):hashes[str(path)]=digest(path)
    decision=select(baseline,trials,p['settings'],[r['index'] for r in p['trials']])
    return p,q,c,decision,hashes


def prepare(prior,out,driver):
    p,q,c,decision,hashes=fitting_evidence(prior)
    if decision['selected'] is None:raise ValueError('No improved group-mean fitting candidate')
    tests=out.parent/'reference-profile-mean-selection-tests.log';log=tests.read_text(encoding='utf-8-sig')
    if 'Ran 2 tests' not in log or '\nOK\n' not in log or 'skipped' in log.lower():raise ValueError('Mean selector tests missing')
    execution_tests=out.parent/'reference-profile-mean-evaluation-tests.log';execution_log=execution_tests.read_text(encoding='utf-8-sig')
    if 'Ran 2 tests' not in execution_log or '\nOK\n' not in execution_log or 'skipped' in execution_log.lower():raise ValueError('Mean execution tests missing')
    out.mkdir(exist_ok=False)
    protocol=dict(p)
    protocol.update(prior=str(prior),prior_evidence_hashes=hashes,source=sources(),driver=str(driver),driver_sha256=digest(driver),
        selection_tests=str(tests),selection_tests_sha256=digest(tests),mean_decision=decision,
        execution_tests=str(execution_tests),execution_tests_sha256=digest(execution_tests),
        selection='Separate fitting-only mean selection after all original trials failed extra per-clip checks; reserved8 and unchanged26 follow without further scale search',
        control='Reuse pinned exact24 GPU and exact26 CPU zero-profile controls from the unchanged initial model',
        previous_protocol_revised=False,goal_achieved=False,global_model_selection_changed=False)
    write(out/'protocol.json',protocol)
    write(out/'selection.json',dict(index=decision['selected'],decision=decision,protocol_sha256=digest(out/'protocol.json'),
        selection_basis=decision['basis'],goal_achieved=False))


def verify(out):
    p=read(out/'protocol.json');old,q,c,decision,hashes=fitting_evidence(Path(p['prior']))
    if (p['source']!=sources() or p['runtime']!=direct.initial.collection.runtime()
            or p['numeric_config']!=collection.numeric_config() or digest(p['driver'])!=p['driver_sha256']
            or p['prior_evidence_hashes']!=hashes or p['mean_decision']!=decision
            or digest(p['selection_tests'])!=p['selection_tests_sha256']
            or digest(p['execution_tests'])!=p['execution_tests_sha256']
            or digest(p['tests'])!=p['tests_sha256']):raise ValueError('Mean follow-up registration changed')
    for key in ('proposals','proposal_protocol_sha256','proposal_results_sha256','trials','settings',
                'fit_cases','validation_cases','screen_cases','generation','decoding','config'):
        if p[key]!=old[key]:raise ValueError('Original generation scope or candidate identity changed')
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
    if kind not in ('validation','screen'):raise ValueError('No new fitting or scale search in mean follow-up')
    chosen=read(out/'selection.json')
    if (chosen['index']!=index or p['mean_decision']['selected']!=index
            or chosen['decision']!=p['mean_decision'] or chosen['protocol_sha256']!=digest(out/'protocol.json')):
        raise ValueError('Mean-selected fitting candidate changed')


def generate(out,kind,index):
    import soundfile as sf
    from .preference_rollouts import capture_generation
    p,q,c,wrapper,profile,state,expected,identity=load(out,index,'cpu' if kind=='screen' else 'cuda')
    permitted(out,p,index,kind)
    dest=directory(out,kind,index);dest.mkdir(exist_ok=False);records=[];prompts={};cases=conditions(p,kind)
    for condition in cases:
        row={k:condition[k] for k in CONDITION};seed=row['seed']
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
    if kind=='control':
        base=Path(q['data'])/'baseline/results.json';old={(r['case'],r['seed']):r for r in read(base)['records']}
        if any(r['audio_sha256']!=old[r['case'],r['seed']]['audio_sha256']
               or abs(r['cosine']-old[r['case'],r['seed']]['cosine'])>2e-6 for r in records):raise ValueError('Native GPU parity failed')
        write(out/'control-verification.json',dict(verified=True,recordings=24,exact_waveform_parity=True,baseline_sha256=digest(base)))


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
    if kind=='screen':
        baseline={(r['case'],r['seed']):r for r in read(Path(c['initial_run'])/'eval-000000/results.json')['records']}
        if any(r['audio_sha256']!=baseline[r['case'],r['seed']]['audio_sha256'] for r in records if r['group']=='development'):
            raise ValueError('Unrelated development voices changed')
        receipt.update(goal_screen=goal_screen(report['summary']) and report['truncated']==0,inactive_development_exact_parity=True)
        write(out/f'screen-verification-{index:02d}.json',receipt)
    else:
        group='profile-fit' if kind=='fit' else 'profile-validation'
        before=[r for r in read(Path(q['data'])/'baseline/results.json')['records'] if r['group']==group]
        gate=compare(before,records,p['settings'],group);gate['audit']=receipt
        path=out/(f'gate-{index:02d}.json' if kind=='fit' else f'validation-verification-{index:02d}.json');write(path,gate)
        if kind=='fit' and gate['passed']:
            write(out/'selection.json',dict(index=index,gate_sha256=digest(path),selection_basis='First fitting-only gate pass',goal_achieved=False))
        print(gate,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase');parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--prior',type=Path);parser.add_argument('--driver',type=Path);parser.add_argument('--index',type=int)
    args=parser.parse_args();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    phase=args.phase
    if phase=='prepare':prepare(args.prior,args.out,args.driver)
    elif phase in ('validation','screen'):generate(args.out,phase,args.index)
    elif phase in ('score-validation','score-screen'):score(args.out,phase.removeprefix('score-'),args.index)
    elif phase in ('audit-validation','audit-screen'):audit(args.out,phase.removeprefix('audit-'),args.index)
    else:raise ValueError('Unknown evaluation phase')


if __name__=='__main__':main()
