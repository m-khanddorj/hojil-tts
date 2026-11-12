"""Fresh multi-draw personalization data with joint pronunciation/voice preferences."""
import argparse,os
from pathlib import Path
import numpy as np
import torch
from .engine import read,write,save,digest,ids
from .policy import fingerprint,external
from . import direct_voice_training as direct
from . import middle_attention_adapter as attention
from . import reference_voice_profile as profiles
from .targeted_voice import aggregate
from . import reference_profile_collection as previous
from .voice_preferences import joint_pairs

SETTINGS=dict(fitting_texts=16,seeds=[41301,41302,41303,41304],radius_fraction=.05,min_cosine_margin=.001)
numeric_config=previous.numeric_config
unchanged=previous.unchanged


def sources():
    result=previous.sources()
    for name in ('reference_profile_preference_collection.py','voice_preferences.py'):
        result[name]=digest(Path(__file__).with_name(name))
    result['test_training_reference_profile_preference_collection.py']=digest(Path(__file__).parent.parent/'tests/test_training_reference_profile_preference_collection.py')
    return result


def fresh_texts(records,panel,old_cases):
    excluded=[text for case in panel['cases'] for text in (case['text'],case['reference_text'])]
    excluded += [case['text'] for case in old_cases]
    return previous.select_texts(records,excluded,SETTINGS['fitting_texts'])


def preference_records(p,report):
    expected={(case['case'],seed):case for case in p['cases'] for seed in p['settings']['seeds']}
    records=report['records']
    if (len(expected)!=64 or len(records)!=64 or {(r['case'],r['seed']) for r in records}!=set(expected)
            or any(not r['terminated'] for r in records)):raise ValueError('Complete64 terminated training records required')
    for r in records:
        if r['group']!='profile-preference-train' or any(r[k]!=v for k,v in expected[r['case'],r['seed']].items()):
            raise ValueError('Preference training condition changed')
    return [dict(r,sample_id=f"{r['seed']}:{r['case']}",partition='train',origin='model_generated') for r in records]


def prepare(old_run,run,driver):
    p=previous.verify(old_run);panel=read(p['panel']);ref=p['profile_reference']
    audit=read(old_run/'verification.json')
    if audit['status']!='verified' or audit['truncated']:raise ValueError('Verified original collection required')
    selected=fresh_texts(read(Path(p['pool'])/'results.json')['records'],panel,p['cases'])
    cases=[dict(row,case=f'profile-preference-{i:03d}',group='profile-preference-train',**ref) for i,row in enumerate(selected)]
    tests=run.parent/'reference-profile-preference-collection-tests.log';log=tests.read_text(encoding='utf-8-sig')
    if 'Ran 2 tests' not in log or '\nOK\n' not in log or 'skipped' in log.lower():raise ValueError('Preference collection tests missing')
    run=external(run.resolve());run.mkdir(exist_ok=False)
    write(run/'protocol.json',dict(previous=str(old_run),previous_protocol_sha256=digest(old_run/'protocol.json'),
        previous_results_sha256=digest(old_run/'baseline/results.json'),previous_audit_sha256=digest(old_run/'verification.json'),
        source=sources(),runtime=direct.initial.collection.runtime(),numeric_config=numeric_config(),settings=SETTINGS,
        profile_reference=ref,cases=cases,generation=p['generation'],decoding=p['decoding'],config=p['config'],
        tests=str(tests),tests_sha256=digest(tests),driver=str(driver),driver_sha256=digest(driver),
        text_separation='16 fresh unpaired preference-training texts exclude all114 evaluation targets/reference transcripts and all12 earlier profile fitting/validation texts',
        policy='Original zero-profile model; shared acoustic/attention/speaker weights frozen, no update during collection',
        pair_rule='Within each text, chosen has no more ASR character errors and no lower original-reference cosine; at least one fewer character error or cosine gain>=.001',
        interpretation='Automated metric-derived preferences, not human labels; data preparation for a new reference-specific preference objective',
        goal_achieved=False,global_model_selection_changed=False))


def verify(run):
    p=read(run/'protocol.json');old=Path(p['previous']);c=previous.verify(old)
    if (p['source']!=sources() or p['runtime']!=direct.initial.collection.runtime() or p['numeric_config']!=numeric_config()
            or p['settings']!=SETTINGS or digest(p['driver'])!=p['driver_sha256'] or digest(p['tests'])!=p['tests_sha256']
            or digest(old/'protocol.json')!=p['previous_protocol_sha256']
            or digest(old/'baseline/results.json')!=p['previous_results_sha256']
            or digest(old/'verification.json')!=p['previous_audit_sha256']):raise ValueError('Preference collection registration changed')
    if digest(p['profile_reference']['reference'])!=p['profile_reference']['reference_sha256']:raise ValueError('Original reference changed')
    expected=fresh_texts(read(Path(c['pool'])/'results.json')['records'],read(c['panel']),c['cases'])
    expected=[dict(row,case=f'profile-preference-{i:03d}',group='profile-preference-train',**c['profile_reference']) for i,row in enumerate(expected)]
    if p['cases']!=expected:raise ValueError('Fresh text partition changed')
    return p


def load(run,device='cuda'):
    p=verify(run);_,wrapper,profile,state=previous.load(Path(p['previous']),device)
    return p,wrapper,profile,state


def control(run):
    import soundfile as sf
    p,wrapper,profile,state=load(run,'cuda');old=Path(p['previous'])
    row=read(old/'baseline/results.json')['records'][0]
    with torch.inference_mode():prompt=wrapper.create_voice_clone_prompt(ref_audio=row['reference'],ref_text=row['reference_text'],x_vector_only_mode=False)
    torch.manual_seed(row['seed']);np.random.seed(row['seed'])
    with profile.use_reference(row['reference_sha256']),torch.inference_mode():
        waves,sr=wrapper.generate_voice_clone(text=row['text'],language='Auto',voice_clone_prompt=prompt,non_streaming_mode=True,**p['generation'])
    path=run/'control.wav';sf.write(path,waves[0],sr,subtype='PCM_16')
    if sr!=24000 or digest(path)!=row['audio_sha256']:raise ValueError('Native original-profile control failed')
    unchanged(wrapper.model,profile,state)
    write(run/'control-verification.json',dict(verified=True,exact_waveform_parity=True,all_weights_unchanged=True,
        previous_results_sha256=p['previous_results_sha256'],case=row['case'],seed=row['seed']))


def generate(run,control=False):
    import soundfile as sf
    from .preference_rollouts import capture_generation
    p,wrapper,profile,state=load(run,'cpu' if control else 'cuda')
    if not control and not read(run/'control-verification.json')['verified']:raise ValueError('Zero-profile control missing')
    cases=p['control_cases'] if control else p['cases'];seeds=p['control_seeds'] if control else p['settings']['seeds']
    out=run/('control' if control else 'baseline');out.mkdir(exist_ok=False);records=[];prompts={}
    for seed in seeds:
        for c in cases:
            if digest(c['reference'])!=c['reference_sha256']:raise ValueError('Reference changed')
            if c['reference'] not in prompts:
                with torch.inference_mode():prompts[c['reference']]=wrapper.create_voice_clone_prompt(
                    ref_audio=c['reference'],ref_text=c['reference_text'],x_vector_only_mode=False)
            prompt=prompts[c['reference']];torch.manual_seed(seed);np.random.seed(seed)
            with profile.use_reference(c['reference_sha256']),torch.inference_mode(),capture_generation(wrapper.model) as captured:
                waves,sr=wrapper.generate_voice_clone(text=c['text'],language='Auto',voice_clone_prompt=prompt,
                    non_streaming_mode=True,**p['generation'])
            if sr!=24000 or not len(waves[0]) or not np.isfinite(waves[0]).all() or len(waves[0])/sr>30:raise ValueError('Invalid generated speech')
            path=out/f"{seed}-{c['case']}.wav";sf.write(path,waves[0],sr,subtype='PCM_16');wave,sr=sf.read(path,dtype='float32')
            with torch.inference_mode():
                vector=wrapper.model.extract_speaker_embedding(audio=wave,sr=sr).flatten();ref=prompt[0].ref_spk_embedding.flatten()
                cosine=float(torch.nn.functional.cosine_similarity(vector[None],ref[None]))
            trajectory=path.with_suffix('.pt');save(trajectory,dict(codes=captured['codes'],first_sequence=captured['first_sequence'],
                terminated=captured['terminated'],ref_codes=prompt[0].ref_code.detach().cpu(),ref_xvec=ref.detach().cpu(),
                input_ids=ids(wrapper,c['text']).cpu(),ref_ids=ids(wrapper,c['reference_text'],True).cpu(),reference_sha256=c['reference_sha256']))
            r=dict(c,seed=seed,wav=path.name,audio_sha256=digest(path),trajectory=trajectory.name,trajectory_sha256=digest(trajectory),
                cosine=cosine,seconds=len(wave)/sr,terminated=captured['terminated'],protocol_sha256=digest(run/'protocol.json'))
            write(path.with_suffix('.json'),r);records.append(r);print(f'{out.name}: {len(records)}/{len(cases)*len(seeds)}',flush=True)
    unchanged(wrapper.model,profile,state);write(out/'generated.json',dict(records=records,all_weights_unchanged=True))
    if control:
        old={(r['case'],r['seed']):r for r in read(Path(p['initial_run'])/'eval-000000/results.json')['records']}
        if any(r['audio_sha256']!=old[r['case'],r['seed']]['audio_sha256']
               or abs(r['cosine']-old[r['case'],r['seed']]['cosine'])>2e-6 for r in records):raise ValueError('Native zero-profile parity failed')
        write(run/'control-verification.json',dict(verified=True,recordings=len(records),exact_waveform_parity=True,
            inactive_development_recordings=20,active_zero_custom_recordings=6,all_weights_unchanged=True))


def score(run):
    import soundfile as sf
    import torchaudio
    from transformers import WhisperForConditionalGeneration,WhisperProcessor
    from .evaluate import ASR_REV
    from scripts.score_latest_audio import score as score_text
    p=verify(run);out=run/'baseline';data=read(out/'generated.json');records=data['records']
    if not data['all_weights_unchanged']:raise ValueError('Generation state changed')
    asr=Path(p['config']['asr']);manifest=read(asr/'download-manifest.json')
    if manifest['repo']!='BuzzASR/mongolian' or manifest['revision']!=ASR_REV:raise ValueError('ASR mismatch')
    for name,item in manifest['files'].items():
        if digest(asr/name)!=item['sha256']:raise ValueError('ASR weights changed')
    processor=WhisperProcessor.from_pretrained(asr,local_files_only=True)
    model=WhisperForConditionalGeneration.from_pretrained(asr,torch_dtype=torch.float32,local_files_only=True).to('cuda').eval()
    for r in records:
        if digest(out/r['wav'])!=r['audio_sha256']:raise ValueError('Audio changed')
        wave,sr=sf.read(out/r['wav'],dtype='float32');wave=torchaudio.functional.resample(torch.from_numpy(wave),sr,16000).numpy()
        batch=processor(wave,sampling_rate=16000,return_tensors='pt',return_attention_mask=True).to('cuda')
        with torch.inference_mode():tokens=model.generate(batch.input_features,attention_mask=batch.attention_mask,**p['decoding'])
        hypothesis=processor.batch_decode(tokens,skip_special_tokens=True)[0].strip()
        r.update(hypothesis=hypothesis,score=score_text(r['text'],hypothesis,True),decoding=p['decoding'],asr_revision=ASR_REV)
    write(out/'results.json',dict(records=records,summary=aggregate(records),truncated=sum(not r['terminated'] for r in records)))


def audit(run):
    import soundfile as sf
    from scripts.score_latest_audio import score as score_text
    from .evaluate import ASR_REV
    p,wrapper,profile,state=load(run,'cuda');out=run/'baseline';report=read(out/'results.json');records=report['records']
    expected={(c['case'],s):c for s in p['settings']['seeds'] for c in p['cases']}
    if len(records)!=64 or {(r['case'],r['seed']) for r in records}!=set(expected):raise ValueError('Incomplete profile collection')
    with torch.inference_mode():
        ref,sr=sf.read(p['profile_reference']['reference'],dtype='float32');reference=wrapper.model.extract_speaker_embedding(audio=ref,sr=sr).flatten()
    maximum=0.
    for r in records:
        if any(r[k]!=v for k,v in expected[r['case'],r['seed']].items()):raise ValueError('Collection condition changed')
        if (digest(out/r['wav'])!=r['audio_sha256'] or r['protocol_sha256']!=digest(run/'protocol.json')
                or r['decoding']!=p['decoding'] or r['asr_revision']!=ASR_REV):raise ValueError('Collection provenance differs')
        if r['score']!=score_text(r['text'],r['hypothesis'],True):raise ValueError('ASR edit counts differ')
        t=direct.trajectory(out,r);codes,seq=t['codes'],t['first_sequence'];eos=wrapper.model.config.talker_config.codec_eos_token_id
        if codes.ndim!=2 or codes.shape[1]!=16 or bool(seq[0,-1]==eos)!=r['terminated']:raise ValueError('EOS/code coverage differs')
        if r['terminated'] and not torch.equal(seq[0,-(len(codes)+1):-1],codes[:,0]):raise ValueError('Sampled main codes differ')
        wave,sr=sf.read(out/r['wav'],dtype='float32')
        with torch.inference_mode():
            vector=wrapper.model.extract_speaker_embedding(audio=wave,sr=sr).flatten()
            cosine=float(torch.nn.functional.cosine_similarity(vector[None],reference[None]))
        maximum=max(maximum,abs(cosine-r['cosine']))
    if maximum>2e-6 or report['summary']!=aggregate(records):raise ValueError('Independent collection scores differ')
    unchanged(wrapper.model,profile,state)
    write(run/'verification.json',dict(status='verified',recordings=64,summary=report['summary'],truncated=report['truncated'],
        maximum_cosine_recomputation_error=maximum,all_weights_unchanged=True,profile_parameters=1024,
        fitting_recordings=64,reserved_validation_recordings=0,goal_achieved=False))


def pairs(run):
    p=verify(run);report=read(run/'baseline/results.json');audit=read(run/'verification.json')
    if audit['status']!='verified' or audit['truncated'] or audit['summary']!=report['summary']:
        raise ValueError('Complete independent64 audit required')
    records=preference_records(p,report);pairs=[]
    for case in p['cases']:
        rows=[r for r in records if r['case']==case['case']]
        pairs.extend(joint_pairs(rows,p['settings']['min_cosine_margin']))
    write(run/'pairs.json',dict(records=records,pairs=pairs,pair_count=len(pairs),cases_with_pairs=len({r['case'] for r in pairs}),
        total_cases=16,results_sha256=digest(run/'baseline/results.json'),audit_sha256=digest(run/'verification.json'),
        rule=p['pair_rule'],goal_achieved=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=['prepare','control','collect','score','audit','pairs'])
    parser.add_argument('--run',type=Path,required=True);parser.add_argument('--previous',type=Path);parser.add_argument('--driver',type=Path)
    args=parser.parse_args();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if args.phase=='prepare':prepare(args.previous,args.run,args.driver)
    elif args.phase=='collect':generate(args.run)
    else:{'control':control,'score':score,'audit':audit,'pairs':pairs}[args.phase](args.run)


if __name__=='__main__':main()
