"""Disjoint-text generated data for a reference-specific conditioning experiment."""
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

SETTINGS=dict(fitting_texts=8,validation_texts=4,seeds=[41201,41202],radius_fraction=.05)


def numeric_config():
    return dict(cpu_threads=torch.get_num_threads(),matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                cudnn_tf32=torch.backends.cudnn.allow_tf32)


def sources():
    result=direct.sources()
    for name in ('reference_voice_profile.py','reference_profile_collection.py'):
        result[name]=digest(Path(__file__).with_name(name))
    for name in ('test_training_reference_voice_profile.py','test_training_reference_profile_collection.py'):
        result[name]=digest(Path(__file__).parent.parent/'tests'/name)
    return result


def select_texts(records,excluded,count=12):
    from data.pipeline import text_key
    forbidden={text_key(t) for t in excluded};unique={}
    for r in sorted(records,key=lambda r:fingerprint(['reference-profile-source-v1',r['sample_id']])):
        key=text_key(r['text'])
        if r['preference_split']=='train' and 30<=len(key)<=140 and key not in forbidden:
            unique.setdefault(key,r)
    keys=sorted(unique,key=lambda k:fingerprint(['reference-profile-text-v1',k]))[:count]
    if len(keys)!=count:raise ValueError('Insufficient distinct fitting-source texts')
    return [dict(text=unique[k]['text'],source_case=unique[k]['case'],source_sample_id=unique[k]['sample_id'],
                 source_speaker_id=unique[k]['speaker_id']) for k in keys]


def prepare(initial_run,run,driver):
    p=direct.verify(initial_run);panel=read(p['panel'])
    if run.exists():raise ValueError('Do not overwrite a registered collection')
    custom=[c for c in p['cases'] if c['group']=='custom']
    if len({c['reference_sha256'] for c in custom})!=1:raise ValueError('Expected one requested reference')
    ref=custom[0]
    if digest(ref['reference'])!=ref['reference_sha256']:raise ValueError('Reference changed')
    if any(c['reference_sha256']==ref['reference_sha256'] for c in p['cases'] if c['group']=='development'):
        raise ValueError('Profile reference must be distinct from development references')
    pool=Path(p['pool']);records=read(pool/'results.json')['records']
    excluded=[text for c in panel['cases'] for text in (c['text'],c['reference_text'])]
    selected=select_texts(records,excluded,SETTINGS['fitting_texts']+SETTINGS['validation_texts'])
    cases=[]
    for i,row in enumerate(selected):
        group='profile-fit' if i<SETTINGS['fitting_texts'] else 'profile-validation'
        cases.append(dict(row,case=f'{group}-{i:03d}',group=group,speaker_id=ref['speaker_id'],
            reference=ref['reference'],reference_sha256=ref['reference_sha256'],reference_text=ref['reference_text']))
    tests={}
    for name,count in [('reference-voice-profile-tests.log',3),('reference-profile-collection-tests.log',2)]:
        path=run.parent/name;log=path.read_text(encoding='utf-8-sig')
        if f'Ran {count} tests' not in log or '\nOK\n' not in log or 'skipped' in log.lower():
            raise ValueError('Reference profile tests missing')
        tests[str(path)]=digest(path)
    run=external(run.resolve());run.mkdir()
    write(run/'protocol.json',dict(source=sources(),runtime=direct.initial.collection.runtime(),numeric_config=numeric_config(),
        initial_run=str(initial_run),initial_protocol_sha256=digest(initial_run/'protocol.json'),
        initial_checkpoint_sha256=digest(initial_run/'checkpoint-000000.pt'),
        initial_screen_sha256=digest(initial_run/'eval-000000/results.json'),initial_audit_sha256=digest(initial_run/'verification-000000.json'),
        pool=str(pool),pool_results_sha256=digest(pool/'results.json'),panel=p['panel'],panel_sha256=digest(p['panel']),
        profile_reference={k:ref[k] for k in ('reference','reference_sha256','reference_text','speaker_id')},
        settings=SETTINGS,cases=cases,control_cases=p['cases'],control_seeds=p['seeds'],
        generation=p['generation'],decoding=p['decoding'],config=p['config'],tests=tests,
        driver=str(driver),driver_sha256=digest(driver),
        text_separation='All12 adaptation/validation texts hash-selected from original preference-training texts, excluding every text and reference transcript in the complete114 confirmation panel',
        personalization='Requested reference audio/transcript deliberately conditions adaptation; no evaluation target transcript, target audio or output enters fitting. Reserved validation texts never enter gradients.',
        scope='Shared acoustic, attention and speaker-adapter weights frozen; only a reference-identified1024D residual may later train. Other references take the exact shared fallback.',
        collection_only=True,goal_achieved=False,global_model_selection_changed=False))


def verify(run):
    p=read(run/'protocol.json');initial=Path(p['initial_run'])
    if (p['source']!=sources() or p['runtime']!=direct.initial.collection.runtime()
            or p['numeric_config']!=numeric_config() or numeric_config()!=dict(cpu_threads=4,matmul_tf32=False,cudnn_tf32=False)
            or p['settings']!=SETTINGS or digest(p['driver'])!=p['driver_sha256']):raise ValueError('Registered execution changed')
    for path,key in [(initial/'protocol.json','initial_protocol_sha256'),(initial/'checkpoint-000000.pt','initial_checkpoint_sha256'),
        (initial/'eval-000000/results.json','initial_screen_sha256'),(initial/'verification-000000.json','initial_audit_sha256'),
        (Path(p['pool'])/'results.json','pool_results_sha256'),(Path(p['panel']),'panel_sha256')]:
        if digest(path)!=p[key]:raise ValueError('Registered data/model changed')
    for path,expected in p['tests'].items():
        if digest(path)!=expected:raise ValueError('Registered tests changed')
    return p


def load(run,device='cpu'):
    p=verify(run);_,wrapper,state=direct.load(Path(p['initial_run']),0,device)
    wrapper.model.requires_grad_(False)
    profile=profiles.install(wrapper.model,p['profile_reference']['reference_sha256'],p['settings']['radius_fraction'])
    if sum(v.numel() for v in wrapper.model.parameters() if v.requires_grad)!=1024:raise ValueError('Unexpected trainable profile coverage')
    unchanged(wrapper.model,profile,state)
    return p,wrapper,profile,state


def unchanged(model,profile,state):
    if attention.frozen_digest(model)!=state['frozen_backbone_sha256'] or profiles.shared_speaker_digest(profile)!=state['frozen_speaker_sha256']:
        raise ValueError('Original weights or shared speaker adapter changed')
    for name,m in attention.modules(model).items():
        for key in ('lora_A','lora_B'):
            if not torch.equal(getattr(m,key).detach().cpu(),state['adapter'][name+'.'+key]):raise ValueError('Attention adapter changed')
    if profile._active or not torch.equal(profile.raw.detach(),torch.zeros_like(profile.raw)):
        raise ValueError('Collection must leave the zero profile inactive')
    if any(v.grad is not None for v in model.parameters()):raise ValueError('Collection created a parameter gradient')


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
    if len(records)!=24 or {(r['case'],r['seed']) for r in records}!=set(expected):raise ValueError('Incomplete profile collection')
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
    write(run/'verification.json',dict(status='verified',recordings=24,summary=report['summary'],truncated=report['truncated'],
        maximum_cosine_recomputation_error=maximum,all_weights_unchanged=True,profile_parameters=1024,
        fitting_recordings=16,reserved_validation_recordings=8,goal_achieved=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=['prepare','control','collect','score','audit'])
    parser.add_argument('--run',type=Path,required=True);parser.add_argument('--initial-run',type=Path);parser.add_argument('--driver',type=Path)
    args=parser.parse_args();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if args.phase=='prepare':prepare(args.initial_run,args.run,args.driver)
    elif args.phase in ('control','collect'):generate(args.run,args.phase=='control')
    else:{'score':score,'audit':audit}[args.phase](args.run)


if __name__=='__main__':main()
