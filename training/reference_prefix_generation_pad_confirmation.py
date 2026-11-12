"""Full original 114-recording confirmation of an audited reference PAD screen.

Conditional only: no training, scale search, selection or automatic promotion.
CPU FP32 generation/audit and GPU ASR execute in separate processes.
"""
import argparse
import math
from pathlib import Path
import numpy as np
import torch
from .engine import read,write,save,digest,ids
from .policy import external
from . import reference_prefix_generation_pad_evaluation as parent
from .reference_prefix_generation_pad import tensor_sha256
from .confirm_joint_candidate import validate_panel,validate_records
from .voice_strength_search import goal_screen

require=parent.require


def sources():
    result=parent.sources()
    for name in ('reference_prefix_generation_pad_confirmation.py','confirm_joint_candidate.py','joint_confirmation.py'):
        result[name]=digest(Path(__file__).with_name(name))
    result['test_training_reference_prefix_generation_pad_confirmation.py']=digest(Path(__file__).parent.parent/'tests/test_training_reference_prefix_generation_pad_confirmation.py')
    return result


def validate_parent(status,cases,reports,audits):
    require(status['status']=='complete' and status['reason']=='needs_broader_confirmation','Terminal passing PAD driver required')
    for stage in parent.STAGES:
        report=reports[stage];audit=audits[stage];rows=report['records'];summary=parent.validate_records(cases,rows)
        truncated=sum(not r['terminated'] for r in rows);error=audit['maximum_cosine_recomputation_error']
        require(audit['status']=='verified' and audit['stage']==stage and audit['recordings']==26
            and audit['all_weights_unchanged'] and audit['all_recomputed_cosines_finite']
            and math.isfinite(error) and 0<=error<=2e-6 and audit['optimizer_updates']==0
            and audit['summary']==report['summary']==summary and audit['truncated']==report['truncated']==truncated
            and audit['baseline_waveforms_checked']==(26 if stage=='control' else 20)
            and audit['paired_prefill_and_first_frame_verified']==(0 if stage=='control' else 26)
            and audit['goal_screen']==(goal_screen(summary) and truncated==0),'Complete independent PAD screen audit required')
    full=reports['full-pad']
    require(full['truncated']==0 and goal_screen(full['summary']),'Strict complete two-group PAD screen pass required')
    require(audits['control']['model_identity']==audits['full-pad']['model_identity'],'Control and candidate model identities differ')


def validate_scope(panel,cases,reference_sha256):
    validate_panel(panel)
    expanded={(c['case'],seed):dict(c,seed=seed) for seed in panel['seeds'] for c in panel['cases']}
    require(len(cases)==26 and len({(c['case'],c['seed']) for c in cases})==26,'Complete original screen required')
    for c in cases:
        full=expanded.get((c['case'],c['seed']))
        require(full is not None and all(full.get(k)==v for k,v in c.items()),'Original panel must preserve all screen metadata')
    require(all((c['reference_sha256']==reference_sha256)==(c['group']=='custom') for c in panel['cases']),
        'Full-panel reference dispatch scope changed')


def validate_panel_registration(panel_path,registration):
    require(panel_path.resolve()==Path(registration['panel']).resolve() and digest(panel_path)==registration['panel_sha256'],
        'Original previously registered 114-panel hash changed')


def measurement(panel,rows,screen):
    require(all(math.isfinite(r['cosine']) and -1<=r['cosine']<=1 for r in rows),'Nonfinite or invalid confirmation cosine')
    measured=validate_records(panel,rows)
    require(len(screen)==26 and len({(r['case'],r['seed']) for r in screen})==26,'Complete screening parity required')
    current={(r['case'],r['seed']):r for r in rows}
    for old in screen:
        r=current.get((old['case'],old['seed']))
        require(r is not None and r['audio_sha256']==old['audio_sha256'] and r['score']==old['score']
            and math.isfinite(old['cosine']) and abs(r['cosine']-old['cosine'])<=2e-6,
            'Screen waveform or score parity failed')
    return dict(measured,screening_recordings_reproduced_exactly=26,screening_scores_reproduced=26)


def evidence(run):
    from .evaluate import ASR_REV
    from scripts.score_latest_audio import score as score_text
    status_path=run.parent/'reference-prefix-generation-pad-status.json';status=read(status_path)
    require(status['status']=='complete' and status['reason']=='needs_broader_confirmation','Terminal passing PAD driver required')
    p,old,baseline=parent.verify(run);reports={};audits={};paths=[status_path,run/'protocol.json']
    for stage in parent.STAGES:
        dest=run/stage;report_path=dest/'results.json';audit_path=dest/'verification.json'
        report=read(report_path);audit=read(audit_path)
        require(audit['protocol_sha256']==digest(run/'protocol.json') and audit['results_sha256']==digest(report_path),
            'Parent audit/report binding changed')
        if stage=='full-pad':require(audit['control_audit_sha256']==digest(run/'control/verification.json'),'Parent control audit binding changed')
        for r in report['records']:
            require(r['stage']==stage and r['protocol_sha256']==digest(run/'protocol.json') and r['model_identity']==audit['model_identity']
                and r['asr_revision']==ASR_REV and r['decoding']==p['decoding'] and digest(dest/r['wav'])==r['audio_sha256']
                and digest(dest/r['trajectory'])==r['trajectory_sha256'] and digest(r['reference'])==r['reference_sha256']
                and r['score']==score_text(r['text'],r['hypothesis'],True),'Parent native speech evidence changed')
        parent.parity(baseline,report['records'],stage)
        reports[stage]=report;audits[stage]=audit;paths.extend((report_path,audit_path))
    validate_parent(status,p['cases'],reports,audits)
    training,_,_=parent.dpo.verify(Path(p['dpo_run']))
    previous_run=Path(training['previous']);previous=parent.dpo.previous.verify(previous_run)
    panel_path=run.parent/'joint-goal-confirmation/protocol.json';validate_panel_registration(panel_path,previous)
    panel=read(panel_path);validate_scope(panel,p['cases'],p['reference_sha256'])
    require(panel['generation']==p['generation'] and panel['decoding']==p['decoding']
        and panel['parent_protocol_sha256']==old['parent_protocol_sha256']
        and panel['preparation_source_sha256']==digest(Path(__file__).with_name('joint_confirmation.py')),
        'Original 114-panel registration changed')
    manifest=Path(p['config']['asr'])/'download-manifest.json'
    require(str(manifest) in p['evidence'] and digest(manifest)==p['evidence'][str(manifest)],'Inherited ASR manifest changed')
    paths.extend((panel_path,manifest,previous_run/'protocol.json'))
    return p,panel,reports['full-pad']['records'],audits['full-pad']['model_identity'],{str(path):digest(path) for path in paths}


def prepare(run,out,driver):
    p,panel,_,identity,hashes=evidence(run)
    tests=out.parent/'reference-prefix-generation-pad-confirmation-tests.log';content=tests.read_text(encoding='utf-8-sig')
    require('Ran 4 tests' in content and '\nOK\n' in content and 'skipped' not in content.lower(),'Confirmation CPU tests missing')
    require(parent.numeric_config()==parent.NUMERIC,'CPU FP32 numerical settings required')
    out=external(out.resolve());out.mkdir(exist_ok=False)
    write(out/'protocol.json',dict(parent_run=str(run),parent_protocol_sha256=digest(run/'protocol.json'),evidence_hashes=hashes,
        source=sources(),runtime=parent.prefix.runtime(),numeric_config=parent.NUMERIC,model_identity=identity,
        panel=str(run.parent/'joint-goal-confirmation/protocol.json'),reference_sha256=p['reference_sha256'],
        cases=[dict(c,seed=seed) for seed in panel['seeds'] for c in panel['cases']],
        config=p['config'],generation=p['generation'],decoding=p['decoding'],
        asr_manifest_sha256=p['evidence'][str(Path(p['config']['asr'])/'download-manifest.json')],
        tests=str(tests),tests_sha256=digest(tests),driver=str(driver),driver_sha256=digest(driver),
        scope='Original 114 recordings: 88 development and 26 custom; preserve both seeds and every original case field',
        goal_achieved=False,global_model_selection_changed=False,optimizer_updates=0))


def verify(out):
    p=read(out/'protocol.json')
    require(digest(Path(p['config']['asr'])/'download-manifest.json')==p['asr_manifest_sha256'],'Registered ASR manifest changed')
    original,panel,screen,identity,hashes=evidence(Path(p['parent_run']))
    require(p['source']==sources() and p['runtime']==parent.prefix.runtime() and p['numeric_config']==parent.numeric_config()==parent.NUMERIC
        and p['parent_protocol_sha256']==digest(Path(p['parent_run'])/'protocol.json') and p['evidence_hashes']==hashes
        and p['model_identity']==identity and digest(p['tests'])==p['tests_sha256'] and digest(p['driver'])==p['driver_sha256'],
        'Confirmation execution registration changed')
    require(p['panel']==str(Path(p['parent_run']).parent/'joint-goal-confirmation/protocol.json')
        and p['cases']==[dict(c,seed=seed) for seed in panel['seeds'] for c in panel['cases']]
        and all(p[k]==original[k] for k in ('reference_sha256','config','generation','decoding')),'Original confirmation scope/settings changed')
    return p,original,panel,screen


def load(out):
    p,original,panel,screen=verify(out);_,old,_=parent.verify(Path(p['parent_run']))
    wrapper,patch,identity=parent.load(original,old,'full-pad')
    require(identity==p['model_identity'],'Screened model identity changed')
    return p,panel,screen,wrapper,patch,identity


def generate(out):
    import soundfile as sf
    from .preference_rollouts import capture_generation
    p,panel,screen,wrapper,patch,identity=load(out);dest=out/'speech';dest.mkdir(exist_ok=False)
    records=[];prompts={};original={(r['case'],r['seed']):r for r in screen}
    for c in p['cases']:
        require(digest(c['reference'])==c['reference_sha256'],'Reference changed')
        if c['reference'] not in prompts:
            with torch.inference_mode():prompts[c['reference']]=wrapper.create_voice_clone_prompt(ref_audio=c['reference'],ref_text=c['reference_text'],x_vector_only_mode=False)
        prompt=prompts[c['reference']];torch.manual_seed(c['seed']);np.random.seed(c['seed'])
        with patch.use_reference(c['reference_sha256']),torch.inference_mode(),capture_generation(wrapper.model) as captured:
            waves,sr=wrapper.generate_voice_clone(text=c['text'],language='Auto',voice_clone_prompt=prompt,non_streaming_mode=True,**p['generation'])
        require(sr==24000 and len(waves[0])>0 and np.isfinite(waves[0]).all() and len(waves[0])/sr<=30,'Invalid generated waveform')
        path=dest/f"{c['seed']}-{c['case']}.wav";sf.write(path,waves[0],sr,subtype='PCM_16');wave,sr=sf.read(path,dtype='float32')
        with torch.inference_mode():
            vector=wrapper.model.extract_speaker_embedding(audio=wave,sr=sr).flatten();ref=prompt[0].ref_spk_embedding.flatten()
            cosine=float(torch.nn.functional.cosine_similarity(vector[None],ref[None]))
        require(math.isfinite(cosine),'Nonfinite generated cosine');trajectory=path.with_suffix('.pt')
        trace=patch.last_trace;routing=wrapper.model.reference_prefix_routing.last_trace
        save(trajectory,dict(codes=captured['codes'],first_sequence=captured['first_sequence'],terminated=captured['terminated'],
            ref_codes=prompt[0].ref_code.detach().cpu(),ref_xvec=ref.detach().cpu(),input_ids=ids(wrapper,c['text']).cpu(),
            ref_ids=ids(wrapper,c['reference_text'],True).cpu(),reference_sha256=c['reference_sha256'],pad_trace=trace,routing=routing))
        row=dict(c,wav=path.name,audio_sha256=digest(path),trajectory=trajectory.name,trajectory_sha256=digest(trajectory),
            cosine=cosine,seconds=len(wave)/sr,frames=len(captured['codes']),terminated=captured['terminated'],
            pad_trace=trace,routing=routing,model_identity=identity,protocol_sha256=digest(out/'protocol.json'))
        key=c['case'],c['seed']
        if key in original:require(row['audio_sha256']==original[key]['audio_sha256'],'Screening waveform parity failed')
        write(path.with_suffix('.json'),row);records.append(row);print(f'PAD confirmation: {len(records)}/114',flush=True)
    parent.unchanged(wrapper,patch,identity)
    write(dest/'generated.json',dict(records=records,all_weights_unchanged=True,screening_recordings_reproduced_exactly=26,
        model_identity=identity,protocol_sha256=digest(out/'protocol.json')))


def score(out):
    import soundfile as sf
    import torchaudio
    from transformers import WhisperForConditionalGeneration,WhisperProcessor
    from .evaluate import ASR_REV
    from scripts.score_latest_audio import score as score_text
    p,_,panel,screen=verify(out);dest=out/'speech';require(not (dest/'results.json').exists(),'Do not overwrite scored confirmation')
    generated=read(dest/'generated.json');require(generated['all_weights_unchanged'] and generated['screening_recordings_reproduced_exactly']==26
        and generated['model_identity']==p['model_identity'] and generated['protocol_sha256']==digest(out/'protocol.json'),'Generation receipt differs')
    asr=Path(p['config']['asr']);manifest=read(asr/'download-manifest.json')
    require(manifest['repo']=='BuzzASR/mongolian' and manifest['revision']==ASR_REV,'ASR mismatch')
    for name,item in manifest['files'].items():require(digest(asr/name)==item['sha256'],'ASR weights changed')
    processor=WhisperProcessor.from_pretrained(asr,local_files_only=True)
    model=WhisperForConditionalGeneration.from_pretrained(asr,torch_dtype=torch.float32,local_files_only=True).to('cuda').eval()
    records=generated['records']
    for r in records:
        require(digest(dest/r['wav'])==r['audio_sha256'],'Audio changed')
        wave,sr=sf.read(dest/r['wav'],dtype='float32');wave=torchaudio.functional.resample(torch.from_numpy(wave),sr,16000).numpy()
        batch=processor(wave,sampling_rate=16000,return_tensors='pt',return_attention_mask=True).to('cuda')
        with torch.inference_mode():tokens=model.generate(batch.input_features,attention_mask=batch.attention_mask,**p['decoding'])
        hypothesis=processor.batch_decode(tokens,skip_special_tokens=True)[0].strip()
        r.update(hypothesis=hypothesis,score=score_text(r['text'],hypothesis,True),decoding=p['decoding'],asr_revision=ASR_REV)
    measured=measurement(panel,records,screen)
    write(dest/'results.json',dict(records=records,summary=measured['summary'],truncated=measured['truncated']))


def cosine_error(recomputed,stored):
    require(math.isfinite(recomputed) and math.isfinite(stored),'Nonfinite independent cosine')
    return abs(recomputed-stored)


def audit(out):
    import soundfile as sf
    from .evaluate import ASR_REV
    from scripts.score_latest_audio import score as score_text
    require(not (out/'verification.json').exists(),'Do not overwrite independent confirmation audit')
    p,panel,screen,wrapper,patch,identity=load(out);dest=out/'speech';report=read(dest/'results.json');rows=report['records']
    measured=measurement(panel,rows,screen);refs={};maximum=0.;screen_map={(r['case'],r['seed']):r for r in screen}
    special_ids,full_pad,half_pad=patch._pads();paired=0
    for r in rows:
        require(r['protocol_sha256']==digest(out/'protocol.json') and r['model_identity']==identity and r['asr_revision']==ASR_REV
            and r['decoding']==p['decoding'] and digest(dest/r['wav'])==r['audio_sha256']
            and digest(dest/r['trajectory'])==r['trajectory_sha256'] and digest(r['reference'])==r['reference_sha256']
            and r['score']==score_text(r['text'],r['hypothesis'],True),'Confirmation speech provenance changed')
        t=torch.load(dest/r['trajectory'],map_location='cpu',weights_only=True);codes=t['codes'];seq=t['first_sequence'][0];eos=wrapper.model.config.talker_config.codec_eos_token_id
        require(codes.shape==(r['frames'],16) and len(codes)>0 and len(seq)==len(codes)+1 and torch.equal(seq[:len(codes)],codes[:,0])
            and bool(seq[-1]==eos)==r['terminated']==t['terminated'] and not bool((seq[:-1]==eos).any())
            and t['reference_sha256']==r['reference_sha256'],'Native sampled trajectory/EOS differs')
        require(torch.equal(t['input_ids'],ids(wrapper,r['text']).cpu()) and torch.equal(t['ref_ids'],ids(wrapper,r['reference_text'],True).cpu()),'Transcript token IDs differ')
        if r['reference'] not in refs:
            with torch.inference_mode():
                prompt=wrapper.create_voice_clone_prompt(ref_audio=r['reference'],ref_text=r['reference_text'],x_vector_only_mode=False)
                ref,refsr=sf.read(r['reference'],dtype='float32');embedding=wrapper.model.extract_speaker_embedding(audio=ref,sr=refsr).flatten()
            refs[r['reference']]=(prompt[0],embedding)
        prompt,embedding=refs[r['reference']]
        require(torch.equal(t['ref_codes'],prompt.ref_code.cpu()) and torch.equal(t['ref_xvec'],prompt.ref_spk_embedding.flatten().cpu()),'Native original reference conditioning differs')
        parent.validate_trace(r,t,p,'full-pad')
        require(r['pad_trace']['special_token_ids_sha256']==tensor_sha256(special_ids) and r['pad_trace']['full_pad_sha256']==tensor_sha256(full_pad)
            and r['pad_trace']['half_pad_sha256']==tensor_sha256(half_pad),'Native PAD reconstruction differs')
        key=r['case'],r['seed']
        if key in screen_map:
            previous=screen_map[key];previous_path=Path(p['parent_run'])/'full-pad'/previous['trajectory']
            require(digest(previous_path)==previous['trajectory_sha256'],'Screen trajectory changed')
            prior=torch.load(previous_path,map_location='cpu',weights_only=True)
            parent.paired_prefill(r,t,previous,prior);require(torch.equal(t['codes'],prior['codes']),'Screen codec trajectory differs');paired+=1
        wave,sr=sf.read(dest/r['wav'],dtype='float32');require(sr==24000 and wave.ndim==1 and len(wave)>0 and np.isfinite(wave).all(),'Invalid confirmation waveform')
        with torch.inference_mode():
            vector=wrapper.model.extract_speaker_embedding(audio=wave,sr=sr).flatten()
            cosine=float(torch.nn.functional.cosine_similarity(vector[None],embedding[None]))
        maximum=max(maximum,cosine_error(cosine,r['cosine']))
    require(paired==26 and maximum<=2e-6 and report['summary']==measured['summary'] and report['truncated']==measured['truncated'],'Independent complete confirmation differs')
    parent.unchanged(wrapper,patch,identity);verify(out)
    write(out/'verification.json',dict(status='verified',recordings=114,**measured,all_weights_unchanged=True,
        model_identity=identity,native_conditioning_verified=True,pad_traces_verified=114,all_recomputed_cosines_finite=True,
        maximum_cosine_recomputation_error=maximum,screen_prefill_and_codecs_verified=paired,optimizer_updates=0,
        protocol_sha256=digest(out/'protocol.json'),results_sha256=digest(dest/'results.json'),
        parent_audit_sha256=digest(Path(p['parent_run'])/'full-pad/verification.json'),goal_achieved=False,global_model_selection_changed=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=['prepare','generate','score','audit'])
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--run',type=Path);parser.add_argument('--driver',type=Path)
    args=parser.parse_args();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if args.phase=='prepare':prepare(args.run,args.out,args.driver)
    else:{'generate':generate,'score':score,'audit':audit}[args.phase](args.out)


if __name__=='__main__':main()
