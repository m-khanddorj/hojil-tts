"""Registered inference-only 2x2 reference-routing ablation; never fits/promotes.

Only the two missing half-reference development arms are generated. The full
development and custom arms are shared historical controls, not fresh repeats.
"""
from dataclasses import asdict
import hashlib
import importlib.util
import json
import math
from pathlib import Path

PROTOCOL_SHA = '89463810f5f9fc5d75d3eabf16144e30872aa1488b0605418f4f604abee12651'
HELPER_SHA = '612aff5b6088cad3ccd6da7f6ab000cbf458c851efa26a840c8fc6042c051191'
STEPS = (0, 32)
FLAGS = dict(diagnostic_only=True, optimizer_updates=0, goal_achieved=False,
             global_model_selection_changed=False, original114_qualification=False)
SETTINGS = dict(checkpoints=list(STEPS), new_development_recordings_per_checkpoint=64,
    reused_full_development_per_checkpoint=64, reused_custom_per_checkpoint=16,
    new_half_reference=True, continuation_pad='full', seeds=[2026, 2027],
    controls='shared historical recordings, not independent repeats',
    confirmation='Changed routing requires a separately registered original114 confirmation; old gates unchanged')


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)


def helper(path):
    require(digest(path) == HELPER_SHA, 'Immutable parent evidence helper changed')
    spec = importlib.util.spec_from_file_location('routing_parent_evidence', path)
    api = importlib.util.module_from_spec(spec); spec.loader.exec_module(api)
    return api


def parent_evidence(train, helper_path):
    train = Path(train); api = helper(helper_path)
    require(train.resolve() == api.TRAIN.resolve() and digest(train/'protocol.json') == PROTOCOL_SHA,
            'Exact registered v2 parent required')
    p, e, _ = api.metadata()
    evidence, receipt = api.parent_evidence(p)
    require(receipt['step'] == 32 and receipt['reason'] in
        ('maximum_updates_reached', 'consecutive_pareto_regressions'),
        'Closed failed terminal32 with complete audited history required')
    return p, e, evidence, receipt


def panel(cases):
    """The only changed case field is half_reference on all64 development rows."""
    require(len(cases) == 80 and len({(c['case'], c['seed']) for c in cases}) == 80,
            'Complete original80 metadata required')
    dev = [c for c in cases if c['group'] == 'development']
    custom = [c for c in cases if c['group'] == 'custom']
    require(len(dev) == 64 and len(custom) == 16
        and len({c['case'] for c in dev}) == 32 and len({c['speaker_id'] for c in dev}) == 16
        and len({c['case'] for c in custom}) == 8 and len({c['speaker_id'] for c in custom}) == 1,
        'Exact64/16 panel required')
    for c in cases:
        require(type(c['half_reference']) is bool and c['half_reference'] == (c['group'] == 'custom'),
                'Original routing changed')
    for group in (dev, custom):
        for name in {c['case'] for c in group}:
            pair = [c for c in group if c['case'] == name]
            require({c['seed'] for c in pair} == {2026, 2027} and len(pair) == 2
                and {k:v for k,v in pair[0].items() if k != 'seed'} ==
                    {k:v for k,v in pair[1].items() if k != 'seed'}, 'Paired metadata/seeds differ')
    return [dict(c, half_reference=True) for c in dev]


def validate_development(cases, records):
    from scripts.score_latest_audio import score as score_text
    from .targeted_voice import aggregate
    expected = {(c['case'], c['seed']): c for c in cases}
    require(len(cases) == len(expected) == len(records) == 64
        and {(r['case'], r['seed']) for r in records} == set(expected), 'Exact64 new records required')
    for r in records:
        require(all(r.get(k) == v for k,v in expected[r['case'],r['seed']].items())
            and r['group'] == 'development' and r['half_reference'] is True, 'Changed development metadata/route')
        require(type(r['terminated']) is bool and math.isfinite(r['cosine']) and -1 <= r['cosine'] <= 1,
                'Invalid termination or voice cosine')
        require(r['score'] == score_text(r['text'], r['hypothesis'], True), 'Independent ASR edit counts differ')
    summary = aggregate(records)
    require(set(summary) == {'development'} and summary['development']['speakers'] == 16,
            'Development speaker coverage changed')
    return summary


def combined(cases, historical, new_records):
    from .reference_masked_supervision_evaluation import validate_records
    original = validate_records(cases, historical)
    validate_development(panel(cases), new_records)
    custom = [r for r in historical if r['group'] == 'custom']
    routed_cases = [dict(c, half_reference=True) for c in cases]
    candidate = validate_records(routed_cases, list(new_records)+custom)
    truncated = sum(not r['terminated'] for r in list(new_records)+custom)
    passed = truncated == 0 and all(v['pooled_cer'] < .03 and v['speaker_macro_cosine'] > .984
                                   for v in candidate.values())
    return dict(original_summary=original, half_development_shared_custom_summary=candidate,
        truncated=truncated, calibration_thresholds_met=passed, **FLAGS)


def validate_test_log(log):
    require(any(line.startswith('Ran 8 tests in ') for line in log.splitlines())
        and log.strip().endswith('OK') and 'FAILED' not in log and 'skipped' not in log.lower(),
        'Eight passing focused tests without skips required')


def prepare(train, out, driver, helper_path, test_log):
    train, out = Path(train), Path(out)
    require(out.parent.resolve() == train.parent.resolve() and out.name == 'masked-supervision-routing-ablation-v1'
        and not out.exists(), 'New external directory required; no implicit restart')
    p, e, evidence, receipt = parent_evidence(train, helper_path)
    cases = panel(e['cases'])
    log = Path(test_log).read_text(encoding='utf-8-sig')
    validate_test_log(log)
    paths = [Path(__file__), Path(__file__).parents[1]/'tests/test_reference_masked_supervision_routing_ablation.py',
             Path(driver), Path(helper_path), Path(test_log), Path(e['asr_manifest'])]
    # Pin recognizer implementation bytes without importing or loading it.
    from importlib.metadata import distribution
    for package, names in {
        'transformers': ('transformers/models/whisper/modeling_whisper.py',
            'transformers/models/whisper/processing_whisper.py','transformers/models/whisper/feature_extraction_whisper.py',
            'transformers/models/whisper/tokenization_whisper.py','transformers/models/whisper/generation_whisper.py',
            'transformers/generation/utils.py'),
        'torchaudio': ('torchaudio/functional/functional.py',), 'soundfile': ('soundfile.py',),
    }.items():
        installed = distribution(package)
        paths.extend(Path(installed.locate_file(name)) for name in names)
    for step in STEPS:
        dest = train/f'eval-{step:06d}'
        for r in read(dest/'results.json')['records']:
            for path, sha in ((dest/r['wav'],r['audio_sha256']), (dest/r['trajectory'],r['trajectory_sha256']),
                              (Path(r['reference']),r['reference_sha256'])):
                require(digest(path) == sha, 'Historical control artifact changed')
                evidence[str(path.resolve())] = sha
    for path in paths: evidence[str(path.resolve())] = digest(path)
    protocol = dict(train=str(train.resolve()), helper=str(Path(helper_path).resolve()), parent=receipt,
        evidence=evidence, cases=cases, original_cases=e['cases'], settings=SETTINGS,
        training_protocol_sha256=PROTOCOL_SHA, speech_protocol_sha256=digest(train/'speech-protocol.json'),
        config=p['config'], features=p['features'], features_audit_sha256=e['features_audit_sha256'],
        generation=e['generation'], decoding=e['decoding'], asr_manifest=e['asr_manifest'],
        asr_manifest_sha256=e['asr_manifest_sha256'], numeric_config=p['numeric_config'], runtime=p['runtime'], **FLAGS)
    out.mkdir(); write(out/'protocol.json', protocol)
    return protocol


def verify(out, protocol_sha):
    out = Path(out)
    require(digest(out/'protocol.json') == protocol_sha, 'Registered ablation protocol changed')
    e = read(out/'protocol.json')
    require(e['settings'] == SETTINGS and all(e.get(k) == v for k,v in FLAGS.items()), 'Ablation contract changed')
    for path, sha in e['evidence'].items(): require(digest(path) == sha, 'Bound evidence changed: '+path)
    p, original, evidence, receipt = parent_evidence(e['train'], e['helper'])
    require(e['parent'] == receipt and all(e['evidence'].get(k) == v for k,v in evidence.items())
        and e['training_protocol_sha256'] == PROTOCOL_SHA
        and e['speech_protocol_sha256'] == digest(Path(e['train'])/'speech-protocol.json')
        and e['original_cases'] == original['cases'] and e['cases'] == panel(original['cases']), 'Inherited scope changed')
    for field in ('config','features','numeric_config','runtime'):
        require(e[field] == p[field], 'Inherited runtime/feature identity changed')
    for field in ('features_audit_sha256','generation','decoding','asr_manifest','asr_manifest_sha256'):
        require(e[field] == original[field], 'Inherited scoring/generation identity changed')
    return p, e


def native_verify(out, protocol_sha, step):
    from . import reference_masked_supervision_speech_v2 as speech
    require(step in STEPS, 'Unregistered endpoint')
    p,e = verify(out, protocol_sha)
    original_p, original_e = speech.verify(Path(e['train']))
    require(original_p == p, 'Native parent verification differs')
    for s in (0,8,16,32): speech.verified_receipt(Path(e['train']),p,original_e,s)
    speech.permitted(Path(e['train']),p,original_e,step)
    return p,e


def directory(out, step):
    require(step in STEPS, 'Unregistered endpoint')
    return Path(out)/f'half-development-{step:06d}'


def bound_generation(out, e, step, scored=None):
    dest = directory(out,step); generated = read(dest/'generated.json')
    checkpoint_sha = digest(Path(e['train'])/f'checkpoint-{step:06d}.pt')
    require(generated['all_weights_unchanged'] is True and generated['checkpoint_sha256'] == checkpoint_sha
        and generated['ablation_protocol_sha256'] == digest(Path(out)/'protocol.json'), 'Generation receipt changed')
    expected = {(c['case'],c['seed']):c for c in e['cases']}; records = generated['records']
    require(len(records) == 64 and {(r['case'],r['seed']) for r in records} == set(expected), 'Incomplete new64 generation')
    current = {} if scored is None else {(r['case'],r['seed']):r for r in scored}
    if scored is not None: require(len(scored) == 64 and set(current) == set(expected), 'Scored scope differs')
    for r in records:
        key = r['case'],r['seed']
        require(all(r.get(k) == v for k,v in expected[key].items()) and r['step'] == step
            and r['checkpoint_sha256'] == checkpoint_sha
            and r['ablation_protocol_sha256'] == digest(Path(out)/'protocol.json')
            and digest(dest/r['wav']) == r['audio_sha256'] and digest(dest/r['trajectory']) == r['trajectory_sha256']
            and digest(r['reference']) == r['reference_sha256'], 'Generation artifact/metadata changed')
        if scored is not None:
            require(set(current[key]) == set(r)|{'hypothesis','score','asr_revision','decoding'}
                and all(current[key].get(k) == v for k,v in r.items()), 'Scoring modified native record')
    return generated



def generate(run, step, protocol_sha):
    from .engine import ids, save
    from .reference_masked_supervision_speech_v2 import capture_inputs, unchanged
    import numpy as np
    import soundfile as sf
    import torch
    from . import reference_masked_supervision_training_v2 as trainer
    from . import reference_masked_supervision_features as features
    from .preference_rollouts import capture_generation
    run=Path(run); p,e=native_verify(run,protocol_sha,step)
    _,wrapper,core,state=trainer.load(Path(e['train']),step,device='cpu',training=False)
    features.assert_native_components(wrapper,p['features'])
    _,assets=features.load_features(p['features'])
    dest=directory(run,step); dest.mkdir(exist_ok=True)
    require(not (dest/'generation-started.json').exists(), 'No implicit generation restart')
    write(dest/'generation-started.json', dict(protocol_sha256=protocol_sha,step=step))
    require(not (dest/'generated.json').exists(), 'Complete generation already exists')
    checkpoint_sha=digest(trainer.checkpoint_path(Path(e['train']),step)); protocol_sha=digest(run/'protocol.json')
    prompts={}; records=[]
    for c in e['cases']:
        path=dest/f"{c['seed']}-{c['case']}.wav"; meta=path.with_suffix('.json')
        require(not meta.exists(), 'No prior partial generation')
        require(not path.exists() and not path.with_suffix('.pt').exists(), 'Orphan generation artifact requires inspection')
        require(digest(c['reference'])==c['reference_sha256'], 'Reference WAV changed')
        asset=assets[c['feature_id']]
        if c['reference'] not in prompts:
            with torch.inference_mode():
                prompt=wrapper.create_voice_clone_prompt(ref_audio=c['reference'],ref_text=c['reference_text'],x_vector_only_mode=False)
            require(torch.equal(prompt[0].ref_code.cpu(),asset['codes'])
                and torch.equal(prompt[0].ref_spk_embedding.flatten().cpu(),asset['speaker'])
                and torch.equal(ids(wrapper,c['reference_text'],True).cpu(),asset['ref_ids']), 'Fresh reference feature parity failed')
            prompts[c['reference']]=prompt
        prompt=prompts[c['reference']]
        torch.manual_seed(c['seed']); np.random.seed(c['seed'])
        with core.native(c['half_reference']), torch.inference_mode(), capture_generation(wrapper.model) as captured, \
                capture_inputs(wrapper.model.talker.model) as chunks:
            waves,sr=wrapper.generate_voice_clone(text=c['text'],language='Auto',voice_clone_prompt=prompt,
                non_streaming_mode=True,**e['generation'])
        require(sr==24000 and len(waves)==1 and len(waves[0])>0 and np.isfinite(waves[0]).all()
            and len(waves[0])/sr<=30, 'Invalid generated waveform')
        sf.write(path,waves[0],sr,subtype='PCM_16'); wave,sr=sf.read(path,dtype='float32')
        with torch.inference_mode():
            vector=wrapper.model.extract_speaker_embedding(audio=wave,sr=sr).flatten()
            cosine=float(torch.nn.functional.cosine_similarity(vector[None],asset['speaker'][None]))
        require(math.isfinite(cosine), 'Nonfinite generated voice cosine')
        trace=dict(core.last_trace,layout=asdict(core.last_trace['layout']))
        trajectory=path.with_suffix('.pt')
        save(trajectory,dict(codes=captured['codes'],first_sequence=captured['first_sequence'],
            terminated=captured['terminated'],native_inputs=torch.cat(chunks,dim=1)[0],
            input_ids=ids(wrapper,c['text']).cpu(),ref_ids=asset['ref_ids'],ref_codes=asset['codes'],
            ref_xvec=asset['speaker'],reference_sha256=c['reference_sha256'],trace=trace))
        r=dict(c,wav=path.name,audio_sha256=digest(path),trajectory=trajectory.name,trajectory_sha256=digest(trajectory),
            cosine=cosine,seconds=len(wave)/sr,frames=len(captured['codes']),terminated=captured['terminated'],
            step=step,checkpoint_sha256=checkpoint_sha,ablation_protocol_sha256=protocol_sha,trace=trace)
        write(meta,r); records.append(r); print(f'checkpoint {step}: {len(records)}/64',flush=True)
    unchanged(wrapper.model,core,state)
    require(len(records)==64, 'Incomplete calibration generation')
    write(dest/'generated.json',dict(records=records,all_weights_unchanged=True,checkpoint_sha256=checkpoint_sha,
        ablation_protocol_sha256=protocol_sha))


def score(run, step, protocol_sha):
    import soundfile as sf
    import torch, torchaudio
    from transformers import WhisperForConditionalGeneration,WhisperProcessor
    from .evaluate import ASR_REV
    from scripts.score_latest_audio import score as score_text
    from . import reference_masked_supervision_training_v2 as trainer
    run=Path(run); p,e=native_verify(run,protocol_sha,step); dest=directory(run,step)
    require(not (dest/'results.json').exists(), 'Do not overwrite scored calibration')
    generated=bound_generation(run,e,step); records=generated['records']
    manifest=read(e['asr_manifest']); asr=Path(p['config']['asr'])
    require(manifest['repo']=='BuzzASR/mongolian' and manifest['revision']==ASR_REV, 'Original ASR required')
    for name,item in manifest['files'].items(): require(digest(asr/name)==item['sha256'], 'ASR file changed')
    processor=WhisperProcessor.from_pretrained(asr,local_files_only=True)
    model=WhisperForConditionalGeneration.from_pretrained(asr,torch_dtype=torch.float32,local_files_only=True).to('cuda').eval()
    for r in records:
        require(digest(dest/r['wav'])==r['audio_sha256'], 'Generated waveform changed')
        wave,sr=sf.read(dest/r['wav'],dtype='float32')
        wave=torchaudio.functional.resample(torch.from_numpy(wave),sr,16000).numpy()
        batch=processor(wave,sampling_rate=16000,return_tensors='pt',return_attention_mask=True).to('cuda')
        with torch.inference_mode(): tokens=model.generate(batch.input_features,attention_mask=batch.attention_mask,**e['decoding'])
        hypothesis=processor.batch_decode(tokens,skip_special_tokens=True)[0].strip()
        r.update(hypothesis=hypothesis,score=score_text(r['text'],hypothesis,True),asr_revision=ASR_REV,decoding=e['decoding'])
    summary=validate_development(e['cases'],records)
    write(dest/'results.json',dict(records=records,summary=summary,truncated=sum(not r['terminated'] for r in records)))


def audit(run, step, protocol_sha):
    from .engine import ids
    from .reference_masked_supervision_speech_v2 import unchanged
    import numpy as np
    import soundfile as sf
    import torch
    from .evaluate import ASR_REV
    from . import reference_masked_supervision_training_v2 as trainer
    from . import reference_masked_supervision_policy_v2 as policy
    from . import reference_masked_supervision_features as features
    from .reference_masked_supervision import NativeReference
    run=Path(run); p,e=native_verify(run,protocol_sha,step); dest=directory(run,step)
    path=run/f'verification-{step:06d}.json'; require(not path.exists(), 'Do not overwrite native speech audit')
    _,wrapper,core,state=trainer.load(Path(e['train']),step,device='cpu',training=False)
    features.assert_native_components(wrapper,p['features']); _,assets=features.load_features(p['features'])
    report=read(dest/'results.json'); records=report['records']; summary=validate_development(e['cases'],records)
    bound_generation(run,e,step,records)
    checkpoint_sha=digest(trainer.checkpoint_path(Path(e['train']),step)); speech_sha=digest(run/'protocol.json')
    maximum=0.; references={}
    for r in records:
        require(r['step']==step and r['checkpoint_sha256']==checkpoint_sha and r['ablation_protocol_sha256']==speech_sha
            and r['asr_revision']==ASR_REV and r['decoding']==e['decoding']
            and digest(dest/r['wav'])==r['audio_sha256'] and digest(dest/r['trajectory'])==r['trajectory_sha256']
            and digest(r['reference'])==r['reference_sha256'], 'Speech artifact identity changed')
        t=torch.load(dest/r['trajectory'],map_location='cpu',weights_only=True)
        codes=t['codes']; seq=t['first_sequence'][0]; eos=wrapper.model.config.talker_config.codec_eos_token_id
        require(codes.dtype==torch.long and seq.dtype==torch.long and codes.shape==(r['frames'],16) and len(codes)>0
            and bool(((codes>=0)&(codes<2048)).all()) and seq.shape==(len(codes)+1,)
            and torch.equal(seq[:-1],codes[:,0]) and bool(seq[-1]==eos)==r['terminated']==t['terminated']
            and t['reference_sha256']==r['reference_sha256'], 'Native codec/termination record differs')
        asset=assets[r['feature_id']]
        require(all(t[k].dtype==torch.long for k in ('input_ids','ref_ids','ref_codes'))
            and t['ref_xvec'].dtype==t['native_inputs'].dtype==torch.float32
            and torch.isfinite(t['native_inputs']).all() and torch.isfinite(t['ref_xvec']).all()
            and torch.equal(t['input_ids'],ids(wrapper,r['text']).cpu())
            and torch.equal(t['ref_ids'],ids(wrapper,r['reference_text'],True).cpu())
            and torch.equal(t['ref_ids'],asset['ref_ids']) and torch.equal(t['ref_codes'],asset['codes'])
            and torch.equal(t['ref_xvec'],asset['speaker']), 'Native conditioning feature differs')
        if r['reference'] not in references:
            with torch.inference_mode():
                prompt=wrapper.create_voice_clone_prompt(ref_audio=r['reference'],ref_text=r['reference_text'],x_vector_only_mode=False)
            require(torch.equal(prompt[0].ref_code.cpu(),asset['codes'])
                and torch.equal(prompt[0].ref_spk_embedding.flatten().cpu(),asset['speaker']), 'Independent reference extraction differs')
            references[r['reference']]=asset['speaker']
        ref=NativeReference(asset['ref_ids'],asset['codes'],asset['speaker'],r['reference_sha256'],e['features_audit_sha256'])
        layout=core._layout(t['ref_ids'].shape[1]-5,t['input_ids'].shape[1]-8,len(t['ref_codes']))
        # Conditioning-only reconstruction, including for truncated outputs:
        # no likelihood is scored and no synthetic EOS is treated as observed.
        with torch.no_grad(),core._context('conditioning-audit',r['half_reference'],layout):
            rebuilt=core._build(t['input_ids'],codes,ref)
            require(torch.equal(rebuilt.embeddings,t['native_inputs']), 'Actual native prefix/continuation inputs differ')
        require(t['trace']==r['trace'] and r['trace']==dict(kind='native',half_reference=r['half_reference'],
            layout=asdict(layout),calls=1), 'Native routing trace differs')
        wave,sr=sf.read(dest/r['wav'],dtype='float32')
        require(sr==24000 and wave.ndim==1 and len(wave)>0 and np.isfinite(wave).all()
            and r['seconds']==len(wave)/sr and r['seconds']<=30, 'Invalid audited waveform or duration')
        with torch.inference_mode():
            vector=wrapper.model.extract_speaker_embedding(audio=wave,sr=sr).flatten()
            value=float(torch.nn.functional.cosine_similarity(vector[None],references[r['reference']][None]))
        require(math.isfinite(value) and math.isfinite(r['cosine']), 'Nonfinite independently extracted cosine')
        maximum=max(maximum,abs(value-r['cosine']))
    truncated=sum(not r['terminated'] for r in records)
    require(maximum<=2e-6 and summary==report['summary'] and truncated==report['truncated'], 'Independent metrics differ')
    unchanged(wrapper.model,core,state)
    movement=policy.movement(core)
    require(movement==state['adapter_movement'] and state['frozen_identity']==p['frozen_identity'],
        'Canonical checkpoint movement or policy identity differs')
    receipt=dict(status='verified',step=step,recordings=64,summary=summary,truncated=truncated,
        frozen_weights_verified=True,native_conditioning_verified=True,all_artifact_hashes_verified=True,
        all_recomputed_cosines_finite=True,max_cosine_recomputation_error=maximum,
        adapter_movement=movement,checkpoint_sha256=checkpoint_sha,
        protocol_sha256=digest(run/'protocol.json'),ablation_protocol_sha256=speech_sha,
        results_sha256=digest(dest/'results.json'),generated_sha256=digest(dest/'generated.json'),
        native_verification_sha256=digest(Path(e['train'])/'native-verification.json'),frozen_identity=state['frozen_identity'],
        features_audit_sha256=e['features_audit_sha256'],
        native_inputs_verified=64,**FLAGS)
    require(truncated == 0, 'Truncated ablation cannot be a verified complete arm')
    verify(run,protocol_sha)
    write(path,receipt)


def audited_arm(out, p, e, step):
    """Pure artifact recheck; does not repeat the already bound native audit."""
    from .evaluate import ASR_REV
    out = Path(out); dest = directory(out,step); train = Path(e['train'])
    audit = read(out/f'verification-{step:06d}.json'); report = read(dest/'results.json')
    state = read(train/f'checkpoint-{step:06d}.json')
    require(audit['status'] == 'verified' and audit['step'] == step and audit['recordings'] == 64
        and audit['native_inputs_verified'] == 64 and audit['truncated'] == 0
        and all(audit.get(k) is True for k in ('frozen_weights_verified','native_conditioning_verified',
            'all_artifact_hashes_verified','all_recomputed_cosines_finite'))
        and all(audit.get(k) == v for k,v in FLAGS.items()), 'Incomplete native arm audit')
    maximum = audit['max_cosine_recomputation_error']
    require(math.isfinite(maximum) and 0 <= maximum <= 2e-6, 'Native cosine recomputation differs')
    require(audit['protocol_sha256'] == audit['ablation_protocol_sha256'] == digest(out/'protocol.json')
        and audit['results_sha256'] == digest(dest/'results.json')
        and audit['generated_sha256'] == digest(dest/'generated.json')
        and audit['native_verification_sha256'] == digest(train/'native-verification.json')
        and audit['features_audit_sha256'] == e['features_audit_sha256']
        and audit['checkpoint_sha256'] == state['sha256'] == digest(train/f'checkpoint-{step:06d}.pt')
        and state['step'] == step and state['protocol_sha256'] == PROTOCOL_SHA
        and audit['frozen_identity'] == state['frozen_identity'] == p['frozen_identity']
        and audit['adapter_movement'] == state['adapter_movement'], 'Native arm provenance changed')
    records = report['records']; bound_generation(out,e,step,records)
    summary = validate_development(e['cases'],records)
    require(summary == report['summary'] == audit['summary'] and report['truncated'] == 0
        and all(r['terminated'] and r['asr_revision'] == ASR_REV and r['decoding'] == e['decoding'] for r in records),
        'Native arm scored evidence changed')
    return audit, records


def finalize(out, protocol_sha):
    out = Path(out); p,e = verify(out,protocol_sha)
    require(not (out/'results.json').exists(), 'Do not overwrite completed ablation')
    arms = {}; evidence = {}
    for step in STEPS:
        audit, new = audited_arm(out,p,e,step)
        previous = Path(e['train'])/f'eval-{step:06d}/results.json'
        historical = read(previous)['records']
        comparison = combined(e['original_cases'],historical,new)
        a = {(r['case'],r['seed']):r for r in historical if r['group']=='development'}
        b = {(r['case'],r['seed']):r for r in new}
        changes = {}
        for label, sign in (('better',-1),('same',0),('worse',1)):
            changes[label] = sum(((b[k]['score']['character_errors'] > a[k]['score']['character_errors'])
                -(b[k]['score']['character_errors'] < a[k]['score']['character_errors'])) == sign for k in a)
        arms[str(step)] = dict(comparison, paired_development_character_error_change=changes,
            original_records=str(previous), original_results_sha256=digest(previous),
            original_audit_sha256=digest(Path(e['train'])/f'verification-{step:06d}.json'),
            new_records=str(directory(out,step)/'results.json'), new_results_sha256=audit['results_sha256'],
            new_audit_sha256=digest(out/f'verification-{step:06d}.json'),
            custom_records_reused=16, full_development_records_reused=64,
            shared_controls=True, independent_control_repeat=False)
        for path in (out/f'verification-{step:06d}.json',directory(out,step)/'generated.json',
                     directory(out,step)/'results.json'):
            evidence[str(path.resolve())] = digest(path)
    verify(out,protocol_sha)
    write(out/'results.json',dict(status='verified',new_recordings=128,reused_recordings=160,
        protocol_sha256=protocol_sha,arms=arms,evidence=evidence,
        scope='2x2 development conditioning comparison; custom controls shared; no original114 claim', **FLAGS))


def main():
    import argparse, os
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('prepare','generate','score','audit','finalize'))
    parser.add_argument('--out',required=True,type=Path); parser.add_argument('--train',type=Path)
    parser.add_argument('--driver',type=Path); parser.add_argument('--helper',type=Path)
    parser.add_argument('--test-log',type=Path); parser.add_argument('--protocol-sha')
    parser.add_argument('--step',type=int,choices=STEPS)
    args = parser.parse_args()
    if args.phase == 'prepare':
        require(all((args.train,args.driver,args.helper,args.test_log)), 'Preparation bindings required')
        prepare(args.train,args.out,args.driver,args.helper,args.test_log); return
    require(args.protocol_sha is not None, 'Explicit registered protocol hash required')
    p,e = verify(args.out,args.protocol_sha)
    if args.phase == 'finalize': finalize(args.out,args.protocol_sha); return
    require(args.step in STEPS, 'Endpoint required')
    os.environ['NUMBA_CACHE_DIR'] = p['config']['numba_cache']
    os.environ['TEMP'] = os.environ['TMP'] = p['config']['tmp']
    os.environ['HF_HUB_OFFLINE'] = os.environ['TRANSFORMERS_OFFLINE'] = '1'
    import torch
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    write(args.out/f'{args.phase}-{args.step:06d}-started.json',dict(protocol_sha256=args.protocol_sha))
    {'generate':generate,'score':score,'audit':audit}[args.phase](args.out,args.step,args.protocol_sha)


if __name__ == '__main__': main()
