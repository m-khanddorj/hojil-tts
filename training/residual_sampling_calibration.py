"""Paired residual-temperature calibration with an unchanged attention checkpoint.

This is a decoding experiment, not training or a passing development screen.
Every case and both seeds are retained. Artifacts must remain outside Git.
"""
import argparse
from contextlib import contextmanager
import os
from pathlib import Path

from .engine import digest, read, save, write
from .policy import external, fingerprint
from .targeted_voice import aggregate
from . import middle_attention_training as attention
from . import middle_attention_adapter as adapter


ARMS = {'control': .9, 'residual_06': .6}


def select_cases(cases):
    """Four hash-selected preference-validation speakers, two distinct cases each."""
    groups = {}
    for case in cases:
        if case['preference_split'] == 'validation':
            groups.setdefault(case['speaker_id'], []).append(case)
    eligible = [s for s, cs in groups.items() if len({c['case'] for c in cs}) >= 2]
    speakers = sorted(eligible, key=lambda s: fingerprint(['residual-temperature-v1', s]))[:4]
    if len(speakers) != 4:
        raise ValueError('Need four validation speakers with two cases each')
    selected = []
    for speaker in speakers:
        cs = sorted(groups[speaker], key=lambda c: fingerprint(['residual-temperature-v1', c['case']]))[:2]
        selected.extend(dict(c, group='calibration') for c in cs)
    if len({c['case'] for c in selected}) != 8:
        raise ValueError('Duplicate selected cases')
    return selected


def generation(parent, arm):
    if arm not in ARMS:
        raise ValueError('Unregistered decoding arm')
    defaults = dict(subtalker_dosample=True, subtalker_top_k=50,
                    subtalker_top_p=1., subtalker_temperature=.9)
    if any(k.startswith('subtalker_') and (k not in defaults or v != defaults[k])
           for k, v in parent.items()):
        raise ValueError('Parent residual settings differ from the registered control')
    return dict(parent, **dict(defaults,subtalker_temperature=ARMS[arm]))


@contextmanager
def observe_residual(model, expected):
    """Verify the actual predictor calls; do not change tensors or RNG state."""
    predictor = model.talker.code_predictor
    original = predictor.generate
    receipt = dict(calls=0, settings=dict(expected))

    def checked(*args, **kwargs):
        if any(kwargs.get(k) != v for k, v in expected.items()):
            raise ValueError('Actual residual generation settings differ')
        receipt['calls'] += 1
        return original(*args, **kwargs)

    predictor.generate = checked
    try:
        yield receipt
    finally:
        predictor.generate = original
    if not receipt['calls']:
        raise ValueError('Residual predictor was not called')


def gate(cases, seeds, control, candidate):
    """Calibration only: no dropped failures, no best-of-N, no goal promotion."""
    expected = {(c['case'], s) for c in cases for s in seeds}
    maps = []
    for records in (control, candidate):
        rows = {(r['case'], r['seed']): r for r in records}
        if len(records) != len(expected) or set(rows) != expected:
            raise ValueError('Incomplete or duplicate paired calibration')
        for c in cases:
            for seed in seeds:
                if any(rows[c['case'], seed][k] != v for k, v in c.items()):
                    raise ValueError('Calibration condition changed')
        maps.append(rows)
    a, b = (aggregate(records)['calibration'] for records in (control, candidate))
    worst = min(maps[1][k]['cosine'] - maps[0][k]['cosine'] for k in expected)
    terminated = all(r['terminated'] for records in (control, candidate) for r in records)
    passed = (terminated and b['speaker_macro_cosine'] > a['speaker_macro_cosine']
              and b['pooled_cer'] <= a['pooled_cer'] and b['pooled_wer'] <= a['pooled_wer']
              and worst >= -.005)
    return dict(control=a, candidate=b, worst_clip_cosine_change=worst,
                all_terminated=terminated, calibration_pass=passed, goal_achieved=False)


def sources():
    import importlib.util
    result = attention.source_hashes()
    result['residual_sampling_calibration.py'] = digest(Path(__file__))
    result['preference_rollouts.py'] = digest(Path(__file__).with_name('preference_rollouts.py'))
    result['confirm_joint_candidate.py'] = digest(Path(__file__).with_name('confirm_joint_candidate.py'))
    result['residual_sampling_tests'] = digest(Path(__file__).parent.parent/'tests/test_training_residual_sampling_calibration.py')
    for name in ('qwen_tts.core.models.modeling_qwen3_tts', 'qwen_tts.inference.qwen3_tts_model'):
        result[name] = digest(importlib.util.find_spec(name).origin)
    return result


def prepare(candidate, run, panel):
    from data.pipeline import text_key
    p = attention.verify(candidate)
    state = attention.read_checkpoint(candidate, 4, p)
    prior = read(candidate/'verification-000004.json')
    if prior['status'] != 'verified' or prior['recordings'] != 26:
        raise ValueError('Frozen candidate is not independently audited')
    pool = read(Path(p['pool'])/'protocol.json')
    cases = select_cases(pool['cases'])
    test_log = run.parent/'residual-sampling-tests.log'
    test_output = test_log.read_text(encoding='utf-8-sig')
    if 'Ran 3 tests' not in test_output or '\nOK\n' not in test_output or 'skipped' in test_output.lower():
        raise ValueError('Residual calibration tests have not passed')
    expanded = read(panel)
    from .confirm_joint_candidate import validate_panel
    validate_panel(expanded)
    for c in cases:
        if text_key(c['text']) == text_key(c['reference_text']):
            raise ValueError('Target and reference transcript overlap')
        for e in expanded['cases']:
            if (c['speaker_id'] == e['speaker_id'] or c['reference_sha256'] == e['reference_sha256']
                    or {text_key(c['text']), text_key(c['reference_text'])}
                    & {text_key(e['text']), text_key(e['reference_text'])}):
                raise ValueError('Calibration overlaps development/custom confirmation')
    run = external(run.resolve()); run.mkdir(exist_ok=False)
    write(run/'protocol.json', dict(candidate=str(candidate), step=4,
        candidate_protocol_sha256=digest(candidate/'protocol.json'),
        checkpoint_sha256=digest(candidate/'checkpoint-000004.pt'),
        candidate_audit_sha256=digest(candidate/'verification-000004.json'),
        frozen_backbone_sha256=state['frozen_backbone_sha256'],
        frozen_speaker_sha256=state['frozen_speaker_sha256'],
        pool_protocol_sha256=digest(Path(p['pool'])/'protocol.json'),
        panel=str(panel), panel_sha256=digest(panel), source=sources(),
        tests=str(test_log), tests_sha256=digest(test_log),
        runtime=attention.collection.runtime(), config=p['config'],
        cases=cases, seeds=[2026, 2027], arms={a:generation(p['generation'], a) for a in ARMS},
        decoding=p['decoding'],
        selection='Four hash-selected preference-validation speakers, two distinct cases each; original supervised training speakers, not unseen-test evidence',
        method='Frozen unfiltered middle-attention update4; change residual-code sampling temperature .9 to .6; main sampling and all weights unchanged',
        caveat='Residual embeddings feed future main steps; this is not acoustic-only intervention. Same seeds do not guarantee identical later main codes.',
        gate='Strict mean cosine gain, no pooled CER/WER increase, no clip cosine drop exceeding .005, all recordings terminated; passing only authorizes full26 screen then unchanged114 conditions with this declared generation policy',
        goal_achieved=False, global_model_selection_changed=False))


def verify(run):
    p = read(run/'protocol.json'); candidate = Path(p['candidate'])
    original = attention.verify(candidate)
    if (p['source'] != sources() or p['runtime'] != attention.collection.runtime()
            or p['candidate_protocol_sha256'] != digest(candidate/'protocol.json')
            or p['checkpoint_sha256'] != digest(candidate/'checkpoint-000004.pt')
            or p['candidate_audit_sha256'] != digest(candidate/'verification-000004.json')
            or p['tests_sha256'] != digest(p['tests'])
            or p['panel_sha256'] != digest(p['panel'])
            or p['pool_protocol_sha256'] != digest(Path(original['pool'])/'protocol.json')):
        raise ValueError('Calibration provenance changed')
    if p['arms'] != {a:generation(original['generation'], a) for a in ARMS}:
        raise ValueError('Generation policy changed')
    if p['cases'] != select_cases(read(Path(original['pool'])/'protocol.json')['cases']):
        raise ValueError('Calibration selection changed')
    for c in p['cases']:
        if digest(c['reference']) != c['reference_sha256']:
            raise ValueError('Reference changed')
    return p, original


def load(p, original):
    wrapper = attention.load(original)
    state = attention.read_checkpoint(Path(p['candidate']), 4, original)
    adapter.restore(wrapper.model, state['adapter'])
    wrapper.model.eval().requires_grad_(False)
    if (adapter.frozen_digest(wrapper.model) != p['frozen_backbone_sha256']
            or adapter.speaker_digest(wrapper.model) != p['frozen_speaker_sha256']):
        raise ValueError('Frozen candidate identity differs')
    return wrapper


def generate(run):
    import numpy as np
    import soundfile as sf
    import torch
    from .preference_rollouts import capture_generation
    p, original = verify(run); wrapper = load(p, original)
    # Digest includes acoustic weights, both adapters and speaker encoder.
    from .targeted_voice import frozen_digest as all_weights
    before = all_weights(wrapper.model)
    prompts = {}; records = []
    control = read(Path(p['candidate'])/'eval-000004/results.json')['records'][0]
    requests = [('native_control', control, control['seed'], original['generation'])]
    requests += [(arm, c, seed, settings) for arm, settings in p['arms'].items()
                 for seed in p['seeds'] for c in p['cases']]
    for index, (arm, c, seed, settings) in enumerate(requests):
        out = run/arm; out.mkdir(exist_ok=True)
        ref = c['reference']
        if digest(ref) != c['reference_sha256']:
            raise ValueError('Generation reference changed')
        if ref not in prompts:
            prompts[ref] = wrapper.create_voice_clone_prompt(ref_audio=ref,
                ref_text=c['reference_text'], x_vector_only_mode=False)
        wav = out/f"{seed}-{c['case']}.wav"
        if wav.exists():
            raise ValueError('Do not overwrite or implicitly resume generation')
        torch.manual_seed(seed); np.random.seed(seed)
        expected = dict(do_sample=True, temperature=settings.get('subtalker_temperature', .9),
                        top_k=50, top_p=1., max_new_tokens=15)
        with torch.inference_mode(), observe_residual(wrapper.model, expected) as observed, capture_generation(wrapper.model) as captured:
            waves, sr = wrapper.generate_voice_clone(text=c['text'], language='Auto',
                voice_clone_prompt=prompts[ref], non_streaming_mode=True, **settings)
        if sr != 24000 or not len(waves[0]) or not np.isfinite(waves[0]).all() or len(waves[0])/sr > 30:
            raise ValueError('Invalid audio')
        sf.write(wav, waves[0], sr, subtype='PCM_16')
        if arm == 'native_control':
            if digest(wav) != control['audio_sha256']:
                raise ValueError('Native frozen-candidate control does not reproduce prior waveform')
            write(out/'verification.json', dict(exact_waveform_parity=True,
                  audio_sha256=digest(wav), original=control['wav'], residual=observed))
            continue
        wave, sr = sf.read(wav, dtype='float32')
        with torch.inference_mode():
            vector = wrapper.model.extract_speaker_embedding(audio=wave, sr=sr).flatten()
            refvector = prompts[ref][0].ref_spk_embedding.flatten()
            cosine = float(torch.nn.functional.cosine_similarity(vector[None], refvector[None]))
        trajectory = wav.with_suffix('.pt')
        save(trajectory, dict(codes=captured['codes'], first_sequence=captured['first_sequence']))
        r = dict(c, arm=arm, seed=seed, wav=str(wav.relative_to(run)), audio_sha256=digest(wav),
                 trajectory=str(trajectory.relative_to(run)), trajectory_sha256=digest(trajectory),
                 cosine=cosine, seconds=len(wave)/sr, terminated=captured['terminated'], residual=observed,
                 protocol_sha256=digest(run/'protocol.json'), checkpoint_sha256=p['checkpoint_sha256'])
        records.append(r); write(wav.with_suffix('.json'), r)
        print(f'Calibration speech {index}/{len(requests)-1}: {arm} {c["case"]} {seed}', flush=True)
    if all_weights(wrapper.model) != before:
        raise ValueError('Frozen weights changed during generation')
    write(run/'generated.json', dict(records=records, full_model_sha256=before, weights_unchanged=True))


def score(run):
    import soundfile as sf
    import torch, torchaudio
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    from scripts.score_latest_audio import score as score_text
    from .evaluate import ASR_REV
    p, _ = verify(run); records = read(run/'generated.json')['records']
    asr = Path(p['config']['asr']); manifest = read(asr/'download-manifest.json')
    if manifest['repo'] != 'BuzzASR/mongolian' or manifest['revision'] != ASR_REV:
        raise ValueError('ASR identity differs')
    for name, item in manifest['files'].items():
        if digest(asr/name) != item['sha256']: raise ValueError('ASR artifact changed')
    processor = WhisperProcessor.from_pretrained(asr, local_files_only=True)
    model = WhisperForConditionalGeneration.from_pretrained(asr, torch_dtype=torch.float32,
                local_files_only=True).to('cuda').eval()
    for r in records:
        if digest(run/r['wav']) != r['audio_sha256']: raise ValueError('Audio changed')
        wave, sr = sf.read(run/r['wav'], dtype='float32')
        wave = torchaudio.functional.resample(torch.from_numpy(wave), sr, 16000).numpy()
        batch = processor(wave, sampling_rate=16000, return_tensors='pt', return_attention_mask=True).to('cuda')
        with torch.inference_mode():
            tokens = model.generate(batch.input_features, attention_mask=batch.attention_mask, **p['decoding'])
        hypothesis = processor.batch_decode(tokens, skip_special_tokens=True)[0].strip()
        r.update(hypothesis=hypothesis, score=score_text(r['text'], hypothesis, True),
                 asr_revision=ASR_REV, decoding=p['decoding'])
    result = gate(p['cases'], p['seeds'], *[[r for r in records if r['arm'] == a] for a in ARMS])
    write(run/'results.json', dict(records=records, result=result, asr_manifest_sha256=digest(asr/'download-manifest.json')))
    print(result, flush=True)


def audit(run):
    import numpy as np
    import soundfile as sf
    import torch
    from scripts.score_latest_audio import score as score_text
    p, original = verify(run); report = read(run/'results.json'); wrapper = load(p, original)
    from .targeted_voice import frozen_digest as all_weights
    generated = read(run/'generated.json')
    if not generated['weights_unchanged'] or all_weights(wrapper.model) != generated['full_model_sha256']:
        raise ValueError('Generation model identity differs')
    control = read(Path(p['candidate'])/'eval-000004/results.json')['records'][0]
    native = run/'native_control'/f"{control['seed']}-{control['case']}.wav"
    receipt = read(run/'native_control/verification.json')
    if not receipt['exact_waveform_parity'] or digest(native) != control['audio_sha256'] or receipt['audio_sha256'] != digest(native):
        raise ValueError('Native waveform control changed')
    prompts = {}; discrepancy = 0.
    originals = {(r['arm'],r['case'],r['seed']):r for r in generated['records']}
    for r in report['records']:
        if any(r[k] != v for k,v in originals[r['arm'],r['case'],r['seed']].items()):
            raise ValueError('Generated record changed')
        if (digest(run/r['wav']) != r['audio_sha256'] or digest(run/r['trajectory']) != r['trajectory_sha256']
                or r['protocol_sha256'] != digest(run/'protocol.json') or r['checkpoint_sha256'] != p['checkpoint_sha256']):
            raise ValueError('Speech artifact changed')
        if r['score'] != score_text(r['text'],r['hypothesis'],True) or r['decoding'] != p['decoding']:
            raise ValueError('ASR edit scoring differs')
        trajectory = torch.load(run/r['trajectory'], map_location='cpu', weights_only=True)
        codes, seq = trajectory['codes'], trajectory['first_sequence']
        eos = wrapper.model.config.talker_config.codec_eos_token_id
        if codes.ndim != 2 or codes.shape[1] != 16 or not len(codes) or bool(seq[0,-1] == eos) != r['terminated']:
            raise ValueError('Codec trajectory or EOS differs')
        if r['terminated'] and not torch.equal(seq[0,-(len(codes)+1):-1], codes[:,0]):
            raise ValueError('Main and residual trajectory differ')
        observed = r['residual']; expected = dict(do_sample=True,temperature=ARMS[r['arm']],top_k=50,top_p=1.,max_new_tokens=15)
        if observed['settings'] != expected or observed['calls'] < len(codes):
            raise ValueError('Residual sampling observation incomplete')
        wave, sr = sf.read(run/r['wav'],dtype='float32')
        if sr != 24000 or not np.isfinite(wave).all() or len(wave)/sr != r['seconds']:
            raise ValueError('Invalid waveform metadata')
        with torch.inference_mode():
            if r['reference'] not in prompts:
                refwave, refsr = sf.read(r['reference'],dtype='float32')
                prompts[r['reference']] = wrapper.model.extract_speaker_embedding(audio=refwave,sr=refsr).flatten()
            vector = wrapper.model.extract_speaker_embedding(audio=wave,sr=sr).flatten()
            cosine = float(torch.nn.functional.cosine_similarity(vector[None],prompts[r['reference']][None]))
        discrepancy = max(discrepancy,abs(cosine-r['cosine']))
    if discrepancy > 2e-6: raise ValueError('Original-reference cosine differs')
    result = gate(p['cases'],p['seeds'],*[[r for r in report['records'] if r['arm']==a] for a in ARMS])
    if result != report['result']: raise ValueError('Calibration gate differs')
    write(run/'verification.json',dict(status='verified',recordings=len(report['records']),result=result,
          source_sha256=digest(Path(__file__)),max_cosine_recomputation_error=discrepancy,
          full_model_sha256=generated['full_model_sha256'],all_artifact_hashes_verified=True,
          residual_calls_and_trajectories_verified=True,asr_edit_counts_recomputed=True,
          asr_inference_rerun=False,goal_achieved=False))
    print(result,flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=['prepare','generate','score','audit'],required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--candidate',type=Path)
    parser.add_argument('--panel',type=Path)
    args = parser.parse_args()
    config = read((args.candidate if args.phase=='prepare' else args.output)/'protocol.json')['config']
    os.environ['NUMBA_CACHE_DIR']=config['numba_cache']; os.environ['TEMP']=os.environ['TMP']=config['tmp']
    os.environ['HF_HUB_OFFLINE']=os.environ['TRANSFORMERS_OFFLINE']='1'
    import torch
    torch.set_num_threads(4)
    if args.phase=='prepare': prepare(args.candidate,args.output,args.panel)
    else: globals()[args.phase](args.output)


if __name__ == '__main__': main()
