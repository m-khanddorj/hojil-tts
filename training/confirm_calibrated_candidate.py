"""Execute the pre-registered 114-recording panel for a verified screen pass.

No selection, optimizer or automatic promotion. Generation, GPU ASR and the
independent audio audit are separate phases to respect local memory limits.
"""
import argparse
import os
from pathlib import Path

from .engine import digest, read, write
from .policy import external
from .targeted_voice import aggregate, frozen_digest
from .voice_strength_search import goal_screen


def validate_panel(panel):
    cases = panel['cases']
    if (len(cases) != 57 or len({c['case'] for c in cases}) != 57
            or panel['seeds'] != [2026, 2027]
            or sum(c['group'] == 'development' for c in cases) != 44
            or sum(c['group'] == 'custom' for c in cases) != 13
            or panel['development_recordings'] != 88 or panel['custom_recordings'] != 26
            or panel['criteria'] != dict(pooled_cer_strictly_below=.03,
                speaker_macro_cosine_strictly_above=.984, groups=['development', 'custom'],
                require_complete_panel=True)):
        raise ValueError('Confirmation scope or thresholds changed')


def validate_records(panel, records):
    validate_panel(panel)
    expected = {(c['case'], s) for c in panel['cases'] for s in panel['seeds']}
    if len(records) != 114 or {(r['case'], r['seed']) for r in records} != expected:
        raise ValueError('Incomplete or duplicate confirmation panel')
    cases = {c['case']: c for c in panel['cases']}
    for r in records:
        if any(r[k] != v for k, v in cases[r['case']].items()):
            raise ValueError('Confirmation case changed')
    summary = aggregate(records)
    return dict(summary=summary, truncated=sum(not r['terminated'] for r in records),
                confirmation_pass=goal_screen(summary) and all(r['terminated'] for r in records))


def candidate_info(kind, candidate, step):
    if kind in ('speaker', 'refreshed', 'centered', 'attention', 'calibrated'):
        if kind == 'speaker':
            from . import speaker_preference_training as trainer
        elif kind == 'refreshed':
            from . import refreshed_speaker_training as trainer
        elif kind == 'centered':
            from . import centered_speaker_training as trainer
        elif kind == 'calibrated':
            from . import calibrated_quality_training as trainer
        else:
            from . import middle_attention_training as trainer
        p = trainer.verify(candidate)
        state = trainer.read_checkpoint(candidate, step, p)
        report = candidate/f'eval-{step:06d}/results.json'
        audit = candidate/f'verification-{step:06d}.json'
        identity = dict(checkpoint_sha256=digest(candidate/f'checkpoint-{step:06d}.pt'),
                        frozen_backbone_sha256=state['frozen_backbone_sha256'])
        if kind in ('attention', 'calibrated'):identity['frozen_speaker_sha256'] = state['frozen_speaker_sha256']
    elif kind in ('precision', 'prefix', 'codec', 'condition'):
        if kind == 'precision':
            from .fp32_merge_comparison import verify
        elif kind == 'prefix':
            from .reference_prefix_comparison import verify
        elif kind == 'codec':
            from .reference_codec_comparison import verify
        else:
            from .reference_condition_comparison import verify
        p = verify(candidate)
        report, audit = candidate/'results.json', candidate/'verification.json'
        identity = read(candidate/'model.json')
    else:
        raise ValueError('Unsupported candidate kind')
    r, a = read(report), read(audit)
    if kind in ('prefix', 'codec', 'condition') and (a.get('reference_matrices_verified') != (198 if kind == 'prefix' else 196)
            or not a.get('sampled_codecs_eos_and_routing_spans_verified')
            or not a.get('acoustic_and_speaker_adapter_verified')):
        raise ValueError('Reference-prefix audit coverage is incomplete')
    if kind == 'codec' and not a.get('recorded_full_text_prefix_checks_verified'):
        raise ValueError('Full text-prefix checks were not audited')
    if kind == 'condition' and not a.get('recorded_leading_control_checks_verified'):
        raise ValueError('Leading control checks were not audited')
    if kind in ('attention', 'calibrated') and (not a.get('frozen_speaker_verified') or not a.get('middle_attention_coverage_verified')):
        raise ValueError('Attention LoRA or frozen speaker audit is incomplete')
    if kind == 'calibrated' and (not a.get('quality_fitting_subset_verified') or not a.get('full_validation_verified') or not a.get('fixed_probe_telemetry_verified')):
        raise ValueError('Quality fitting or full-validation audit is incomplete')
    if (a['status'] != 'verified' or not a['goal_screen'] or not goal_screen(aggregate(r['records']))
            or r['truncated'] or len(r['records']) != 26
            or not all(v['terminated'] for v in r['records'])):
        raise ValueError('Candidate has no independently verified screening pass')
    expected = {(c['case'], s) for c in p['cases'] for s in p['seeds']}
    if {(v['case'], v['seed']) for v in r['records']} != expected:
        raise ValueError('Screening coverage differs')
    for v in r['records']:
        if digest(report.parent/v['wav']) != v['audio_sha256']:
            raise ValueError('Screening audio changed')
    return p, dict(report=str(report), report_sha256=digest(report), audit=str(audit),
                  audit_sha256=digest(audit), identity=identity)


def prepare(kind, candidate, step, panel_path, output):
    p, evidence = candidate_info(kind, candidate, step)
    panel = read(panel_path)
    validate_panel(panel)
    if (panel['parent_protocol_sha256'] != p['parent_protocol_sha256']
            or any(panel[k] != p[k] for k in ('seeds', 'generation', 'decoding'))):
        raise ValueError('Confirmation registration differs from screening')
    cases = {c['case']: c for c in panel['cases']}
    if any(cases.get(c['case']) != c for c in p['cases']):
        raise ValueError('Confirmation no longer contains all screening cases')
    if panel['preparation_source_sha256'] != digest(Path(__file__).with_name('joint_confirmation.py')):
        raise ValueError('Panel preparation implementation changed')
    output = external(output.resolve())
    output.mkdir(exist_ok=False)
    write(output/'protocol.json', dict(kind=kind, candidate=str(candidate), step=step,
        candidate_protocol_sha256=digest(candidate/'protocol.json'), evidence=evidence,
        panel=str(panel_path), panel_sha256=digest(panel_path), config=p['config'],
        source_sha256=digest(Path(__file__)), goal_achieved=False))
    write(output/'status.json', dict(status='prepared', recordings=114, goal_achieved=False))


def verify(run):
    p = read(run/'protocol.json')
    if (digest(Path(__file__)) != p['source_sha256'] or digest(p['panel']) != p['panel_sha256']
            or digest(Path(p['candidate'])/'protocol.json') != p['candidate_protocol_sha256']):
        raise ValueError('Confirmation provenance changed')
    original, evidence = candidate_info(p['kind'], Path(p['candidate']), p['step'])
    if evidence != p['evidence']:
        raise ValueError('Candidate evidence changed')
    panel = read(p['panel'])
    validate_panel(panel)
    return p, panel, original


def acoustic_digest(p, model):
    if p['kind'] in ('attention', 'calibrated'):
        from .middle_attention_adapter import frozen_digest as digest_acoustic
        return digest_acoustic(model)
    if p['kind'] in ('centered', 'prefix', 'codec', 'condition'):
        from .centered_speaker_training import frozen_digest as digest_acoustic
        return digest_acoustic(model)
    return frozen_digest(model)


def load_candidate(p, original):
    if p['kind'] in ('speaker', 'refreshed', 'centered', 'attention', 'calibrated'):
        if p['kind'] == 'speaker':
            from . import speaker_preference_training as trainer
        elif p['kind'] == 'refreshed':
            from . import refreshed_speaker_training as trainer
        elif p['kind'] == 'centered':
            from . import centered_speaker_training as trainer
        elif p['kind'] == 'calibrated':
            from . import calibrated_quality_training as trainer
        else:
            from . import middle_attention_training as trainer
        wrapper = trainer.load(original)
        state = trainer.read_checkpoint(Path(p['candidate']), p['step'], original)
        if p['kind'] == 'centered':
            trainer.restore_adapter(wrapper.model, state['adapter'])
        elif p['kind'] in ('attention', 'calibrated'):
            from .middle_attention_adapter import restore, speaker_digest
            restore(wrapper.model, state['adapter'])
            if (speaker_digest(wrapper.model) != state['frozen_speaker_sha256']
                    or state['frozen_speaker_sha256'] != p['evidence']['identity']['frozen_speaker_sha256']):
                raise ValueError('Confirmation frozen speaker adapter differs')
        else:
            wrapper.model.speaker_preference_adapter.load_state_dict(state['adapter'], strict=True)
    elif p['kind'] in ('prefix', 'codec', 'condition'):
        if p['kind'] == 'prefix':
            from .reference_prefix_comparison import load, model_identity
        elif p['kind'] == 'codec':
            from .reference_codec_comparison import load, model_identity
        else:
            from .reference_condition_comparison import load, model_identity
        wrapper = load(original)
        if any(p['evidence']['identity'].get(k) != v for k, v in model_identity(wrapper).items()):
            raise ValueError('Confirmation reference-routing model differs')
    else:
        from .fp32_merge_comparison import load
        wrapper = load(original)
    wrapper.model.eval()
    if acoustic_digest(p, wrapper.model) != p['evidence']['identity']['frozen_backbone_sha256']:
        raise ValueError('Confirmation acoustic model differs')
    return wrapper


def generate(run):
    import torch
    import numpy as np
    import soundfile as sf
    from .preference_rollouts import capture_generation
    p, panel, original = verify(run)
    wrapper = load_candidate(p, original)
    trainable = {n: v.detach().clone() for n, v in wrapper.model.named_parameters() if v.requires_grad}
    fixed_adapter = wrapper.model.speaker_preference_adapter.frozen_state() if p['kind'] in ('centered', 'prefix', 'codec', 'condition') else None
    if p['kind'] in ('attention', 'calibrated'):
        from .middle_attention_adapter import speaker_digest
        fixed_speaker_digest = speaker_digest(wrapper.model)
    screening = {(r['case'], r['seed']): r for r in read(p['evidence']['report'])['records']}
    prompts, records = {}, []
    for seed in panel['seeds']:
        for c in panel['cases']:
            ref = c['reference']
            if digest(ref) != c['reference_sha256']:
                raise ValueError('Reference audio changed')
            if ref not in prompts:
                with torch.inference_mode():
                    prompts[ref] = wrapper.create_voice_clone_prompt(ref_audio=ref,
                        ref_text=c['reference_text'], x_vector_only_mode=False)
            wav = run/f"{seed}-{c['case']}.wav"
            meta = wav.with_suffix('.json')
            if meta.exists():
                r = read(meta)
                if (any(r[k] != v for k, v in c.items()) or r['seed'] != seed
                        or r['protocol_sha256'] != digest(run/'protocol.json')
                        or digest(wav) != r['audio_sha256']):
                    raise ValueError('Cached audio changed')
            else:
                torch.manual_seed(seed)
                np.random.seed(seed)
                with torch.inference_mode(), capture_generation(wrapper.model) as captured:
                    waves, sr = wrapper.generate_voice_clone(text=c['text'], language='Auto',
                        voice_clone_prompt=prompts[ref], non_streaming_mode=True, **panel['generation'])
                if sr != 24000 or not len(waves[0]) or not np.isfinite(waves[0]).all() or len(waves[0])/sr > 30:
                    raise ValueError('Invalid waveform')
                sf.write(wav, waves[0], sr, subtype='PCM_16')
                wave, sr = sf.read(wav, dtype='float32')
                with torch.inference_mode():
                    vector = wrapper.model.extract_speaker_embedding(audio=wave, sr=sr).flatten()
                    cosine = float(torch.nn.functional.cosine_similarity(vector[None],
                        prompts[ref][0].ref_spk_embedding.flatten()[None]))
                r = dict(c, seed=seed, wav=wav.name, audio_sha256=digest(wav), cosine=cosine,
                    seconds=len(wave)/sr, terminated=captured['terminated'],
                    protocol_sha256=digest(run/'protocol.json'))
                if p['kind'] in ('prefix', 'codec', 'condition'):
                    router = getattr(wrapper.model, f"reference_{p['kind']}_routing")
                    r.update(routing=router.last_trace,
                             frames=len(captured['codes']))
                write(meta, r)
            if (c['case'], seed) in screening and r['audio_sha256'] != screening[c['case'], seed]['audio_sha256']:
                raise ValueError('Screening waveform not reproduced in confirmation')
            records.append(r)
            write(run/'status.json', dict(status='generating', completed=len(records), total=114))
            print(f'Confirmation speech {len(records)}/114', flush=True)
    if acoustic_digest(p, wrapper.model) != p['evidence']['identity']['frozen_backbone_sha256']:
        raise ValueError('Acoustic weights changed during generation')
    if fixed_adapter is not None:
        wrapper.model.speaker_preference_adapter.verify_frozen(fixed_adapter)
    if p['kind'] in ('attention', 'calibrated') and speaker_digest(wrapper.model) != fixed_speaker_digest:
        raise ValueError('Frozen speaker adapter changed during confirmation')
    if p['kind'] in ('prefix', 'codec', 'condition'):
        if p['kind'] == 'prefix':
            from .reference_prefix_comparison import model_identity
        elif p['kind'] == 'codec':
            from .reference_codec_comparison import model_identity
        else:
            from .reference_condition_comparison import model_identity
        if any(p['evidence']['identity'].get(k) != v for k, v in model_identity(wrapper).items()):
            raise ValueError('Reference-routing matrices changed during confirmation')
    if any(not torch.equal(v, trainable[n]) for n, v in wrapper.model.named_parameters() if n in trainable):
        raise ValueError('Adapter changed during generation')
    write(run/'generated.json', dict(records=records, screening_waveform_parity=True))
    write(run/'status.json', dict(status='generated_waiting_for_asr', recordings=114))


def score_all(run):
    import torch
    import torchaudio
    import soundfile as sf
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    from scripts.score_latest_audio import score
    from .evaluate import ASR_REV
    p, panel, _ = verify(run)
    records = read(run/'generated.json')['records']
    asr = Path(p['config']['asr'])
    manifest = read(asr/'download-manifest.json')
    if manifest['repo'] != 'BuzzASR/mongolian' or manifest['revision'] != ASR_REV:
        raise ValueError('ASR identity differs')
    for name, item in manifest['files'].items():
        if digest(asr/name) != item['sha256']:
            raise ValueError('ASR file changed')
    processor = WhisperProcessor.from_pretrained(asr, local_files_only=True)
    model = WhisperForConditionalGeneration.from_pretrained(asr,
        torch_dtype=torch.float32, local_files_only=True).to('cuda').eval()
    for r in records:
        if digest(run/r['wav']) != r['audio_sha256'] or r['protocol_sha256'] != digest(run/'protocol.json'):
            raise ValueError('Generated audio provenance differs')
        path = run/f"score-{r['seed']}-{r['case']}.json"
        if path.exists():
            value = read(path)
            if (any(value[k] != v for k, v in r.items()) or value['asr_revision'] != ASR_REV
                    or value['decoding'] != panel['decoding']):
                raise ValueError('ASR cache changed')
        else:
            wave, sr = sf.read(run/r['wav'], dtype='float32')
            wave = torchaudio.functional.resample(torch.from_numpy(wave), sr, 16000).numpy()
            batch = processor(wave, sampling_rate=16000, return_tensors='pt', return_attention_mask=True).to('cuda')
            with torch.inference_mode():
                tokens = model.generate(batch.input_features, attention_mask=batch.attention_mask, **panel['decoding'])
            hypothesis = processor.batch_decode(tokens, skip_special_tokens=True)[0].strip()
            value = dict(r, hypothesis=hypothesis, score=score(r['text'], hypothesis, True),
                         asr_revision=ASR_REV, decoding=panel['decoding'])
            write(path, value)
        r.update(value)
    outcome = validate_records(panel, records)
    write(run/'results.json', dict(outcome, records=records, goal_achieved=False,
        global_model_selection_changed=False, protocol_sha256=digest(run/'protocol.json')))
    write(run/'status.json', dict(status='scored_waiting_for_audit', **outcome))
    print(outcome)


def audit(run):
    import torch
    import numpy as np
    import soundfile as sf
    from scripts.score_latest_audio import score
    from .evaluate import ASR_REV
    p, panel, original = verify(run)
    report = read(run/'results.json')
    if report['protocol_sha256'] != digest(run/'protocol.json'):
        raise ValueError('Report provenance differs')
    wrapper = load_candidate(p, original)
    screening = {(r['case'], r['seed']): r for r in read(p['evidence']['report'])['records']}
    refs, errors, route_refs = {}, [], {}
    def embed(path):
        wave, sr = sf.read(path, dtype='float32')
        if sr != 24000 or wave.ndim != 1 or not len(wave) or not np.isfinite(wave).all():
            raise ValueError('Invalid audio')
        with torch.inference_mode():
            return wrapper.model.extract_speaker_embedding(audio=wave, sr=sr).flatten()
    for r in report['records']:
        if (digest(run/r['wav']) != r['audio_sha256'] or digest(r['reference']) != r['reference_sha256']
                or r['protocol_sha256'] != digest(run/'protocol.json')
                or r['asr_revision'] != ASR_REV or r['decoding'] != panel['decoding']):
            raise ValueError('Audio or ASR provenance differs')
        if score(r['text'], r['hypothesis'], True) != r['score']:
            raise ValueError('ASR edit counts differ')
        key = (r['case'], r['seed'])
        if key in screening and r['audio_sha256'] != screening[key]['audio_sha256']:
            raise ValueError('Confirmation does not reproduce screening audio')
        ref = r['reference']
        if p['kind'] in ('prefix', 'codec', 'condition'):
            from .engine import ids
            if ref not in route_refs:
                with torch.inference_mode():
                    prompt = wrapper.create_voice_clone_prompt(ref_audio=ref,
                        ref_text=r['reference_text'], x_vector_only_mode=False)
                route_refs[ref] = (len(prompt[0].ref_code), ids(wrapper, r['reference_text'], True).shape[1]-5)
            ref_frames, nr = route_refs[ref]
            nt = ids(wrapper, r['text']).shape[1]-8
            prefix = 8+nr+nt+1+1+ref_frames
            expected_trace = dict(target_start=8+nr, target_stop=8+nr+nt,
                prefix_length=prefix, prefills=1, decode_steps=r['frames'], protected=True,
                main_matrices=196, reference_positions=prefix-nt)
            if p['kind'] == 'codec':
                expected_trace.update(acoustic_start=8+nr+nt+1, text_prefix_exact=True,
                                      reference_positions=ref_frames+1)
            elif p['kind'] == 'condition':
                expected_trace.update(acoustic_start=8+nr+nt+1, leading_control_exact=True,
                                      speaker_position=6, reference_positions=ref_frames+2)
            if r['frames'] < 1 or r['routing'] != expected_trace:
                raise ValueError('Confirmation reference-routing spans differ')
        if ref not in refs:
            refs[ref] = embed(ref)
        with torch.inference_mode():
            actual = float(torch.nn.functional.cosine_similarity(embed(run/r['wav'])[None], refs[ref][None]))
        errors.append(abs(actual-r['cosine']))
        if errors[-1] > 2e-6:
            raise ValueError('Original-reference cosine differs')
    outcome = validate_records(panel, report['records'])
    if any(report[k] != v for k, v in outcome.items()):
        raise ValueError('Confirmation aggregate differs')
    write(run/'verification.json', dict(outcome, status='verified', recordings=114,
        report_sha256=digest(run/'results.json'), protocol_sha256=digest(run/'protocol.json'),
        max_cosine_error=max(errors), asr_edit_counts_recomputed=True, asr_inference_rerun=False,
        goal_achieved=False, global_model_selection_changed=False))
    write(run/'status.json', dict(status='verified', **outcome, goal_achieved=False))
    print(outcome)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', required=True, choices=['prepare', 'generate', 'score', 'audit'])
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--kind', choices=['speaker', 'refreshed', 'centered', 'attention', 'calibrated', 'precision', 'prefix', 'codec', 'condition'])
    parser.add_argument('--candidate', type=Path)
    parser.add_argument('--step', type=int)
    parser.add_argument('--panel', type=Path)
    args = parser.parse_args()
    cfg = read((args.candidate if args.phase == 'prepare' else args.output)/'protocol.json')['config']
    os.environ['NUMBA_CACHE_DIR'] = cfg['numba_cache']
    os.environ['TEMP'] = os.environ['TMP'] = cfg['tmp']
    os.environ['HF_HUB_OFFLINE'] = os.environ['TRANSFORMERS_OFFLINE'] = '1'
    import torch
    torch.set_num_threads(4)
    if args.phase == 'prepare':
        prepare(args.kind, args.candidate, args.step, args.panel, args.output)
    else:
        {'generate': generate, 'score': score_all, 'audit': audit}[args.phase](args.output)


if __name__ == '__main__':
    main()
