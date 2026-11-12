"""Training-only generated speech pairs with original sampled codec trajectories.

This collects data; it does not optimize weights or promote a checkpoint.
GPU FP32 rollouts are explicitly distinct from the fixed CPU evaluation backend.
"""
import argparse
from collections import defaultdict
from contextlib import contextmanager
import os
from pathlib import Path
import subprocess
import sys
import time

from .engine import digest, read, rows, save, write
from .policy import external, fingerprint, validate_cohorts
from data.pipeline import text_key


def select_cases(train, excluded_texts, speakers=16, per_speaker=2):
    """Hash-select speakers before observing any generated quality scores."""
    excluded = {text_key(t) for t in excluded_texts}
    grouped = defaultdict(list)
    for row in train:
        if (2 <= row['duration'] <= 9 and text_key(row['text']) not in excluded
                and text_key(row['ref_text']) not in excluded):
            grouped[row['speaker_id']].append(row)
    eligible = [s for s, group in grouped.items() if len(group) >= per_speaker]
    selected = sorted(eligible, key=lambda s: fingerprint(['joint-preference-v1', s]))[:speakers]
    if len(selected) != speakers or speakers < 4:
        raise ValueError('Insufficient distinct training speakers')
    result = []
    for i, speaker in enumerate(selected):
        chosen = sorted(grouped[speaker], key=lambda r: fingerprint(['joint-preference-v1', r['id']]))[:per_speaker]
        for row in chosen:
            result.append(dict(row, preference_split='validation' if i < speakers//4 else 'train'))
    return result


@contextmanager
def capture_generation(model):
    """Observe original generation calls without altering sampling or decoding."""
    import torch
    captured = {}
    original, talker_original = model.generate, model.talker.generate

    def talker(*args, **kwargs):
        result = talker_original(*args, **kwargs)
        captured['first_sequence'] = result.sequences.detach().cpu().clone()
        return result

    def full(*args, **kwargs):
        result = original(*args, **kwargs)
        if len(result[0]) != 1:
            raise ValueError('Only single-example rollouts supported')
        captured['codes'] = result[0][0].detach().cpu().clone()
        return result

    model.generate, model.talker.generate = full, talker
    try:
        yield captured
    finally:
        model.generate, model.talker.generate = original, talker_original
    codes, seq = captured['codes'], captured['first_sequence']
    eos = model.config.talker_config.codec_eos_token_id
    captured['terminated'] = bool(seq[0, -1] == eos)
    if codes.ndim != 2 or codes.shape[1] != 16 or not len(codes):
        raise ValueError('Malformed generated codec trajectory')
    if captured['terminated'] and not torch.equal(seq[0, -(len(codes)+1):-1], codes[:, 0]):
        raise ValueError('Generated codec frames disagree with sampled first-codebook sequence')


def prepare(parent, output):
    from . import reference_protected as protected
    protected.configure_helpers()
    p = read(parent/'protocol.json')
    if p['source'] != protected.source_hashes():
        raise ValueError('Parent source changed')
    checkpoint = parent/'checkpoint-000000.pt'
    if digest(checkpoint) != read(parent/'checkpoint-000000.json')['sha256']:
        raise ValueError('Initial halfway checkpoint changed')
    backup = Path(p['config']['backup'])
    recovery = read(backup/'recovery-verification.json')['cohorts']
    for name in ('train', 'unseen-validation', 'familiar-validation'):
        if digest(backup/f'cohorts/{name}.jsonl') != recovery[name]['sha256']:
            raise ValueError('Cohort changed')
    inventory = read(backup/'dataset-v1/dataset-inventory.json')['files']
    if digest(backup/'dataset-v1/manifest.jsonl') != inventory['manifest.jsonl']['sha256']:
        raise ValueError('Canonical manifest changed')
    train = rows(backup/'cohorts/train.jsonl')
    unseen = rows(backup/'cohorts/unseen-validation.jsonl')
    familiar = [r for r in rows(backup/'cohorts/familiar-validation.jsonl') if r.get('eligible')]
    canonical = validate_cohorts(train, unseen+familiar, rows(backup/'dataset-v1/manifest.jsonl'))
    excluded = [t for c in p['cases'] for t in (c['text'], c['reference_text'])]
    selected = select_cases(train, excluded)
    cases = []
    for i, row in enumerate(selected):
        ref = canonical[row['ref_id']]
        reference = backup/'dataset-v1'/ref['audio']
        if digest(reference) != inventory[ref['audio']]['sha256']:
            raise ValueError('Training reference changed')
        cases.append(dict(case=f'train-{i:03d}', row_id=row['id'], ref_id=row['ref_id'],
            speaker_id=row['speaker_id'], text=row['text'], reference=str(reference),
            reference_text=ref['text'], reference_sha256=digest(reference),
            partition='train', preference_split=row['preference_split']))
    output = external(output.resolve())
    output.mkdir(exist_ok=False)
    source = {name:digest(Path(__file__).parent/name) for name in ('preference_rollouts.py', 'voice_preferences.py')}
    protocol = dict(parent=str(parent), parent_protocol_sha256=digest(parent/'protocol.json'),
        checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint), source=source,
        canonical_manifest_sha256=digest(backup/'dataset-v1/manifest.jsonl'),
        train_cohort_sha256=recovery['train']['sha256'], cases=cases, seeds=[3401,3402,3403,3404],
        generation=p['generation'], decoding=p['decoding'],
        generation_backend='CUDA FP32 SDPA, TF32 disabled; codec decoder CPU; distinct from CPU selection evaluations',
        selection='16 equally weighted training speakers, two hash-selected 2–9s texts per speaker; first four speakers held out from preference fitting',
        objective='Generated versus generated; chosen no worse CER or cosine; voice-only gain >=0.001',
        goal_achieved=False)
    write(output/'protocol.json', protocol)
    write(output/'status.json', dict(status='prepared', completed=0, total=len(cases)*4))
    return protocol


def verify(output):
    p = read(output/'protocol.json')
    parent = Path(p['parent'])
    if (digest(parent/'protocol.json') != p['parent_protocol_sha256']
            or digest(p['checkpoint']) != p['checkpoint_sha256']):
        raise ValueError('Parent provenance changed')
    for name, sha in p['source'].items():
        if digest(Path(__file__).parent/name) != sha:
            raise ValueError('Rollout implementation changed')
    return p, read(parent/'protocol.json')


def generate(output, limit=None):
    import numpy as np
    import soundfile as sf
    import torch
    from . import reference_protected as protected
    from .targeted_voice import frozen_digest
    p, parent = verify(output)
    protected.configure_helpers()
    wrapper = protected.load_model(parent)
    state = torch.load(p['checkpoint'], map_location='cpu', weights_only=True)
    protected.audited_load_adapter(wrapper.model.talker, state['adapter'], output/'routing.json')
    if frozen_digest(wrapper.model) != state['frozen_backbone_sha256']:
        raise ValueError('Frozen model mismatch')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    wrapper.model.to('cuda').eval()
    wrapper.device = torch.device('cuda')
    torch.cuda.set_per_process_memory_fraction(0.84)
    prompts = {}
    done = 0
    for case in p['cases']:
        ref = case['reference']
        if digest(ref) != case['reference_sha256']:
            raise ValueError('Reference changed')
        for seed in p['seeds']:
            sample = f"{case['case']}-{seed}"
            path = output/f'{sample}.json'
            if path.exists():
                record = read(path)
                for field, sha in (('wav','audio_sha256'), ('trajectory','trajectory_sha256')):
                    if digest(output/record[field]) != record[sha]:
                        raise ValueError('Rollout cache changed')
                if record['protocol_sha256'] != digest(output/'protocol.json'):
                    raise ValueError('Cached protocol differs')
            else:
                if ref not in prompts:
                    with torch.inference_mode():
                        prompts[ref] = wrapper.create_voice_clone_prompt(ref_audio=ref,
                            ref_text=case['reference_text'], x_vector_only_mode=False)
                prompt = prompts[ref]
                torch.manual_seed(seed)
                np.random.seed(seed)
                started = time.monotonic()
                with torch.inference_mode(), capture_generation(wrapper.model) as captured:
                    waves, sr = wrapper.generate_voice_clone(text=case['text'], language='Auto',
                        voice_clone_prompt=prompt, non_streaming_mode=True, **p['generation'])
                wave = np.asarray(waves[0])
                if sr != 24000 or not len(wave) or not np.isfinite(wave).all() or len(wave)/sr > 30:
                    raise ValueError('Invalid rollout waveform')
                wav = output/f'{sample}.wav'
                sf.write(wav, wave, sr, subtype='PCM_16')
                wave, sr = sf.read(wav, dtype='float32')
                with torch.inference_mode():
                    vec = wrapper.model.extract_speaker_embedding(audio=wave, sr=sr)
                refvec = prompt[0].ref_spk_embedding
                cosine = torch.nn.functional.cosine_similarity(vec.flatten()[None], refvec.flatten()[None]).item()
                trajectory = output/f'{sample}.pt'
                from .engine import ids
                save(trajectory, dict(codes=captured['codes'], first_sequence=captured['first_sequence'],
                    terminated=captured['terminated'], ref_codes=prompt[0].ref_code.detach().cpu(),
                    ref_xvec=refvec.detach().cpu(), input_ids=ids(wrapper, case['text']),
                    ref_ids=ids(wrapper, case['reference_text'], True),
                    protocol_sha256=digest(output/'protocol.json')))
                record = dict(case, sample_id=sample, seed=seed, origin='model_generated',
                    wav=wav.name, audio_sha256=digest(wav), trajectory=trajectory.name,
                    trajectory_sha256=digest(trajectory), terminated=captured['terminated'],
                    frames=len(captured['codes']), seconds=len(wave)/sr, cosine=cosine,
                    generation_seconds=time.monotonic()-started,
                    protocol_sha256=digest(output/'protocol.json'))
                write(path, record)
            done += 1
            write(output/'status.json', dict(status='generating', completed=done,
                total=len(p['cases'])*len(p['seeds']), last=sample, updated=time.time()))
            print(f'Preference rollouts: {done}/{len(p["cases"])*len(p["seeds"])}', flush=True)
            if limit and done >= limit:
                return
    write(output/'status.json', dict(status='generated', completed=done, updated=time.time()))


def score_all(output):
    import soundfile as sf
    import torch
    import torchaudio
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    from scripts.score_latest_audio import score
    from .evaluate import ASR_REV, pooled
    from .voice_preferences import joint_pairs
    p, parent = verify(output)
    asr = Path(parent['config']['asr'])
    manifest = read(asr/'download-manifest.json')
    if manifest['repo'] != 'BuzzASR/mongolian' or manifest['revision'] != ASR_REV:
        raise ValueError('ASR revision mismatch')
    for name, item in manifest['files'].items():
        if digest(asr/name) != item['sha256']:
            raise ValueError('ASR file changed')
    processor = WhisperProcessor.from_pretrained(asr, local_files_only=True)
    model = WhisperForConditionalGeneration.from_pretrained(asr, torch_dtype=torch.float32,
        local_files_only=True).to('cuda').eval()
    records, pairs = [], []
    for case in p['cases']:
        candidates = []
        for seed in p['seeds']:
            sample = f"{case['case']}-{seed}"
            record = read(output/f'{sample}.json')
            if record['protocol_sha256'] != digest(output/'protocol.json'):
                raise ValueError('Rollout protocol changed')
            for field, sha in (('wav','audio_sha256'), ('trajectory','trajectory_sha256')):
                if digest(output/record[field]) != record[sha]:
                    raise ValueError('Rollout changed before scoring')
            scored = output/f'score-{sample}.json'
            if scored.exists():
                result = read(scored)
                if (any(result[k] != v for k,v in record.items()) or result['asr_revision'] != ASR_REV
                        or result['decoding'] != p['decoding']):
                    raise ValueError('Score cache mismatch')
            else:
                wave, sr = sf.read(output/record['wav'], dtype='float32')
                wave = torchaudio.functional.resample(torch.from_numpy(wave), sr, 16000).numpy()
                batch = processor(wave, sampling_rate=16000, return_tensors='pt', return_attention_mask=True).to('cuda')
                with torch.inference_mode():
                    tokens = model.generate(batch.input_features, attention_mask=batch.attention_mask, **p['decoding'])
                hypothesis = processor.batch_decode(tokens, skip_special_tokens=True)[0].strip()
                result = dict(record, hypothesis=hypothesis, score=score(record['text'], hypothesis, True),
                    asr_revision=ASR_REV, decoding=p['decoding'])
                write(scored, result)
            records.append(result)
            # Truncation is observed, never relabeled as a sampled EOS target.
            if result['terminated']:
                candidates.append(result)
        pairs.extend(dict(pair, preference_split=case['preference_split']) for pair in joint_pairs(candidates))
        write(output/'status.json', dict(status='scoring', completed=len(records), pairs=len(pairs), updated=time.time()))
    summaries = {}
    for split in ('train', 'validation'):
        selected = [r for r in records if r['preference_split'] == split]
        summaries[split] = dict(**pooled([r['score'] for r in selected]),
            mean_cosine=sum(r['cosine'] for r in selected)/len(selected),
            recordings=len(selected), speakers=len({r['speaker_id'] for r in selected}),
            usable_pairs=sum(pair['preference_split']==split for pair in pairs),
            cases_with_pairs=len({pair['case'] for pair in pairs if pair['preference_split']==split}))
    write(output/'results.json', dict(records=records, pairs=pairs, summary=summaries,
        truncated=sum(not r['terminated'] for r in records), goal_achieved=False,
        scope='Training-data collection; not development/test quality evidence and not a trained checkpoint'))
    write(output/'status.json', dict(status='complete', recordings=len(records), pairs=len(pairs), updated=time.time()))
    print(summaries, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--phase', choices=['prepare','generate','score','run'], default='run')
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    if args.phase == 'prepare' or not (args.output/'protocol.json').exists():
        if args.parent is None:
            raise ValueError('--parent required for preparation')
        prepare(args.parent, args.output)
    if args.phase == 'prepare':
        return
    _, parent = verify(args.output)
    cfg = parent['config']
    os.environ['HF_HUB_OFFLINE'] = os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['NUMBA_CACHE_DIR'] = cfg['numba_cache']
    os.environ['TEMP'] = os.environ['TMP'] = cfg['tmp']
    import torch
    torch.set_num_threads(4)
    if args.phase == 'generate':
        generate(args.output, args.limit)
    elif args.phase == 'score':
        score_all(args.output)
    else:
        for phase in ('generate', 'score'):
            subprocess.run([sys.executable, '-m', 'training.preference_rollouts', '--output', str(args.output),
                '--phase', phase], check=True)


if __name__ == '__main__':
    main()
