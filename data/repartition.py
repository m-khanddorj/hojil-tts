"""Rebuild evaluation splits from immutable cleaning checkpoints, without rescoring."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import wave

from .audio import convert
from .adaptation import pcm_hash
from .bulk import atomic_json
from .io import digest, external, read_rows, write_json
from .pipeline import finalize, split_records, text_key, validated_config
from .preprocessing import process


def restore_selected(row, target):
    """Recreate the previously selected full waveform, before boundary trimming."""
    audit = row['audio_processing']
    original = row['original_audio']
    if digest(original) != row['original_audio_sha256']:
        raise ValueError(f'Original audio changed: {row["id"]}')
    convert(original, target)
    if pcm_hash(target) != row['original_pcm_sha256']:
        raise ValueError(f'Canonical source PCM changed: {row["id"]}')
    decision = audit['denoise_decision']
    if decision in {'kept_quality_gain', 'applied_without_quality_gate'}:
        options = deepcopy(audit['settings'])
        for section in ('vad', 'quality', 'trim'):
            options[section]['enabled'] = False
        cfg = options['denoise']
        for key in ('model', 'executable'):
            if cfg.get(key) and digest(cfg[key]) != cfg[key + '_sha256']:
                raise ValueError(f'Denoiser {key} changed')
        process(target, options)
    elif decision not in {'disabled', 'skipped_clean_background', 'reverted_quality_gate', 'reverted_speech_loss'}:
        raise ValueError(f'Unsupported saved denoising decision: {decision}')


def restore(row, target):
    """Replay only the selected enhancement and exact saved trim offsets."""
    if target.exists():
        if pcm_hash(target) != row['pcm_sha256']:
            raise ValueError(f'Restored audio checksum mismatch: {target}')
        return 'reused'
    if Path(row['audio']).is_file():
        if pcm_hash(row['audio']) != row['pcm_sha256']:
            raise ValueError(f'Existing cleaned audio changed: {row["id"]}')
        shutil.copyfile(row['audio'], target)
        return 'copied'
    audit = row['audio_processing']
    temporary = target.with_suffix('.restoring.wav')
    temporary.unlink(missing_ok=True)
    try:
        restore_selected(row, temporary)
        with wave.open(str(temporary), 'rb') as audio:
            raw = audio.readframes(audio.getnframes())
        start = audit['leading_trim_samples']
        stop = len(raw) // 2 - audit['trailing_trim_samples']
        trimmed = raw[start * 2:stop * 2]
        if hashlib.sha256(trimmed).hexdigest() != row['pcm_sha256']:
            raise ValueError(f'Replayed audio differs from saved checkpoint: {row["id"]}')
        with wave.open(str(temporary), 'wb') as audio:
            audio.setparams((1, 2, 24000, 0, 'NONE', 'not compressed'))
            audio.writeframes(trimmed)
        temporary.replace(target)
        return 'reconstructed'
    finally:
        temporary.unlink(missing_ok=True)


def overlap_audit(rows):
    result = {}
    for label, key in [('text', lambda r: text_key(r['text'])),
                       ('audio', lambda r: r['pcm_sha256']),
                       ('original_audio', lambda r: r.get('original_pcm_sha256')),
                       ('speaker', lambda r: r.get('speaker_id'))]:
        splits = {}
        for row in rows:
            value = key(row)
            if value:
                splits.setdefault(value, set()).add(row['split'])
        result[label + '_keys_across_splits'] = sum(len(value) > 1 for value in splits.values())
    return result


def rebuild(prepared, output, workers=6):
    prepared, out = Path(prepared).resolve(), external(output)
    if out == prepared:
        raise ValueError('Repartition must use a separate output directory')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'audio').mkdir(exist_ok=True)
    original_report = json.loads((prepared / 'report.json').read_text(encoding='utf-8'))
    original_config = original_report['config']
    config = deepcopy(original_config)
    config.update(speaker_disjoint=False, preserve_source_splits=False)
    config['evaluation_policy'] = 'Rebuilt text/audio-disjoint splits; familiar speakers allowed; not an official benchmark'
    processing, filters = validated_config(config)
    accepted, rejected = [], []
    for checkpoint in sorted((prepared / 'checkpoints').glob('*.json')):
        row = json.loads(checkpoint.read_text(encoding='utf-8'))['row']
        (rejected if 'rejection' in row else accepted).append(row)
    if len(accepted) + len(rejected) != original_report['accepted'] + original_report['rejected']:
        raise ValueError('Incomplete checkpoint inventory')
    identity = dict(config=config, original_manifest_sha256=digest(prepared / 'manifest.jsonl'),
                    checkpoints=len(accepted) + len(rejected))
    state = out / 'repartition-state.json'
    if state.exists() and json.loads(state.read_text(encoding='utf-8')) != identity:
        raise ValueError('Existing repartition output has a different identity')
    write_json(state, identity)
    # Validate the new assignment before reconstructing any audio.
    assigned, leaked = split_records(accepted, config.get('seed', 2026), config['splits'], False, False)
    if leaked or any(overlap_audit(assigned)[k] for k in ('text_keys_across_splits', 'audio_keys_across_splits', 'original_audio_keys_across_splits')):
        raise ValueError('Rebuilt splits failed overlap checks')
    write_json(out / 'planned-splits.json', {
        split: dict(rows=sum(r['split'] == split for r in assigned),
                    hours=sum(r['duration'] for r in assigned if r['split'] == split) / 3600)
        for split in ('train', 'validation', 'test')})
    completed, actions = 0, Counter()
    def progress(stage, **extra):
        atomic_json(out / 'progress.json', dict(status=stage, processed=completed, total=len(accepted),
                    actions=dict(actions), updated=datetime.now(timezone.utc).isoformat(), **extra))
    def work(row):
        target = out / 'audio' / (hashlib.sha256(row['id'].encode()).hexdigest() + '.wav')
        action = restore(row, target)
        return dict(row, audio=str(target)), action
    progress('restoring_audio')
    restored = []
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for row, action in pool.map(work, accepted):
                restored.append(row)
                actions[action] += 1
                completed += 1
                if completed % 50 == 0:
                    progress('restoring_audio')
        progress('finalizing')
        report = finalize(config, out, restored, rejected, filters, processing)
        rows = list(read_rows(out / 'manifest.jsonl'))
        audit = overlap_audit(rows)
        if any(audit[k] for k in ('text_keys_across_splits', 'audio_keys_across_splits', 'original_audio_keys_across_splits')):
            raise ValueError('Final split overlap detected')
        progress('exporting')
        from .adaptation import export_adaptation
        adaptation = export_adaptation(out / 'manifest.jsonl', out / 'adaptation')
        write_json(out / 'completion-report.json', dict(
            accepted=report['accepted'], rejected=report['rejected'], accepted_hours=report['accepted_hours'],
            source=report['source'], split=report['split'], reasons=report['reasons'], overlap_audit=audit,
            adaptation=adaptation, reconstruction=dict(actions), evaluation_policy=config['evaluation_policy']))
        progress('complete')
        return report
    except BaseException as error:
        progress('failed', error=f'{type(error).__name__}: {error}')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        parser.error('workers must be between 1 and 16')
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('RAYON_NUM_THREADS', '1')
    rebuild(args.prepared, args.output, args.workers)
