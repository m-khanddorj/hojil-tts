"""Canonical corpus preparation, conservative filters, and reproducible leakage groups."""
from collections import Counter, defaultdict
import hashlib
import json
import math
import wave
import re
import shutil
import subprocess
import unicodedata
from pathlib import Path

from . import __version__
from .audio import convert, inspect, ffmpeg_executable
from .io import digest, fresh_dir, write_json, write_rows
from .sources import records

DEFAULT_FILTERS = dict(min_duration=1.0, max_duration=20.0, min_rms_dbfs=-45.0,
                       max_clipping_fraction=0.01, max_silence_fraction=0.8,
                       min_cyrillic_fraction=0.6, max_chars=500)


def normalize(text):
    # Keep punctuation, case, digits, Ө/Ү and spelling; never transliterate or guess pronunciation.
    return ' '.join(unicodedata.normalize('NFC', text).split())


def text_key(text):
    return ''.join(c for c in normalize(text).casefold() if c.isalnum())


def text_error(text, cfg):
    if not isinstance(text, str) or not text.strip():
        return 'empty_text'
    if any(unicodedata.category(c).startswith('C') and c not in '\n\r\t' for c in text):
        return 'control_character'
    if '<|' in text or '|>' in text:
        return 'model_control_token'
    if len(text) > cfg['max_chars']:
        return 'text_too_long'
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 'no_letters'
    cyrillic = sum('CYRILLIC' in unicodedata.name(c, '') for c in letters) / len(letters)
    if cyrillic < cfg['min_cyrillic_fraction']:
        return 'insufficient_cyrillic'


def split_records(rows, seed, ratios, speaker_disjoint=True):
    """Union text/speaker/session groups; reserve official holdouts before assignment.

    A train row connected to any official holdout is rejected, never promoted into it.
    Test takes priority over validation if upstream sets themselves overlap.
    """
    parent = list(range(len(rows)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(i, j):
        parent[find(j)] = find(i)
    keys = {}
    for i, r in enumerate(rows):
        group_keys = [('text', text_key(r['text'])), ('audio', r['pcm_sha256'])]
        if speaker_disjoint and r['speaker_id']:
            group_keys.append(('speaker', r['speaker_id']))
        if r.get('session_id'):
            group_keys.append(('session', r['source'], r['session_id']))
        for key in group_keys:
            if key in keys:
                union(i, keys[key])
            keys[key] = i
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        groups[find(i)].append(r)
    accepted, rejected = [], []
    for group in groups.values():
        fixed = {r['source_split'] for r in group} & {'validation', 'test'}
        if fixed:
            chosen = 'test' if 'test' in fixed else 'validation'
        else:
            identity = '|'.join(sorted(r['id'] for r in group))
            fraction = int(hashlib.sha256(f'{seed}:{identity}'.encode()).hexdigest()[:16], 16) / 2**64
            chosen = 'test' if fraction < ratios['test'] else ('validation' if fraction < ratios['test'] + ratios['validation'] else 'train')
        for r in group:
            if fixed and r['source_split'] != chosen:
                rejected.append(dict(r, rejection='heldout_leakage'))
            else:
                accepted.append(dict(r, split=chosen))
    return accepted, rejected


def prepare(config, output):
    overrides = config.get('filters', {})
    if set(overrides) - DEFAULT_FILTERS.keys():
        raise ValueError('Unknown quality filter')
    filters = DEFAULT_FILTERS | overrides
    if any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in filters.values()):
        raise ValueError('Filter thresholds must be finite numbers')
    if not 0 < filters['min_duration'] <= filters['max_duration'] or filters['max_chars'] < 1:
        raise ValueError('Invalid duration/text bounds')
    for name in ('max_clipping_fraction', 'max_silence_fraction', 'min_cyrillic_fraction'):
        if not 0 <= filters[name] <= 1:
            raise ValueError(f'{name} must be between zero and one')
    ratios = config.get('splits', {'validation': 0.05, 'test': 0.05})
    if set(ratios) != {'validation', 'test'} or any(not 0 <= v <= 1 for v in ratios.values()) or sum(ratios.values()) >= 1:
        raise ValueError('Splits require validation/test fractions whose sum is below one')
    sources = config['sources']
    if len({s['id'] for s in sources}) != len(sources):
        raise ValueError('Duplicate source IDs')
    out = fresh_dir(output)
    (out / 'audio').mkdir()
    accepted, rejected, seen_ids = [], [], set()
    for source in sources:
        if not source.get('enabled', True):
            continue
        for row in records(source):
            if row['id'] in seen_ids:
                raise ValueError(f"Duplicate record ID: {row['id']}")
            seen_ids.add(row['id'])
            if row['source_split'] not in {'train', 'validation', 'test', 'unspecified', 'withheld'}:
                raise ValueError(f"Unknown source split: {row['source_split']}")
            if source['kind'] == 'mbspeech' and not source.get('rights_clearance'):
                rejected.append(dict(row, rejection='rights_quarantine'))
                continue
            if row['source_split'] == 'withheld':
                rejected.append(dict(row, rejection='source_withheld'))
                continue
            reason = text_error(row['original_text'], filters)
            if reason:
                rejected.append(dict(row, rejection=reason))
                continue
            row['text'] = normalize(row['original_text'])
            target = out / 'audio' / (hashlib.sha256(row['id'].encode()).hexdigest() + '.wav')
            try:
                row['original_audio_sha256'] = digest(row['audio'])
                row.update(convert(row['audio'], target))
                row.update(inspect(target))
            except RuntimeError:
                # Missing system dependency is a run error, not thousands of bad audio clips.
                raise
            except (OSError, ValueError, EOFError, wave.Error, subprocess.CalledProcessError) as exc:
                target.unlink(missing_ok=True)
                rejected.append(dict(row, rejection='audio_decode', detail=type(exc).__name__))
                continue
            row['original_audio'] = row['audio']
            row['audio'] = str(target)
            row['language'] = 'mn-MN'  # metadata only; NOT a Qwen codec language ID
            reason = None
            if not filters['min_duration'] <= row['duration'] <= filters['max_duration']:
                reason = 'duration'
            elif row['rms_dbfs'] < filters['min_rms_dbfs']:
                reason = 'quiet'
            elif row['clipping_fraction'] > filters['max_clipping_fraction']:
                reason = 'clipping'
            elif row['silence_fraction'] > filters['max_silence_fraction']:
                reason = 'silence'
            if reason:
                rejected.append(dict(row, rejection=reason))
            else:
                accepted.append(row)
    # Build leakage relationships BEFORE removing duplicates; a duplicate may connect
    # another speaker/session's training rows to an official holdout.
    accepted, leaked = split_records(accepted, config.get('seed', 2026), ratios,
                                    config.get('speaker_disjoint', True))
    rejected.extend(leaked)
    # Prefer official test, then validation when exact waveforms recur.
    priority = {'test': 0, 'validation': 1, 'train': 2, 'unspecified': 3}
    seen_audio, unique = {}, []
    for row in sorted(accepted, key=lambda r: (priority[r['source_split']], r['id'])):
        key = row['pcm_sha256']
        if key in seen_audio:
            rejected.append(dict(row, rejection='duplicate_audio', duplicate_of=seen_audio[key]))
        else:
            seen_audio[key] = row['id']
            unique.append(row)
    accepted = unique
    # Per-speaker, per-partition references prevent evaluation audio conditioning training.
    refs = {}
    for row in sorted(accepted, key=lambda r: r['id']):
        if row['speaker_id']:
            refs.setdefault((row['speaker_id'], row['split']), row['audio'])
    for row in accepted:
        row['ref_audio'] = refs.get((row['speaker_id'], row['split']))
        row['reference_policy'] = 'same-speaker-within-split' if row['ref_audio'] else 'unknown-speaker-no-reference'
    accepted.sort(key=lambda r: r['id'])
    rejected.sort(key=lambda r: r['id'])
    # Rejected output waveforms are disposable; original sources remain untouched.
    keep = {r['audio'] for r in accepted}
    for p in (out / 'audio').glob('*.wav'):
        if str(p) not in keep:
            p.unlink()
    write_rows(out / 'manifest.jsonl', accepted)
    write_rows(out / 'rejected.jsonl', rejected)
    for split in ('train', 'validation', 'test'):
        write_rows(out / f'{split}.jsonl', [r for r in accepted if r['split'] == split])
    report = dict(version=__version__, accepted=len(accepted), rejected=len(rejected),
                  measured_rejected_hours=sum(r.get('duration', 0) for r in rejected)/3600,
                  unmeasured_rejections=sum('duration' not in r for r in rejected),
                  reasons=dict(Counter(r['rejection'] for r in rejected)),
                  unknown_speaker_rows=sum(r['speaker_id'] is None for r in accepted),
                  config=config, filters=filters,
                  limitations=['No perceptual near-duplicate detection', 'Unknown speakers prevent certified speaker-disjoint evaluation',
                               'Cyrillic screening does not certify Khalkha', 'Human transcript/dialect review still required'])
    for field in ('split', 'source'):
        report[field] = {value: dict(rows=sum(r[field] == value for r in accepted),
                                    hours=sum(r['duration'] for r in accepted if r[field] == value)/3600)
                         for value in sorted({r[field] for r in accepted})}
    report['accepted_hours'] = sum(r['duration'] for r in accepted)/3600
    report['manifest_sha256'] = digest(out / 'manifest.jsonl')
    report['source_metadata_sha256'] = {}
    for source in sources:
        if not source.get('enabled', True):
            continue
        root = Path(source['path']).expanduser().resolve()
        names = ['train.tsv', 'dev.tsv', 'test.tsv', 'validated.tsv',
                 source.get('manifest', 'metadata.jsonl'), 'acquisition.json', 'source.json']
        report['source_metadata_sha256'][source['id']] = {
            name: digest(root / name) for name in names if (root / name).is_file()}
    report['config_sha256'] = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    report['python_version'] = __import__('platform').python_version()
    try:
        report['ffmpeg'] = subprocess.run([ffmpeg_executable(), '-version'], capture_output=True, text=True).stdout.splitlines()[0]
    except RuntimeError:
        report['ffmpeg'] = None
    write_json(out / 'report.json', report)
    return report
