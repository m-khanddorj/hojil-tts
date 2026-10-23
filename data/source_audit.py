"""FLEURS source-anomaly screening; suspected slowdown is not a verified speed label."""
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import statistics
import struct

from .io import read_rows, write_json
from .pipeline import text_key
from .sources import records


def read_source_wav(path):
    """Read original PCM16 or IEEE-float32 RIFF/WAVE without changing its clock."""
    import numpy as np
    raw = Path(path).read_bytes()
    if raw[:4] != b'RIFF' or raw[8:12] != b'WAVE':
        raise ValueError('Expected RIFF/WAVE source')
    pos, fmt, payload = 12, None, None
    while pos + 8 <= len(raw):
        tag, size = struct.unpack_from('<4sI', raw, pos)
        chunk = raw[pos+8:pos+8+size]
        if len(chunk) != size:
            raise ValueError('Truncated WAV chunk')
        if tag == b'fmt ':
            fmt = chunk
        if tag == b'data':
            payload = chunk
        pos += 8 + size + size % 2
    if fmt is None or payload is None:
        raise ValueError('Missing WAV format/data')
    kind, channels, rate, _, _, bits = struct.unpack_from('<HHIIHH', fmt)
    if kind == 65534:
        kind = struct.unpack_from('<I', fmt, 24)[0]
    if (rate, channels) != (16000, 1):
        raise ValueError('FLEURS audit expects native mono 16000 Hz')
    if (kind, bits) == (3, 32):
        samples = np.frombuffer(payload, dtype='<f4').astype(np.float64)
    elif (kind, bits) == (1, 16):
        samples = np.frombuffer(payload, dtype='<i2').astype(np.float64) / 32768
    else:
        raise ValueError('Unsupported source WAV sample format')
    if len(samples) < 1024 or not np.isfinite(samples).all():
        raise ValueError('Invalid or too-short source audio')
    return samples, hashlib.sha256(raw).hexdigest()


def source_features(row, prepared=None):
    import numpy as np
    samples, sha = read_source_wav(row['audio'])
    frames = np.lib.stride_tricks.sliding_window_view(samples, 1024)[::512]
    frames = frames - frames.mean(axis=1, keepdims=True)
    energy = np.mean(frames**2, axis=1)
    selected = frames[energy >= np.quantile(energy, .5)]
    spectrum = np.sum(np.abs(np.fft.rfft(selected*np.hanning(1024), axis=1))**2, axis=0)
    frequencies = np.fft.rfftfreq(1024, 1/16000)
    total = max(float(spectrum.sum()), 1e-30)
    cumulative = np.cumsum(spectrum) / total
    active = np.flatnonzero(energy >= max(float(energy.max())*.01, 1e-12))
    result = dict(id=row['id'], text=row['original_text'], path=row['audio'], source_sha256=sha,
        duration=len(samples)/16000, retained=prepared is not None,
        active_span_seconds=float(((active[-1]-active[0])*512+1024)/16000) if len(active) else 0,
        rolloff={str(q): float(frequencies[min(np.searchsorted(cumulative, q), len(frequencies)-1)])
                 for q in (.5, .9, .95, .99)},
        band_fraction={str(cut): float(spectrum[frequencies < cut].sum()/total)
                       for cut in (20, 50, 80, 200, 500, 1000, 2000, 4000)})
    if prepared:
        audit = prepared['audio_processing']
        result.update(processed_duration=prepared['duration'], flags=audit['review_flags'],
                      quality_before=audit.get('quality_before'), vad=audit.get('vad_before'))
    return result


def classify_sources(rows):
    """Quarantine severe anomalies; independent cues prevent a low-pitch-only rule.

    Thresholds are conservative engineering heuristics for this diagnosed corpus,
    not a calibrated multilingual speed detector. Unmatched cases remain uncertain.
    """
    groups = defaultdict(list)
    for row in rows:
        groups[text_key(row['text'])].append(row)
    paired, unresolved = [], []
    for peers in groups.values():
        references = [p for p in peers if p['rolloff']['0.99'] >= 4000 and p['rolloff']['0.95'] >= 1500
                      and (p.get('quality_before') or {}).get('speech', 0) >= 3
                      and p['active_span_seconds'] >= 2]
        for row in peers:
            if not row['retained']:
                continue
            if not (row['rolloff']['0.99'] <= 2000 and row['rolloff']['0.95'] <= 1000
                    and row['band_fraction']['4000'] >= .995):
                continue
            coverage = (row.get('vad') or {}).get('speech_seconds', row['duration']) / row['duration']
            weak_speech = coverage < .5 or (row.get('quality_before') or {}).get('speech', 5) < 2.8
            if not weak_speech:
                continue
            evidence = dict(id=row['id'], vad_coverage=coverage, rolloff=row['rolloff'],
                            below_4000hz_fraction=row['band_fraction']['4000'],
                            source_sha256=row.get('source_sha256'))
            if not references:
                unresolved.append(dict(evidence, reason='source_slowdown_unresolved',
                                       confidence='uncertain severe source anomaly; no suitable same-text reference'))
                continue
            span = statistics.median(p['active_span_seconds'] for p in references)
            ratio = row['active_span_seconds']/span
            if ratio >= 1.6:
                paired.append(dict(evidence, reason='suspected_slowed_source', duration_ratio=ratio,
                                   peer_span_seconds=span, reference_ids=[p['id'] for p in references],
                                   confidence='multiple-cue suspicion; human confirmation required for a definitive speed label'))
    return dict(paired=paired, unresolved=unresolved,
                limitations=['Not a universal or validated slowdown detector',
                             'Natural voice differences and channel filtering can resemble slowdown',
                             'Quarantine is reversible; original recordings are retained'])


def audit_fleurs(prepared, output, workers=6):
    prepared, output = Path(prepared), Path(output)
    config = json.loads((prepared/'report.json').read_text(encoding='utf-8'))['config']
    source = next(s for s in config['sources'] if s['kind'] == 'fleurs')
    retained = {r['id']: r for r in read_rows(prepared/'manifest.jsonl') if r['source'] == source['id']}
    inputs = list(records(source))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        features = list(pool.map(lambda r: source_features(r, retained.get(r['id'])), inputs))
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'features.json', features)
    result = classify_sources(features)
    result.update(scanned=len(features), retained_scanned=len(retained), source=source)
    write_json(output/'source-exclusions.json', result)
    return result
