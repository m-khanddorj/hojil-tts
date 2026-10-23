"""Review exclusions and conservative trim repair, preserving existing partitions."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import wave

from .adaptation import export_adaptation
from .audio import inspect
from .bulk import atomic_json
from .io import digest, fresh_dir, read_rows, write_json
from .pipeline import audio_error, finalize, validated_config
from .preprocessing import trim_edges
from .repartition import overlap_audit, restore_selected
from .source_audit import audit_fleurs


def repair_trim(row, output, filters):
    if row.get('human_review', {}).get('decision') == 'keep':
        return row, 'human_approved'
    old = row['audio_processing']
    removed = old.get('leading_trim_seconds', 0) + old.get('trailing_trim_seconds', 0)
    if removed <= old['before']['duration'] * .4:
        return row, 'unchanged'
    out = Path(output)
    with tempfile.TemporaryDirectory(prefix='trim-check-', dir=out/'audio') as temp:
        selected = Path(temp)/'selected.wav'
        restore_selected(row, selected)
        with wave.open(str(selected), 'rb') as audio:
            raw = audio.readframes(audio.getnframes())
        start, stop = old['leading_trim_samples'], len(raw)//2 - old['trailing_trim_samples']
        if hashlib.sha256(raw[start*2:stop*2]).hexdigest() != row['pcm_sha256']:
            raise ValueError('Selected-waveform replay differs from old cleaned PCM: ' + row['id'])
        options = deepcopy(old['settings'])
        section = 'vad' if old.get('trim_backend') == 'silero-vad' else 'trim'
        options[section].update(max_removed_duration_fraction=.4, max_removed_energy_fraction=.1)
        bounds = None
        if section == 'vad':
            segments = old['vad_selected']['segments']
            bounds = (math.floor(segments[0]['start']*1.5), math.ceil(segments[-1]['end']*1.5))
        target = Path(temp)/'guarded.wav'
        guard = trim_edges(selected, target, options[section], bounds)
        if guard['trim_decision'] != 'reverted_edge_energy':
            return row, 'checked_safe'
        revised = deepcopy(row)
        audit = revised['audio_processing']
        audit['trim_safety_revision'] = dict(version=1, previous_audio=row['audio'], previous_pcm_sha256=row['pcm_sha256'],
            previous_leading_trim_samples=start, previous_trailing_trim_samples=len(raw)//2-stop)
        audit.update(guard)
        audit['settings'] = options
        audit['review_flags'] = [f for f in audit['review_flags'] if f not in ('large_edge_trim', 'low_predicted_quality')]
        audit['review_flags'].append('unsafe_edge_trim_reverted')
        quality = old.get('quality_candidate') if old['denoise_decision'] in ('kept_quality_gain', 'applied_without_quality_gate') else old.get('quality_before')
        if quality:
            audit['quality_after'] = quality
            audit['quality_after_origin'] = 'previous score of the identical untrimmed selected waveform'
            if quality['overall'] < 3 or quality['speech'] < 3:
                audit['review_flags'].append('low_predicted_quality')
        revised.update(inspect(target))
        destination = out/'audio'/(hashlib.sha256(row['id'].encode()).hexdigest()+'.wav')
        target.replace(destination)
        revised['audio'] = str(destination)
        reason = audio_error(revised, filters)
        if reason:
            revised['rejection'] = reason
        return revised, 'trim_reverted'


def revise(prepared, output, review, workers=6):
    prepared = Path(prepared).resolve()
    parent = json.loads((prepared/'report.json').read_text(encoding='utf-8'))
    config = deepcopy(parent['config'])
    processing, filters = validated_config(config)
    accepted = list(read_rows(prepared/'manifest.jsonl'))
    by_id = {r['id']: r for r in accepted}
    human = json.loads(Path(review).read_text(encoding='utf-8-sig'))
    if human.get('schema') != 'hojil-listening-review-v1':
        raise ValueError('Unsupported human review schema')
    for sample in human['samples']:
        row = by_id.get(sample['id'])
        if not row or row['pcm_sha256'] != sample['pcm_sha256']:
            raise ValueError('Review does not match the parent canonical manifest')
        if sample['review']['decision'] not in {'keep', 'review', 'reject', 'unreviewed'}:
            raise ValueError('Unsupported human review decision')
    out = fresh_dir(output)
    (out/'audio').mkdir()
    def progress(stage, **extra):
        atomic_json(out/'progress.json', dict(status=stage, updated=datetime.now(timezone.utc).isoformat(), **extra))
    try:
        progress('auditing_fleurs_sources')
        screening = audit_fleurs(prepared, out/'source-audit', workers)
        exclusions = {r['id']: r for r in screening['paired'] + screening['unresolved']}
        for sample in human['samples']:
            if sample['review']['decision'] == 'reject':
                exclusions[sample['id']] = dict(id=sample['id'], reason='human_rejected',
                    review=sample['review'], source_screening=exclusions.get(sample['id']),
                    source_sha256=by_id[sample['id']]['original_audio_sha256'])
        for ident, evidence in exclusions.items():
            row = by_id[ident]
            if digest(row['original_audio']) != row['original_audio_sha256'] or evidence['source_sha256'] != row['original_audio_sha256']:
                raise ValueError('Source changed since original preparation: ' + ident)
        write_json(out/'review-exclusions.json', list(exclusions.values()))
        shutil.copyfile(review, out/'human-review.json')
        config['quality_revision'] = dict(parent_manifest=str(prepared/'manifest.jsonl'),
            parent_manifest_sha256=digest(prepared/'manifest.jsonl'), review_sha256=digest(review),
            source_features_sha256=digest(out/'source-audit'/'features.json'),
            policy='Quarantine severe FLEURS source anomalies; revert large energetic edge trims; preserve existing partitions')
        rejected = list(read_rows(prepared/'rejected.jsonl'))
        eligible = []
        human_by_id = {s['id']: s['review'] for s in human['samples'] if s['review']['decision'] != 'unreviewed'}
        for row in accepted:
            if row['id'] in exclusions:
                rejected.append(dict(row, rejection=exclusions[row['id']]['reason'], review_evidence=exclusions[row['id']]))
            else:
                eligible.append(dict(row, human_review=human_by_id[row['id']]) if row['id'] in human_by_id else row)
        results, counts = [], Counter()
        progress('checking_large_trims', processed=0, total=len(eligible), source_exclusions=len(exclusions))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for index, (row, action) in enumerate(pool.map(lambda r: repair_trim(r, out, filters), eligible), 1):
                counts[action] += 1
                (rejected if 'rejection' in row else results).append(row)
                if index % 100 == 0:
                    progress('checking_large_trims', processed=index, total=len(eligible), actions=dict(counts))
        audit = overlap_audit(results)
        if any(audit[k] for k in ('text_keys_across_splits', 'audio_keys_across_splits', 'original_audio_keys_across_splits')):
            raise ValueError('Revision introduced cross-partition overlap')
        progress('finalizing', actions=dict(counts))
        report = finalize(config, out, results, rejected, filters, processing, preserve_assigned_splits=True)
        if any(r['split'] != by_id[r['id']]['split'] for r in read_rows(out/'manifest.jsonl')):
            raise ValueError('Revision changed an existing partition assignment')
        progress('verifying_export')
        adaptation = export_adaptation(out/'manifest.jsonl', out/'adaptation')
        summary = dict(accepted=report['accepted'], rejected=report['rejected'], accepted_hours=report['accepted_hours'],
            source=report['source'], split=report['split'], reasons=report['reasons'], overlap_audit=audit,
            adaptation=adaptation, source_screening=dict(scanned=screening['scanned'],
                paired_suspicions=len(screening['paired']), unresolved_quarantined=len(screening['unresolved']),
                human_rejected=sum(s['review']['decision']=='reject' for s in human['samples'])),
            trim_safety=dict(counts), evaluation_policy=config.get('evaluation_policy'),
            asset_policy='Unchanged audio is referenced from the immutable parent run; corrected WAVs are in this run/audio')
        write_json(out/'completion-report.json', summary)
        progress('complete', accepted=report['accepted'], rejected=report['rejected'])
        return summary
    except BaseException as error:
        progress('failed', error=f'{type(error).__name__}: {error}')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--review', required=True)
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        parser.error('workers must be between 1 and 16')
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('RAYON_NUM_THREADS', '1')
    revise(args.prepared, args.output, args.review, args.workers)
