"""Deterministic training policy; no GPU dependencies."""
import hashlib
import json
import math
from pathlib import Path

from data.pipeline import text_key


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':')).encode()).hexdigest()


def conditioning(row_id, seed, epoch, non_streaming_only=False):
    n = int(fingerprint([seed, epoch, row_id, 'conditioning']), 16)
    return {'icl': n % 10 < 8,
            'non_streaming': non_streaming_only or (n // 10) % 2 == 0}


def update_groups(rows, size):
    """Fold a tiny tail into the preceding update, instead of a one-clip step."""
    if size < 1:
        raise ValueError('Update size must be positive')
    for start in range(0, len(rows), size):
        if start and len(rows) - start < max(2, size // 2):
            break
        end = min(start + size, len(rows))
        if 0 < len(rows) - end < max(2, size // 2):
            end = len(rows)
        yield rows[start:end]


def microbatches(items, frame_limit, token_limit, max_batch=6):
    """Items carry target-frame count and FULL input length (including reference).

    Frame count controls the residual predictor; padded length controls talker
    activations. A singleton can exceed the budget and is never silently dropped.
    """
    if min(frame_limit, token_limit, max_batch) < 1:
        raise ValueError('Batch budgets must be positive')
    batch, frames, longest = [], 0, 0
    for item in sorted(items, key=lambda x: x['length']):
        length = max(longest, item['length'])
        if batch and (frames + item['frames'] > frame_limit
                      or length * (len(batch) + 1) > token_limit
                      or len(batch) == max_batch):
            yield batch
            batch, frames, longest = [], 0, 0
        batch.append(item)
        frames += item['frames']
        longest = max(longest, item['length'])
    if batch:
        yield batch


def learning_rate(step, total, peak, end, warmup):
    if not 1 <= step <= total or not 0 <= warmup < total or not 0 < end <= peak:
        raise ValueError('Invalid learning-rate schedule')
    if step <= warmup:
        return peak * step / warmup
    progress = (step - warmup - 1) / max(1, total - warmup - 1)
    return end + (peak - end) * 0.5 * (1 + math.cos(math.pi * progress))


def external(path):
    path = Path(path).resolve()
    if any((parent / '.git').exists() for parent in (path, *path.parents)):
        raise ValueError('Training artifacts must be outside every Git checkout')
    return path


def validate_cohorts(train, validation, manifest, *, heldout_split='validation'):
    """Check both target AND reference identity, transcript, audio and split."""
    if heldout_split not in ('validation', 'test'):
        raise ValueError('Invalid held-out split')
    by_id = {r['id']: r for r in manifest}
    if len(by_id) != len(manifest) or not train or not validation:
        raise ValueError('Duplicate manifest IDs or empty cohort')
    train_ids = {r['id'] for r in train}
    if len(train_ids) != len(train):
        raise ValueError('Duplicate training IDs')
    used = {'train': {}, 'validation': {}}
    for split, rows in (('train', train), ('validation', validation)):
        expected_split = heldout_split if split == 'validation' else 'train'
        if len({r['id'] for r in rows}) != len(rows):
            raise ValueError('Duplicate cohort IDs')
        for row in rows:
            target, ref = by_id[row['id']], by_id[row['ref_id']]
            fields = ('text', 'audio', 'speaker_id', 'split', 'pcm_sha256')
            if any(row[k] != target[k] for k in fields):
                raise ValueError('Cohort differs from canonical manifest')
            if (target['split'] != expected_split or ref['split'] != expected_split
                    or not target['speaker_id'] or target['speaker_id'] != ref['speaker_id']
                    or row['ref_text'] != ref['text'] or row['ref_audio'] != ref['audio']
                    or target['id'] == ref['id'] or target['pcm_sha256'] == ref['pcm_sha256']
                    or text_key(target['text']) == text_key(ref['text'])
                    or (split == 'train' and ref['id'] not in train_ids)):
                raise ValueError('Invalid distinct same-speaker/same-split reference')
            used[split].update({target['id']: target, ref['id']: ref})
    for key in ('id', 'audio', 'pcm_sha256', 'original_pcm_sha256', 'text'):
        def values(split):
            return {text_key(r[key]) if key == 'text' else r[key]
                    for r in used[split].values() if r.get(key)}
        if values('train') & values('validation'):
            raise ValueError(f'Train/validation leakage: {key}')
    return by_id
