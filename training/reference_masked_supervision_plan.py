"""Deterministic data-only plan for the bounded real-speech masked-LoRA pilot.

No native imports, feature extraction, model selection, or optimizer updates.
The caller must verify the input manifests and bind the returned plan before
extracting features or observing candidate quality.
"""
from collections import defaultdict
import hashlib


SEED = 20261111
SETTINGS = dict(seed=SEED, updates=32, checkpoints=[0, 8, 16, 32],
    pairs_per_update=8, fitting_probe_pairs=16, custom_calibration_texts=8,
    rank=4, layers=list(range(10, 18)), learning_rate=2e-5,
    weight_decay=0.0, gradient_clip=1.0, residual_mean_ce_weight=0.3,
    forward_dtype='float32', optimizer_dtype='float32', dropout=0.0)


def rank_key(namespace, value, seed=SEED):
    return hashlib.sha256(f'{namespace}:{seed}:{value}'.encode('utf-8')).hexdigest()


def transcript_key(text):
    import unicodedata
    return ''.join(c for c in unicodedata.normalize('NFC', text).casefold() if c.isalnum())


def fitting_schedule(rows, *, updates=32, pairs_per_update=8, seed=SEED):
    if updates < 1 or pairs_per_update < 1:
        raise ValueError('Positive update and pair counts required')
    indexed = {r['id']: r for r in rows}
    if not rows or len(indexed) != len(rows):
        raise ValueError('Nonempty unique fitting IDs required')
    groups = defaultdict(list)
    for r in rows:
        reference = indexed.get(r['ref_id'])
        if (reference is None or reference['id'] == r['id']
                or reference['speaker_id'] != r['speaker_id']):
            raise ValueError('Fitting references must be distinct same-speaker rows in fitting')
        groups[r['speaker_id']].append(r['id'])
    speakers = sorted(groups, key=lambda s: (rank_key('masked-supervision-speakers-v1', s, seed), s))
    if len(speakers) < pairs_per_update:
        raise ValueError('Insufficient distinct speakers per update')
    queues = {s: sorted(ids, key=lambda i: (rank_key('masked-supervision-pairs-v1', i, seed), i))
        for s, ids in groups.items()}
    visits = defaultdict(int)
    result = []
    # A fixed shuffled cycle cannot repeat a speaker within an update when
    # there are at least pairs_per_update speakers, including at its boundary.
    for update in range(1, updates + 1):
        chosen = []
        for offset in range(pairs_per_update):
            s = speakers[((update - 1) * pairs_per_update + offset) % len(speakers)]
            queue = queues[s]
            chosen.append(queue[visits[s] % len(queue)])
            visits[s] += 1
        modes = ['full', 'half'] if update % 2 else ['half', 'full']
        result.append(dict(update=update, pair_ids=chosen, modes=modes,
            pair_weight=1.0 / pairs_per_update, mode_weight=0.5))
    return result


def fitting_probe(rows, count=16, seed=SEED):
    groups = defaultdict(list)
    for r in rows:
        groups[r['speaker_id']].append(r['id'])
    if count < 1 or len(groups) < count:
        raise ValueError('Probe requires one pair per distinct speaker')
    speakers = sorted(groups, key=lambda s: (rank_key('masked-supervision-probe-speakers-v1', s, seed), s))[:count]
    return [min(groups[s], key=lambda i: (rank_key('masked-supervision-probe-pairs-v1', i, seed), i)) for s in speakers]


def custom_calibration_texts(calibration, count=8, seed=SEED):
    unique = {}
    for r in sorted(calibration, key=lambda r: r['case']):
        key = transcript_key(r['text'])
        if not key:
            raise ValueError('Empty calibration transcript')
        unique.setdefault(key, r)
    if count < 1 or len(unique) < count:
        raise ValueError('Insufficient distinct reserved calibration texts')
    keys = sorted(unique, key=lambda k: (rank_key('reference-masked-supervision-custom-calibration-v1', k, seed), k))[:count]
    return [dict(case=f'masked-custom-calibration-{i:03d}', source_case=unique[k]['case'],
        source_pair_id=unique[k]['id'], text=unique[k]['text'], text_key=k)
        for i, k in enumerate(keys)]


def build_plan(fitting, calibration):
    schedule = fitting_schedule(fitting)
    probe = fitting_probe(fitting)
    if len(calibration) != 32 or len({r['case'] for r in calibration}) != 32:
        raise ValueError('All 32 reserved calibration cases required')
    if {r['speaker_id'] for r in fitting} & {r['speaker_id'] for r in calibration}:
        raise ValueError('Fitting and calibration speakers overlap')
    indexed = {r['id']: r for r in fitting}
    selected = {i for step in schedule for i in step['pair_ids']} | set(probe)
    required = [indexed[i] for i in sorted(selected)] + list(calibration)
    feature_ids = sorted({r[k] for r in required for k in ('id', 'ref_id')})
    return dict(kind='masked_supervision_pre_quality_data_schedule', settings=dict(SETTINGS),
        updates=schedule, fitting_probe_pair_ids=probe,
        custom_calibration=custom_calibration_texts(calibration),
        calibration_cases=[r['case'] for r in calibration],
        selected_fitting_pair_ids=sorted(selected), real_feature_ids=feature_ids,
        generation_seeds=[2026, 2027], calibration_recordings=80,
        feature_policy='Fresh native FP32 features for every scheduled/probe/calibration target and reference; custom reference separately bound. No unplanned row may enter fitting.',
        baseline_is_selection_candidate=False, optimizer_updates_executed=0)
