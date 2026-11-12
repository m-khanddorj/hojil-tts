"""Complete calibration scope and checkpoint decisions for masked real-speech SFT.

Quality decisions require independently verified complete 80-recording audits.
Calibration is a selection gate; the unchanged original 114 panel is still
required to establish the user's goal. Step zero is diagnostic only.
"""
import math
from .reference_masked_supervision_plan import custom_calibration_texts
from .targeted_voice import aggregate
from scripts.score_latest_audio import score as score_text


CHECKPOINTS = [0, 8, 16, 32]
COUNTS = {'development': 64, 'custom': 16}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def calibration_cases(plan, calibration, reference):
    require(len(calibration) == 32 and len({r['case'] for r in calibration}) == 32
        and len({r['speaker_id'] for r in calibration}) == 16, 'Complete reserved 32-case/16-speaker panel required')
    require(plan['calibration_cases'] == [r['case'] for r in calibration]
        and plan['custom_calibration'] == custom_calibration_texts(calibration)
        and plan['generation_seeds'] == [2026, 2027] and plan['calibration_recordings'] == 80,
        'Bound calibration selection changed')
    require(len(reference['reference_sha256']) == 64 and reference['reference_text'].strip(),
        'Original custom reference identity required')
    cases = [dict(case=r['case'], group='development', speaker_id=r['speaker_id'], text=r['text'],
        reference=r['ref_audio'], reference_sha256=r['ref_audio_sha256'], reference_text=r['ref_text'],
        feature_id=r['ref_id'], source_pair_id=r['id'], half_reference=False) for r in calibration]
    require(all(c['reference_sha256'] != reference['reference_sha256'] for c in cases),
        'Custom reference overlaps development reference')
    for r in plan['custom_calibration']:
        cases.append(dict(case=r['case'], group='custom', speaker_id=reference['speaker_id'], text=r['text'],
            reference=reference['reference'], reference_sha256=reference['reference_sha256'],
            reference_text=reference['reference_text'], feature_id='__custom_reference__',
            source_pair_id=r['source_pair_id'], source_case=r['source_case'], half_reference=True))
    return [dict(c, seed=seed) for seed in plan['generation_seeds'] for c in cases]


def validate_records(cases, records):
    expected = {(c['case'], c['seed']): c for c in cases}
    require(len(expected) == len(cases) == len(records) == 80
        and {(r['case'], r['seed']) for r in records} == set(expected), 'Incomplete or duplicated calibration speech')
    require({g: sum(c['group'] == g for c in cases) for g in COUNTS} == COUNTS,
        'Both complete calibration groups required')
    for r in records:
        require(all(r.get(k) == v for k, v in expected[r['case'], r['seed']].items()), 'Calibration metadata changed')
        require(type(r['terminated']) is bool and math.isfinite(r['cosine']) and -1 <= r['cosine'] <= 1,
            'Invalid finite voice or termination value')
        require(r['score'] == score_text(r['text'], r['hypothesis'], True), 'ASR edit counts differ')
    return aggregate(records)


def verified_summary(audit):
    require(audit['status'] == 'verified' and audit['recordings'] == 80
        and all(audit.get(k) is True for k in ('frozen_weights_verified', 'native_conditioning_verified',
            'all_artifact_hashes_verified', 'all_recomputed_cosines_finite')), 'Complete native calibration audit required')
    error = audit['max_cosine_recomputation_error']
    require(math.isfinite(error) and 0 <= error <= 2e-6, 'Independent voice audit differs')
    summary = audit['summary']
    require(set(summary) == set(COUNTS), 'Both calibration groups required')
    for group, count in COUNTS.items():
        r = summary[group]
        require(r['recordings'] == count and r['speakers'] == (16 if group == 'development' else 1)
            and all(math.isfinite(r[k]) for k in ('pooled_cer', 'pooled_wer', 'speaker_macro_cosine'))
            and r['pooled_cer'] >= 0 and r['pooled_wer'] >= 0 and -1 <= r['speaker_macro_cosine'] <= 1,
            'Invalid or incomplete group metrics')
    require(type(audit['truncated']) is int and 0 <= audit['truncated'] <= 80, 'Invalid truncation count')
    movement = audit['adapter_movement']['maximum']
    require(math.isfinite(movement) and movement >= 0 and (audit['step'] != 0 or movement == 0),
        'Invalid new-adapter movement or nonzero baseline')
    return summary


def dominates(earlier, current):
    comparisons = []
    for group in COUNTS:
        a, b = earlier[group], current[group]
        comparisons.extend([(a['pooled_cer'], b['pooled_cer']),
                            (-a['speaker_macro_cosine'], -b['speaker_macro_cosine'])])
    return all(a <= b for a, b in comparisons) and any(a < b for a, b in comparisons)


def progression(history):
    require(history and [r['step'] for r in history] == CHECKPOINTS[:len(history)],
        'Contiguous audited checkpoints starting at zero required')
    summaries = [verified_summary(r) for r in history]
    streak = 0
    for index, (audit, summary) in enumerate(zip(history, summaries)):
        step = audit['step']
        qualified = (step > 0 and audit['adapter_movement']['maximum'] > 0 and audit['truncated'] == 0
            and all(r['pooled_cer'] < .03 and r['speaker_macro_cosine'] > .984 for r in summary.values()))
        if qualified:
            require(index == len(history)-1, 'Fitting continued after calibration qualification')
            return dict(stop=True, reason='calibration_qualified_needs_original114', step=step, goal_achieved=False)
        dominated = index > 0 and any(dominates(old, summary) for old in summaries[:index])
        streak = streak+1 if dominated else 0
        if streak >= 2:
            require(index == len(history)-1, 'Fitting continued after consecutive Pareto regressions')
            return dict(stop=True, reason='consecutive_pareto_regressions', step=step, goal_achieved=False)
    step = history[-1]['step']
    return dict(stop=step == CHECKPOINTS[-1], reason='maximum_updates_reached' if step == CHECKPOINTS[-1]
        else 'continue', step=step, goal_achieved=False)
