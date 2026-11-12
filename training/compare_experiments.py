"""Compare fixed generation seeds, clustering uncertainty by speaker across seeds."""
import argparse
from pathlib import Path

from .engine import digest, read, write
from .evaluate import ASR_REV, compare
from .policy import external
from .test_protocol import evaluation_folder


def verified_evaluation(run, seed, cohort='unseen-validation'):
    folder = run / evaluation_folder(cohort, seed)
    plan, scores = read(folder / 'plan.json'), read(folder / 'scores.json')
    model_sha = read(run / 'merged-model/training-provenance.json')['model_sha256']
    if (digest(run / 'merged-model/model.safetensors') != model_sha or plan['seed'] != seed
            or plan.get('cohort') != cohort):
        raise ValueError('Model or evaluation seed mismatch')
    cases = {c['case']: c for c in plan['cases']}
    if len(scores) != len(cases) or {s['case'] for s in scores} != set(cases):
        raise ValueError('Incomplete evaluation')
    for s in scores:
        audio = (folder / s['wav']).resolve()
        if (not audio.is_relative_to(folder.resolve()) or digest(audio) != s['audio_sha256']
                or s['model_sha256'] != model_sha or s['reference'] != cases[s['case']]['text']
                or s['asr_revision'] != ASR_REV or s['decoding'] != plan['decoding']):
            raise ValueError('Score provenance mismatch')
    return plan, scores


def combine(pairs):
    """pairs: (plan, baseline scores, candidate scores); all seeds equally retained."""
    old, new, cases, seeds, template = [], [], [], [], None
    per_seed = []
    for plan, baseline, candidate in pairs:
        seed = plan['seed']
        if seed in seeds:
            raise ValueError('Duplicate generation seed')
        settings = {k: v for k, v in plan.items() if k != 'seed'}
        if template is not None and settings != template:
            raise ValueError('Evaluation settings/cases differ across seeds')
        template = settings
        seeds.append(seed)
        per_seed.append(dict(seed=seed, **compare(baseline, candidate, plan['cases'], 'baseline')))
        old.extend(dict(s, case=f"{seed}:{s['case']}") for s in baseline)
        new.extend(dict(s, case=f"{seed}:{s['case']}") for s in candidate)
        cases.extend(dict(c, case=f"{seed}:{c['case']}") for c in plan['cases'])
    result = compare(old, new, cases, 'baseline')
    result.update(seeds=seeds, per_seed=per_seed,
                  interpretation='Development ASR proxy; seeds pooled, bootstrap clusters whole speakers across seeds; not independent test or perceptual quality evidence')
    if template.get('cohort') == 'unseen-test':
        result['interpretation'] = 'Reserved unseen-speaker test ASR proxy; bootstrap clusters speakers across both fixed seeds; not a perceptual-quality verdict'
        for item in per_seed:
            item['interpretation'] = 'Reserved unseen-speaker test ASR proxy; one fixed seed'
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cohort', choices=('unseen-validation','unseen-test'), default='unseen-validation')
    args = parser.parse_args()
    output = external(args.output)
    pairs = []
    for seed in args.seeds:
        plan, baseline = verified_evaluation(args.baseline, seed, args.cohort)
        other_plan, candidate = verified_evaluation(args.candidate, seed, args.cohort)
        if plan != other_plan:
            raise ValueError('Baseline/candidate plans differ')
        pairs.append((plan, baseline, candidate))
    result = combine(pairs)
    result.update(baseline_run=str(args.baseline), candidate_run=str(args.candidate))
    write(output, result)
    print(result)


if __name__ == '__main__':
    main()
