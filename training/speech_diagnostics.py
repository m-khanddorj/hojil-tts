"""Expose paired generation failures that average teacher-forced loss can hide."""
import argparse
from collections import defaultdict
from pathlib import Path

from .compare_experiments import verified_evaluation
from .engine import write
from .evaluate import compare
from .policy import external


def diagnose(pairs):
    records, per_seed = [], []
    speaker_counts = defaultdict(lambda: [0,0,0])
    for plan, baseline, candidate in pairs:
        comparison = compare(baseline,candidate,plan['cases'],'baseline')
        per_seed.append(dict(seed=plan['seed'],baseline=comparison['baseline'],candidate=comparison['candidate']))
        old, new = {s['case']:s for s in baseline}, {s['case']:s for s in candidate}
        for case in plan['cases']:
            a,b = old[case['case']],new[case['case']]
            x,y = a['without_punctuation'],b['without_punctuation']
            records.append(dict(seed=plan['seed'],case=case['case'],speaker_id=case['speaker_id'],
                delta_word_errors=y['word_errors']-x['word_errors'],
                delta_character_errors=y['character_errors']-x['character_errors'],
                baseline_wer=x['wer'],candidate_wer=y['wer'],
                baseline_cer=x['cer'],candidate_cer=y['cer'],
                baseline_seconds=a['seconds'],candidate_seconds=b['seconds'],
                reference=a.get('reference'),baseline_transcript=a.get('hypothesis'),candidate_transcript=b.get('hypothesis')))
            s = speaker_counts[case['speaker_id']]
            s[0] += x['word_errors']; s[1] += y['word_errors']; s[2] += x['words']
    if not records:
        raise ValueError('No speech evaluations supplied')
    summary = dict(recordings=len(records),seeds=len(per_seed),
        unique_prompts=len({r['case'] for r in records}),speakers=len(speaker_counts),
        delta_word_errors=sum(r['delta_word_errors'] for r in records),
        delta_character_errors=sum(r['delta_character_errors'] for r in records),
        baseline_high_error_recordings=sum(r['baseline_wer']>=.5 for r in records),
        candidate_high_error_recordings=sum(r['candidate_wer']>=.5 for r in records),
        improved_recordings=sum(r['delta_word_errors']<0 for r in records),
        unchanged_recordings=sum(r['delta_word_errors']==0 for r in records),
        worsened_recordings=sum(r['delta_word_errors']>0 for r in records))
    return dict(summary=summary,per_seed=per_seed,
        per_speaker={s:dict(baseline_wer=a/n,candidate_wer=b/n,words=n) for s,(a,b,n) in speaker_counts.items()},
        largest_regressions=sorted((r for r in records if r['delta_word_errors']>0),key=lambda r:r['delta_word_errors'],reverse=True)[:5],
        recordings=records,
        interpretation='ASR proxy only. High error means WER >= 50%; it is a review flag, not a perceptual verdict. All recordings remain in pooled scores.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--seeds',type=int,nargs='+',required=True)
    parser.add_argument('--cohort',choices=('unseen-validation','unseen-test'),default='unseen-validation')
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError('Duplicate generation seeds')
    pairs = []
    for seed in args.seeds:
        a,baseline = verified_evaluation(args.baseline,seed,args.cohort)
        b,candidate = verified_evaluation(args.candidate,seed,args.cohort)
        if a != b:
            raise ValueError('Evaluation plans differ')
        pairs.append((a,baseline,candidate))
    report = diagnose(pairs)
    report.update(baseline_run=str(args.baseline),candidate_run=str(args.candidate),cohort=args.cohort)
    filename = 'speech-diagnostics.json' if args.cohort == 'unseen-validation' else 'speech-test-diagnostics.json'
    write(external(args.candidate)/filename,report)
    print(report['summary'])


if __name__ == '__main__':
    main()
