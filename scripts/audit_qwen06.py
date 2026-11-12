"""Read the historical 0.6B run without loading a model or modifying artifacts."""
import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import mean, median


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--continuation', type=Path, required=True)
    parser.add_argument('--cohort', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import torch
    from training.engine import digest, read, rows, write
    from training.policy import external
    torch.set_num_threads(4)
    checkpoint = torch.load(args.run / 'resume.pt', map_location='cpu', weights_only=True)
    later = torch.load(args.continuation / 'final-adapter/adapter.pt', map_location='cpu', weights_only=True)
    metrics = rows(args.run / 'metrics.jsonl')
    cohort = rows(args.cohort)
    speakers = Counter(r['speaker_id'] for r in cohort)
    unchanged = Counter()
    elements = Counter()
    for name, tensor in checkpoint['adapter'].items():
        kind = 'lora_A' if 'lora_A' in name else 'lora_B'
        elements[kind] += tensor.numel()
        unchanged[kind] += int((tensor == later[name]).sum())
    intervals = [(b['seconds']-a['seconds']) / (b['update']-a['update'])
                 for a, b in zip(metrics, metrics[1:])
                 if b['seconds'] >= a['seconds'] and b['update'] > a['update']]
    event_tail = metrics[-min(20, len(metrics)):]
    report = {
        'kind': 'LoRA; base parameters frozen',
        'config': checkpoint['config'],
        'source_sha256': {name: digest(args.run / name)
                          for name in ('train_full.py', 'run.json', 'metrics.jsonl', 'resume.pt')},
        'adapter_dtype_tensors': dict(Counter(str(t.dtype) for t in checkpoint['adapter'].values())),
        'optimizer_moment_dtypes': dict(Counter(str(s['exp_avg'].dtype)
                                             for s in checkpoint['optimizer']['state'].values())),
        'trainable_parameters': sum(t.numel() for t in checkpoint['adapter'].values()),
        'cohort': {'clips': len(cohort), 'hours': sum(r['duration'] for r in cohort)/3600,
                   'sources': dict(Counter(r['source'] for r in cohort)), 'speakers': len(speakers),
                   'top_10_speaker_clip_fraction': sum(n for _, n in speakers.most_common(10))/len(cohort),
                   'median_speaker_clips': median(speakers.values())},
        'training': {'logged_points': len(metrics), 'last_update': metrics[-1],
                     'median_seconds_per_update_between_logs': median(intervals),
                     'last_20_logged_mean_loss': mean(r['loss'] for r in event_tail),
                     'logged_gradient_clip_fraction': mean(r['gradient_norm'] > 1 for r in metrics)},
        'unchanged_fraction_after_176_more_updates': {k: unchanged[k]/elements[k] for k in elements},
        'evaluation': read(args.run / 'validation/summary.json'),
        'continuation_comparison': read(args.continuation / 'comparison.json'),
        'teacher_forcing': read(args.run / 'fit-diagnostic.json')['summary'],
        'limitations': ['Logged gradient fractions are not all-update statistics.',
                       'Unchanged adapter elements alone do not establish causal quality impact.',
                       'One generation seed and only ten unseen validation speakers.',
                       'Training loss is not a speech intelligibility or naturalness metric.'],
    }
    output = external(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write(output, report)
    print(json.dumps({k: report[k] for k in ('adapter_dtype_tensors', 'optimizer_moment_dtypes',
                     'cohort', 'training', 'unchanged_fraction_after_176_more_updates')}, indent=2))


if __name__ == '__main__':
    main()
