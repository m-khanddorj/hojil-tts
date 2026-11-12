"""Comparable loss probes and low-overhead training diagnostics."""
from collections import defaultdict
import math

from .policy import fingerprint


def matched_probe(train, validation, features, seed):
    """One retained training clip per familiar validation speaker, length matched.

    Only frame lengths and IDs affect selection, never losses. References remain
    the training cohort's own references. Persist IDs for repeatable monitoring.
    """
    by_speaker = defaultdict(list)
    for row in train:
        by_speaker[row['speaker_id']].append(row)
    probe = []
    for val in sorted(validation, key=lambda r: r['id']):
        if val['validation_group'] != 'familiar':
            continue
        candidates = by_speaker[val['speaker_id']]
        if not candidates:
            raise ValueError('Familiar validation speaker has no training examples')
        frames = len(features[val['id']]['codes'])
        chosen = min(candidates, key=lambda r: (
            abs(len(features[r['id']]['codes']) - frames), fingerprint([seed, r['id']])))
        probe.append(dict(chosen, validation_group='train_matched'))
    if not probe or len({r['id'] for r in probe}) != len(probe):
        raise ValueError('Empty or duplicate training probe')
    return probe


def summarize_losses(records):
    groups = defaultdict(list)
    for row in records:
        groups['all'].append(row)
        groups[row['group']].append(row)
    result = {}
    for group, items in groups.items():
        first = sum(r['first_sum'] for r in items) / sum(r['first_tokens'] for r in items)
        residual = sum(r['residual_sum'] for r in items) / sum(r['residual_tokens'] for r in items)
        speakers = defaultdict(list)
        for row in items:
            speakers[row['speaker_id']].append(row)
        per_speaker = {}
        for speaker, subset in speakers.items():
            a = sum(r['first_sum'] for r in subset) / sum(r['first_tokens'] for r in subset)
            b = sum(r['residual_sum'] for r in subset) / sum(r['residual_tokens'] for r in subset)
            per_speaker[speaker] = a + .3*b
        result[group] = dict(first=first, residual=residual, combined=first+.3*residual,
                             clips=len(items), speakers=len(speakers),
                             speaker_macro_loss=sum(per_speaker.values())/len(per_speaker),
                             per_speaker=per_speaker)
        if not all(math.isfinite(v) for v in (first, residual, *per_speaker.values())):
            raise ValueError('Nonfinite evaluation loss')
    # Explicit cohort names prevent train probes/metadata entering selection.
    cohorts = [result[k]['combined'] for k in ('familiar', 'unseen') if k in result]
    result['selection_loss'] = sum(cohorts)/len(cohorts) if cohorts else result['all']['combined']
    return result


def component(name):
    if name.startswith('code_predictor.'):
        return 'residual_predictor'
    if name.startswith('text_projection.'):
        return 'text_projection'
    return 'talker_decoder'


def gradient_metrics(named_parameters):
    """Called before clipping, only at logged steps; no RNG or graph mutation."""
    import torch
    parts = defaultdict(list)
    for name, param in named_parameters:
        if param.requires_grad and param.grad is not None:
            parts[component(name)].append(param.grad.detach().float().square().sum())
    return {name: torch.stack(values).sum().sqrt().item() for name, values in parts.items()}


def adapter_movement(previous, current):
    """Actual checkpoint-to-checkpoint movement, not LR*gradient approximation."""
    import torch
    parts = defaultdict(lambda: [0.0, 0.0, 0, 0])
    for name, value in current.items():
        old = previous[name].float()
        value = value.float()
        part = parts[component(name)]
        part[0] += torch.sum((value-old).double().square()).item()
        part[1] += torch.sum(old.double().square()).item()
        part[2] += torch.count_nonzero(value == old).item()
        part[3] += value.numel()
    return {name: dict(delta_l2=math.sqrt(a),
                       relative_delta_l2=math.sqrt(a/b) if b else None,
                       unchanged_fraction=c/n) for name, (a, b, c, n) in parts.items()}


def window_summary(events):
    first = sum(e['first_sum'] for e in events)/sum(e['first_tokens'] for e in events)
    residual = sum(e['residual_sum'] for e in events)/sum(e['residual_tokens'] for e in events)
    seconds = sum(e['step_seconds'] for e in events)
    return dict(window_updates=len(events), first_loss=first, residual_loss=residual,
                window_loss=first+.3*residual,
                mean_gradient_norm=sum(e['gradient_norm'] for e in events)/len(events),
                max_gradient_norm=max(e['gradient_norm'] for e in events),
                clipping_fraction=sum(e['gradient_norm'] > 1 for e in events)/len(events),
                mean_step_seconds=seconds/len(events),
                clips_per_second=sum(e['clips'] for e in events)/seconds,
                codec_frames_per_second=sum(e['codec_frames'] for e in events)/seconds)


def fit_signals(probes, tolerance=.002):
    """Descriptive trends, never a definitive diagnosis or selection criterion."""
    if len(probes) < 3:
        return ['Need at least three comparable probes to assess a trend.']
    a, b = probes[-3], probes[-1]
    train_change = b['train'] - a['train']
    val_change = b['familiar'] - a['familiar']
    unseen_change = b['unseen'] - a['unseen']
    if train_change < -tolerance and val_change > tolerance:
        message = 'Possible overfitting: matched training loss falls while familiar validation loss rises.'
    elif train_change < -tolerance and val_change < -tolerance:
        message = 'Learning continues: matched training and familiar validation losses both fall.'
    elif abs(train_change) <= tolerance and abs(val_change) <= tolerance:
        message = 'Plateau: inspect learning rate, capacity and data; plateau alone does not prove underfitting.'
    else:
        message = 'Mixed trend: inspect component losses and generated speech before changing the recipe.'
    result = [message]
    if unseen_change > tolerance:
        result.append('Unseen-speaker loss is rising over the last three probes.')
    return result
