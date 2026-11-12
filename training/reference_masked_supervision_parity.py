"""Fixed numerical gates for independent native-cached/teacher FP32 replay.

These gates check implementation parity, never generated speech quality. They
are registered before v2 probes run, after the preserved v1 failure diagnosis.
Comparisons use CPU float64 probabilities to avoid measuring the comparison's
own FP32 softmax error. Exact conditioning and frame/EOS coverage are checked
separately by the caller.
"""
import math

import torch


VERSION = 'native-teacher-fp32-probability-parity-v1'
LIMITS = dict(maximum_absolute_logit_error=5e-4,
              mean_absolute_logit_error=5e-5,
              maximum_absolute_log_probability_error=5e-4,
              maximum_chosen_log_probability_error=2e-4,
              maximum_forward_kl=1e-8,
              maximum_reverse_kl=1e-8,
              maximum_forward_argmax_regret=2e-4,
              maximum_reverse_argmax_regret=2e-4)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_metrics(metrics):
    """Reject incomplete/nonfinite/out-of-bound serialized parity evidence."""
    require(metrics.get('version') == VERSION and metrics.get('limits') == LIMITS,
            'Numerical parity registration differs')
    shape = metrics.get('shape')
    require(isinstance(shape, list) and len(shape) in (2, 3)
            and all(type(n) is int and n > 0 for n in shape) and shape[-1] >= 2,
            'Invalid parity logit shape')
    count = math.prod(shape[:-1])
    require(type(metrics.get('distributions')) is int and metrics['distributions'] == count
            and type(metrics.get('labels')) is int and metrics['labels'] == count
            and metrics.get('finite') is True, 'Incomplete or nonfinite parity coverage')
    for key, maximum in LIMITS.items():
        value = metrics.get(key)
        require(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= maximum,
                f'Numerical parity limit exceeded: {key}; observed={value!r}; maximum={maximum!r}')
    for key in ('argmax_disagreements', 'stable_argmax_disagreements'):
        require(type(metrics.get(key)) is int and 0 <= metrics[key] <= count,
                'Invalid argmax disagreement count')
    require(metrics['stable_argmax_disagreements'] == 0,
            'Argmax differs outside the observed numerical tie margin')
    require(metrics['stable_argmax_disagreements'] <= metrics['argmax_disagreements'],
            'Inconsistent argmax disagreement counts')
    for key in ('minimum_forward_kl', 'minimum_reverse_kl'):
        value = metrics.get(key)
        require(type(value) in (int, float) and math.isfinite(value) and -1e-12 <= value <= 1e-8,
                'Invalid minimum KL metric')
        require(value <= metrics[key.replace('minimum_', 'maximum_')] + 1e-12,
                'Inconsistent KL extrema')
    require(metrics['mean_absolute_logit_error'] <= metrics['maximum_absolute_logit_error'],
            'Inconsistent logit error metrics')
    require(metrics['maximum_chosen_log_probability_error'] <= metrics['maximum_absolute_log_probability_error'],
            'Inconsistent chosen-token probability metrics')


def compare_logits(native, teacher, labels):
    """Return validated JSON metrics; the registered thresholds are not tunable."""
    require(isinstance(native, torch.Tensor) and isinstance(teacher, torch.Tensor)
            and isinstance(labels, torch.Tensor), 'Parity requires tensors')
    require(native.dtype == teacher.dtype == torch.float32 and labels.dtype == torch.long,
            'Parity requires FP32 logits and integer labels')
    require(native.shape == teacher.shape and native.ndim in (2, 3)
            and native.numel() > 0 and native.shape[-1] >= 2
            and tuple(labels.shape) == tuple(native.shape[:-1]), 'Parity tensor shape mismatch')
    require(bool(torch.isfinite(native).all() and torch.isfinite(teacher).all()),
            'Nonfinite parity logits')
    require(bool(((labels >= 0) & (labels < native.shape[-1])).all()), 'Parity label out of range')
    with torch.no_grad():
        a = native.detach().to(device='cpu', dtype=torch.float64)
        b = teacher.detach().to(device='cpu', dtype=torch.float64)
        label = labels.detach().cpu()
        delta = (a - b).abs()
        la, lb = a.log_softmax(-1), b.log_softmax(-1)
        log_delta = (la - lb).abs()
        chosen = log_delta.gather(-1, label.unsqueeze(-1))
        # KL can be slightly negative from summation roundoff in float64.
        # Reject a material negative before clamping roundoff to zero.
        forward = (la.exp() * (la - lb)).sum(-1)
        reverse = (lb.exp() * (lb - la)).sum(-1)
        require(float(forward.min()) >= -1e-12 and float(reverse.min()) >= -1e-12,
                'Invalid negative KL divergence')
        winner_a, winner_b = a.argmax(-1), b.argmax(-1)
        disagreement = winner_a != winner_b
        regret_a = la.gather(-1, winner_a.unsqueeze(-1)) - la.gather(-1, winner_b.unsqueeze(-1))
        regret_b = lb.gather(-1, winner_b.unsqueeze(-1)) - lb.gather(-1, winner_a.unsqueeze(-1))
        top = a.topk(2, dim=-1).values
        # A class ordering can legitimately flip if the native top-two margin
        # is at most twice that distribution's observed absolute logit error.
        stable = (top[..., 0] - top[..., 1]) > 2 * delta.amax(-1)
        metrics = dict(version=VERSION, limits=dict(LIMITS), shape=list(a.shape),
                       distributions=label.numel(), labels=label.numel(), finite=True,
                       maximum_absolute_logit_error=float(delta.max()),
                       mean_absolute_logit_error=float(delta.mean()),
                       maximum_absolute_log_probability_error=float(log_delta.max()),
                       maximum_chosen_log_probability_error=float(chosen.max()),
                       maximum_forward_kl=max(0., float(forward.max())),
                       maximum_reverse_kl=max(0., float(reverse.max())),
                       minimum_forward_kl=float(forward.min()),
                       minimum_reverse_kl=float(reverse.min()),
                       maximum_forward_argmax_regret=float(regret_a.max()),
                       maximum_reverse_argmax_regret=float(regret_b.max()),
                       argmax_disagreements=int(disagreement.sum()),
                       stable_argmax_disagreements=int((disagreement & stable).sum()))
    validate_metrics(metrics)
    return metrics
