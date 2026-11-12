"""DPO on sampled codec trajectories, with a one-graph-at-a-time backward pass.

The policy is the raw model distribution, not its temperature/top-k sampler.
Sequence likelihood includes all 16 codebooks and the sampled terminal EOS.
Reference log probabilities are fixed at initialization.
"""
import math


def log_probability(model, trajectory):
    from .sequence import build, loss_sums
    if not trajectory['terminated']:
        raise ValueError('Truncated generations cannot receive a fabricated EOS label')
    sequence = build(model, trajectory['input_ids'], trajectory['codes'],
        trajectory['ref_xvec'], trajectory['ref_ids'], trajectory['ref_codes'], True)
    first, residual, n, m = loss_sums(model, [sequence])
    # No historical 0.3 residual weighting or token-count normalization here.
    return -(first.double() + residual.double()), n + m


def dpo_loss(chosen, rejected, reference_chosen, reference_rejected, beta=0.1):
    import torch.nn.functional as F
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError('DPO beta must be positive and finite')
    margin = beta*((chosen-reference_chosen)-(rejected-reference_rejected))
    return -F.logsigmoid(margin), margin


def deterministic_training(model):
    """Keep gradient checkpointing active while removing stochastic dropout."""
    import torch
    model.talker.train()
    for module in model.talker.modules():
        if isinstance(module, torch.nn.Dropout):
            module.eval()
        if getattr(module, 'attention_dropout', 0) != 0:
            raise ValueError('Attention dropout must be zero for replayed likelihoods')


def backward_pair(model, chosen, rejected, references, beta=0.1, weight=1.0):
    """Exact DPO gradient, without retaining chosen and rejected graphs together.

First compute the scalar derivative, then recompute each deterministic trajectory
and apply its coefficient. Replay equality is checked before every backward.
"""
    import torch
    if not math.isfinite(weight) or weight <= 0:
        raise ValueError('Pair weight must be positive and finite')
    with torch.no_grad():
        chosen_value, chosen_tokens = log_probability(model, chosen)
        rejected_value, rejected_tokens = log_probability(model, rejected)
        loss, margin = dpo_loss(chosen_value, rejected_value, *references, beta)
        if not torch.isfinite(loss) or not torch.isfinite(margin):
            raise ValueError('Nonfinite preference objective')
        coefficient = -beta*torch.sigmoid(-margin)*weight
    for sample, expected, coefficient_value in ((chosen,chosen_value,coefficient),
                                                (rejected,rejected_value,-coefficient)):
        value, _ = log_probability(model, sample)
        # Deterministic evaluation with and without autograd can choose different
        # kernels; admit only a small absolute FP32 reduction discrepancy.
        if not torch.allclose(value.detach(), expected, atol=0.002, rtol=0):
            raise ValueError(f'Likelihood replay differs: {value.item()} versus {expected.item()}')
        (coefficient_value*value).backward()
    return dict(loss=loss.item(), margin=margin.item(),
        chosen_logp=chosen_value.item(), rejected_logp=rejected_value.item(),
        chosen_logratio=chosen_value.item()-references[0],
        rejected_logratio=rejected_value.item()-references[1],
        chosen_tokens=chosen_tokens, rejected_tokens=rejected_tokens,
        chosen_frames=len(chosen['codes']), rejected_frames=len(rejected['codes']))
