"""Prospective absolute-quality filtering and CE-anchored codec DPO.

Not used by registered historical runs. Selection must follow a pool provenance
audit. The optional anchor is mean chosen-token NLL over all codebooks and EOS;
the DPO term retains raw sequence log probabilities. No evaluation audio fits.
"""
import math


def quality_pairs(records, pairs, max_cer=0.03, min_cosine=0.984):
    if not math.isfinite(max_cer) or max_cer <= 0:
        raise ValueError('Invalid CER bound')
    if not math.isfinite(min_cosine) or not -1 <= min_cosine <= 1:
        raise ValueError('Invalid cosine bound')
    by_id = {r['sample_id']: r for r in records}
    if len(by_id) != len(records):
        raise ValueError('Duplicate sample IDs')
    kept = []
    for pair in pairs:
        if pair['preference_split'] not in ('train', 'validation'):
            raise ValueError('Unknown preference split')
        a, b = (by_id[pair[k]] for k in ('chosen', 'rejected'))
        for r in (a, b):
            if r['partition'] != 'train' or r['origin'] != 'model_generated' or not r['terminated']:
                raise ValueError('Only terminated generated training trajectories are eligible')
            if (r['case'] != pair['case'] or r['speaker_id'] != pair['speaker_id']
                    or r['preference_split'] != pair['preference_split']):
                raise ValueError('Pair provenance differs')
            s = r['score']
            if (s['characters'] <= 0 or s['character_errors'] < 0
                    or not math.isclose(s['cer'], s['character_errors']/s['characters'], abs_tol=1e-12)
                    or not math.isfinite(r['cosine']) or not -1 <= r['cosine'] <= 1):
                raise ValueError('Invalid quality metrics')
        if any(a[k] != b[k] for k in ('text', 'reference_sha256')):
            raise ValueError('Pair conditions differ')
        if a['score']['cer'] > b['score']['cer']:
            raise ValueError('Invalid joint preference')
        if a['cosine'] < b['cosine']:
            raise ValueError('Chosen voice is worse')
        if a['score']['cer'] < max_cer and a['cosine'] > min_cosine:
            kept.append(dict(pair))
    return kept


def backward_pair(model, chosen, rejected, references, beta=0.1, weight=1.0, sft_weight=0.1):
    """Exact gradient of weight * (DPO + sft_weight * chosen mean NLL)."""
    import torch
    from .speaker_preference_adapter import log_probability
    from .preference_objective import dpo_loss
    if not math.isfinite(weight) or weight <= 0:
        raise ValueError('Invalid pair weight')
    if not math.isfinite(sft_weight) or sft_weight < 0:
        raise ValueError('Invalid SFT weight')
    with torch.no_grad():
        a, na = log_probability(model, chosen)
        b, nb = log_probability(model, rejected)
        if na <= 0 or nb <= 0:
            raise ValueError('Empty likelihood target')
        loss, margin = dpo_loss(a, b, *references, beta)
        ce = -a / na
        total = loss + sft_weight * ce
        if not torch.isfinite(total) or not torch.isfinite(margin):
            raise ValueError('Nonfinite preference objective')
        dpo_coefficient = -beta * torch.sigmoid(-margin)
        coefficients = ((dpo_coefficient - sft_weight/na)*weight, -dpo_coefficient*weight)
    for sample, expected, count, coefficient in zip((chosen, rejected), (a, b), (na, nb), coefficients):
        value, tokens = log_probability(model, sample)
        if tokens != count or not torch.allclose(value.detach(), expected, atol=0.002, rtol=0):
            raise ValueError('Likelihood replay differs')
        (value * coefficient).backward()
    return dict(loss=float(loss), total_loss=float(total), chosen_mean_nll=float(ce),
                sft_weight=sft_weight, margin=float(margin), chosen_logp=float(a), rejected_logp=float(b),
                chosen_logratio=float(a)-references[0], rejected_logratio=float(b)-references[1],
                chosen_tokens=na, rejected_tokens=nb)
