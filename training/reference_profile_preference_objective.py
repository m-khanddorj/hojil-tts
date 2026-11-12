"""Case-balanced DPO for a frozen model with one reference-specific profile.

Likelihoods are raw all-codebook sequence probabilities including EOS, not
temperature/top-k sampling probabilities. Unique trajectories are replayed once
per batch after accumulating their exact scalar loss derivatives.
"""
import math
from collections import Counter
import torch
from .preference_objective import dpo_loss
from .speaker_preference_adapter import log_probability as speaker_log_probability


def log_probability(model,profile,trajectory):
    if trajectory['reference_sha256']!=profile.reference_sha256:
        raise ValueError('Preference trajectory belongs to another reference')
    with profile.use_reference(trajectory['reference_sha256']):
        return speaker_log_probability(model,trajectory)


def objective(values,references,pairs,beta=.1):
    if not pairs:raise ValueError('Joint training preferences are required')
    keys=[(p['case'],p['chosen'],p['rejected']) for p in pairs]
    if len(set(keys))!=len(keys) or any(a==b for _,a,b in keys):raise ValueError('Duplicate or self preference')
    counts=Counter(p['case'] for p in pairs);owners={};coefficients={};losses=[];margins=[]
    for p in pairs:
        a,b=p['chosen'],p['rejected']
        for key in (a,b):
            if key not in values or key not in references:raise ValueError('Missing policy/reference likelihood')
            if key in owners and owners[key]!=p['case']:raise ValueError('Cross-text preference trajectory')
            owners[key]=p['case']
        weight=1/(len(counts)*counts[p['case']])
        loss,margin=dpo_loss(values[a],values[b],references[a],references[b],beta)
        if not torch.isfinite(loss) or not torch.isfinite(margin):raise ValueError('Nonfinite preference objective')
        derivative=-weight*beta*torch.sigmoid(-margin)
        coefficients[a]=coefficients.get(a,0)+derivative
        coefficients[b]=coefficients.get(b,0)-derivative
        losses.append(weight*loss);margins.append(dict(case=p['case'],margin=float(margin.detach()),weight=weight))
    return sum(losses),coefficients,margins


def backward_batch(model,profile,trajectories,references,pairs,beta=.1):
    involved=sorted({p[k] for p in pairs for k in ('chosen','rejected')})
    if not involved:raise ValueError('No preference trajectories')
    values={};tokens={}
    with torch.no_grad():
        for key in involved:
            values[key],tokens[key]=log_probability(model,profile,trajectories[key])
        loss,coefficients,margins=objective(values,references,pairs,beta)
    errors=[]
    for key in involved:
        value,count=log_probability(model,profile,trajectories[key])
        error=float((value.detach()-values[key]).abs());errors.append(error)
        if count!=tokens[key] or error>.002:raise ValueError('Native likelihood replay differs')
        (coefficients[key].detach()*value).backward()
    if profile.raw.grad is None or not torch.isfinite(profile.raw.grad).all():raise ValueError('Invalid profile preference gradient')
    counts=Counter(p['case'] for p in pairs);chosen_ratios=[];rejected_ratios=[];both_decrease=0.
    for pair in pairs:
        weight=1/(len(counts)*counts[pair['case']])
        chosen=float(values[pair['chosen']])-references[pair['chosen']]
        rejected=float(values[pair['rejected']])-references[pair['rejected']]
        chosen_ratios.append(weight*chosen);rejected_ratios.append(weight*rejected)
        both_decrease+=weight*(chosen<0 and rejected<0)
    return dict(loss=float(loss),margins=margins,maximum_likelihood_replay_error=max(errors),
        unique_trajectories=len(involved),pairs=len(pairs),cases=len({p['case'] for p in pairs}),
        weighted_preference_accuracy=sum(r['weight']*(r['margin']>0) for r in margins),
        weighted_chosen_logratio=sum(chosen_ratios),weighted_rejected_logratio=sum(rejected_ratios),
        weighted_both_likelihoods_decreased_fraction=both_decrease,
        log_probabilities={key:float(value) for key,value in values.items()},token_counts=tokens)


def project_input_radius(profile,maximum_fraction):
    """Project only the personal residual; the original speaker vector stays fixed."""
    if not math.isfinite(maximum_fraction) or not 0<maximum_fraction<profile.radius_fraction:
        raise ValueError('Input trust region must lie inside the smooth profile bound')
    limit=maximum_fraction/math.sqrt(profile.radius_fraction**2-maximum_fraction**2)
    with torch.no_grad():
        norm=float(profile.raw.norm())
        if not torch.isfinite(profile.raw).all() or not math.isfinite(norm):
            raise ValueError('Nonfinite personal profile cannot be projected')
        if norm>limit:profile.raw.mul_(limit/norm)
        actual=float(profile.radius_fraction*profile.raw.norm()/torch.sqrt(1+profile.raw.square().sum()))
    return dict(projected=norm>limit,actual_input_fraction=actual,maximum_input_fraction=maximum_fraction)
