"""Low-rank, bounded speaker conditioning for generated-speech preferences.

No reference reconstruction objective. The frozen full-LoRA policy supplies
the acoustic model; this module adapts only its speaker input. No trained
checkpoint or quality claim is implied by installing the zero-output adapter.
"""
from dataclasses import replace
from types import MethodType
import math

import torch
from torch import nn


class SpeakerAdapter(nn.Module):
    def __init__(self,dimension,rank=16,radius_fraction=0.05):
        super().__init__()
        if dimension<1 or not 1<=rank<=dimension or not 0<radius_fraction<1:
            raise ValueError('Invalid speaker adapter configuration')
        self.dimension=dimension;self.rank=rank;self.radius_fraction=radius_fraction
        self.down=nn.Parameter(torch.randn(rank,dimension,dtype=torch.float32)/math.sqrt(dimension))
        self.up=nn.Parameter(torch.zeros(dimension,rank,dtype=torch.float32))

    def forward(self,speaker):
        if speaker.shape[-1]!=self.dimension or speaker.dtype!=torch.float32:
            raise ValueError('Expected FP32 speaker vector')
        norm=speaker.norm(dim=-1,keepdim=True).clamp_min(1e-12)
        raw=(speaker/norm)@self.down.T@self.up.T
        # Smooth L2 trust region, including at zero initialization. Unlike hard
        # clipping this keeps a gradient before and after approaching the bound.
        delta=self.radius_fraction*norm*raw/torch.sqrt(1+raw.square().sum(-1,keepdim=True))
        return speaker+delta


def install(model,rank=16,radius_fraction=0.05):
    if hasattr(model,'speaker_preference_adapter'):raise ValueError('Already installed')
    if any(p.requires_grad for p in model.parameters()):raise ValueError('Acoustic model must be frozen')
    device=model.talker.get_input_embeddings().weight.device
    adapter=SpeakerAdapter(model.config.talker_config.hidden_size,rank,radius_fraction).to(device)
    model.add_module('speaker_preference_adapter',adapter)
    original=model.generate_speaker_prompt
    def generate_speaker_prompt(self,voice_clone_prompt):
        return [self.speaker_preference_adapter(vector) for vector in original(voice_clone_prompt)]
    model.generate_speaker_prompt=MethodType(generate_speaker_prompt,model)
    return adapter


def preference_sequence(model,trajectory):
    """Restore input gradients at the native speaker slot in full ICL layout."""
    from .sequence import build
    device=model.talker.get_input_embeddings().weight.device
    original=trajectory['ref_xvec'].to(device=device,dtype=torch.float32).flatten()
    adapted=model.speaker_preference_adapter(original)
    seq=build(model,trajectory['input_ids'],trajectory['codes'],adapted,
        trajectory['ref_ids'],trajectory['ref_codes'],True)
    # build detaches the vector; the numerical prefix is already correct.
    slot=seq.embeddings[6:7]+(adapted-adapted.detach()).reshape(1,-1)
    return replace(seq,embeddings=torch.cat([seq.embeddings[:6],slot,seq.embeddings[7:]]))


def log_probability(model,trajectory):
    from .sequence import loss_sums
    if not trajectory['terminated']:raise ValueError('No fabricated EOS on truncated speech')
    seq=preference_sequence(model,trajectory)
    first,residual,n,m=loss_sums(model,[seq])
    return -(first.double()+residual.double()),n+m


def backward_pair(model,chosen,rejected,references,beta=0.1,weight=1.0):
    """Exact DPO gradient with deterministic one-trajectory graph replay."""
    from .preference_objective import dpo_loss
    if not math.isfinite(weight) or weight<=0:raise ValueError('Invalid pair weight')
    with torch.no_grad():
        a,na=log_probability(model,chosen);b,nb=log_probability(model,rejected)
        loss,margin=dpo_loss(a,b,*references,beta)
        if not torch.isfinite(loss) or not torch.isfinite(margin):raise ValueError('Nonfinite DPO loss')
        coefficient=-beta*torch.sigmoid(-margin)*weight
    for sample,expected,multiplier in ((chosen,a,coefficient),(rejected,b,-coefficient)):
        value,_=log_probability(model,sample)
        if not torch.allclose(value.detach(),expected,atol=0.002,rtol=0):raise ValueError('Likelihood replay differs')
        (value*multiplier).backward()
    return dict(loss=float(loss),margin=float(margin),chosen_logp=float(a),rejected_logp=float(b),
        chosen_logratio=float(a)-references[0],rejected_logratio=float(b)-references[1],
        chosen_tokens=na,rejected_tokens=nb)
