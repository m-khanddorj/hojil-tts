"""Explicit per-reference conditioning, with an unchanged shared-model fallback."""
from contextlib import contextmanager
import hashlib
import re
import torch
from torch import nn


class ReferenceVoiceProfile(nn.Module):
    def __init__(self,base,reference_sha256,dimension,radius_fraction=.05):
        super().__init__()
        if not re.fullmatch('[0-9a-f]{64}',reference_sha256) or dimension<1 or not 0<radius_fraction<1:
            raise ValueError('Invalid reference profile configuration')
        if any(p.requires_grad for p in base.parameters()):raise ValueError('Shared speaker adapter must remain frozen')
        self.base=base;self.reference_sha256=reference_sha256;self.radius_fraction=radius_fraction
        parameter=next(base.parameters())
        self.raw=nn.Parameter(torch.zeros(dimension,dtype=torch.float32,device=parameter.device))
        self._active=False

    @contextmanager
    def use_reference(self,reference_sha256):
        if not re.fullmatch('[0-9a-f]{64}',reference_sha256):raise ValueError('Reference identity must be a SHA256')
        previous=self._active;self._active=reference_sha256==self.reference_sha256
        try:yield self
        finally:self._active=previous

    def forward(self,speaker):
        result=self.base(speaker)
        if not self._active:return result
        if speaker.dtype!=torch.float32 or speaker.shape[-1]!=self.raw.numel():
            raise ValueError('Expected original FP32 speaker input')
        residual=self.raw/torch.sqrt(1+self.raw.square().sum())
        return result+self.radius_fraction*speaker.norm(dim=-1,keepdim=True)*residual


def install(model,reference_sha256,radius_fraction=.05):
    if any(p.requires_grad for p in model.parameters()):raise ValueError('Freeze all existing weights before profile installation')
    if isinstance(model.speaker_preference_adapter,ReferenceVoiceProfile):raise ValueError('Profile already installed')
    profile=ReferenceVoiceProfile(model.speaker_preference_adapter,reference_sha256,
                                  model.config.talker_config.hidden_size,radius_fraction)
    model.speaker_preference_adapter=profile
    return profile


def shared_speaker_digest(profile):
    h=hashlib.sha256()
    for name,value in profile.base.state_dict().items():
        h.update(name.encode());h.update(str((tuple(value.shape),value.dtype)).encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def snapshot(profile):
    return dict(reference_sha256=profile.reference_sha256,radius_fraction=profile.radius_fraction,
                raw=profile.raw.detach().cpu().clone())


def restore(profile,state):
    value=state['raw']
    if (state['reference_sha256']!=profile.reference_sha256 or state['radius_fraction']!=profile.radius_fraction
            or value.dtype!=torch.float32 or value.shape!=profile.raw.shape or not torch.isfinite(value).all()):
        raise ValueError('Profile identity or parameter coverage differs')
    with torch.no_grad():profile.raw.copy_(value)
