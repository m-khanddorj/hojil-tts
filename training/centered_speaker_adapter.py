"""A bounded, speaker-dependent residual around an audited speaker adapter.

The PCA basis uses fitting references only, with equal speaker weights. This
is an experimental parameterization, not an identified speaker circuit or a
claim of better generated speech. Original acoustic and adapter weights freeze.
"""
import copy
import math
import torch
from torch import nn


def fit_basis(speaker_vectors, rank=16):
    """Fit centered, whitened coordinates from one mean unit vector per speaker."""
    if len(speaker_vectors) <= rank or rank < 1:
        raise ValueError('Need more fitting speakers than basis directions')
    means = []
    for speaker in sorted(speaker_vectors):
        x = torch.stack(speaker_vectors[speaker]).detach().cpu().double()
        if x.ndim != 2 or not torch.isfinite(x).all() or (x.norm(dim=1) <= 0).any():
            raise ValueError('Invalid fitting reference vectors')
        means.append(torch.nn.functional.normalize(x, dim=-1).mean(0))
    x = torch.stack(means)
    center = x.mean(0)
    _, singular, vh = torch.linalg.svd(x-center, full_matrices=False)
    if singular[rank-1] <= 1e-10:
        raise ValueError('Requested basis exceeds observed reference rank')
    basis = vh[:rank].clone()
    # Fix arbitrary SVD signs for stable artifacts across identical runs.
    for row in basis:
        if row[row.abs().argmax()] < 0:
            row.neg_()
    scale = singular[:rank]/math.sqrt(len(x)-1)
    return dict(center=center.float(), basis=basis.float(), scale=scale.float(),
                explained_variance=float(singular[:rank].square().sum()/singular.square().sum()),
                speakers=len(x), rank=rank)


class CenteredSpeakerAdapter(nn.Module):
    def __init__(self, initial, coordinates):
        super().__init__()
        self.initial = copy.deepcopy(initial).requires_grad_(False)
        self.dimension = initial.dimension
        self.rank = coordinates['rank']
        self.radius_fraction = initial.radius_fraction
        center, basis, scale = (coordinates[k] for k in ('center','basis','scale'))
        if (center.shape != (self.dimension,) or basis.shape != (self.rank,self.dimension)
                or scale.shape != (self.rank,) or (scale <= 0).any()
                or any(v.dtype != torch.float32 or not torch.isfinite(v).all()
                       for v in (center,basis,scale))):
            raise ValueError('Invalid centered speaker coordinates')
        self.register_buffer('center', center.detach().clone())
        self.register_buffer('basis', basis.detach().clone())
        self.register_buffer('scale', scale.detach().clone())
        self.residual = nn.Parameter(torch.zeros(self.dimension,self.rank,dtype=torch.float32))

    def coordinates(self, speaker):
        norm = speaker.norm(dim=-1,keepdim=True).clamp_min(1e-12)
        return ((speaker/norm-self.center)@self.basis.T)/self.scale/math.sqrt(self.rank)

    def forward(self, speaker):
        if speaker.shape[-1] != self.dimension or speaker.dtype != torch.float32:
            raise ValueError('Expected FP32 speaker vector')
        norm = speaker.norm(dim=-1,keepdim=True).clamp_min(1e-12)
        raw = (speaker/norm)@self.initial.down.T@self.initial.up.T
        raw = raw + self.coordinates(speaker)@self.residual.T
        delta = self.radius_fraction*norm*raw/torch.sqrt(1+raw.square().sum(-1,keepdim=True))
        return speaker+delta

    def frozen_state(self):
        return {k:v.detach().cpu().clone() for k,v in self.state_dict().items() if k != 'residual'}

    def verify_frozen(self, expected):
        actual = self.frozen_state()
        if set(actual) != set(expected) or any(not torch.equal(actual[k],expected[k]) for k in actual):
            raise ValueError('Original speaker adapter or fitted coordinates changed')
