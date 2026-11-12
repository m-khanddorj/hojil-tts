"""Frozen-codec checkpointing and a pronunciation-anchor descent constraint."""
import torch
from torch.utils.checkpoint import checkpoint
from .differentiable_codec_probe import expected_quantized


def checkpointed_waveform(decoder,codes,probabilities,chunk_size=300,left_context=25):
    """Native chunk boundaries; checkpoint activations, never shorten the audio."""
    if (codes.dtype!=torch.long or tuple(codes.shape)!=tuple(probabilities.shape[:3])
            or codes.min()<0 or codes.max()>=decoder.config.codebook_size
            or chunk_size<1 or left_context<0):
        raise ValueError('Invalid codec coverage or chunk policy')
    waves=[]
    for start in range(0,codes.shape[-1],chunk_size):
        context=min(start,left_context);end=min(start+chunk_size,codes.shape[-1])
        part=codes[...,start-context:end]
        soft=expected_quantized(decoder,probabilities[:,:,start-context:end])
        with torch.no_grad():hard=decoder.quantizer.decode(part)
        latent=hard+(soft-soft.detach())
        hidden=checkpoint(decoder.pre_conv,latent,use_reentrant=False).transpose(1,2)
        hidden=checkpoint(lambda x:decoder.pre_transformer(inputs_embeds=x).last_hidden_state,
                          hidden,use_reentrant=False).permute(0,2,1)
        for blocks in decoder.upsample:
            for block in blocks:hidden=checkpoint(block,hidden,use_reentrant=False)
        for block in decoder.decoder:hidden=checkpoint(block,hidden,use_reentrant=False)
        waves.append(hidden.clamp(-1,1)[...,context*decoder.total_upsample:])
    return torch.cat(waves,dim=-1)


def constrained_direction(proposal,voice_gradient,anchor_gradient):
    """Project an update onto non-increasing anchor loss; retain voice descent.

All arguments are flat parameter-space vectors. This is a local first-order
condition, followed by actual anchor-loss backtracking in the caller. It is
not a guarantee of lower ASR CER or higher generated-voice similarity.
"""
    if (proposal.ndim!=1 or proposal.shape!=voice_gradient.shape or proposal.shape!=anchor_gradient.shape
            or not all(torch.isfinite(x).all() for x in (proposal,voice_gradient,anchor_gradient))):
        raise ValueError('Invalid gradient coverage')
    dtype=proposal.dtype
    d,v,a=(x.double() for x in (proposal,voice_gradient,anchor_gradient))
    norm=d.norm();aa=a.dot(a)
    if norm==0 or v.norm()==0:raise ValueError('No update or voice gradient')
    def project(x):
        return x-a*(x.dot(a).clamp(min=0)/aa) if aa>0 else x
    d=project(d);fallback=bool(d.dot(v)>=0)
    if fallback:
        d=project(-v)
        if d.norm()>0:d=d*min(1.,float(norm/d.norm()))
    if d.dot(v)>=0 or d.dot(a)>1e-12*max(1.,float(d.norm()*a.norm())):
        raise ValueError('No common descent direction')
    return d.to(dtype),dict(fallback_to_projected_voice_gradient=fallback,
        predicted_voice_change=float(d.dot(v)),predicted_anchor_change=float(d.dot(a)),
        original_update_norm=float(norm),constrained_update_norm=float(d.norm()))
