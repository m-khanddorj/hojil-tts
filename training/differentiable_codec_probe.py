"""Experimental gradient bridge for Qwen's frozen 12-Hz codec.

No optimizer, training recipe, inference-policy change or reward-model update.
The straight-through gradient is a biased local surrogate at fixed hard codes;
it does not differentiate the actual discrete autoregressive sampling process.
"""
import torch


def expected_quantized(decoder, probabilities):
    """Map [batch, codebook, frame, vocabulary] probabilities to codec latents."""
    if probabilities.ndim != 4:
        raise ValueError('Expected batch/codebook/frame/vocabulary probabilities')
    config = decoder.config
    if (probabilities.shape[1] != config.num_quantizers
            or probabilities.shape[-1] != config.codebook_size
            or probabilities.shape[0] < 1 or probabilities.shape[2] < 1
            or not torch.isfinite(probabilities).all()
            or (probabilities.detach() < 0).any()
            or not torch.allclose(probabilities.detach().sum(-1),
                                  torch.ones_like(probabilities[...,0]),atol=1e-5,rtol=1e-5)):
        raise ValueError('Invalid probability simplex or codec coverage')
    if any(p.requires_grad for p in decoder.parameters()):
        raise ValueError('Codec must remain frozen')
    quantizer = decoder.quantizer
    groups = ((quantizer.rvq_first,0,quantizer.n_q_semantic),
              (quantizer.rvq_rest,quantizer.n_q_semantic,config.num_quantizers))
    outputs = []
    for rvq, start, stop in groups:
        total = None
        for local, index in enumerate(range(start,stop)):
            layer = rvq.vq.layers[local]
            book = layer._codebook
            embedding = book.embedding_sum/book.cluster_usage.clamp(min=book.epsilon)[:,None]
            if embedding.device != probabilities.device or embedding.dtype != probabilities.dtype:
                raise ValueError('Probability and codec device/dtype must match')
            value = layer.project_out(probabilities[:,index] @ embedding).transpose(1,2)
            total = value if total is None else total + value
        outputs.append(rvq.output_proj(total))
    return outputs[0] + outputs[1]


def waveform_from_quantized(decoder, quantized):
    """The native frozen decoder after its discrete codebook lookup."""
    if any(p.requires_grad for p in decoder.parameters()):
        raise ValueError('Codec must remain frozen')
    hidden = decoder.pre_conv(quantized).transpose(1,2)
    hidden = decoder.pre_transformer(inputs_embeds=hidden).last_hidden_state.permute(0,2,1)
    for blocks in decoder.upsample:
        for block in blocks:
            hidden = block(hidden)
    for block in decoder.decoder:
        hidden = block(hidden)
    return hidden.clamp(min=-1,max=1)


def straight_through_waveform(decoder, codes, probabilities):
    """Exactly native hard-code forward values; soft-code surrogate backward.

Codes have native [batch,codebook,frame] layout. EOS is not an acoustic token
and must be removed before this operation. No chunking or PCM quantization is
applied here; full runtime equivalence must be measured separately.
"""
    if (codes.dtype != torch.long or tuple(codes.shape) != tuple(probabilities.shape[:3])
            or codes.device != probabilities.device or codes.min() < 0
            or codes.max() >= decoder.config.codebook_size):
        raise ValueError('Invalid discrete acoustic codes')
    soft = expected_quantized(decoder,probabilities)
    with torch.no_grad(): hard = decoder.quantizer.decode(codes)
    return waveform_from_quantized(decoder,hard+(soft-soft.detach()))
