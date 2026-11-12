"""Teacher-forced acoustic posteriors and a memory-separated gradient replay.

The first-codebook posterior is conditional on an acoustic (non-EOS) token.
Residual posteriors condition on the recorded earlier codebooks. These are
raw-model softmax probabilities, not temperature/top-k sampling probabilities.
The hard trajectory and reference codes stay fixed. This is a biased local
straight-through objective, not an unbiased discrete policy gradient.
"""
import torch
import torch.nn.functional as F
from .speaker_preference_adapter import preference_sequence


def policy_outputs(model, trajectory):
    if not trajectory['terminated']:
        raise ValueError('No fabricated EOS for truncated trajectory')
    seq = preference_sequence(model,trajectory)
    embeddings = seq.embeddings[None]
    mask = torch.ones(embeddings.shape[:2],device=embeddings.device,dtype=torch.long)
    positions = (mask.cumsum(-1)-1).unsqueeze(0).expand(3,-1,-1)
    talker = model.talker
    hidden = talker.model(inputs_embeds=embeddings,attention_mask=mask,
        position_ids=positions,use_cache=False,return_dict=True).last_hidden_state[0]
    supervised = seq.labels != -100
    main_logits = talker.codec_head(hidden[supervised]).float()
    codes = seq.codes
    predictor = talker.code_predictor
    pieces = [hidden[seq.frame_positions,None],talker.get_input_embeddings()(codes[:,:1])]
    pieces += [predictor.get_input_embeddings()[i-1](codes[:,i:i+1]) for i in range(1,codes.shape[1]-1)]
    residual_logits = predictor.forward_finetune(inputs_embeds=torch.cat(pieces,dim=1),
        labels=None,use_cache=False,return_dict=True).logits.float()
    vocabulary = residual_logits.shape[-1]
    if (main_logits.shape[0] != len(codes)+1 or residual_logits.shape[:2] != codes[:,1:].shape
            or main_logits.shape[-1] <= vocabulary or codes.min() < 0 or codes.max() >= vocabulary
            or seq.labels[supervised][-1] != model.config.talker_config.codec_eos_token_id):
        raise ValueError('Codec alignment, vocabulary or EOS differs')
    main_nll = F.cross_entropy(main_logits,seq.labels[supervised],reduction='sum')
    residual_nll = F.cross_entropy(residual_logits.reshape(-1,vocabulary),codes[:,1:].reshape(-1),reduction='sum')
    logits = torch.cat([main_logits[:-1,:vocabulary,None].transpose(1,2),residual_logits],dim=1)
    probabilities = logits.softmax(-1).permute(1,0,2)[None]
    mass = (main_logits[:-1,:vocabulary].logsumexp(-1)-main_logits[:-1].logsumexp(-1)).exp()
    return dict(probabilities=probabilities,
                log_probability=-(main_nll.double()+residual_nll.double()),
                token_count=codes.numel()+1,mean_nll=(main_nll+residual_nll)/(codes.numel()+1),
                main_acoustic_mass=mass)


def combine_reference_probabilities(reference_codes, generated_probabilities):
    """Reference code probabilities are fixed one-hot constants, never optimized."""
    if reference_codes.ndim != 2 or generated_probabilities.ndim != 4 or generated_probabilities.shape[0] != 1:
        raise ValueError('Expected one reference and one generated trajectory')
    reference_codes = reference_codes.to(device=generated_probabilities.device,dtype=torch.long)
    if (reference_codes.shape[1] != generated_probabilities.shape[1] or reference_codes.min() < 0
            or reference_codes.max() >= generated_probabilities.shape[-1]):
        raise ValueError('Reference codebook coverage differs')
    fixed = F.one_hot(reference_codes.T,generated_probabilities.shape[-1]).to(generated_probabilities.dtype)[None]
    if fixed.requires_grad: raise ValueError('Reference probabilities must stay fixed')
    return torch.cat([fixed,generated_probabilities],dim=2)


def backward_probability_gradient(model, trajectory, cached_probabilities, gradient):
    """Replay the exact local vector-Jacobian product; no optimizer step."""
    if (cached_probabilities.shape != gradient.shape or cached_probabilities.dtype != torch.float32
            or gradient.dtype != torch.float32 or not torch.isfinite(gradient).all()
            or not torch.isfinite(cached_probabilities).all()):
        raise ValueError('Invalid cached probability/gradient tensor')
    output = policy_outputs(model,trajectory)
    probabilities = output['probabilities']
    expected = cached_probabilities.to(probabilities.device)
    if probabilities.shape != expected.shape:
        raise ValueError('Replay coverage differs')
    maximum = float((probabilities.detach()-expected).abs().max())
    if not torch.allclose(probabilities.detach(),expected,atol=2e-6,rtol=2e-4):
        raise ValueError(f'Posterior replay differs; max absolute discrepancy {maximum}')
    torch.autograd.backward(probabilities,gradient.to(probabilities.device))
    return dict(maximum_probability_replay_error=maximum,
                raw_log_probability=float(output['log_probability'].detach()),
                main_acoustic_mass_min=float(output['main_acoustic_mass'].detach().min()),
                main_acoustic_mass_mean=float(output['main_acoustic_mass'].detach().mean()))
