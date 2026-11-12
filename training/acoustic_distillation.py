"""Keep halfway-model acoustic predictions while learning target pronunciation.

This is a teacher-forced residual-codebook KL surrogate, not a speaker loss.
Actual generated-speech CER and speaker cosine remain the selection criteria.
"""
from contextlib import contextmanager


@contextmanager
def halfway_teacher(model):
    """Use the same frozen backbone with the new target-text LoRA disabled."""
    talker = model.talker
    modules = [talker.text_projection.linear_fc1, talker.text_projection.linear_fc2]
    if any(set(layer.scaling) != {'default'} or layer.merged for layer in modules):
        raise ValueError('Expected two unmerged default target-text adapters')
    scales = [layer.scaling['default'] for layer in modules]
    modes = [(module, module.training) for module in talker.modules()]
    try:
        talker.eval()
        for layer in modules:
            layer.scaling['default'] = 0.0
        import torch
        with torch.no_grad():
            yield
    finally:
        for layer, scale in zip(modules, scales):
            layer.scaling['default'] = scale
        for module, training in modes:
            module.training = training


def forward_terms(model, sequences):
    """Same next-frame alignment and CE as sequence.loss_sums; expose RVQ logits."""
    import torch
    import torch.nn.functional as F
    from .sequence import collate
    embeddings, labels, mask = collate(sequences)
    talker = model.talker
    positions = (mask.cumsum(-1)-1).masked_fill(mask == 0, 1).unsqueeze(0).expand(3, -1, -1)
    hidden = talker.model(inputs_embeds=embeddings, attention_mask=mask,
        position_ids=positions, use_cache=False, return_dict=True).last_hidden_state
    supervised = labels != -100
    first = F.cross_entropy(talker.codec_head(hidden[supervised]).float(),
        labels[supervised], reduction='sum')
    h = torch.cat([hidden[i, seq.frame_positions] for i, seq in enumerate(sequences)])
    codes = torch.cat([seq.codes for seq in sequences])
    predictor = talker.code_predictor
    pieces = [h[:, None], talker.get_input_embeddings()(codes[:, :1])]
    pieces += [predictor.get_input_embeddings()[i-1](codes[:, i:i+1])
               for i in range(1, codes.shape[1]-1)]
    logits = predictor.forward_finetune(inputs_embeds=torch.cat(pieces, dim=1),
        labels=None, use_cache=False, return_dict=True).logits.float()
    residual = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
        codes[:, 1:].reshape(-1), reduction='sum')
    return first, residual, logits.log_softmax(-1), int(supervised.sum()), codes.shape[0]*(codes.shape[1]-1)


def distillation_terms(model, make_sequences):
    """Teacher and student receive identical real codec history and text IDs."""
    import torch.nn.functional as F
    with halfway_teacher(model):
        _, _, teacher_logp, _, _ = forward_terms(model, make_sequences())
    first, residual, student_logp, n, m = forward_terms(model, make_sequences())
    if teacher_logp.requires_grad or teacher_logp.shape != student_logp.shape:
        raise ValueError('Invalid teacher supervision')
    kl = F.kl_div(student_logp, teacher_logp, reduction='sum', log_target=True)
    return first, residual, kl, n, m


def backward(model, assigned, frame_limit, token_limit, kl_weight=1.0):
    import torch
    from .engine import sequences
    from .policy import microbatches
    frames = sum(x['frames'] for x in assigned)
    first_count, residual_count = frames + len(assigned), frames * 15
    total = torch.zeros((), device='cuda', dtype=torch.float32)
    first_sum, residual_sum, kl_sum = total.clone(), total.clone(), total.clone()
    for batch in microbatches(assigned, frame_limit, token_limit):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            first, residual, kl, _, _ = distillation_terms(model, lambda: sequences(model, batch))
            loss = first/first_count + 0.3*residual/residual_count + kl_weight*kl/residual_count
        loss.backward()
        total += loss.detach()
        first_sum += first.detach()
        residual_sum += residual.detach()
        kl_sum += kl.detach()
    if not torch.isfinite(total):
        raise ValueError('Nonfinite distillation objective')
    return dict(loss=total.item(), task_loss=(first_sum/first_count+0.3*residual_sum/residual_count).item(),
        residual_kl=(kl_sum/residual_count).item(), kl_weight=kl_weight,
        first_sum=first_sum.item(), residual_sum=residual_sum.item(),
        first_tokens=first_count, residual_tokens=residual_count)
