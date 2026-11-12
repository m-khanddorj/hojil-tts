"""Teacher forcing matching Qwen Base inference, with explicit loss alignment.

This module deliberately bypasses upstream implicit causal-loss shifting. Each
supervised position predicts the NEXT target frame; residual predictions use
that same pre-frame hidden state, never the state after consuming the frame.
"""
from dataclasses import dataclass
import torch
import torch.nn.functional as F


@dataclass
class Sequence:
    embeddings: torch.Tensor
    labels: torch.Tensor
    frame_positions: torch.Tensor
    codes: torch.Tensor
    prefix_length: int


def build(model, input_ids, target_codes, speaker_embedding, ref_ids=None,
          ref_codes=None, non_streaming=False):
    talker = model.talker
    cfg = model.config
    tc = cfg.talker_config
    device = next(talker.parameters()).device
    input_ids = input_ids.to(device)
    target_codes = target_codes.to(device=device, dtype=torch.long)
    if target_codes.ndim != 2 or target_codes.shape[1] != tc.num_code_groups or not len(target_codes):
        raise ValueError('Expected nonempty target codes T x num_code_groups')
    if input_ids.shape[1] <= 8:
        raise ValueError('Empty or malformed assistant transcript')
    def ids(values):
        return torch.tensor([values], device=device, dtype=torch.long)
    def text(values):
        return talker.text_projection(talker.get_text_embeddings()(values))
    codec = talker.get_input_embeddings()
    bos, eos, pad = text(ids([cfg.tts_bos_token_id, cfg.tts_eos_token_id, cfg.tts_pad_token_id])).chunk(3, dim=1)
    tags = codec(ids([tc.codec_nothink_id, tc.codec_think_bos_id, tc.codec_think_eos_id]))
    speaker = speaker_embedding.detach().to(device=device, dtype=tags.dtype).reshape(1, 1, -1)
    codec_prefix = torch.cat([tags, speaker, codec(ids([tc.codec_pad_id, tc.codec_bos_id]))], dim=1)
    prefix = torch.cat([text(input_ids[:, :3]),
                        torch.cat([pad.expand(-1, codec_prefix.shape[1]-2, -1), bos], dim=1)
                        + codec_prefix[:, :-1]], dim=1)
    if ref_codes is not None:
        if ref_ids is None:
            raise ValueError('ICL requires reference transcript')
        context, trailing = model.generate_icl_prompt(
            text_id=input_ids[:, 3:-5], ref_id=ref_ids.to(device)[:, 3:-2],
            ref_code=ref_codes.to(device=device, dtype=torch.long),
            tts_pad_embed=pad, tts_eos_embed=eos, non_streaming_mode=non_streaming)
    elif non_streaming:
        txt = torch.cat([text(input_ids[:, 3:-5]), eos], dim=1)
        context = torch.cat([txt + codec(ids([tc.codec_pad_id] * txt.shape[1])),
                             pad + codec(ids([tc.codec_bos_id]))], dim=1)
        trailing = pad
    else:
        context = text(input_ids[:, 3:4]) + codec(ids([tc.codec_bos_id]))
        trailing = torch.cat([text(input_ids[:, 4:-5]), eos], dim=1)
    prefix = torch.cat([prefix, context], dim=1)
    frame_embed = codec(target_codes[:, 0])
    for group in range(1, tc.num_code_groups):
        frame_embed = frame_embed + talker.code_predictor.get_input_embeddings()[group-1](target_codes[:, group])
    # Generation consumes frame zero with trailing_text_hidden[0], and so on.
    trailing_frames = torch.cat([trailing[:, :len(target_codes)],
                                 pad.expand(-1, max(0, len(target_codes)-trailing.shape[1]), -1)], dim=1)
    embeddings = torch.cat([prefix, frame_embed.unsqueeze(0) + trailing_frames], dim=1).squeeze(0)
    labels = torch.full((len(embeddings),), -100, device=device, dtype=torch.long)
    start = prefix.shape[1]-1
    labels[start:start+len(target_codes)] = target_codes[:, 0]
    labels[start+len(target_codes)] = tc.codec_eos_token_id
    positions = torch.arange(start, start+len(target_codes), device=device)
    return Sequence(embeddings, labels, positions, target_codes, prefix.shape[1])


def collate(sequences):
    embeddings = torch.nn.utils.rnn.pad_sequence([s.embeddings for s in sequences], batch_first=True)
    labels = torch.nn.utils.rnn.pad_sequence([s.labels for s in sequences], batch_first=True, padding_value=-100)
    mask = torch.arange(embeddings.shape[1], device=embeddings.device)[None, :] < torch.tensor(
        [len(s.labels) for s in sequences], device=embeddings.device)[:, None]
    return embeddings, labels, mask.long()


def loss_sums(model, sequences):
    embeddings, labels, mask = collate(sequences)
    talker = model.talker
    # Decoder directly: no generate state or implicit causal-label shift.
    positions = (mask.cumsum(-1)-1).masked_fill(mask == 0, 1).unsqueeze(0).expand(3, -1, -1)
    out = talker.model(inputs_embeds=embeddings, attention_mask=mask,
                       position_ids=positions, use_cache=False, return_dict=True)
    hidden = out.last_hidden_state
    # Context and padding have no labels. Do not materialize vocabulary logits
    # for those positions; keep exactly the same summed next-frame objective.
    supervised = labels != -100
    logits = talker.codec_head(hidden[supervised]).float()
    first = F.cross_entropy(logits, labels[supervised], reduction='sum')
    h = torch.cat([hidden[i, s.frame_positions] for i, s in enumerate(sequences)])
    codes = torch.cat([s.codes for s in sequences])
    predictor = talker.code_predictor
    pieces = [h[:, None], talker.get_input_embeddings()(codes[:, :1])]
    pieces += [predictor.get_input_embeddings()[i-1](codes[:, i:i+1]) for i in range(1, codes.shape[1]-1)]
    sub = predictor.forward_finetune(inputs_embeds=torch.cat(pieces, dim=1), labels=None,
                                     use_cache=False, return_dict=True)
    residual = F.cross_entropy(sub.logits.float().reshape(-1, sub.logits.shape[-1]),
                               codes[:, 1:].reshape(-1), reduction='sum')
    return first, residual, int((labels != -100).sum()), codes.shape[0] * (codes.shape[1]-1)
