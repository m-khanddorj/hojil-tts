"""Read-only local gradient diagnostics for speaker and attention adapters.

These quantities depend on parameterization. They do not rank audible quality
or establish that a layer is responsible for speaker identity or pronunciation.
"""
import re
import torch
from . import middle_attention_adapter as attention


def enable_adapter_gradients(model):
    """Enable the existing speaker adapter alongside the existing attention LoRA."""
    attention.parameters(model)  # Require the original attention-only fitting state.
    attention.frozen_digest(model)
    model.speaker_preference_adapter.requires_grad_(True)
    expected={id(p) for m in attention.modules(model).values() for p in (m.lora_A,m.lora_B)}
    expected.update(id(p) for p in model.speaker_preference_adapter.parameters())
    actual={id(p) for p in model.parameters() if p.requires_grad}
    if actual!=expected:raise ValueError('Unexpected trainable coverage')


def capture_gradients(model):
    result={}
    for name,value in model.named_parameters():
        if not value.requires_grad:
            if value.grad is not None:raise ValueError('Frozen original parameter received a gradient')
            continue
        if value.grad is None or not torch.isfinite(value.grad).all():
            raise ValueError('Missing or invalid adapter gradient')
        result[name]=value.grad.detach().cpu().clone()
    if not result:raise ValueError('No adapter gradients')
    return result


def compare_gradients(voice,anchor):
    """Summarize common local descent geometry without choosing a learning rate."""
    if not voice or set(voice)!=set(anchor):raise ValueError('Gradient coverage differs')
    groups={'all_adapters':[], 'speaker_adapter':[], 'attention_adapter':[]}
    for name in sorted(voice):
        v,a=voice[name],anchor[name]
        if v.shape!=a.shape or not torch.isfinite(v).all() or not torch.isfinite(a).all():
            raise ValueError('Invalid gradient shape/value')
        groups['all_adapters'].append(name)
        if name.startswith('speaker_preference_adapter.'):
            groups['speaker_adapter'].append(name)
        else:
            match=re.fullmatch(r'talker\.model\.layers\.(\d+)\.self_attn\.[qv]_proj\.lora_[AB]',name)
            if match is None:raise ValueError('Unexpected adapter parameter')
            groups['attention_adapter'].append(name)
            groups.setdefault('attention_layer_'+match[1],[]).append(name)
    results={}
    for group,names in groups.items():
        if not names:raise ValueError('Missing adapter group')
        v=torch.cat([voice[n].flatten().double() for n in names])
        a=torch.cat([anchor[n].flatten().double() for n in names])
        vn,an=v.norm(),a.norm();va=v.dot(a)
        direction=-v
        if an>0:direction=direction-a*(direction.dot(a).clamp(min=0)/a.dot(a))
        results[group]=dict(parameters=v.numel(),tensors=len(names),voice_norm=float(vn),anchor_norm=float(an),
            cosine=float(va/(vn*an)) if vn>0 and an>0 else None,
            projected_voice_gradient_fraction=float(direction.norm()/vn) if vn>0 else None,
            projected_voice_change=float(direction.dot(v)),projected_anchor_change=float(direction.dot(a)))
    return results
