"""Small FP32 LoRA on selected main-talker Q/V projections.

The previously fitted speaker adapter and every original acoustic parameter
stay frozen. Generated-trajectory likelihoods use the native speaker input.
"""
import hashlib
import math
import torch
from torch import nn
from .speaker_preference_adapter import log_probability,backward_pair


class LowRankLinear(nn.Module):
    def __init__(self,base,rank):
        super().__init__()
        if not isinstance(base,nn.Linear) or base.weight.dtype!=torch.float32 or not 1<=rank<=min(base.weight.shape):
            raise ValueError('Expected FP32 linear and valid rank')
        self.base_layer=base.requires_grad_(False)
        self.lora_A=nn.Parameter(torch.randn(rank,base.in_features,device=base.weight.device,dtype=torch.float32)/math.sqrt(base.in_features))
        self.lora_B=nn.Parameter(torch.zeros(base.out_features,rank,device=base.weight.device,dtype=torch.float32))

    def forward(self,x):
        return self.base_layer(x)+torch.nn.functional.linear(torch.nn.functional.linear(x,self.lora_A),self.lora_B)


def modules(model):
    return {name:module for name,module in model.named_modules() if isinstance(module,LowRankLinear)}


def install(model,layers,rank=4):
    if modules(model) or not hasattr(model,'speaker_preference_adapter'):
        raise ValueError('Requires unwrapped model with original speaker adapter')
    if not layers or len(layers)!=len(set(layers)):
        raise ValueError('Expected unique nonempty layer indices')
    decoder=model.talker.model.layers
    if any(i<0 or i>=len(decoder) for i in layers):raise ValueError('Layer outside main decoder')
    targets=[(decoder[i].self_attn,name) for i in layers for name in ('q_proj','v_proj')]
    if any(not isinstance(getattr(attn,name),nn.Linear) for attn,name in targets):
        raise ValueError('Already wrapped or unsupported attention projection')
    model.requires_grad_(False)
    for attn,name in targets:setattr(attn,name,LowRankLinear(getattr(attn,name),rank))
    if len(modules(model))!=2*len(layers):raise ValueError('Unexpected LoRA coverage')


def parameters(model):
    expected={id(p) for m in modules(model).values() for p in (m.lora_A,m.lora_B)}
    actual={id(p) for p in model.parameters() if p.requires_grad}
    if not expected or expected!=actual:raise ValueError('Unexpected trainable parameter')
    return [p for p in model.parameters() if p.requires_grad]


def snapshot(model):
    parameters(model)
    return {name+'.'+key:getattr(m,key).detach().cpu().clone()
            for name,m in modules(model).items() for key in ('lora_A','lora_B')}


def restore(model,state):
    if set(state)!=set(snapshot(model)):raise ValueError('LoRA state coverage differs')
    with torch.no_grad():
        for name,m in modules(model).items():
            for key in ('lora_A','lora_B'):
                value=state[name+'.'+key];parameter=getattr(m,key)
                if value.shape!=parameter.shape or value.dtype!=torch.float32 or not torch.isfinite(value).all():
                    raise ValueError('Invalid FP32 LoRA state')
                parameter.copy_(value)


def speaker_digest(model):
    h=hashlib.sha256()
    for name,value in model.speaker_preference_adapter.state_dict().items():
        h.update(name.encode());h.update(str((tuple(value.shape),value.dtype)).encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def frozen_digest(model):
    """Canonical original acoustic names; separate frozen-speaker identity."""
    h=hashlib.sha256()
    for name,value in model.named_parameters():
        if name.startswith('speaker_preference_adapter.') or name.endswith(('.lora_A','.lora_B')):continue
        if value.requires_grad:raise ValueError('Original acoustic parameter is trainable')
        name=name.replace('.base_layer.','.')
        h.update(name.encode());h.update(str((tuple(value.shape),value.dtype)).encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def movement(model):
    rows={}
    with torch.no_grad():
        for name,m in modules(model).items():
            delta=m.lora_B@m.lora_A
            rows[name]=float(delta.norm()/m.base_layer.weight.norm().clamp_min(1e-12))
    return dict(per_matrix=rows,maximum=max(rows.values()),mean=sum(rows.values())/len(rows))


def project(model,radius):
    if not 0<radius<1:raise ValueError('Invalid relative update bound')
    with torch.no_grad():
        for m in modules(model).values():
            ratio=(m.lora_B@m.lora_A).norm()/m.base_layer.weight.norm().clamp_min(1e-12)
            if not torch.isfinite(ratio):raise ValueError('Nonfinite weight update')
            if ratio>radius:m.lora_B.mul_(radius/ratio)
