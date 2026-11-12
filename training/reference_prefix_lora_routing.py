"""Native reference-prefix routing around frozen outer Q/V LoRA modules.

Reference/control rows use supplied alternative matrices; target-text rows and
cached generated frames use the original full model. Attention still couples
these representations. This is a hypothesis, not a voice-preservation claim.
"""
from types import MethodType
import torch
import torch.nn.functional as F
from .middle_attention_adapter import LowRankLinear


def base_linear(module):
    """Return the acoustic linear without merging or modifying LoRA weights."""
    base = module.base_layer if isinstance(module, LowRankLinear) else module
    if not isinstance(base, torch.nn.Linear):
        raise ValueError('Expected plain acoustic linear or supported outer LoRA')
    return base


def weight_modules(model, expected_layers=tuple(range(10,18))):
    """Canonical registry; never include wrapped .base_layer descendants."""
    layers = model.talker.model.layers
    expected_layers = tuple(expected_layers)
    if (not expected_layers or len(set(expected_layers)) != len(expected_layers)
            or any(not isinstance(i, int) or not 0 <= i < len(layers) for i in expected_layers)):
        raise ValueError('Invalid expected LoRA layers')
    all_modules = dict(model.named_modules())
    modules = {}
    expected_wrapped = set()
    for i in range(len(layers)):
        for suffix in ('self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj',
                       'self_attn.o_proj', 'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj'):
            name = f'talker.model.layers.{i}.{suffix}'
            module = all_modules.get(name)
            wrapped = i in expected_layers and suffix in ('self_attn.q_proj', 'self_attn.v_proj')
            if wrapped:
                expected_wrapped.add(name)
                if (not isinstance(module, LowRankLinear)
                        or module.lora_A.shape != (4, module.base_layer.in_features)
                        or module.lora_B.shape != (module.base_layer.out_features, 4)):
                    raise ValueError('Expected rank-4 outer Q/V LoRA')
                for tensor in (module.lora_A, module.lora_B):
                    if tensor.dtype != torch.float32 or tensor.device.type != 'cpu' or not torch.isfinite(tensor).all():
                        raise ValueError('Expected finite CPU FP32 LoRA')
            elif not isinstance(module, torch.nn.Linear):
                raise ValueError('Unexpected wrapped decoder projection')
            modules[name] = module
    actual_wrapped = {name for name, module in all_modules.items() if isinstance(module, LowRankLinear)}
    if actual_wrapped != expected_wrapped:
        raise ValueError('Outer LoRA coverage differs')
    for i in (1,2):
        name=f'talker.text_projection.linear_fc{i}'
        module=all_modules.get(name)
        if not isinstance(module,torch.nn.Linear):
            raise ValueError('Requires merged plain text-projection linears')
        modules[name]=module
    if len(modules) != 7 * len(layers) + 2:
        raise ValueError('Canonical decoder/text projection coverage differs')
    return modules


class ReferencePrefixLoRARouting:
    def __init__(self,model,alternative_weights,protected=True,expected_layers=tuple(range(10,18))):
        if hasattr(model,'reference_prefix_routing') or model.training:
            raise ValueError('Requires fresh evaluation model')
        if any(p.requires_grad or p.dtype != torch.float32 or p.device.type != 'cpu'
               for p in model.parameters()):
            raise ValueError('Requires frozen CPU FP32 model')
        self.model=model;self.protected=bool(protected)
        self.modules=weight_modules(model,expected_layers)
        if set(alternative_weights)!={n+'.weight' for n in self.modules}:
            raise ValueError('Alternative matrix coverage differs')
        self.weights={}
        for name,module in self.modules.items():
            value=alternative_weights[name+'.weight']
            if (value.shape!=base_linear(module).weight.shape or value.dtype!=torch.float32
                    or value.device!=base_linear(module).weight.device or value.device.type!='cpu'
                    or not torch.isfinite(value).all()):
                raise ValueError('Expected finite same-device CPU FP32 matrices')
            self.weights[name]=value.detach()
        self.active=False;self.text_span=None;self.main_mask=None;self.layout=None
        self.prefills=0;self.decodes=0;self.last_trace=None
        self.original_generate=model.generate
        self.original_icl=model.generate_icl_prompt
        self.original_main=model.talker.model.forward
        self.original_projection=model.talker.text_projection.forward
        self.original_linears={n:m.forward for n,m in self.modules.items() if n.startswith('talker.model.')}
        router=self

        def generate(this,*args,**kwargs):
            if (router.active or args or not kwargs.get('non_streaming_mode',False)
                    or len(kwargs.get('input_ids',[]))!=1 or len(kwargs.get('ref_ids',[]))!=1
                    or kwargs.get('languages')!=['Auto']):
                raise ValueError('Routing supports one full-ICL Auto-language nonstreaming sample')
            router.active=True;router.layout=None;router.prefills=0;router.decodes=0
            router.text_span=None;router.main_mask=None;router.last_trace=None
            try:
                result=router.original_generate(**kwargs)
                if router.prefills!=1 or router.layout is None or router.decodes<1:
                    raise ValueError('Expected one prefill and cached decoding')
                router.last_trace=dict(router.layout,prefills=router.prefills,decode_steps=router.decodes,
                    protected=router.protected,main_matrices=len(router.original_linears),
                    reference_positions=router.layout['prefix_length']-(router.layout['target_stop']-router.layout['target_start']))
                return result
            finally:
                router.active=False;router.text_span=None;router.main_mask=None

        def icl(this,*args,**kwargs):
            if not router.active:return router.original_icl(*args,**kwargs)
            if args or router.layout is not None or not kwargs['non_streaming_mode']:
                raise ValueError('Unexpected ICL layout or repeated prompt')
            nr=kwargs['ref_id'].shape[-1];nt=kwargs['text_id'].shape[-1]
            if nr<1 or nt<1:raise ValueError('Empty reference or target text')
            router.text_span=(nr,nr+nt)
            try:result=router.original_icl(**kwargs)
            finally:router.text_span=None
            # The native Auto-language prefix contains three assistant text
            # positions plus five codec/control/speaker positions.
            router.layout=dict(target_start=8+nr,target_stop=8+nr+nt,
                               prefix_length=8+result[0].shape[1])
            return result

        def projection(this,x):
            full=router.original_projection(x)
            if not router.active or not router.protected:return full
            if x.ndim!=3 or x.shape[0]!=1:raise ValueError('Unexpected projection batch')
            a,b=this.linear_fc1,this.linear_fc2
            other=F.linear(this.act_fn(F.linear(x,router.weights['talker.text_projection.linear_fc1'],a.bias)),
                           router.weights['talker.text_projection.linear_fc2'],b.bias)
            if router.text_span is None:return other
            start,stop=router.text_span
            if x.shape[1]!=stop:raise ValueError('Unexpected reference/target concatenation')
            indices=torch.arange(stop,device=x.device)
            return torch.where(((indices>=start)&(indices<stop))[None,:,None],full,other)

        def main(this,*args,**kwargs):
            if not router.active:return router.original_main(*args,**kwargs)
            if args:raise ValueError('Expected keyword decoder invocation')
            x=kwargs.get('inputs_embeds')
            if x is None or x.ndim!=3 or x.shape[0]!=1:
                raise ValueError('Unexpected native decoder input')
            cache=kwargs.get('past_key_values')
            cached=0 if cache is None else cache.get_seq_length()
            if x.shape[1]>1:
                if (router.prefills or cached or router.layout is None
                        or x.shape[1]!=router.layout['prefix_length']):
                    raise ValueError('Unexpected prefill/cache layout')
                router.prefills+=1
                mask=torch.ones(x.shape[:2],dtype=torch.bool,device=x.device)
                mask[:,router.layout['target_start']:router.layout['target_stop']]=False
                router.main_mask=mask if router.protected else None
            else:
                if router.prefills!=1 or cached<1:raise ValueError('Decode lacks reference cache')
                router.decodes+=1;router.main_mask=None
            try:return router.original_main(**kwargs)
            finally:router.main_mask=None

        def linear_forward(name,original):
            def forward(this,x):
                full=original(x);mask=router.main_mask
                if mask is None:return full
                if x.shape[:-1]!=mask.shape:raise ValueError('Main projection span differs')
                other=F.linear(x,router.weights[name],base_linear(this).bias)
                return torch.where(mask[...,None],other,full)
            return forward

        model.generate=MethodType(generate,model)
        model.generate_icl_prompt=MethodType(icl,model)
        model.talker.text_projection.forward=MethodType(projection,model.talker.text_projection)
        model.talker.model.forward=MethodType(main,model.talker.model)
        for name,original in self.original_linears.items():
            self.modules[name].forward=MethodType(linear_forward(name,original),self.modules[name])
        model.reference_prefix_routing=self

    def remove(self):
        if self.active:raise ValueError('Cannot remove routing during generation')
        self.model.generate=self.original_generate
        self.model.generate_icl_prompt=self.original_icl
        self.model.talker.model.forward=self.original_main
        self.model.talker.text_projection.forward=self.original_projection
        for name,original in self.original_linears.items():self.modules[name].forward=original
        del self.model.reference_prefix_routing

