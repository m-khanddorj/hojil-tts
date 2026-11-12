"""Single-example native routing and teacher replay for NEW masked Q/V LoRA.

No trainer, data selection or checkpoint choice is implemented here. Callers
must verify fresh reference extraction and target-code provenance. Backward
must finish inside replay(); its immutable row scope survives checkpointing.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import math
import re
from types import MethodType

import torch
from torch import nn
import torch.nn.functional as F

from .middle_attention_adapter import LowRankLinear
from .sequence import Sequence


@dataclass(frozen=True)
class NativeReference:
    ref_ids: torch.Tensor
    ref_codes: torch.Tensor
    speaker: torch.Tensor
    reference_sha256: str
    provenance: str

    def validate(self, hidden_size, groups):
        if (not re.fullmatch('[0-9a-f]{64}', self.reference_sha256)
                or not isinstance(self.provenance, str) or not self.provenance.strip()):
            raise ValueError('Explicit reference identity and extraction provenance required')
        if (self.speaker.dtype != torch.float32 or self.speaker.shape != (hidden_size,)
                or self.speaker.requires_grad or not torch.isfinite(self.speaker).all()):
            raise ValueError('Fresh native FP32 speaker feature required; never silently cast cached FP16')
        if (self.ref_ids.dtype != torch.long or self.ref_ids.ndim != 2
                or self.ref_ids.shape[0] != 1 or self.ref_ids.shape[1] <= 5
                or self.ref_codes.dtype != torch.long or self.ref_codes.ndim != 2
                or self.ref_codes.shape[1] != groups or not len(self.ref_codes)):
            raise ValueError('Malformed single-reference IDs or codec frames')


@dataclass(frozen=True)
class Layout:
    target_start: int
    target_stop: int
    prefix_length: int

    def masks(self, positions):
        target = (positions >= self.target_start) & (positions < self.target_stop)
        generated = positions >= self.prefix_length
        return target | generated, (positions < self.prefix_length) & ~target


class MaskedDelta(nn.Module):
    def __init__(self, base, rank):
        super().__init__()
        self.A = nn.Parameter(torch.randn(rank, base.in_features, device=base.weight.device,
                                         dtype=torch.float32) / math.sqrt(base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device,
                                         dtype=torch.float32))

    def forward(self, x):
        return F.linear(F.linear(x, self.A), self.B)


def base_linear(module):
    value = module.base_layer if isinstance(module, LowRankLinear) else module
    if not isinstance(value, nn.Linear):
        raise ValueError('Unsupported acoustic projection')
    return value


class MaskedReferenceSupervision:
    def __init__(self, model, half_weights, layers=tuple(range(10,18)), rank=4):
        if any(hasattr(model, name) for name in ('reference_prefix_routing', 'reference_masked_supervision')):
            raise ValueError('Requires clean model without an existing routing controller')
        if rank != 4 or not layers or len(layers) != len(set(layers)):
            raise ValueError('Expected new rank-4 LoRA on distinct layers')
        decoder = model.talker.model.layers
        if any(not isinstance(i, int) or not 0 <= i < len(decoder) for i in layers):
            raise ValueError('Layer outside decoder')
        if any(p.dtype != torch.float32 for p in model.parameters()):
            raise ValueError('FP32 original parameters required')
        for sub in (model.talker.model, model.talker.code_predictor.model):
            if sub.config.attention_dropout != 0:
                raise ValueError('Deterministic attention required')
        self.model = model
        self.modules = {}
        named = dict(model.named_modules())
        for i in range(len(decoder)):
            for suffix in ('self_attn.q_proj','self_attn.k_proj','self_attn.v_proj','self_attn.o_proj',
                           'mlp.gate_proj','mlp.up_proj','mlp.down_proj'):
                name = f'talker.model.layers.{i}.{suffix}'
                module = named[name]
                base_linear(module)
                if isinstance(module, LowRankLinear) and suffix not in ('self_attn.q_proj','self_attn.v_proj'):
                    raise ValueError('Only old Q/V LoRA wrappers supported')
                self.modules[name] = module
        for i in (1,2):
            name = f'talker.text_projection.linear_fc{i}'
            module = named[name]
            if not isinstance(module, nn.Linear):
                raise ValueError('Text projections must remain plain frozen linears')
            self.modules[name] = module
        if set(half_weights) != {n+'.weight' for n in self.modules}:
            raise ValueError('Half-matrix canonical coverage differs')
        bank = nn.Module()
        self.half_keys = {}
        for index, (name, module) in enumerate(self.modules.items()):
            value = half_weights[name+'.weight']
            base = base_linear(module)
            if value.dtype != torch.float32 or value.shape != base.weight.shape or not torch.isfinite(value).all():
                raise ValueError('Invalid immutable FP32 half matrix')
            key = f'w{index}'
            bank.register_buffer(key, value.detach().to(base.weight.device).clone())
            self.half_keys[name] = key
        model.requires_grad_(False)
        self.adapter_keys = {}
        adapters = nn.ModuleDict()
        for i in layers:
            for projection in ('q_proj', 'v_proj'):
                name = f'talker.model.layers.{i}.self_attn.{projection}'
                key = f'layer{i}_{projection}'
                adapters[key] = MaskedDelta(base_linear(self.modules[name]), rank)
                self.adapter_keys[name] = key
        model.add_module('reference_masked_adapters', adapters)
        model.add_module('reference_masked_half', bank)
        self.adapters, self.bank = adapters, bank
        self.scope = None
        self.positions = None
        self.projection_span = None
        self.last_trace = None
        self.removed = False
        self.original_generate = model.generate
        self.original_icl = model.generate_icl_prompt
        self.original_main = model.talker.model.forward
        self.original_projection = model.talker.text_projection.forward
        self.original_talker_generate = model.talker.generate
        self.original_linears = {n:m.forward for n,m in self.modules.items() if n.startswith('talker.model.')}
        self._install()
        model.reference_masked_supervision = self

    def parameters(self):
        expected = {id(p) for p in self.adapters.parameters()}
        if {id(p) for p in self.model.parameters() if p.requires_grad} != expected:
            raise ValueError('Only NEW masked LoRA parameters may train')
        return list(self.adapters.parameters())

    def _half(self, name):
        return getattr(self.bank, self.half_keys[name])

    def _need_scope(self):
        if self.scope is None or self.removed:
            raise ValueError('Explicit native/replay scope required, including backward')
        return self.scope

    @contextmanager
    def _context(self, kind, half_reference, layout=None):
        if self.scope is not None or self.removed or type(half_reference) is not bool:
            raise ValueError('Fresh nonnested explicit mode scope required')
        self.parameters()
        scope = dict(kind=kind, half_reference=half_reference, layout=layout, calls=0, sequence=None)
        self.scope = scope
        try:
            yield scope
        finally:
            self.last_trace = dict(kind=kind, half_reference=half_reference, layout=scope['layout'], calls=scope['calls'])
            self.scope = self.positions = self.projection_span = None

    @contextmanager
    def native(self, half_reference=False):
        if self.model.training:
            raise ValueError('Native generation requires evaluation mode')
        with self._context('native', half_reference) as scope:
            yield self
            if scope['calls'] != 1:
                raise ValueError('Exactly one native generation required')

    def _install(self):
        owner = self
        model = self.model

        def generate(this, *args, **kwargs):
            scope = owner._need_scope()
            if (scope['kind'] != 'native' or scope['calls'] or args
                    or not kwargs.get('non_streaming_mode') or kwargs.get('languages') != ['Auto']
                    or len(kwargs.get('input_ids', [])) != 1 or len(kwargs.get('ref_ids', [])) != 1):
                raise ValueError('Only one native full-ICL Auto sample supported')
            prompt = kwargs['voice_clone_prompt']
            if prompt['icl_mode'] != [True] or prompt['x_vector_only_mode'] != [False]:
                raise ValueError('Full ICL reference required')
            nr, nt = kwargs['ref_ids'][0].shape[-1]-5, kwargs['input_ids'][0].shape[-1]-8
            scope['layout'] = owner._layout(nr, nt, len(prompt['ref_code'][0]))
            scope['calls'] += 1
            return owner.original_generate(**kwargs)

        def icl(this, *args, **kwargs):
            scope = owner._need_scope()
            if args or not kwargs['non_streaming_mode'] or owner.projection_span is not None:
                raise ValueError('Unexpected ICL projection scope')
            nr, nt = kwargs['ref_id'].shape[-1], kwargs['text_id'].shape[-1]
            if owner._layout(nr, nt, len(kwargs['ref_code'])) != scope['layout']:
                raise ValueError('ICL layout differs')
            owner.projection_span = (nr, nr+nt)
            try:
                value = owner.original_icl(**kwargs)
            finally:
                owner.projection_span = None
            if value[0].shape[1]+8 != scope['layout'].prefix_length:
                raise ValueError('Native prefix length differs')
            return value

        def projection(this, x):
            scope = owner._need_scope()
            full = owner.original_projection(x)
            if not scope['half_reference']:
                return full
            a,b = this.linear_fc1,this.linear_fc2
            half = F.linear(this.act_fn(F.linear(x,owner._half('talker.text_projection.linear_fc1'),a.bias)),
                            owner._half('talker.text_projection.linear_fc2'),b.bias)
            if owner.projection_span is None:
                return half
            start, stop = owner.projection_span
            if x.ndim != 3 or x.shape[:2] != (1,stop):
                raise ValueError('Unexpected reference/target projection shape')
            positions = torch.arange(stop,device=x.device)
            return torch.where(((positions>=start)&(positions<stop))[None,:,None],full,half)

        def main(this, *args, **kwargs):
            scope = owner._need_scope()
            x = kwargs.get('inputs_embeds')
            if args or x is None or x.ndim != 3 or x.shape[0] != 1:
                raise ValueError('Single-example decoder embeddings required')
            if scope['kind'] == 'replay':
                if (kwargs.get('use_cache') is not False or kwargs.get('past_key_values') is not None
                        or x.shape[1] != len(scope['sequence'].embeddings)):
                    raise ValueError('Replay must use entire immutable sequence without cache')
                # DO NOT clear positions on forward exit: checkpoint backward
                # recomputes decoder layers after this frame has returned.
                if owner.positions is None or len(owner.positions) != x.shape[1]:
                    raise ValueError('Replay row scope missing')
                return owner.original_main(**kwargs)
            if torch.is_grad_enabled():
                raise ValueError('Cached/native decoding is verification-only, without gradients')
            cache = kwargs.get('past_key_values')
            start = 0 if cache is None else cache.get_seq_length()
            if (start == 0 and x.shape[1] != scope['layout'].prefix_length
                    or start > 0 and (start < scope['layout'].prefix_length or x.shape[1] != 1)):
                raise ValueError('Unexpected prefill or cached-frame positions')
            previous = owner.positions
            owner.positions = torch.arange(start,start+x.shape[1],device=x.device)
            try:
                return owner.original_main(**kwargs)
            finally:
                owner.positions = previous

        def linear(name, original):
            def forward(this, x):
                scope = owner._need_scope()
                if owner.positions is None or x.ndim != 3 or x.shape[:2] != (1,len(owner.positions)):
                    raise ValueError('Projection outside immutable decoder row scope')
                allowed, protected = scope['layout'].masks(owner.positions)
                full = original(x)  # preserves old outer base(x)+B(A(x)) exactly
                value = full
                if scope['half_reference']:
                    half = F.linear(x,owner._half(name),base_linear(this).bias)
                    value = torch.where(protected[None,:,None],half,full)
                if name in owner.adapter_keys:
                    delta = owner.adapters[owner.adapter_keys[name]](x)
                    value = torch.where(allowed[None,:,None],value+delta,value)
                return value
            return forward

        def talker_generate(this, *args, **kwargs):
            scope = owner._need_scope()
            if args or scope['kind'] != 'native' or scope['layout'] is None:
                raise ValueError('Native prefix must precede talker generation')
            _,_,pad = owner._full_special()
            incoming = kwargs['tts_pad_embed']
            cfg=owner.model.config
            special=owner.model.talker.get_text_embeddings()(owner._ids([cfg.tts_bos_token_id,cfg.tts_eos_token_id,cfg.tts_pad_token_id]))
            expected=owner.model.talker.text_projection(special).chunk(3,dim=1)[2]
            if (incoming.shape != pad.shape or kwargs['trailing_text_hidden'].shape != pad.shape
                    or not torch.equal(incoming,expected)
                    or not torch.equal(kwargs['trailing_text_hidden'],expected)):
                raise ValueError('Only single trailing PAD supported')
            outgoing = dict(kwargs,trailing_text_hidden=pad,tts_pad_embed=pad)
            return owner.original_talker_generate(**outgoing)

        model.generate = MethodType(generate,model)
        model.generate_icl_prompt = MethodType(icl,model)
        model.talker.text_projection.forward = MethodType(projection,model.talker.text_projection)
        model.talker.model.forward = MethodType(main,model.talker.model)
        model.talker.generate = MethodType(talker_generate,model.talker)
        for name, original in self.original_linears.items():
            self.modules[name].forward = MethodType(linear(name,original),self.modules[name])

    @staticmethod
    def _layout(nr, nt, frames):
        if nr < 1 or nt < 1 or frames < 1:
            raise ValueError('Nonempty reference/target conditioning required')
        return Layout(8+nr,8+nr+nt,8+nr+nt+frames+2)

    def _ids(self, values):
        return torch.tensor([values],device=self.model.talker.get_input_embeddings().weight.device,dtype=torch.long)

    def _full_special(self):
        cfg = self.model.config
        embedded = self.model.talker.get_text_embeddings()(self._ids([cfg.tts_bos_token_id,cfg.tts_eos_token_id,cfg.tts_pad_token_id]))
        projected=self.original_projection(embedded)
        if projected.dtype != torch.float32 or not torch.isfinite(projected).all():
            raise ValueError('Nonfinite or non-FP32 full special projection')
        return projected.chunk(3,dim=1)

    def _build(self, input_ids, codes, reference):
        talker, cfg = self.model.talker,self.model.config
        tc = cfg.talker_config
        device = talker.get_input_embeddings().weight.device
        ids = input_ids.to(device)
        codes = codes.to(device)
        text = lambda values: talker.text_projection(talker.get_text_embeddings()(values))
        codec = talker.get_input_embeddings()
        bos,eos,prefix_pad = text(self._ids([cfg.tts_bos_token_id,cfg.tts_eos_token_id,cfg.tts_pad_token_id])).chunk(3,dim=1)
        full_pad = self._full_special()[2]
        speaker = reference.speaker.to(device)
        if hasattr(self.model,'speaker_preference_adapter'):
            speaker = self.model.speaker_preference_adapter(speaker)
        tags = codec(self._ids([tc.codec_nothink_id,tc.codec_think_bos_id,tc.codec_think_eos_id]))
        codec_prefix = torch.cat([tags,speaker.reshape(1,1,-1),codec(self._ids([tc.codec_pad_id,tc.codec_bos_id]))],dim=1)
        prefix = torch.cat([text(ids[:,:3]),torch.cat([prefix_pad.expand(-1,codec_prefix.shape[1]-2,-1),bos],dim=1)+codec_prefix[:,:-1]],dim=1)
        context,_ = self.model.generate_icl_prompt(text_id=ids[:,3:-5],ref_id=reference.ref_ids.to(device)[:,3:-2],
            ref_code=reference.ref_codes.to(device),tts_pad_embed=prefix_pad,tts_eos_embed=eos,non_streaming_mode=True)
        prefix = torch.cat([prefix,context],dim=1)
        # Native cached decode concatenates all16 codebook embeddings, then sums.
        pieces = [codec(codes[:,:1])]+[talker.code_predictor.get_input_embeddings()[i-1](codes[:,i:i+1]) for i in range(1,tc.num_code_groups)]
        suffix = torch.cat(pieces,dim=1).sum(1).unsqueeze(0)+full_pad
        embeddings = torch.cat([prefix,suffix],dim=1)[0]
        P = prefix.shape[1]
        if P != self.scope['layout'].prefix_length:
            raise ValueError('Teacher prefix layout differs')
        labels = torch.full((len(embeddings),),-100,device=device,dtype=torch.long)
        labels[P-1:P-1+len(codes)] = codes[:,0]
        labels[P-1+len(codes)] = tc.codec_eos_token_id
        return Sequence(embeddings,labels,torch.arange(P-1,P-1+len(codes),device=device),codes,P)

    @contextmanager
    def replay(self, input_ids, codes, reference, half_reference=False, terminated=False, _cached=False):
        tc = self.model.config.talker_config
        reference.validate(tc.hidden_size,tc.num_code_groups)
        if (terminated is not True or input_ids.dtype != torch.long or input_ids.ndim != 2
                or input_ids.shape[0] != 1 or input_ids.shape[1] <= 8
                or codes.dtype != torch.long or codes.ndim != 2 or not len(codes)
                or codes.shape[1] != tc.num_code_groups):
            raise ValueError('Complete terminated single target trajectory required')
        vocab = self.model.talker.code_predictor.config.vocab_size
        if any(t.min()<0 or t.max()>=vocab for t in (codes,reference.ref_codes)):
            raise ValueError('Invalid residual codec vocabulary')
        layout = self._layout(reference.ref_ids.shape[1]-5,input_ids.shape[1]-8,len(reference.ref_codes))
        with self._context('cached' if _cached else 'replay',half_reference,layout) as scope:
            seq = self._build(input_ids,codes,reference)
            scope['sequence'] = seq
            if not _cached:
                self.positions = torch.arange(len(seq.embeddings),device=seq.embeddings.device)
            yield seq

    def _checkpoint_guard(self):
        for decoder in (self.model.talker.model,self.model.talker.code_predictor.model):
            if decoder.training and decoder.gradient_checkpointing:
                if getattr(decoder._gradient_checkpointing_func,'keywords',{}).get('use_reentrant') is not False:
                    raise ValueError('Frozen-input replay requires explicit nonreentrant checkpointing')

    def _outputs(self, seq, hidden):
        scope = self._need_scope()
        if hidden.requires_grad:
            def guard(gradient):
                if self.scope is not scope:
                    raise ValueError('Backward must finish inside its original replay scope')
                return gradient
            hidden.register_hook(guard)
        talker = self.model.talker
        main = talker.codec_head(hidden[seq.labels!=-100]).float()
        codes = seq.codes
        predictor = talker.code_predictor
        pieces = [hidden[seq.frame_positions,None],talker.get_input_embeddings()(codes[:,:1])]
        pieces += [predictor.get_input_embeddings()[i-1](codes[:,i:i+1]) for i in range(1,codes.shape[1]-1)]
        residual = predictor.forward_finetune(inputs_embeds=torch.cat(pieces,dim=1),labels=None,use_cache=False,return_dict=True).logits.float()
        if main.shape[0] != len(codes)+1 or residual.shape[:2] != codes[:,1:].shape:
            raise ValueError('Pre-frame codec/EOS alignment differs')
        first = F.cross_entropy(main,seq.labels[seq.labels!=-100],reduction='sum')
        rest = F.cross_entropy(residual.reshape(-1,residual.shape[-1]),codes[:,1:].reshape(-1),reduction='sum')
        return dict(main_logits=main,residual_logits=residual,main_nll=first,residual_nll=rest,
                    log_probability=-(first.double()+rest.double()),token_count=codes.numel()+1,hidden=hidden)

    def outputs(self, seq):
        scope = self._need_scope()
        if scope['kind'] != 'replay' or scope['sequence'] is not seq:
            raise ValueError('Outputs require their own replay sequence')
        self._checkpoint_guard()
        mask = torch.ones(1,len(seq.embeddings),device=seq.embeddings.device,dtype=torch.long)
        positions = (mask.cumsum(-1)-1).unsqueeze(0).expand(3,-1,-1)
        hidden = self.model.talker.model(inputs_embeds=seq.embeddings[None],attention_mask=mask,
            position_ids=positions,use_cache=False,return_dict=True).last_hidden_state[0]
        return self._outputs(seq,hidden)

    def cached_logits(self, input_ids, codes, reference, half_reference=False, terminated=False):
        """No-grad cache diagnostic; main prefix/suffix uses the explicit builder.

        Residual logits independently traverse native predictor.forward caches,
        rather than forward_finetune. Actual native generation prefix/suffix
        parity remains a separate mandatory control, covered by tiny tests here.
        """
        if self.model.training or torch.is_grad_enabled():
            raise ValueError('Cached verification requires eval and no_grad')
        with self.replay(input_ids,codes,reference,half_reference,terminated,_cached=True) as seq:
            pieces=[]
            cache=None
            for start,stop in [(0,seq.prefix_length)]+[(i,i+1) for i in range(seq.prefix_length,len(seq.embeddings))]:
                mask=torch.ones(1,stop,device=seq.embeddings.device,dtype=torch.long)
                positions=torch.arange(start,stop,device=seq.embeddings.device)[None,None].expand(3,1,-1)
                result=self.model.talker.model(inputs_embeds=seq.embeddings[None,start:stop],attention_mask=mask,
                    position_ids=positions,past_key_values=cache,use_cache=True,return_dict=True)
                cache=result.past_key_values
                pieces.append(result.last_hidden_state[0])
            hidden=torch.cat(pieces,dim=0)
            output=self._outputs(seq,hidden)
            predictor=self.model.talker.code_predictor
            residual=[]
            for frame,position in enumerate(seq.frame_positions):
                initial=torch.cat([hidden[position][None,None],self.model.talker.get_input_embeddings()(seq.codes[frame:frame+1,:1])],dim=1)
                current=predictor(inputs_embeds=initial,use_cache=True,return_dict=True)
                row=[current.logits[:,-1]]
                for group in range(1,seq.codes.shape[1]-1):
                    current=predictor(input_ids=seq.codes[frame:frame+1,group:group+1],
                        past_key_values=current.past_key_values,generation_steps=current.generation_steps,
                        use_cache=True,return_dict=True)
                    row.append(current.logits[:,-1])
                residual.append(torch.stack(row,dim=1))
            output['residual_logits']=torch.cat(residual,dim=0).float()
            logits=output['residual_logits']
            output['residual_nll']=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),seq.codes[:,1:].reshape(-1),reduction='sum')
            output['log_probability']=-(output['main_nll'].double()+output['residual_nll'].double())
            return output

    def remove(self):
        if self.scope is not None or self.removed:
            raise ValueError('Cannot remove active or removed controller')
        model=self.model
        model.generate=self.original_generate
        model.generate_icl_prompt=self.original_icl
        model.talker.model.forward=self.original_main
        model.talker.text_projection.forward=self.original_projection
        model.talker.generate=self.original_talker_generate
        for name,original in self.original_linears.items():
            self.modules[name].forward=original
        del model.reference_masked_supervision
        del model.reference_masked_adapters
        del model.reference_masked_half
        self.removed=True
