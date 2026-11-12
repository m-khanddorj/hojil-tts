"""Reference-gated, inference-only PAD ablation after native prefix construction.

The existing router remains immutable. Only the two generation-time PAD
arguments change; the already constructed reference prefix is retained exactly.
"""
from contextlib import contextmanager
import hashlib
import re
from types import MethodType

import torch
import torch.nn.functional as F


def tensor_sha256(value):
    """Hash tensor shape, dtype and exact contiguous bytes for audit receipts."""
    if not isinstance(value, torch.Tensor):
        raise ValueError('Expected tensor for identity')
    value = value.detach().cpu().contiguous()
    h = hashlib.sha256(str((tuple(value.shape), value.dtype)).encode())
    h.update(value.numpy().tobytes())
    return h.hexdigest()


class ReferencePrefixGenerationPad:
    def __init__(self, router, reference_sha256, full_generation_pad=True):
        self.router = router
        self.model = router.model
        self.reference_sha256 = self._sha(reference_sha256)
        if not isinstance(full_generation_pad, bool):
            raise ValueError('full_generation_pad must be boolean')
        self.full_generation_pad = full_generation_pad
        self.active = False
        self.actual_sha256 = None
        self.calls = 0
        self.last_trace = None
        self.removed = False
        if router.active or hasattr(self.model, 'reference_prefix_generation_pad'):
            raise ValueError('Requires idle router without PAD wrapper')
        self._environment()
        self.original_generate = self.model.talker.generate
        owner = self

        def generate(this, *args, **kwargs):
            return owner._generate(args, kwargs)

        self.wrapped_generate = MethodType(generate, self.model.talker)
        self.model.talker.generate = self.wrapped_generate
        self.model.reference_prefix_generation_pad = self

    @staticmethod
    def _sha(value):
        if not isinstance(value, str) or not re.fullmatch('[0-9a-f]{64}', value):
            raise ValueError('Expected lowercase reference SHA-256')
        return value

    def _environment(self):
        if (self.model.training or self.model.talker.training
                or getattr(self.model, 'reference_prefix_routing', None) is not self.router):
            raise ValueError('Requires registered evaluation router')
        if any(p.device.type != 'cpu' or p.dtype != torch.float32
               for p in self.model.parameters()):
            raise ValueError('Requires CPU FP32 parameters')
        if any(v.device.type != 'cpu' or v.dtype != torch.float32
               for v in self.router.weights.values()):
            raise ValueError('Requires CPU FP32 alternative matrices')

    @contextmanager
    def use_reference(self, actual_sha):
        actual_sha = self._sha(actual_sha)
        if self.removed or self.active or self.router.active:
            raise ValueError('Reference context must be fresh and cannot nest')
        self._environment()
        if self.model.talker.generate is not self.wrapped_generate:
            raise ValueError('PAD wrapper was replaced')
        previous = self.router.protected
        self.active = True
        self.actual_sha256 = actual_sha
        self.calls = 0
        self.last_trace = None
        self.router.protected = actual_sha == self.reference_sha256
        try:
            yield self
            if self.calls != 1 or self.last_trace is None or self.last_trace['status'] != 'complete':
                raise ValueError('Reference context requires exactly one completed generation')
        finally:
            self.router.protected = previous
            self.active = False
            self.actual_sha256 = None

    def _pads(self):
        config = self.model.config
        ids = torch.tensor([[config.tts_bos_token_id, config.tts_eos_token_id,
                             config.tts_pad_token_id]], dtype=torch.long, device='cpu')
        with torch.no_grad():
            embeddings = self.model.talker.get_text_embeddings()(ids)
            projection = self.model.talker.text_projection
            full = self.router.original_projection(embeddings).chunk(3, dim=1)[2]
            half = F.linear(projection.act_fn(F.linear(
                embeddings, self.router.weights['talker.text_projection.linear_fc1'],
                projection.linear_fc1.bias)),
                self.router.weights['talker.text_projection.linear_fc2'],
                projection.linear_fc2.bias).chunk(3, dim=1)[2]
        if any(not torch.isfinite(value).all() for value in (full, half)):
            raise ValueError('Nonfinite full or half generation PAD')
        return ids, full, half

    def _generate(self, args, kwargs):
        router = self.router
        if not self.active or self.removed:
            raise ValueError('Generation requires explicit reference context')
        self._environment()
        custom = self.actual_sha256 == self.reference_sha256
        if (args or router.protected != custom or not router.active
                or router.layout is None or router.prefills != 0 or router.decodes != 0
                or self.calls != 0):
            raise ValueError('Expected one native talker call after reference prefix construction')
        ids, full, half = self._pads()
        expected = half if custom else full
        for name in ('trailing_text_hidden', 'tts_pad_embed'):
            value = kwargs.get(name)
            if (not isinstance(value, torch.Tensor) or value.shape != expected.shape
                    or value.shape[:2] != (1, 1) or value.dtype != torch.float32
                    or value.device.type != 'cpu' or not torch.isfinite(value).all()
                    or not torch.equal(value, expected)):
                raise ValueError('Unexpected native PAD argument: ' + name)
        prefix, mask = kwargs.get('inputs_embeds'), kwargs.get('attention_mask')
        if (not isinstance(prefix, torch.Tensor) or prefix.ndim != 3
                or tuple(prefix.shape[:2]) != (1, router.layout['prefix_length'])
                or prefix.shape[-1] != expected.shape[-1] or prefix.dtype != torch.float32
                or prefix.device.type != 'cpu' or not torch.isfinite(prefix).all()
                or not isinstance(mask, torch.Tensor) or mask.shape != prefix.shape[:2]
                or mask.device.type != 'cpu' or mask.dtype != torch.long
                or not torch.all(mask == 1)):
            raise ValueError('Expected single-sample native prefix and attention mask')
        tensor_inputs = {name: tensor_sha256(value) for name, value in kwargs.items()
                         if isinstance(value, torch.Tensor)}
        outgoing = dict(kwargs)
        replace = custom and self.full_generation_pad
        if replace:
            outgoing['trailing_text_hidden'] = full.clone()
            outgoing['tts_pad_embed'] = full.clone()
        if any(outgoing[name] is not value for name, value in kwargs.items()
               if name not in ('trailing_text_hidden', 'tts_pad_embed')):
            raise ValueError('Non-PAD arguments changed')
        self.calls += 1
        self.last_trace = dict(
            status='entered', reference_sha256=self.actual_sha256,
            registered_reference_sha256=self.reference_sha256,
            custom_reference=custom, protected=custom,
            full_generation_pad=self.full_generation_pad, pad_replaced=replace,
            mode=('custom_full_generation_pad' if replace else
                  'custom_original_half_pad' if custom else 'inactive_full_policy'),
            talker_calls=self.calls, layout=dict(router.layout),
            input_tensor_sha256=tensor_inputs,
            output_pad_sha256={name: tensor_sha256(outgoing[name])
                               for name in ('trailing_text_hidden', 'tts_pad_embed')},
            special_token_ids_sha256=tensor_sha256(ids),
            full_pad_sha256=tensor_sha256(full), half_pad_sha256=tensor_sha256(half),
            non_pad_kwargs_identity_verified=True, input_tensors_unchanged=False)
        try:
            result = self.original_generate(**outgoing)
            if any(tensor_sha256(kwargs[name]) != sha for name, sha in tensor_inputs.items()):
                raise ValueError('Native generation mutated incoming tensors')
            self.last_trace.update(status='complete', input_tensors_unchanged=True)
            return result
        except BaseException:
            self.last_trace['status'] = 'failed'
            raise

    def remove(self):
        if self.active or self.router.active or self.removed:
            raise ValueError('Cannot remove active or already removed PAD wrapper')
        if self.model.talker.generate is not self.wrapped_generate:
            raise ValueError('PAD wrapper was replaced')
        self.model.talker.generate = self.original_generate
        del self.model.reference_prefix_generation_pad
        self.removed = True
