"""Optimizer and checkpoint policies for controlled LoRA experiments."""
import math


def split_lora_optimizer(optimizer, named_parameters, ratio):
    """Split an existing Adam optimizer without resetting any parameter moments.

    The schedule denotes B's learning rate. A receives lr / ratio. Keeping B's
    rate fixed avoids confounding the ratio experiment with a larger maximum LR.
    Call after loading a historical single-group optimizer, or before loading a
    new multi-group resume checkpoint. No parameter objects are replaced.
    """
    if not math.isfinite(ratio) or ratio < 1:
        raise ValueError('LoRA+ B/A ratio must be finite and >= 1')
    if ratio == 1:
        return
    if len(optimizer.param_groups) != 1:
        raise ValueError('Expected one optimizer group before LoRA+ split')
    original = optimizer.param_groups[0]
    a, b = [], []
    for name, param in named_parameters:
        if not param.requires_grad:
            continue
        if '.lora_A.' in name:
            a.append(param)
        elif '.lora_B.' in name:
            b.append(param)
        else:
            raise ValueError(f'Unsupported LoRA+ trainable parameter: {name}')
    if not a or not b or {id(p) for p in a+b} != {id(p) for p in original['params']}:
        raise ValueError('LoRA+ groups do not cover the optimizer')
    defaults = {k: v for k, v in original.items() if k != 'params'}
    optimizer.param_groups = [dict(defaults, params=a, lr_scale=1/ratio, group_name='lora_A'),
                              dict(defaults, params=b, lr_scale=1.0, group_name='lora_B')]
    set_learning_rate(optimizer, defaults['lr'])


def set_learning_rate(optimizer, rate):
    for group in optimizer.param_groups:
        group['lr'] = rate * group.get('lr_scale', 1.0)


def selection_update(loss, best, anchor, bad, min_delta):
    """Always retain the lowest loss; use min_delta only for patience."""
    if not math.isfinite(loss) or min_delta < 0:
        raise ValueError('Invalid selection metric')
    improved = loss < best
    significant = loss < anchor - min_delta
    return min(loss, best), loss if significant else anchor, 0 if significant else bad+1, improved
