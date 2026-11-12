"""Stateful evidence guard: distinguish validation divergence from a plateau."""
import copy
import math


def validate_guard(config):
    if (not isinstance(config.get('patience'), int) or config['patience'] < 2
            or any(not math.isfinite(config.get(k, float('nan'))) or config[k] <= 0
                   for k in ('validation_delta', 'training_delta'))):
        raise ValueError('Overfit guard requires patience >= 2 and positive finite deltas')


def update_guard(state, *, step, train, familiar, unseen, config):
    """Require repeated validation regression while training beats its paired anchor.

    Each cohort has its own lowest validation loss and the training loss at
    that same checkpoint. A new minimum resets that cohort's anchor/counter.
    Persist the returned state in checkpoints; identical-step calls are ignored.
    """
    validate_guard(config)
    if not all(math.isfinite(v) for v in (train, familiar, unseen)):
        raise ValueError('Non-finite fit probe; refusing to continue training')
    result = copy.deepcopy(state) if state else dict(groups={}, stopped=False, last_step=-1)
    if step < result['last_step']:
        raise ValueError('Fit probes must arrive in increasing step order')
    if step == result['last_step'] or result['stopped']:
        return result
    result['last_step'] = step
    for name, loss in (('familiar', familiar), ('unseen', unseen)):
        group = result['groups'].get(name)
        if group is None or loss < group['validation']:
            group = dict(validation=loss, train=train, step=step, bad_checks=0)
        else:
            divergence = (loss - group['validation'] >= config['validation_delta']
                          and group['train'] - train >= config['training_delta'])
            group['bad_checks'] = group['bad_checks'] + 1 if divergence else 0
        group['latest_validation'] = loss
        group['latest_train'] = train
        result['groups'][name] = group
    result['triggered_groups'] = [name for name, group in result['groups'].items()
                                  if group['bad_checks'] >= config['patience']]
    result['stopped'] = bool(result['triggered_groups'])
    return result
