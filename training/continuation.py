"""Validate an explicit continuation without discarding optimizer history."""


def validate_parent(state, provenance):
    parent = state.get('provenance')
    cfg = provenance['config']
    if parent is None:
        if (state['config']['features'] != provenance['features']
                or state['config']['rank'] != cfg['rank']):
            raise ValueError('Historical parent model/features/rank differ')
        return False
    for key in ('features','shards','base_config','base_weights','validation','versions'):
        if parent[key] != provenance[key]:
            raise ValueError(f'Continuation parent differs: {key}')
    for key, default in (('method','lora'),('rank',16),('loraplus_b_to_a_ratio',1)):
        if parent['config'].get(key,default) != cfg.get(key,default):
            raise ValueError(f'Continuation optimizer/model structure differs: {key}')
    if parent['config'].get('method','lora') != 'lora':
        raise ValueError('Explicit checkpoint forks currently support LoRA only')
    for key in ('adapter','optimizer','rng_cpu','rng_cuda','step'):
        if key not in state:
            raise ValueError(f'Incomplete continuation checkpoint: {key}')
    return True
