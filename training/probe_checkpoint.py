"""Measure another selected LoRA checkpoint on a run's exact observability probes."""
import argparse
import os
from pathlib import Path

from .engine import assignment, digest, evaluate, ids, inject, load_adapter, read, rows, write
from .policy import external, fingerprint, validate_cohorts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('base','training-run','checkpoint-run','backup','features','output'):
        parser.add_argument('--'+key, type=Path, required=True)
    args = parser.parse_args()
    output = external(args.output)
    if output.exists():
        raise ValueError('Probe output already exists; retain the original result')
    os.environ['HF_HUB_OFFLINE'] = os.environ['TRANSFORMERS_OFFLINE'] = '1'
    import torch
    from qwen_tts import Qwen3TTSModel
    torch.set_num_threads(4)
    run, other = read(args.training_run/'run.json'), read(args.checkpoint_run/'run.json')
    selection = read(args.checkpoint_run/'best.json')
    weights = args.checkpoint_run/selection['weights']
    if (other['config']['method'] != 'lora' or run['base_weights'] != other['base_weights']
            or digest(args.base/'model.safetensors') != run['base_weights']
            or digest(weights) != selection['weights_sha256']
            or read(args.features/'signature.json') != run['features']):
        raise ValueError('Checkpoint/base/features provenance differs')
    train = rows(args.backup/'cohorts/train.jsonl')
    if digest(args.backup/'cohorts/train.jsonl') != run['features']['train_sha256']:
        raise ValueError('Training cohort checksum changed')
    unseen = rows(args.backup/'cohorts/unseen-validation.jsonl')
    familiar = rows(args.backup/'cohorts/familiar-validation.jsonl')
    cases = read(args.training_run/'validation-cases-000000.json')
    heldout = {r['id']:r for r in unseen+familiar}
    selected = [dict(heldout[r['id']],validation_group=r['group']) for r in cases]
    if fingerprint(selected) != run['validation']:
        raise ValueError('Validation selection changed')
    canonical = validate_cohorts(train, selected, rows(args.backup/'dataset-v1/manifest.jsonl'))
    probe = read(args.training_run/'train-probe.json')
    if fingerprint(probe['rows']) != probe['fingerprint']:
        raise ValueError('Training probe changed')
    cohort = {r['id']:r for r in train}
    for row in probe['rows']:
        if {k:v for k,v in row.items() if k!='validation_group'} != cohort[row['id']]:
            raise ValueError('Training probe differs from retained cohort')
    needed = {r[key] for r in probe['rows'] for key in ('id','ref_id')}
    data = {}
    for name, checksum in run['shards'].items():
        path = args.features/name
        if digest(path) != checksum:
            raise ValueError('Feature shard changed')
        for row in torch.load(path,map_location='cpu',weights_only=True):
            if row['id'] in needed:
                data[row['id']] = row
    cache = torch.load(args.training_run/'validation-features.pt',map_location='cpu',weights_only=True)
    if cache['signature'] != fingerprint([run['validation'],run['features']]):
        raise ValueError('Validation feature signature differs')
    data.update(cache['data'])
    wrapper = Qwen3TTSModel.from_pretrained(str(args.base),device_map='cpu',
                torch_dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True)
    wrapper.model.requires_grad_(False)
    inject(wrapper.model.talker, other['config']['rank'], 0)
    load_adapter(wrapper.model.talker,torch.load(weights,map_location='cpu',weights_only=True))
    wrapper.model.talker.to('cuda').eval()
    tokens = {}
    for row in selected+probe['rows']:
        for key, reference in ((row['id'],False),(row['ref_id'],True)):
            if (key,reference) not in tokens:
                tokens[key,reference] = ids(wrapper,canonical[key]['text'],reference)
    def measure(group):
        return evaluate(wrapper.model,[assignment(r,data,tokens,dict(icl=True,non_streaming=True)) for r in group])
    val, fit = measure(selected), measure(probe['rows'])
    result = dict(checkpoint_run=str(args.checkpoint_run),selected_step=selection['step'],
                  adapter_sha256=selection['weights_sha256'],probe_fingerprint=probe['fingerprint'],
                  training_run_sha256=digest(args.training_run/'run.json'),
                  source_sha256=digest(Path(__file__)),validation=val,train_probe=fit,
                  familiar_gap=val['familiar']['combined']-fit['all']['combined'],
                  unseen_gap=val['unseen']['combined']-fit['all']['combined'])
    write(output,result)
    print({k:result[k] for k in ('selected_step','familiar_gap','unseen_gap')},flush=True)


if __name__ == '__main__':
    main()
