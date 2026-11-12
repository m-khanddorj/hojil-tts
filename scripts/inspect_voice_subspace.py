"""Training-only weight-space diagnostic; does not edit or select a model."""
import argparse
from collections import defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.engine import digest, read, rows, save, write
from training.policy import external


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import torch
    torch.set_num_threads(2)
    parent = args.parent.resolve()
    protocol = read(parent/'protocol.json')
    cfg = protocol['config']
    out = external(args.output.resolve())
    out.mkdir(exist_ok=False)
    backup, features = Path(cfg['backup']), Path(cfg['features'])
    train_path = backup/'cohorts/train.jsonl'
    if digest(train_path) != protocol['feature_signature']['train_sha256']:
        raise ValueError('Training cohort changed')
    train = {r['id']:r['speaker_id'] for r in rows(train_path)}
    sums, counts, seen = {}, defaultdict(int), set()
    for name, expected in protocol['feature_shards'].items():
        path = features/name
        if digest(path) != expected:
            raise ValueError('Feature shard changed')
        for row in torch.load(path, map_location='cpu', weights_only=True):
            if row['id'] not in train or row['id'] in seen:
                raise ValueError('Unexpected feature ID')
            seen.add(row['id'])
            speaker = train[row['id']]
            vector = row['xvec'].float().flatten()
            if not torch.isfinite(vector).all():
                raise ValueError('Nonfinite speaker embedding')
            sums[speaker] = sums.get(speaker, torch.zeros_like(vector)) + vector
            counts[speaker] += 1
    if seen != set(train):
        raise ValueError('Incomplete training features')
    means = torch.stack([sums[s]/counts[s] for s in sorted(sums)])
    centered = means - means.mean(0)
    _, singular, vh = torch.linalg.svd(centered, full_matrices=False)
    basis = vh[:32].T.contiguous()
    basis_path = out/'training-speaker-basis.pt'
    save(basis_path, dict(basis=basis, speaker_mean=means.mean(0), singular_values=singular,
        speakers=len(sums), clips=len(seen), train_sha256=digest(train_path)))
    checkpoint = parent/'checkpoint-000044.pt'
    receipt = read(parent/'checkpoint-000044.json')
    if digest(checkpoint) != receipt['sha256']:
        raise ValueError('Checkpoint changed')
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)['adapter']
    a = state['text_projection.linear_fc2.lora_A.weight'].float()
    b = state['text_projection.linear_fc2.lora_B.weight'].float()
    if b.shape[0] != basis.shape[0]:
        raise ValueError('Speaker and projection output spaces differ')
    update = b @ a
    denominator = update.square().sum()
    if denominator <= 0:
        raise ValueError('Zero update')
    results = []
    generator = torch.Generator().manual_seed(20261110)
    for rank in (1, 4, 8, 16, 32):
        u = basis[:, :rank]
        alignment = float((u.T @ update).square().sum()/denominator)
        random = []
        for _ in range(64):
            q, _ = torch.linalg.qr(torch.randn(b.shape[0], rank, generator=generator), mode='reduced')
            random.append(float((q.T @ update).square().sum()/denominator))
        rand = torch.tensor(random)
        results.append(dict(rank=rank, speaker_variance_fraction=float(singular[:rank].square().sum()/singular.square().sum()),
            update_energy_fraction_in_speaker_subspace=alignment,
            random_subspace_mean=float(rand.mean()), random_subspace_95=rand.quantile(torch.tensor([.025,.975])).tolist(),
            note='Geometric association only, not proof these directions cause perceived voice drift'))
    result = dict(training_speakers=len(sums), training_clips=len(seen),
        module='text_projection.linear_fc2', checkpoint_sha256=receipt['sha256'],
        basis_sha256=digest(basis_path), source_sha256=digest(Path(__file__)),
        source_protocol_sha256=digest(parent/'protocol.json'), results=results,
        interpretation='Equal weight per training speaker; no development/custom recordings used; no weights edited')
    write(out/'results.json', result)
    print(result, flush=True)


if __name__ == '__main__':
    main()
