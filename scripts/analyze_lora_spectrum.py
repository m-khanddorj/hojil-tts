"""Measure adapter rank utilization without forming full-size weight updates.

This diagnoses learned update spectra; it does NOT determine the best rank for
speech or measure singular values of the frozen pretrained model.
"""
import argparse
from collections import defaultdict
from pathlib import Path
import statistics

from training.engine import digest, write
from training.policy import external


def spectrum(a, b):
    import torch
    # B A = Qb (Rb Ra^T) Qa^T, with orthonormal Q. Nonzero singular
    # values therefore equal those of the small rank-by-rank middle matrix.
    _, rb = torch.linalg.qr(b.double(), mode='reduced')
    _, ra = torch.linalg.qr(a.double().T, mode='reduced')
    values = torch.linalg.svdvals(rb @ ra.T)
    energy = values.square()
    if energy.sum() == 0:
        return dict(rank90=0, rank99=0, stable_rank=0.0, frobenius_norm=0.0)
    cumulative = energy.cumsum(0) / energy.sum()
    return dict(rank90=int((cumulative < .9).sum()) + 1,
                rank99=int((cumulative < .99).sum()) + 1,
                stable_rank=(energy.sum()/energy.max()).item(),
                frobenius_norm=energy.sum().sqrt().item())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--adapter', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import torch
    torch.set_num_threads(2)
    state = torch.load(args.adapter, map_location='cpu', weights_only=True)
    layers, groups = [], defaultdict(list)
    for key, a in state.items():
        if '.lora_A.' not in key:
            continue
        b = state[key.replace('.lora_A.', '.lora_B.')]
        item = dict(layer=key.split('.lora_A.')[0], rank=a.shape[0], **spectrum(a, b))
        layers.append(item)
        group = ('residual_predictor' if key.startswith('code_predictor.') else
                 'text_projection' if key.startswith('text_projection.') else 'talker_decoder')
        groups[group].append(item)
    if not layers:
        raise ValueError('No LoRA adapters found')
    summary = {name: dict(layers=len(items),
                         median_rank90=statistics.median(x['rank90'] for x in items),
                         median_rank99=statistics.median(x['rank99'] for x in items),
                         median_stable_rank=statistics.median(x['stable_rank'] for x in items))
               for name, items in groups.items()}
    write(external(args.output), dict(adapter_sha256=digest(args.adapter), groups=summary, layers=layers,
          interpretation='Learned adapter update spectrum only; concentrated energy is not proof that lower rank preserves speech quality'))
    print(summary)


if __name__ == '__main__':
    main()
