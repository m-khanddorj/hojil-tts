"""Measure reference-voice similarity on verified, existing speech evaluations.

Uses the original Qwen speaker encoder on CPU. This internal diagnostic is not
an independent perceptual score and must not replace listening or CER/WER.
"""
import argparse
from collections import defaultdict
import math
from pathlib import Path
from statistics import mean


def summarize(records):
    if not records:
        raise ValueError('No speaker measurements supplied')
    speakers, seeds = defaultdict(list), defaultdict(list)
    seen = set()
    for row in records:
        key = row['seed'], row['case']
        value = row['cosine']
        if key in seen:
            raise ValueError('Duplicate recording')
        if not math.isfinite(value) or not -1.000001 <= value <= 1.000001:
            raise ValueError('Invalid speaker cosine')
        seen.add(key)
        speakers[row['speaker_id']].append(value)
        seeds[row['seed']].append(value)
    per_speaker = {key: mean(values) for key, values in speakers.items()}
    return dict(recordings=len(records), speakers=len(speakers),
                mean_cosine=mean(row['cosine'] for row in records),
                speaker_macro_cosine=mean(per_speaker.values()),
                per_speaker=per_speaker,
                per_seed={str(key): mean(values) for key, values in seeds.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True, help='Original frozen Qwen base model')
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--backup', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[2026, 2027])
    parser.add_argument('--cohort', choices=('unseen-validation', 'unseen-test'), default='unseen-validation')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if not args.seeds or len(set(args.seeds)) != len(args.seeds) or args.threads < 1:
        raise ValueError('Supply unique seeds and a positive CPU thread count')

    import numpy as np
    import soundfile as sf
    import torch
    from qwen_tts import Qwen3TTSModel
    from .compare_experiments import verified_evaluation
    from .engine import digest, read, write
    from .policy import external
    from .test_protocol import evaluation_folder

    run = external(args.run)
    evaluations, template = [], None
    for seed in args.seeds:
        plan, scores = verified_evaluation(run, seed, args.cohort)
        settings = {key: value for key, value in plan.items() if key != 'seed'}
        if template is not None and template != settings:
            raise ValueError('Evaluation settings/cases differ across seeds')
        template = settings
        evaluations.append((plan, scores))
    dataset = (args.backup / 'dataset-v1').resolve()
    inventory = read(dataset / 'dataset-inventory.json')['files']
    torch.set_num_threads(args.threads)
    wrapper = Qwen3TTSModel.from_pretrained(str(args.base), device_map='cpu',
        torch_dtype=torch.float32, attn_implementation='sdpa', local_files_only=True)
    wrapper.model.eval()
    references, records = {}, []

    def embedding(path):
        waveform, sr = sf.read(path, dtype='float32')
        if (sr != wrapper.model.speaker_encoder_sample_rate or waveform.ndim != 1
                or not len(waveform) or not np.isfinite(waveform).all()):
            raise ValueError('Invalid speaker-diagnostic waveform')
        with torch.inference_mode():
            value = wrapper.model.extract_speaker_embedding(audio=waveform, sr=sr).detach().cpu().flatten()
        if not torch.isfinite(value).all() or value.norm().item() == 0:
            raise ValueError('Invalid speaker embedding')
        return value

    for plan, scores in evaluations:
        cases = {case['case']: case for case in plan['cases']}
        folder = run / evaluation_folder(args.cohort, plan['seed'])
        for score in scores:
            case = cases[score['case']]
            reference = (dataset / case['ref_audio']).resolve()
            if not reference.is_relative_to(dataset):
                raise ValueError('Reference escaped dataset')
            reference_sha = inventory[case['ref_audio']]['sha256']
            if reference not in references:
                if digest(reference) != reference_sha:
                    raise ValueError('Reference checksum mismatch')
                references[reference] = embedding(reference)
            audio = folder / score['wav']
            cosine = torch.nn.functional.cosine_similarity(
                references[reference][None], embedding(audio)[None]).item()
            records.append(dict(case=case['case'], seed=plan['seed'], speaker_id=case['speaker_id'],
                cosine=cosine, audio=str(audio), audio_sha256=score['audio_sha256'],
                reference_audio=str(reference), reference_sha256=reference_sha))

    report = dict(summary=summarize(records), recordings=records, cohort=args.cohort,
        run=str(run), encoder_base=str(args.base.resolve()),
        encoder_model_sha256=digest(args.base / 'model.safetensors'),
        generation_model_sha256=read(run / 'merged-model/training-provenance.json')['model_sha256'],
        plans={str(plan['seed']): digest(run / evaluation_folder(args.cohort, plan['seed']) / 'plan.json')
               for plan, _ in evaluations},
        source_sha256=digest(Path(__file__)), forward_dtype='float32', device='cpu',
        interpretation='Internal Qwen speaker cosine; not an independent perceptual score, not a percentage, and not a checkpoint promotion criterion.')
    filename = 'speaker-diagnostics.json' if args.cohort == 'unseen-validation' else 'speaker-test-diagnostics.json'
    write(run / filename, report)
    print(report['summary'])


if __name__ == '__main__':
    main()
