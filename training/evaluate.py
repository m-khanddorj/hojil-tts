"""Export a selected LoRA checkpoint and compare speech with verified epoch one.

Run from the repository root; uses the existing scoring script and local models.
The development plan stays fixed. Reserved test evaluation requires --cohort unseen-test.
"""
import argparse
from collections import defaultdict
import gc
import os
from pathlib import Path
import random
import shutil
import time

from .engine import digest, inject, load_adapter, read, rows, write
from .policy import external, validate_cohorts
from .test_protocol import build_test_plan, evaluation_folder
from scripts.score_latest_audio import score

ASR_REV = '185f1bf1f562ad4b1bdbfc71811fb165a9235416'


def export(base, run):
    import torch
    from peft.tuners.lora import LoraLayer
    from qwen_tts import Qwen3TTSModel
    from safetensors.torch import save_file
    selection, provenance = read(run / 'best.json'), read(run / 'run.json')
    if provenance['config']['method'] != 'lora':
        raise ValueError('This export path is for LoRA')
    weights = run / selection['weights']
    if (digest(weights) != selection['weights_sha256']
            or digest(base / 'model.safetensors') != provenance['base_weights']
            or base.name != provenance['features']['base_revision']):
        raise ValueError('Selected adapter or base weights changed')
    signature = {'selection': selection, 'training_run_sha256': digest(run / 'run.json')}
    merged = run / 'merged-model'
    if merged.exists():
        saved = read(merged / 'training-provenance.json')
        if saved['signature'] != signature or saved['model_sha256'] != digest(merged / 'model.safetensors'):
            raise ValueError('Existing merged model differs from selection')
        return merged
    stage = run / 'merged-model-staging'
    stage.mkdir(exist_ok=False)
    wrapper = Qwen3TTSModel.from_pretrained(str(base), device_map='cpu',
                torch_dtype=torch.bfloat16, attn_implementation='sdpa', local_files_only=True)
    talker = wrapper.model.talker
    inject(talker, provenance['config']['rank'], 0)
    load_adapter(talker, torch.load(weights, map_location='cpu', weights_only=True))
    # Match the historical BF16 merge policy; training master weights stay FP32.
    talker.to(torch.bfloat16)
    for name, layer in list(talker.named_modules()):
        if isinstance(layer, LoraLayer):
            layer.merge(safe_merge=True)
            talker.set_submodule(name, layer.get_base_layer())
    save_file({k: v.detach().cpu().contiguous() for k, v in wrapper.model.state_dict().items()},
              stage / 'model.safetensors', metadata={'format': 'pt'})
    for path in base.iterdir():
        if path.is_file() and path.suffix in {'.json', '.txt'} and path.name != 'model.safetensors.index.json':
            shutil.copy2(path, stage / path.name)
    shutil.copytree(base / 'speech_tokenizer', stage / 'speech_tokenizer')
    write(stage / 'training-provenance.json', {'signature': signature,
                                             'model_sha256': digest(stage / 'model.safetensors')})
    stage.rename(merged)
    return merged


def pooled(scores):
    if not scores or sum(s['words'] for s in scores) == 0 or sum(s['characters'] for s in scores) == 0:
        raise ValueError('Empty evaluation')
    return {'pooled_cer': sum(s['character_errors'] for s in scores)/sum(s['characters'] for s in scores),
            'pooled_wer': sum(s['word_errors'] for s in scores)/sum(s['words'] for s in scores),
            'mean_cer': sum(s['cer'] for s in scores)/len(scores),
            'mean_wer': sum(s['wer'] for s in scores)/len(scores)}


def compare(baseline, candidate, cases, baseline_name='epoch_one'):
    old = {r['case']: r for r in baseline}
    new = {r['case']: r for r in candidate}
    if len(old) != len(baseline) or len(new) != len(candidate) or set(old) != set(new) or set(old) != {c['case'] for c in cases}:
        raise ValueError('Incomplete paired evaluation')
    clusters = defaultdict(lambda: [0, 0, 0])
    for case in cases:
        a, b = old[case['case']]['without_punctuation'], new[case['case']]['without_punctuation']
        if a['words'] != b['words'] or a['characters'] != b['characters']:
            raise ValueError('Score denominators differ')
        counts = clusters[case['speaker_id']]
        for i, v in enumerate((a['word_errors'], b['word_errors'], a['words'])):
            counts[i] += v
    rng, deltas = random.Random(2026), []
    for _ in range(4000):
        sample = rng.choices(list(clusters.values()), k=len(clusters))
        deltas.append(sum(b-a for a, b, _ in sample)/sum(n for _, _, n in sample))
    deltas.sort()
    a, b = pooled([r['without_punctuation'] for r in baseline]), pooled([r['without_punctuation'] for r in candidate])
    return {baseline_name: a, 'candidate': b, 'wer_delta': b['pooled_wer']-a['pooled_wer'],
            'cer_delta': b['pooled_cer']-a['pooled_cer'],
            'wer_delta_speaker_bootstrap_95': [deltas[100], deltas[3899]],
            'speakers': len(clusters), 'clips': len(cases),
            'interpretation': 'Development ASR proxy; one seed; not a naturalness or test-set claim'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('base', 'run', 'epoch-one', 'backup', 'asr'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--cohort', choices=('unseen-validation','unseen-test'), default='unseen-validation')
    parser.add_argument('--baseline-evaluation', type=Path,
                        help='Compare against another verified run with the identical plan/seed')
    args = parser.parse_args()
    run = external(args.run)
    output = run / evaluation_folder(args.cohort, args.seed)
    output.mkdir(exist_ok=True)
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio
    from qwen_tts import Qwen3TTSModel
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    torch.set_num_threads(4)
    prior = args.epoch_one / 'validation'
    plan = read(prior / 'plan.json')
    if (plan['cohort'] != 'unseen-validation' or len(plan['cases']) != 44
            or plan['cohort_sha256'] != digest(args.backup / 'cohorts/unseen-validation.jsonl')):
        raise ValueError('Unexpected original validation plan')
    if args.cohort == 'unseen-test':
        plan = build_test_plan(args.backup, plan)
    train = rows(args.backup / 'cohorts/train.jsonl')
    heldout = rows(args.backup / f'cohorts/{args.cohort}.jsonl')
    validate_cohorts(train, heldout, rows(args.backup / 'dataset-v1/manifest.jsonl'),
                     heldout_split='test' if args.cohort == 'unseen-test' else 'validation')
    if {r['speaker_id'] for r in heldout} & {r['speaker_id'] for r in train}:
        raise ValueError('Unseen speakers overlap training')
    for case, row in zip(plan['cases'], heldout, strict=True):
        if any(case[k] != row[k] for k in ('id', 'text', 'speaker_id', 'audio', 'ref_id', 'ref_audio', 'ref_text')):
            raise ValueError('Validation plan changed')
    historical_seed = plan['seed']
    plan['seed'] = args.seed
    if (output / 'plan.json').exists() and read(output / 'plan.json') != plan:
        raise ValueError('Existing evaluation plan differs')
    write(output / 'plan.json', plan)
    original_model = read(args.epoch_one / 'merged-model/training-provenance.json')['model_sha256']
    baseline = []
    for case in plan['cases'] if args.cohort == 'unseen-validation' else []:
        row = read(prior / 'transcriptions' / f"finetuned-{case['case']}.json")
        if (row['audio_sha256'] != digest(prior / row['wav']) or row['reference'] != case['text']
                or row['asr_revision'] != ASR_REV or row['model_sha256'] != original_model
                or row['decoding'] != plan['decoding'] or row['case'] != case['case']):
            raise ValueError('Historical score provenance mismatch')
        baseline.append(row)
    if args.seed != historical_seed:
        baseline = []  # Never compare different generation seeds as a matched pair.
    baseline_name = 'epoch_one'
    if args.baseline_evaluation:
        baseline_name = 'baseline'
        prior_eval = args.baseline_evaluation
        if read(prior_eval / 'plan.json') != plan:
            raise ValueError('Baseline evaluation must use the identical plan and seed')
        baseline = read(prior_eval / 'scores.json')
        baseline_run = prior_eval.parent
        expected_model = read(baseline_run / 'merged-model/training-provenance.json')['model_sha256']
        if digest(baseline_run / 'merged-model/model.safetensors') != expected_model:
            raise ValueError('Baseline model checksum changed')
        cases = {c['case']: c for c in plan['cases']}
        for row in baseline:
            if (row['case'] not in cases or row['reference'] != cases[row['case']]['text']
                    or row['model_sha256'] != expected_model or row['asr_revision'] != ASR_REV
                    or row['decoding'] != plan['decoding']
                    or row['audio_sha256'] != digest(prior_eval / row['wav'])):
                raise ValueError('Baseline evaluation provenance mismatch')
    modeldir = export(args.base, run)
    gc.collect()
    model_sha = digest(modeldir / 'model.safetensors')
    if args.cohort == 'unseen-test':
        provenance = dict(cohort=args.cohort, model_sha256=model_sha,
            selection_sha256=digest(run/'best.json'), plan_sha256=digest(output/'plan.json'),
            source={name:digest(Path(__file__).parent/name) for name in
                    ('evaluate.py','test_protocol.py','policy.py')})
        receipt = output/'evaluation-provenance.json'
        if receipt.exists() and read(receipt) != provenance:
            raise ValueError('Reserved test evaluation provenance changed')
        write(receipt, provenance)
    audio_root = (args.backup / 'dataset-v1').resolve()
    inventory = read(audio_root / 'dataset-inventory.json')['files']
    wrapper, prompts, recordings = None, {}, []
    for case in plan['cases']:
        wav_path = output / f"{case['case']}.wav"
        meta = output / f"recording-{case['case']}.json"
        if meta.exists():
            record = read(meta)
            if record['model_sha256'] != model_sha or record['audio_sha256'] != digest(wav_path):
                raise ValueError('Changed generated recording')
        else:
            if wrapper is None:
                wrapper = Qwen3TTSModel.from_pretrained(str(modeldir), device_map='cuda:0',
                    torch_dtype=torch.bfloat16, attn_implementation='sdpa', local_files_only=True)
                wrapper.model.eval()
            reference = (audio_root / case['ref_audio']).resolve()
            if not reference.is_relative_to(audio_root) or digest(reference) != inventory[case['ref_audio']]['sha256']:
                raise ValueError('Changed reference audio')
            if case['ref_id'] not in prompts:
                prompts[case['ref_id']] = wrapper.create_voice_clone_prompt(ref_audio=str(reference),
                    ref_text=case['ref_text'], x_vector_only_mode=False)
            torch.manual_seed(plan['seed'])
            started = time.monotonic()
            with torch.inference_mode():
                waves, sr = wrapper.generate_voice_clone(text=case['text'],
                    voice_clone_prompt=prompts[case['ref_id']], **plan['generation'])
            wave = waves[0]
            if not len(wave) or not np.isfinite(wave).all() or len(wave)/sr > 30:
                raise ValueError('Malformed synthesized audio')
            sf.write(wav_path, wave, sr, subtype='PCM_16')
            record = {'case': case['case'], 'model_sha256': model_sha, 'wav': wav_path.name,
                      'audio_sha256': digest(wav_path), 'seconds': len(wave)/sr,
                      'generation_seconds': time.monotonic()-started}
            write(meta, record)
        recordings.append(record)
        print(f"Speech {case['case']}/{len(plan['cases']):03d}", flush=True)
    del wrapper, prompts
    gc.collect()
    torch.cuda.empty_cache()
    asr_manifest = read(args.asr / 'download-manifest.json')
    if asr_manifest['repo'] != 'BuzzASR/mongolian' or asr_manifest['revision'] != ASR_REV:
        raise ValueError('Unexpected ASR revision')
    for name, info in asr_manifest['files'].items():
        if digest(args.asr / name) != info['sha256']:
            raise ValueError('Changed ASR weights/tokenizer')
    processor = WhisperProcessor.from_pretrained(args.asr, local_files_only=True)
    model = WhisperForConditionalGeneration.from_pretrained(args.asr, torch_dtype=torch.float32,
                local_files_only=True).to('cuda').eval()
    scored = []
    for record, case in zip(recordings, plan['cases'], strict=True):
        path = output / f"score-{case['case']}.json"
        if path.exists():
            result = read(path)
            if (result['audio_sha256'] != record['audio_sha256'] or result['asr_revision'] != ASR_REV
                    or result['decoding'] != plan['decoding'] or result['reference'] != case['text']
                    or result['model_sha256'] != model_sha or result['case'] != case['case']):
                raise ValueError('Changed cached ASR score')
        else:
            wave, sr = sf.read(output / record['wav'], dtype='float32')
            wave = torchaudio.functional.resample(torch.from_numpy(wave), sr, 16000).numpy()
            inputs = processor(wave, sampling_rate=16000, return_tensors='pt', return_attention_mask=True).to('cuda')
            with torch.inference_mode():
                generated = model.generate(inputs.input_features, attention_mask=inputs.attention_mask, **plan['decoding'])
            hypothesis = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
            result = {**record, 'asr_revision': ASR_REV, 'decoding': plan['decoding'],
                      'reference': case['text'], 'hypothesis': hypothesis,
                      'strict': score(case['text'], hypothesis, False),
                      'without_punctuation': score(case['text'], hypothesis, True)}
            write(path, result)
        scored.append(result)
        print(f"ASR {case['case']}/{len(plan['cases']):03d}", flush=True)
    write(output / 'scores.json', scored)
    comparison = (compare(baseline, scored, plan['cases'], baseline_name) if baseline else
                  {'candidate': pooled([r['without_punctuation'] for r in scored]),
                   'interpretation': 'Unpaired development evaluation; no same-seed baseline supplied'})
    comparison['seed'] = args.seed
    comparison['baseline_evaluation'] = str(args.baseline_evaluation) if args.baseline_evaluation else None
    comparison['selected_step'] = read(run / 'best.json')['step']
    if args.cohort == 'unseen-test':
        comparison['interpretation'] = 'Reserved unseen-speaker test ASR proxy; fixed checkpoint and generation settings; not a perceptual-quality verdict'
    write(output / 'comparison.json', comparison)
    print(comparison, flush=True)


if __name__ == '__main__':
    main()
