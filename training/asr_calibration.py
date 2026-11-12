"""Measure the pinned evaluator on real development speech; never tune the ASR."""
import argparse
import os
from pathlib import Path

from .engine import digest, read, rows, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    protocol = read(args.protocol)
    cfg = protocol['config']
    os.environ['HF_HUB_OFFLINE'] = os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['NUMBA_CACHE_DIR'] = cfg['numba_cache']
    os.environ['TEMP'] = os.environ['TMP'] = cfg['tmp']
    import torch
    import torchaudio
    import soundfile as sf
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    from .evaluate import ASR_REV, pooled
    from .policy import external
    from scripts.score_latest_audio import score
    torch.set_num_threads(2)
    out = external(args.output.resolve())
    out.mkdir(exist_ok=False)
    backup = Path(cfg['backup'])
    cohort = backup/'cohorts/unseen-validation.jsonl'
    if digest(cohort) != read(backup/'recovery-verification.json')['cohorts']['unseen-validation']['sha256']:
        raise ValueError('Changed development cohort')
    inventory = read(backup/'dataset-v1/dataset-inventory.json')['files']
    cases = rows(cohort)
    if len(cases) != 44:
        raise ValueError('Unexpected full development size')
    panel = {c['row_id'] for c in protocol['cases'] if c['group']=='development'}
    if len(panel) != 10 or not panel <= {r['id'] for r in cases}:
        raise ValueError('Screening panel mismatch')
    asr = Path(cfg['asr'])
    manifest = read(asr/'download-manifest.json')
    if manifest['repo'] != 'BuzzASR/mongolian' or manifest['revision'] != ASR_REV:
        raise ValueError('ASR revision mismatch')
    for name, item in manifest['files'].items():
        if digest(asr/name) != item['sha256']:
            raise ValueError('ASR file changed')
    write(out/'protocol.json', dict(source_sha256=digest(Path(__file__)),
        parent_protocol_sha256=digest(args.protocol), cohort_sha256=digest(cohort),
        asr_manifest_sha256=digest(asr/'download-manifest.json'), decoding=protocol['decoding'],
        scope='Real development audio, all 44 clips; existing 10-case screen reported separately. No test data.',
        interpretation='Evaluator calibration, not a mathematical CER floor and not a changed goal.'))
    model = WhisperForConditionalGeneration.from_pretrained(asr, torch_dtype=torch.float32,
        local_files_only=True).to('cuda').eval()
    processor = WhisperProcessor.from_pretrained(asr, local_files_only=True)
    records = []
    for row in cases:
        path = backup/'dataset-v1'/row['audio']
        expected = inventory[row['audio']]['sha256']
        if digest(path) != expected:
            raise ValueError('Real recording changed')
        wave, sr = sf.read(path, dtype='float32')
        wave = torchaudio.functional.resample(torch.from_numpy(wave), sr, 16000).numpy()
        batch = processor(wave, sampling_rate=16000, return_tensors='pt', return_attention_mask=True).to('cuda')
        with torch.inference_mode():
            generated = model.generate(batch.input_features, attention_mask=batch.attention_mask, **protocol['decoding'])
        hypothesis = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
        result = dict(id=row['id'], speaker_id=row['speaker_id'], text=row['text'],
            audio=str(path), audio_sha256=expected, hypothesis=hypothesis,
            score=score(row['text'], hypothesis, True), in_screening_panel=row['id'] in panel)
        records.append(result)
        print(f'Real speech calibration: {len(records)}/{len(cases)}', flush=True)
    result = dict(records=records, all_development=pooled([r['score'] for r in records]),
        fixed_screen=pooled([r['score'] for r in records if r['in_screening_panel']]))
    write(out/'results.json', result)
    print({k:v for k,v in result.items() if k!='records'}, flush=True)


if __name__ == '__main__':
    main()
