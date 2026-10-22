"""Optional official single-speaker export; language adaptation needs a separate trainer."""
import json
import os
from pathlib import Path

from .audio import inspect
from .io import digest, external, fresh_dir, read_rows, write_json, write_rows

UPSTREAM = '022e286b98fbec7e1e916cb940cdf532cd9f488e'


def export(manifest, speaker, output):
    rows = list(read_rows(manifest))
    selected = [r for r in rows if r['speaker_id'] == speaker]
    if not speaker or not selected:
        raise ValueError('Select an existing, known source-qualified speaker_id')
    train = [r for r in selected if r['split'] == 'train']
    if not train:
        raise ValueError('Selected speaker has no training rows; never repurpose held-out speech')
    reference = sorted(train, key=lambda r: r['id'])[0]['audio']
    inspect(reference)
    out = fresh_dir(output)
    for split in ('train', 'validation', 'test'):
        result = []
        for r in selected:
            if r['split'] == split:
                if inspect(r['audio'])['pcm_sha256'] != r['pcm_sha256']:
                    raise ValueError('Canonical audio changed since preparation')
                result.append(dict(audio=str(Path(r['audio']).resolve()), text=r['text'], ref_audio=reference))
        write_rows(out / f'{split}_raw.jsonl', result)
    write_json(out / 'export.json', dict(upstream_commit=UPSTREAM, speaker=speaker,
               manifest_sha256=digest(manifest), reference=reference,
               reference_sha256=digest(reference), purpose='optional-single-speaker-refinement',
               excluded_rows=len(rows)-len(selected), selected_rows=len(selected)))


def validate_codes(row):
    codes = row.get('audio_codes')
    if not isinstance(codes, list) or not codes or any(
        not isinstance(frame, list) or len(frame) != 16 or any(type(x) is not int or x < 0 for x in frame)
        for frame in codes
    ):
        raise ValueError('Expected nonempty audio_codes with shape T x 16 nonnegative integers')
    for key in ('audio', 'ref_audio', 'text'):
        if not row.get(key):
            raise ValueError(f'Missing {key}')


def encode(input_path, output_path, model_path, device='cpu', batch_size=1):
    """Explicit offline tokenizer execution, bounded batches, per-batch checkpoints."""
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    model = Path(model_path).expanduser().resolve()
    if not model.is_dir() or not (model / 'config.json').is_file():
        raise ValueError('Provide an already-downloaded local Qwen3-TTS-Tokenizer-12Hz directory')
    output = external(output_path)
    if output.exists():
        raise ValueError('Completed output exists')
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + '.partial')
    checkpoint = output.with_suffix(output.suffix + '.checkpoint.json')
    model_hashes = {str(p.relative_to(model)): digest(p) for p in sorted(model.rglob('*')) if p.is_file()}
    media_hashes = {}
    count = 0
    for row in read_rows(input_path):
        count += 1
        if not isinstance(row.get('text'), str) or not row['text'].strip():
            raise ValueError('Expected nonempty text')
        for key in ('audio', 'ref_audio'):
            value = row.get(key)
            if not isinstance(value, str) or not Path(value).is_absolute():
                raise ValueError('Code extraction requires absolute audio and known-speaker reference paths')
            if value not in media_hashes:
                media_hashes[value] = digest(value)
    if not count:
        raise ValueError('Empty tokenizer input')
    signature = dict(input_sha256=digest(input_path), model_files=model_hashes,
                     media_files=media_hashes, device=device)
    completed = 0
    if checkpoint.exists():
        state = json.loads(checkpoint.read_text())
        if state['signature'] != signature:
            raise ValueError('Resume input/model/device differs from checkpoint')
        completed = state['rows']
        if not partial.exists() or partial.stat().st_size < state['bytes']:
            raise ValueError('Checkpoint output missing/truncated')
        # Discard an uncommitted tail after an interrupted batch.
        with partial.open('r+b') as f:
            f.truncate(state['bytes'])
    elif partial.exists():
        raise ValueError('Partial output has no checkpoint; inspect and move it before restarting')
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    from qwen_tts import Qwen3TTSTokenizer
    tokenizer = Qwen3TTSTokenizer.from_pretrained(str(model), device_map=device)
    batch = []
    with partial.open('ab') as f:
        def flush():
            nonlocal completed
            encoded = tokenizer.encode([r['audio'] for r in batch])
            if len(encoded.audio_codes) != len(batch):
                raise ValueError('Tokenizer output count mismatch')
            for row, code in zip(batch, encoded.audio_codes):
                row['audio_codes'] = code.cpu().tolist()
                validate_codes(row)
                f.write((json.dumps(row, ensure_ascii=False) + '\n').encode())
            f.flush()
            os.fsync(f.fileno())
            completed += len(batch)
            temp = checkpoint.with_suffix('.tmp')
            write_json(temp, dict(signature=signature, rows=completed, bytes=f.tell(), upstream_commit=UPSTREAM))
            temp.replace(checkpoint)
            batch.clear()
        for i, row in enumerate(read_rows(input_path)):
            if i < completed:
                continue
            for key in ('audio', 'ref_audio'):
                if not Path(row[key]).is_absolute():
                    raise ValueError('Qwen audio paths must be absolute')
                inspect(row[key])
            batch.append(row)
            if len(batch) == batch_size:
                flush()
        if batch:
            flush()
    partial.rename(output)
