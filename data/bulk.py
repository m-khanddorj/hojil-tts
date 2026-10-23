"""Parallel, resumable cleaning; split/leakage decisions remain corpus-wide."""
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import hashlib
import json
import threading
import time
import wave
from pathlib import Path

from .audio import inspect
from .io import digest, external, write_json
from .pipeline import finalize, prepare_record, validated_config
from .preprocessing import check_backend
from .sources import records


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def atomic_json(path, value):
    temp = path.with_suffix('.tmp')
    write_json(temp, value)
    # Windows readers/antivirus can briefly prevent replacing an open status file.
    # Retry the atomic rename; never fall back to a partially written live file.
    for attempt in range(20):
        try:
            temp.replace(path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(min(.02 * (attempt + 1), .25))


def verify_pcm(path, expected):
    try:
        actual = inspect(path)['pcm_sha256']
    except (OSError, EOFError, wave.Error, ValueError) as exc:
        raise ValueError(f'Checkpoint audio is unreadable: {path}') from exc
    if actual != expected:
        raise ValueError(f'Checkpoint audio checksum mismatch: {path}')


def prepare_bulk(config, output, workers=4, resume=False):
    if type(workers) is not int or not 1 <= workers <= 16:
        raise ValueError('workers must be between 1 and 16')
    processing, filters = validated_config(config)
    check_backend(processing)
    out = external(output)
    items, seen = [], set()
    metadata = {}
    for source in config['sources']:
        if not source.get('enabled', True):
            continue
        root = Path(source['path']).expanduser().resolve()
        names = {'train.tsv', 'dev.tsv', 'test.tsv', 'validated.tsv',
                 source.get('manifest', 'metadata.jsonl'), 'acquisition.json', 'source.json'}
        metadata[source['id']] = {name: digest(root/name) for name in sorted(names) if (root/name).is_file()}
        for row in records(source):
            if row['id'] in seen:
                raise ValueError(f"Duplicate record ID: {row['id']}")
            seen.add(row['id'])
            items.append((source, row))
    implementation = {p.name: digest(p) for p in Path(__file__).parent.glob('*.py')}
    identity = dict(config=config, metadata=metadata, implementation=implementation,
                    inputs_sha256=fingerprint([row for _, row in items]), rows=len(items))
    state_file = out/'run-state.json'
    if resume:
        if not (out/'audio').is_dir() or not (out/'checkpoints').is_dir():
            raise ValueError('Resume requires the original audio and checkpoint directories')
        if not state_file.is_file() or json.loads(state_file.read_text(encoding='utf-8'))['identity'] != identity:
            raise ValueError('Resume requires identical config, source metadata, input rows, and implementation')
        progress_file = out/'progress.json'
        complete = progress_file.is_file() and json.loads(progress_file.read_text(encoding='utf-8'))['status'] == 'complete'
        if (out/'report.json').is_file() and complete:
            report = json.loads((out/'report.json').read_text(encoding='utf-8'))
            if digest(out/'manifest.jsonl') != report['manifest_sha256']:
                raise ValueError('Completed manifest checksum mismatch')
            from .io import read_rows
            for row in read_rows(out/'manifest.jsonl'):
                verify_pcm(row['audio'], row['pcm_sha256'])
            return report
    else:
        out.mkdir(parents=True, exist_ok=False)
        (out/'audio').mkdir()
        (out/'checkpoints').mkdir()
        atomic_json(state_file, dict(identity=identity, started=datetime.now(timezone.utc).isoformat()))
    local = threading.local()
    started = time.monotonic()
    completed, reused, accepted, rejected = 0, 0, [], []

    def clean(item):
        source, row = item
        key = hashlib.sha256(row['id'].encode()).hexdigest()
        checkpoint = out/'checkpoints'/(key+'.json')
        input_hash = fingerprint(dict(source=source, row=row))
        if checkpoint.exists():
            saved = json.loads(checkpoint.read_text(encoding='utf-8'))
            if saved['input_sha256'] != input_hash:
                raise ValueError('Checkpoint input mismatch')
            result = saved['row']
            if result.get('original_audio_sha256'):
                if digest(row['audio']) != result['original_audio_sha256']:
                    raise ValueError(f"Source audio changed: {row['id']}")
            if 'rejection' in result:
                return result, True
            target = Path(result['audio'])
            if target.is_file():
                verify_pcm(target, result['pcm_sha256'])
                return result, True
        # Only this deterministic generated path can be replaced after interruption.
        target = out/'audio'/(key+'.wav')
        if target.parent.resolve() != (out/'audio').resolve():
            raise ValueError('Invalid generated audio path')
        target.unlink(missing_ok=True)
        if not hasattr(local, 'neural'):
            local.neural = check_backend(processing)
        result = prepare_record(source, row, out, filters, processing, local.neural)
        atomic_json(checkpoint, dict(input_sha256=input_hash, row=result))
        return result, False

    def progress(status, detail=None):
        elapsed = time.monotonic() - started
        value = dict(status=status, total=len(items), processed=completed, reused=reused,
                     accepted_before_global_split=len(accepted), rejected_before_global_split=len(rejected),
                     elapsed_seconds=round(elapsed, 1), workers=workers,
                     updated=datetime.now(timezone.utc).isoformat())
        if detail:
            value['detail'] = detail
        atomic_json(out/'progress.json', value)

    progress('running')
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            iterator = iter(items)
            pending = set()
            for _ in range(workers * 2):
                item = next(iterator, None)
                if item is not None:
                    pending.add(pool.submit(clean, item))
            last_progress = 0
            while pending:
                done, pending = wait(pending, timeout=10, return_when=FIRST_COMPLETED)
                for future in done:
                    result, cached = future.result()
                    completed += 1
                    reused += cached
                    (rejected if 'rejection' in result else accepted).append(result)
                    item = next(iterator, None)
                    if item is not None:
                        pending.add(pool.submit(clean, item))
                if completed % 25 == 0 or time.monotonic() - last_progress >= 10:
                    progress('running')
                    last_progress = time.monotonic()
        progress('finalizing')
        report = finalize(config, out, accepted, list(rejected), filters, processing)
        progress('complete')
        return report
    except BaseException as exc:
        progress('failed', f'{type(exc).__name__}: {exc}')
        raise
