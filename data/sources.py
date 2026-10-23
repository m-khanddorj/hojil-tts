"""Adapters preserve source partitions and never infer speaker identity from gender."""
import csv
import json
import re
import tempfile
from pathlib import Path

from .io import child, digest, download, extract, external, write_json, write_rows


def validate_source(s):
    if s['kind'] not in {'common_voice', 'fleurs', 'mbspeech', 'manifest'}:
        raise ValueError('Unknown source kind')
    if not re.fullmatch(r'[a-zA-Z0-9_-]+', s['id']):
        raise ValueError('Source ID must be a simple identifier')
    for key in ('revision', 'license', 'source_url'):
        if not s.get(key):
            raise ValueError(f'Missing provenance: {key}')
    if s['kind'] == 'common_voice' and s['revision'] != '27.0':
        raise ValueError('Common Voice adapter requires the selected release 27.0')
    if s['kind'] in {'fleurs', 'mbspeech'} and not re.fullmatch('[0-9a-f]{40}', s['revision']):
        raise ValueError('Pin FLEURS/MBSpeech to a full repository commit SHA')


def acquire(s, root):
    validate_source(s)
    root = external(root)
    dest = root / s['id']
    if dest.exists():
        raise ValueError(f'Acquisition destination already exists: {dest}')
    if s['kind'] == 'mbspeech' and not s.get('rights_clearance'):
        raise ValueError('MBSpeech acquisition is quarantined pending rights_clearance evidence')
    root.mkdir(parents=True, exist_ok=True)
    cache = root / '.acquisition-cache' / s['id']
    cache.mkdir(parents=True, exist_ok=True)
    journal = cache / 'journal.json'
    state = json.loads(journal.read_text()) if journal.exists() else dict(source=s, files={})
    if state['source'] != s:
        raise ValueError('Acquisition configuration changed; use a new output root')
    def persist():
        temp = journal.with_suffix('.tmp')
        write_json(temp, state)
        temp.replace(journal)
    persist()
    def fetch(name, url, expected=None):
        p = cache / name
        known = state['files'].get(name)
        if known:
            if not p.is_file() or digest(p) != known['sha256'] or (expected and expected != known['sha256']):
                raise ValueError('Cached download checksum mismatch; inspect cache before retrying')
        else:
            download(url, p, expected)
            state['files'][name] = dict(file=name, sha256=digest(p), url=url)
            persist()
        return p
    files = []
    # Failed extraction never publishes a half-populated source tree. Completed
    # downloads survive in a journaled cache and are rehashed on the next attempt.
    with tempfile.TemporaryDirectory(prefix=s['id'] + '-extract-', dir=root) as tmp:
        stage = Path(tmp) / 'source'
        if s['kind'] == 'fleurs':
            stage.mkdir()
            import shutil
            for split in ('train', 'dev', 'test'):
                name = f'{split}.tsv'
                url = f"https://huggingface.co/datasets/google/fleurs/resolve/{s['revision']}/data/mn_mn/{name}"
                p = fetch(name, url)
                shutil.copyfile(p, stage / name)
                files.append(state['files'][name])
                name = f'audio/{split}.tar.gz'
                url = f"https://huggingface.co/datasets/google/fleurs/resolve/{s['revision']}/data/mn_mn/{name}"
                p = fetch(name, url)
                files.append(state['files'][name])
                extract(p, stage / 'audio' / split)
        elif s.get('archive'):
            p = Path(s['archive']).expanduser().resolve()
            if not s.get('archive_sha256') or digest(p) != s['archive_sha256']:
                raise ValueError('Local archive needs a matching archive_sha256')
            extract(p, stage)
            files.append(dict(file=str(p), sha256=digest(p)))
        elif s.get('download_url'):
            if not s.get('archive_sha256') or not s.get('authorized_download'):
                raise ValueError('Download needs checksum and explicit authorized_download=true')
            p = fetch('source.tar.gz', s['download_url'], s['archive_sha256'])
            extract(p, stage)
            files.append(state['files']['source.tar.gz'])
        else:
            raise ValueError('Provide a checksummed local archive or authorized public archive URL')
        stage.mkdir(exist_ok=True)
        write_json(stage / 'acquisition.json', dict(source=s, files=files))
        stage.rename(dest)


def base(s, clip, text, audio, split, speaker=None, **extra):
    return dict(id=f"{s['id']}:{clip}", source=s['id'], source_family=s.get('family', s['kind']),
                source_url=s['source_url'], revision=s['revision'], license=s['license'],
                rights_clearance=s.get('rights_clearance'), original_clip_id=str(clip),
                original_text=text, audio=str(audio.resolve()), source_split=split,
                speaker_id=f"{s['id']}:{speaker}" if speaker else None, **extra)


def records(s):
    validate_source(s)
    root = Path(s['path']).expanduser().resolve()
    if s['kind'] == 'common_voice':
        # validated.tsv includes dev/test: reserve their paths before ingesting it.
        reserved = {}
        for file, split in [('dev.tsv', 'validation'), ('test.tsv', 'test')]:
            with (root / file).open(encoding='utf-8', newline='') as f:
                for row in csv.DictReader(f, delimiter='\t'):
                    if row['path'] in reserved and reserved[row['path']] != split:
                        raise ValueError('Clip occurs in both CV dev and test')
                    reserved[row['path']] = split
        seen = set()
        for file in ('test.tsv', 'dev.tsv', 'validated.tsv'):
            with (root / file).open(encoding='utf-8', newline='') as f:
                for row in csv.DictReader(f, delimiter='\t'):
                    clip = row['path']
                    if clip in seen:
                        continue
                    seen.add(clip)
                    yield base(s, clip, row['sentence'], child(root / 'clips', clip),
                               reserved.get(clip, 'train'), row.get('client_id') or None,
                               source_metadata={'up_votes': row.get('up_votes'), 'down_votes': row.get('down_votes')})
    elif s['kind'] == 'fleurs':
        for file, split in [('train', 'train'), ('dev', 'validation'), ('test', 'test')]:
            paths = {}
            for path in (root / 'audio' / file).rglob('*'):
                if path.is_file():
                    paths.setdefault(path.name, []).append(path)
            with (root / f'{file}.tsv').open(encoding='utf-8', newline='') as f:
                # FLEURS is literal TSV: quotation marks belong to transcripts.
                for row in csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE):
                    if not row:
                        continue
                    if len(row) != 7:
                        raise ValueError('FLEURS expects seven headerless TSV columns')
                    # Archives may contain a split directory or bare filenames.
                    if Path(row[1]).name != row[1] or any(c in row[1] for c in '*?[]'):
                        raise ValueError('FLEURS filename must be a plain basename')
                    matches = paths.get(row[1], [])
                    if len(matches) != 1:
                        raise ValueError(f'Expected one FLEURS audio path for {row[1]}')
                    yield base(s, f'{file}/{row[1]}', row[2], matches[0], split,
                               source_metadata={'sentence_id': row[0], 'normalized_text': row[3], 'gender': row[6]})
    else:
        # MBSpeech local interchange: original transcripts and source identifiers required.
        # No fabricated global speaker ID. Explicitly reviewed speaker IDs may be supplied.
        from .io import read_rows
        for row in read_rows(root / s.get('manifest', 'metadata.jsonl')):
            yield base(s, row['id'], row['text'], child(root, row['audio']),
                       row.get('split', 'unspecified'), row.get('speaker_id'),
                       session_id=row.get('session_id'), source_metadata=row.get('metadata', {}))


def import_parquet(source, inputs, output):
    """Explicit optional converter for locally obtained MBSpeech Parquet distributions."""
    validate_source(source)
    if source['kind'] != 'mbspeech':
        raise ValueError('Parquet import currently supports MBSpeech only')
    if not source.get('rights_clearance'):
        raise ValueError('MBSpeech import quarantined pending rights_clearance evidence')
    import pyarrow.parquet as pq
    from .io import fresh_dir
    out = fresh_dir(output)
    (out / 'audio').mkdir()
    rows = []
    for path in sorted(map(Path, inputs)):
        file_hash = digest(path)
        index = 0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=64):
            for row in batch.to_pylist():
                clip = f'{file_hash[:16]}-{index}'
                index += 1
                audio = row['audio']
                if not isinstance(audio, dict) or not audio.get('bytes'):
                    raise ValueError('Expected embedded Hugging Face audio.bytes in Parquet')
                target = out / 'audio' / f'{clip}.wav'
                target.write_bytes(audio['bytes'])
                text = row.get('original_sentence') or row.get('sentence_orig') or row.get('sentence') or row.get('text')
                if not isinstance(text, str):
                    raise ValueError('Expected original_sentence, sentence_orig, sentence, or text column')
                rows.append(dict(id=clip, audio=str(target.relative_to(out)), text=text,
                                 split=row.get('split', 'unspecified'), speaker_id=row.get('speaker_id'),
                                 metadata={'parquet_sha256': file_hash, 'row': index-1}))
    write_rows(out / 'metadata.jsonl', rows)
    write_json(out / 'source.json', source)
