"""Optional offline ASR review. Never replaces or automatically rejects source text."""
import math
import subprocess
import tempfile
import unicodedata
from pathlib import Path

from .audio import ffmpeg_executable, inspect
from .io import digest, fresh_dir, read_rows, write_json
from .pipeline import text_key


def character_error_rate(reference, hypothesis):
    reference, hypothesis = text_key(reference), text_key(hypothesis)
    if not reference:
        raise ValueError('Empty normalized reference')
    previous = list(range(len(hypothesis) + 1))
    for i, left in enumerate(reference, 1):
        current = [i]
        for j, right in enumerate(hypothesis, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j-1] + (left != right)))
        previous = current
    return previous[-1] / len(reference)


def transcript_flags(reference, hypothesis, timestamps, duration, threshold=.3):
    """Separate recognizer failures from possible errors in the supplied text."""
    flags = []
    letters = [c for c in hypothesis if c.isalpha()]
    if not letters:
        flags.append('asr_unreliable_empty')
    elif sum('CYRILLIC' in unicodedata.name(c, '') for c in letters) / len(letters) < .6:
        flags.append('asr_unreliable_script')
    words = hypothesis.casefold().split()
    if len(words) >= 10 and len(set(words)) <= max(2, len(words)//10):
        flags.append('asr_unreliable_repetition')
    if any(s['end'] > duration + .5 or s['start'] < 0 or s['end'] < s['start'] for s in timestamps):
        flags.append('asr_unreliable_timestamps')
    if not flags and character_error_rate(reference, hypothesis) > threshold:
        flags.append('possible_transcript_mismatch')
    return flags


def audit_transcripts(manifest, model_dir, output, device='cpu', threshold=.3,
                      backend='faster-whisper', executable=None):
    if not math.isfinite(threshold) or not 0 <= threshold <= 2:
        raise ValueError('CER threshold must be between 0 and 2')
    model_dir = Path(model_dir).expanduser().resolve()
    acquisition = model_dir / 'acquisition.json'
    if not acquisition.is_file():
        raise ValueError('Local ASR model requires acquisition.json with pinned file hashes')
    import json
    state = json.loads(acquisition.read_text(encoding='utf-8'))
    if not state.get('revision') or not state.get('files'):
        raise ValueError('ASR acquisition record requires revision and file hashes')
    from .io import child
    for item in state['files']:
        if digest(child(model_dir, item['file'])) != item['sha256']:
            raise ValueError('ASR model checksum mismatch')
    hashed_files = {item['file'] for item in state['files']}
    if backend == 'faster-whisper':
        required = {'config.json', 'model.bin', 'tokenizer.json', 'vocabulary.txt'}
        if not required <= hashed_files:
            raise ValueError('ASR acquisition record must hash all required model files')
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError('Install a working asr extra or use the native whisper-cpp backend') from exc
        model = WhisperModel(str(model_dir), device=device,
                             compute_type='int8' if device == 'cpu' else 'float16', local_files_only=True)
    elif backend == 'whisper-cpp':
        if device != 'cpu':
            raise ValueError('whisper-cpp audit currently supports CPU only')
        exe = Path(executable or '').expanduser().resolve()
        if not exe.is_relative_to(model_dir) or str(exe.relative_to(model_dir)) not in hashed_files:
            raise ValueError('ASR executable must be included in the acquisition hashes')
        if state.get('model_file') not in hashed_files:
            raise ValueError('ASR acquisition record must identify and hash model_file')
        model_path = child(model_dir, state['model_file'])
    else:
        raise ValueError('Unknown ASR backend')
    out = fresh_dir(output)
    count, flagged = 0, 0
    # Write per clip so a long audit leaves inspectable partial results if interrupted.
    with (out / 'transcript-audit.jsonl').open('w', encoding='utf-8') as f:
        for row in read_rows(manifest):
            if inspect(row['audio'])['pcm_sha256'] != row['pcm_sha256']:
                raise ValueError('Audio changed since preparation')
            if backend == 'faster-whisper':
                segments, _ = model.transcribe(row['audio'], language='mn', beam_size=5,
                                           condition_on_previous_text=False, vad_filter=False,
                                           temperature=0, word_timestamps=True)
                segments = list(segments)
                hypothesis = ' '.join(s.text.strip() for s in segments)
                timestamps = [dict(word=w.word, start=w.start, end=w.end, probability=w.probability)
                              for s in segments for w in (s.words or [])]
            else:
                with tempfile.TemporaryDirectory(prefix='asr-', dir=out) as temp:
                    wav = Path(temp) / 'input.wav'
                    prefix = Path(temp) / 'transcription'
                    try:
                        subprocess.run([ffmpeg_executable(), '-nostdin', '-v', 'error', '-i', row['audio'],
                                        '-ac', '1', '-ar', '16000', '-c:a', 'pcm_s16le', str(wav)],
                                       capture_output=True, check=True)
                        subprocess.run([str(exe), '-m', str(model_path), '-f', str(wav), '-l', 'mn',
                                        '-ng', '-oj', '-of', str(prefix), '-t', '4', '-bs', '5'],
                                       capture_output=True, check=True)
                    except subprocess.CalledProcessError as exc:
                        raise RuntimeError('ASR failed: ' + exc.stderr.decode(errors='replace')[:1000]) from exc
                    segments = json.loads(prefix.with_suffix('.json').read_text(encoding='utf-8'))['transcription']
                    hypothesis = ' '.join(s['text'].strip() for s in segments)
                    timestamps = [dict(text=s['text'], start=s['offsets']['from']/1000,
                                       end=s['offsets']['to']/1000) for s in segments]
            cer = character_error_rate(row['text'], hypothesis)
            flags = transcript_flags(row['text'], hypothesis, timestamps, row.get('duration', inspect(row['audio'])['duration']), threshold)
            result = dict(id=row['id'], audio=row['audio'], pcm_sha256=row['pcm_sha256'],
                          source_text=row['text'], asr_text=hypothesis, cer=cer, review_flags=flags,
                          asr_timestamps=timestamps,
                          policy='review_only_no_automatic_transcript_replacement')
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
            f.flush()
            count += 1
            flagged += bool(flags)
    report = dict(rows=count, flagged=flagged, cer_threshold=threshold, language='mn', backend=backend,
                  model_provenance=state, input_manifest_sha256=digest(manifest),
                  policy='review_only', limitations=['ASR errors can falsely flag correct Mongolian',
                                                     'ASR word timestamps are not source-text forced alignment'])
    write_json(out / 'report.json', report)
    return report
