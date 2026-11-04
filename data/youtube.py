"""Local YouTube intake: uploaded captions only, review before corpus export."""
import argparse
import html
import json
import re
import secrets
import shutil
import subprocess
import threading
import uuid
import wave
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from .audio import ffmpeg_executable, inspect
from .io import digest, external, write_json, write_rows
from .pipeline import DEFAULT_FILTERS, normalize, prepare, text_error
from .preprocessing import process, processing_options


def video_url(value):
    u = urlsplit(value.strip())
    if u.scheme != 'https' or u.username or u.password or u.port:
        raise ValueError('Use an HTTPS YouTube video link')
    host = u.hostname
    if host == 'youtu.be':
        identity = u.path.strip('/')
    elif host in {'youtube.com', 'www.youtube.com', 'm.youtube.com'}:
        identity = parse_qs(u.query).get('v', [''])[0] if u.path == '/watch' else (
            u.path.split('/')[2] if re.fullmatch(r'/(shorts|embed)/[^/]+', u.path) else '')
    else:
        raise ValueError('Only individual YouTube videos are supported')
    if not re.fullmatch(r'[A-Za-z0-9_-]{11}', identity):
        raise ValueError('Invalid YouTube video ID')
    return f'https://www.youtube.com/watch?v={identity}'


def captions(payload):
    """Parse JSON3 while retaining overlaps as explicit review/rejection evidence."""
    result = []
    for event in payload.get('events', []):
        text = normalize(html.unescape(''.join(s.get('utf8', '') for s in event.get('segs', []))))
        if not text:
            continue
        start = event.get('tStartMs', -1) / 1000
        duration = event.get('dDurationMs', 0) / 1000
        if start < 0 or duration <= 0:
            raise ValueError('Subtitle cue has invalid timing')
        result.append(dict(start=start, end=start + duration, text=text))
    result.sort(key=lambda r: r['start'])
    if not result:
        raise ValueError('Uploaded subtitle track contains no timed text')
    return result


def acquire(url, language, folder, status, retain_video=False):
    try:
        from yt_dlp import YoutubeDL
        import yt_dlp.version
    except ImportError:
        raise RuntimeError('Install the youtube extra: pip install -e ".[youtube]"') from None
    common = dict(noplaylist=True, quiet=True, socket_timeout=30, retries=3,
                  cachedir=False, ffmpeg_location=ffmpeg_executable())
    if shutil.which('node'):
        common['js_runtimes'] = {'node': {'path': shutil.which('node')}}
    options = common | dict(format='best[height<=720]/best' if retain_video else 'bestaudio/best',
                            writesubtitles=True, writeautomaticsub=False, subtitleslangs=[language],
                            subtitlesformat='json3', outtmpl=str(folder / 'original.%(ext)s'),
                            max_filesize=4 * 1024**3)
    status('Checking uploaded subtitles')
    # Keep extraction and transfer in one session: extracted URLs may depend on
    # the extractor's cookies, visitor identity and request headers.
    with YoutubeDL(options) as ydl:
        info = ydl.extract_info(url, download=False)
        if info.get('_type', 'video') != 'video' or info.get('is_live') or info.get('live_status') == 'is_upcoming':
            raise ValueError('Only completed individual videos are supported')
        if not info.get('duration') or info['duration'] > 4 * 3600:
            raise ValueError('Videos must be at most four hours long')
        # Never consult automatic_captions; translated tracks are excluded as well.
        tracks = {k: v for k, v in info.get('subtitles', {}).items()
                  if any(t.get('ext') == 'json3' and not parse_qs(urlsplit(t.get('url', '')).query).get('tlang')
                         for t in v)}
        if language not in tracks:
            raise ValueError(f'No uploaded {language} JSON3 subtitle track. Available: {", ".join(tracks) or "none"}')
        info['subtitles'] = {language: [t for t in tracks[language] if t.get('ext') == 'json3'
                                      and not parse_qs(urlsplit(t.get('url', '')).query).get('tlang')]}
        status('Downloading media and uploaded subtitles')
        info = ydl.process_ie_result(info, download=True)
        media = Path(ydl.prepare_filename(info))
    subtitle = folder / f'original.{language}.json3'
    if not media.is_file() or not subtitle.is_file():
        raise ValueError('Download did not produce media and manual subtitles')
    provenance = dict(url=url, title=info.get('title'), channel=info.get('channel'),
                      video_id=info['id'], subtitle_language=language, subtitle_type='uploaded',
                      media_sha256=digest(media), subtitle_sha256=digest(subtitle),
                      yt_dlp_version=yt_dlp.version.__version__)
    write_json(folder / 'provenance.json', provenance)
    return media, captions(json.loads(subtitle.read_text(encoding='utf-8'))), provenance


def segment(media, cues, folder, denoise=False):
    decoded = folder / 'decoded.wav'
    subprocess.run([ffmpeg_executable(), '-nostdin', '-v', 'error', '-i', str(media),
                    '-vn', '-ac', '1', '-ar', '24000', '-c:a', 'pcm_s16le', '-n', str(decoded)],
                   check=True, capture_output=True, timeout=1800)
    options = processing_options({'denoise': {'enabled': denoise}, 'trim': {'enabled': True}})
    rows = []
    (folder / 'clips').mkdir()
    with wave.open(str(decoded), 'rb') as src:
        for i, cue in enumerate(cues):
            row = dict(cue, id=f'{i:05d}', approved=False, speaker_id=None, flags=[])
            duration = cue['end'] - cue['start']
            reason = text_error(cue['text'], DEFAULT_FILTERS)
            if not 1 <= duration <= 20:
                reason = 'duration_outside_1_to_20_seconds'
            if (i and cue['start'] < cues[i-1]['end']) or (i+1 < len(cues) and cue['end'] > cues[i+1]['start']):
                reason = 'overlapping_subtitle_cues'
            if cue['end'] > src.getnframes() / 24000 + .05:
                reason = 'subtitle_outside_audio'
            if reason:
                row['rejection'] = reason
            else:
                path = folder / 'clips' / f'{i:05d}.wav'
                src.setpos(round(cue['start'] * 24000))
                with wave.open(str(path), 'wb') as dst:
                    dst.setparams((1, 2, 24000, 0, 'NONE', 'not compressed'))
                    dst.writeframes(src.readframes(round(duration * 24000)))
                before = inspect(path)
                row['original_metrics'] = before
                try:
                    row['processing'] = process(path, options)
                    row['metrics'] = inspect(path)
                    row['audio'] = str(path.relative_to(folder))
                    if max(before['clipping_fraction'], row['metrics']['clipping_fraction']) > .01:
                        row['rejection'] = 'clipping'
                    elif row['metrics']['rms_dbfs'] < -45 or row['metrics']['silence_fraction'] > .8:
                        row['rejection'] = 'quiet_or_silent'
                    elif row['metrics']['duration'] < 1:
                        row['rejection'] = 'too_short_after_trim'
                    row['flags'] = ['check_transcript_and_boundaries', 'check_music_and_speakers']
                except ValueError as exc:
                    row['rejection'] = str(exc)
            rows.append(row)
    write_rows(folder / 'review.jsonl', rows)
    return rows


class Intake:
    def __init__(self, root):
        self.root = external(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.jobs = {}
        self.lock = threading.RLock()
        self.pool = ThreadPoolExecutor(max_workers=1)
        for path in self.root.glob('*/job.json'):
            job = json.loads(path.read_text(encoding='utf-8'))
            if job['state'] in {'queued', 'working', 'exporting'}:
                job.update(state='failed', message='Interrupted by server restart; submit again or inspect saved artifacts')
            self.jobs[job['id']] = job

    def save(self, job):
        write_json(self.root / job['id'] / 'job.json', job)

    def submit(self, body):
        urls = list(dict.fromkeys(video_url(u) for u in body['links'].splitlines() if u.strip()))
        if not 1 <= len(urls) <= 20:
            raise ValueError('Paste between 1 and 20 video links')
        language = body.get('language', 'mn')
        if not re.fullmatch(r'[a-z]{2,3}(?:-[A-Za-z0-9]+)*', language):
            raise ValueError('Invalid subtitle language code')
        clearance = body.get('clearance', '').strip()
        if not clearance:
            raise ValueError('Record your permission/license reference')
        with self.lock:
            if sum(j['state'] in {'queued', 'working'} for j in self.jobs.values()) + len(urls) > 20:
                raise ValueError('Queue is full; wait for current downloads')
            for url in urls:
                identity = uuid.uuid4().hex
                (self.root / identity).mkdir()
                job = dict(id=identity, url=url, language=language, clearance=clearance,
                           denoise=body.get('denoise') is True, retain_video=body.get('retain_video') is True,
                           state='queued', message='Queued', clips=[])
                self.jobs[identity] = job
                self.save(job)
                self.pool.submit(self.run, identity)

    def run(self, identity):
        job = self.jobs[identity]
        def status(message):
            with self.lock:
                job.update(state='working', message=message)
                self.save(job)
        try:
            folder = self.root / identity
            media, cues, provenance = acquire(job['url'], job['language'], folder, status, job['retain_video'])
            status('Converting, screening and trimming subtitle clips')
            clips = segment(media, cues, folder, job['denoise'])
            with self.lock:
                job.update(state='review', message='Review clips before export', clips=clips, provenance=provenance)
        except Exception as exc:
            with self.lock:
                job.update(state='failed', message=str(exc)[:2000])
        with self.lock:
            self.save(job)

    def review(self, identity, changes):
        with self.lock:
            job = self.jobs[identity]
            if job['state'] != 'review':
                raise ValueError('Job is not ready for review')
            clip = next(c for c in job['clips'] if c['id'] == changes['id'])
            if clip.get('rejection'):
                raise ValueError('Rejected clips cannot be approved')
            text = normalize(changes['text'])
            if text_error(text, DEFAULT_FILTERS):
                raise ValueError('Transcript fails Mongolian text validation')
            speaker = changes.get('speaker_id', '').strip() or None
            if speaker and not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', speaker):
                raise ValueError('Use a simple stable speaker label (letters, digits, dash, underscore)')
            clip.update(text=text, speaker_id=speaker, approved=changes.get('approved') is True)
            self.save(job)

    def export(self, identity):
        with self.lock:
            job = self.jobs[identity]
            if job['state'] != 'review':
                raise ValueError('Job is not ready for export')
            selected = [dict(c) for c in job['clips'] if c['approved'] and not c.get('rejection')]
            if not selected:
                raise ValueError('Approve at least one reviewed clip')
            job.update(state='exporting', message='Preparing canonical corpus')
            self.save(job)
        try:
            folder = self.root / identity
            export = folder / ('export-' + uuid.uuid4().hex[:12])
            export.mkdir()
            (export / 'audio').mkdir()
            for clip in selected:
                shutil.copyfile(folder / clip['audio'], export / 'audio' / f'{clip["id"]}.wav')
            # Shared source ID allows stable labels to identify a speaker across videos.
            rows = [dict(id=f'{identity}-{c["id"]}', audio=f'audio/{c["id"]}.wav',
                         text=c['text'], speaker_id=c['speaker_id'], session_id=job['provenance']['video_id'],
                         metadata=dict(provenance=job['provenance'], start=c['start'], end=c['end'],
                                       processing=c.get('processing'), original_metrics=c.get('original_metrics'),
                                       reviewed=True)) for c in selected]
            write_rows(export / 'metadata.jsonl', rows)
            source = dict(id='youtube', kind='manifest', family='youtube', revision=identity,
                          path=str(export), source_url=job['url'], license='See rights_clearance',
                          rights_clearance=job['clearance'])
            config = dict(seed=2026, speaker_disjoint=True, sources=[source])
            write_json(export / 'config.json', config)
            prepare(config, export / 'prepared')
            with self.lock:
                job.update(state='review', message='Exported reviewed clips', export=str(export / 'prepared'))
                self.save(job)
            return str(export / 'prepared')
        except Exception:
            with self.lock:
                job.update(state='review', message='Export failed; inspect retained files and retry')
                self.save(job)
            raise


def serve(root, port=8765):
    intake = Intake(root)
    token = secrets.token_urlsafe(32)
    class Handler(BaseHTTPRequestHandler):
        def send(self, status, body, kind='application/json'):
            raw = json.dumps(body, ensure_ascii=False).encode() if kind == 'application/json' else body
            self.send_response(status)
            self.send_header('Content-Type', kind)
            self.send_header('Content-Length', str(len(raw)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; media-src 'self'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(raw)

        def allowed(self):
            return self.headers.get('Host') == f'127.0.0.1:{port}'

        def do_GET(self):
            if not self.allowed():
                return self.send(403, {'error': 'Invalid host'})
            u = urlsplit(self.path)
            if u.path == '/':
                page = Path(__file__).with_name('youtube.html').read_text(encoding='utf-8').replace('__TOKEN__', token)
                return self.send(200, page.encode(), 'text/html; charset=utf-8')
            if u.path == '/api/jobs':
                with intake.lock:
                    return self.send(200, list(intake.jobs.values()))
            match = re.fullmatch(r'/audio/([a-f0-9]{32})/([0-9]{5})', u.path)
            if match:
                identity, clip_id = match.groups()
                path = intake.root / identity / 'clips' / f'{clip_id}.wav'
                if path.is_file():
                    return self.send(200, path.read_bytes(), 'audio/wav')
            self.send(404, {'error': 'Not found'})

        def do_POST(self):
            if not self.allowed() or self.headers.get('X-Intake-Token') != token:
                return self.send(403, {'error': 'Reload the local intake page'})
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 65536:
                    raise ValueError('Invalid request size')
                body = json.loads(self.rfile.read(size))
                if self.path == '/api/jobs':
                    intake.submit(body)
                    return self.send(202, {'ok': True})
                match = re.fullmatch(r'/api/jobs/([a-f0-9]{32})/(review|export)', self.path)
                if not match:
                    return self.send(404, {'error': 'Not found'})
                identity, action = match.groups()
                result = intake.review(identity, body) if action == 'review' else intake.export(identity)
                self.send(200, {'ok': True, 'output': result})
            except (ValueError, KeyError, StopIteration, TypeError) as exc:
                self.send(400, {'error': str(exc)})
            except Exception as exc:
                self.send(500, {'error': str(exc)[:1000]})

    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    print(f'YouTube intake: http://127.0.0.1:{port}', flush=True)
    print(f'Artifacts: {intake.root}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        intake.pool.shutdown(wait=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, help='Artifact directory outside any Git checkout')
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    serve(args.output, args.port)


if __name__ == '__main__':
    main()
