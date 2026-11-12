"""Build the portable visual report; generated HTML belongs outside the checkout.

Usage: python scripts/build_visual_report.py --output <external-directory>/index.html
Requires Markdown 3.11.1 (report tooling only; no training dependencies).
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import html
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import shutil
import wave
from urllib.parse import unquote, urlsplit

import markdown

REPO = Path(__file__).resolve().parents[1]
DOCS = REPO / 'docs/experiments/0.6b'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def doc_id(path):
    return 'evidence-' + re.sub(r'[^a-z0-9]+', '-', path.relative_to(REPO).as_posix().lower()).strip('-')


def discover():
    queue = [DOCS / 'report.md', DOCS / 'README.md', *sorted((DOCS / 'evidence').glob('*.md'))]
    found = []
    while queue:
        path = queue.pop(0).resolve()
        if path in found:
            continue
        if not path.is_relative_to(REPO) or not path.is_file():
            continue
        found.append(path)
        for target in re.findall(r'\]\(([^\s)]+)(?:\s+[^)]*)?\)', path.read_text(encoding='utf-8-sig')):
            parsed = urlsplit(target.strip('<>'))
            if not parsed.scheme and parsed.path.endswith('.md'):
                queue.append((path.parent / unquote(parsed.path)).resolve())
    return found


class LinkRewriter(HTMLParser):
    def __init__(self, path, included):
        super().__init__(convert_charrefs=False)
        self.path, self.included, self.parts = path, included, []

    def handle_starttag(self, tag, attrs):
        output = []
        for key, value in attrs:
            if key == 'id':
                value = doc_id(self.path) + '--' + value
            if key == 'href' and value:
                url = urlsplit(value)
                if not url.scheme:
                    target = (self.path.parent / unquote(url.path)).resolve() if url.path else self.path
                    if target in self.included:
                        # Python-Markdown collapses repeated hyphens in heading slugs.
                        fragment = re.sub(r'-+', '-', url.fragment)
                        value = '#' + doc_id(target) + ('--' + fragment if fragment else '')
                    elif target.exists():
                        value = target.as_uri() + ('#' + url.fragment if url.fragment else '')
            output.append(key if value is None else f'{key}="{html.escape(value, quote=True)}"')
        self.parts.append('<' + tag + (' ' + ' '.join(output) if output else '') + '>')

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)

    def handle_endtag(self, tag):
        self.parts.append(f'</{tag}>')

    def handle_data(self, data):
        self.parts.append(data)

    def handle_entityref(self, name):
        self.parts.append('&' + name + ';')

    def handle_charref(self, name):
        self.parts.append('&#' + name + ';')


def build(output):
    output = output.resolve()
    if output.is_relative_to(REPO):
        raise ValueError('Generated reports must stay outside Git; choose an external output directory.')
    data_path = DOCS / 'visual-data.json'
    data = json.loads(data_path.read_text(encoding='utf-8-sig'))
    for source in data['sources'].values():
        path = Path(source['path'])
        if not path.is_absolute():
            path = REPO / path
        if digest(path) != source['sha256']:
            raise ValueError(f'Source changed since chart-data verification: {path}')
    paths = discover()
    documents, manifest = [], []
    for path in paths:
        source = path.read_text(encoding='utf-8-sig')
        title = next((s.lstrip('# ').strip() for s in source.splitlines() if s.startswith('# ')), path.stem)
        parser = LinkRewriter(path, set(paths))
        parser.feed(markdown.markdown(source, extensions=['tables', 'fenced_code', 'toc', 'sane_lists']))
        identity = doc_id(path)
        documents.append(f'<details class="evidence-doc" id="{identity}"><summary><span>{html.escape(title)}</span>'
                         f'<span class="doc-path">{html.escape(path.relative_to(REPO).as_posix())}</span></summary>'
                         f'<article class="prose">{"".join(parser.parts)}</article></details>')
        manifest.append(dict(id=identity, title=title, path=path.relative_to(REPO).as_posix(), sha256=digest(path)))
    data['document_manifest'] = manifest
    data['visual_data_sha256'] = digest(data_path)
    audio_manifest = json.loads((DOCS / 'visual-audio.json').read_text(encoding='utf-8-sig'))
    audio_dir = output.parent / 'audio'
    audio_dir.mkdir(parents=True, exist_ok=True)
    audio_sections = []
    for group in audio_manifest['groups']:
        clips = [r for r in audio_manifest['clips'] if r['group_id'] == group['id']]
        cards = []
        for clip in clips:
            original = Path(clip['path'])
            if digest(original) != clip['audio_sha256']:
                raise ValueError(f'Audio identity changed: {original}')
            filename = clip['bundle_filename']
            if Path(filename).name != filename or not filename.endswith('.wav'):
                raise ValueError('Unsafe audio bundle filename')
            target = audio_dir / filename
            shutil.copyfile(original, target)
            if digest(target) != clip['audio_sha256']:
                raise ValueError('Audio copy changed')
            with wave.open(str(target), 'rb') as wav:
                if wav.getframerate() != clip['sample_rate'] or wav.getnchannels() != clip['channels']:
                    raise ValueError('Audio header differs from manifest')
            name = clip['label'].replace('Full epoch-2 adapted model', 'Epoch-2 LoRA endpoint')
            scores = '' if clip['cer'] is None else (
                f'<div class="audio-score"><span>CER {clip["cer"]*100:.2f}%</span>'
                f'<span>WER {clip["wer"]*100:.2f}%</span><span>Cosine {clip["cosine"]:.6f}</span></div>')
            tag = 'Human reference' if clip['kind'] == 'human_reference' else f'Seed {clip["seed"]} · individual clip'
            embedded_audio = base64.b64encode(target.read_bytes()).decode('ascii')
            cards.append(f'<div class="audio-card"><span class="badge">{tag}</span><h3>{html.escape(name)}</h3>'
                         f'<audio controls preload="metadata" aria-label="{html.escape(name, quote=True)}">'
                         f'<source src="data:audio/wav;base64,{embedded_audio}" type="audio/wav">'
                         'Your browser does not support HTML audio.</audio>'
                         f'{scores}<div class="small">{clip["seconds"]:.2f} seconds · {clip["sample_rate"]//1000} kHz</div>'
                         f'<details class="chart-details"><summary>Transcript & evidence</summary>'
                         f'<p lang="mn">{html.escape(clip["text"])}</p><p>{html.escape(clip["metric_scope"])}</p>'
                         f'<code style="font-size:10px;overflow-wrap:anywhere">SHA256 {clip["audio_sha256"]}</code></details></div>')
        common_text = clips[0]['text'] if clips and len({r['text'] for r in clips}) == 1 else None
        quote = f'<p lang="mn" style="font-size:17px;margin:13px 0">“{html.escape(common_text)}”</p>' if common_text else ''
        audio_sections.append(f'<div class="audio-group"><h3>{html.escape(group["title"])}</h3>'
                              f'<p class="small">{html.escape(group["note"])}</p>{quote}'
                              f'<div class="audio-grid{" paired" if len(clips)==4 else ""}">{"".join(cards)}</div></div>')
    # Escape HTML-significant characters inside the inert JSON script element.
    encoded = json.dumps(data, ensure_ascii=False, allow_nan=False).replace('&', '\\u0026').replace('<', '\\u003c').replace('>', '\\u003e')
    template = (DOCS / 'visual-report.html').read_text(encoding='utf-8')
    result = template.replace('<!--REPORT_DATA-->', encoded).replace('<!--EVIDENCE_LIBRARY-->', '\n'.join(documents)).replace('<!--AUDIO_LIBRARY-->', '\n'.join(audio_sections))
    if '<!--REPORT_DATA-->' in result or '<!--EVIDENCE_LIBRARY-->' in result:
        raise ValueError('Unreplaced report placeholders')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(result, encoding='utf-8')
    (output.parent / 'audio-manifest.json').write_text(json.dumps(audio_manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(dict(output=str(output), bytes=output.stat().st_size, embedded_documents=len(paths),
                          sha256=digest(output), visual_data_sha256=digest(data_path))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    build(parser.parse_args().output)
