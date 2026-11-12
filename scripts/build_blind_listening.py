"""Build a local blind listening page from complete verified evaluation runs."""
import argparse
import json
import os
from pathlib import Path
import random
from urllib.parse import quote

from training.compare_experiments import verified_evaluation
from training.engine import digest, read, write
from training.policy import external
from training.test_protocol import evaluation_folder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, nargs='+', required=True)
    parser.add_argument('--backup', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--cohort',choices=('unseen-validation','unseen-test'),default='unseen-validation')
    args = parser.parse_args()
    output = external(args.output)
    output.mkdir(parents=True, exist_ok=False)
    if not 2 <= len(args.runs) <= 4:
        raise ValueError('Supply two to four runs')
    plans = [verified_evaluation(run, args.seed,args.cohort)[0] for run in args.runs]
    if any(plan != plans[0] for plan in plans[1:]):
        raise ValueError('Plans differ')
    def url(path):
        if not path.is_file():
            raise ValueError(f'Missing recording: {path}')
        return quote(os.path.relpath(path, output).replace('\\', '/'), safe='/.:')
    rng, cases, key = random.Random(7319), [], {}
    audio_root = (args.backup / 'dataset-v1').resolve()
    inventory = read(audio_root / 'dataset-inventory.json')['files']
    folder = evaluation_folder(args.cohort,args.seed)
    for case in plans[0]['cases']:
        reference = (audio_root / case['ref_audio']).resolve()
        if (not reference.is_relative_to(audio_root)
                or digest(reference) != inventory[case['ref_audio']]['sha256']):
            raise ValueError('Reference checksum mismatch')
        order = list(args.runs)
        rng.shuffle(order)
        tracks = []
        for i, run in enumerate(order):
            label = chr(65+i)
            # Neutral symlinks are not required; the page hides run names from
            # its interface, while the separate key preserves reproducibility.
            tracks.append(dict(label=label, audio=url(run / folder / f"{case['case']}.wav")))
        cases.append(dict(case=case['case'], text=case['text'], reference=url(reference), tracks=tracks))
        key[case['case']] = {chr(65+i): str(run) for i, run in enumerate(order)}
    write(output / 'blind-key.json', dict(seed=args.seed, assignments=key))
    payload = json.dumps(cases, ensure_ascii=False).replace('<', '\\u003c')
    page = '''<!doctype html><meta charset="utf-8"><title>Mongolian speech listening</title>
<style>body{font:18px system-ui;max-width:900px;margin:40px auto;padding:20px;background:#f5f6f8;color:#162033}article{background:white;padding:20px;margin:16px 0;border-radius:12px}audio{display:block;width:100%;margin:12px 0}label{display:inline-block;margin:8px}select,button{font:inherit;padding:8px}p{line-height:1.5}textarea{width:95%;min-height:60px}</style>
<h1>Mongolian speech listening</h1><p>Listen to the reference for voice identity, then rate each version. Use 1 (poor) to 5 (excellent). Leave uncertain scores blank. Letters are randomized separately for each case. Do not open the key until ratings are finished.</p>
<label>Case <select id="caseChoice"></select></label><button id="download">Download ratings</button>
<div id="content"></div><p>Ratings stay in this browser until downloaded. These audio links require the original local files.</p>
<script>
const cases = PAYLOAD;
const storageKey='hojil-blind-'+location.pathname;
let ratings;try{ratings=JSON.parse(localStorage.getItem(storageKey)||'{}')}catch(e){ratings={}}
const choice=document.getElementById('caseChoice'), content=document.getElementById('content');
for(const c of cases){const o=document.createElement('option');o.value=c.case;o.textContent=c.case;choice.append(o)}
function save(){try{localStorage.setItem(storageKey,JSON.stringify(ratings))}catch(e){}}
function audio(url){const a=document.createElement('audio');a.controls=true;a.preload='none';a.src=url;return a}
function show(){content.replaceChildren();const c=cases.find(c=>c.case===choice.value);const p=document.createElement('p');p.textContent=c.text;content.append(p);const ref=document.createElement('article');ref.append('Reference voice',audio(c.reference));content.append(ref);
for(const t of c.tracks){const card=document.createElement('article');const h=document.createElement('h2');h.textContent='Version '+t.label;card.append(h,audio(t.audio));const key=c.case+'-'+t.label;ratings[key]??={};
for(const metric of ['Intelligibility','Naturalness','Voice similarity']){const label=document.createElement('label');label.textContent=metric+' ';const s=document.createElement('select');for(const value of ['',1,2,3,4,5]){const o=document.createElement('option');o.value=value;o.textContent=value||'Unrated';s.append(o)}s.value=ratings[key][metric]??'';s.onchange=()=>{ratings[key][metric]=s.value?Number(s.value):null;save()};label.append(s);card.append(label)}
const notes=document.createElement('textarea');notes.placeholder='Mispronounced words, omissions, voice changes…';notes.value=ratings[key].notes||'';notes.oninput=()=>{ratings[key].notes=notes.value;save()};card.append(notes);content.append(card)}}
choice.onchange=show;show();document.getElementById('download').onclick=()=>{const blob=new Blob([JSON.stringify({page:location.pathname,ratings},null,2)],{type:'application/json'});const u=URL.createObjectURL(blob),a=document.createElement('a');a.href=u;a.download='blind-listening-ratings.json';a.click();setTimeout(()=>URL.revokeObjectURL(u),1000)};
</script>'''.replace('PAYLOAD', payload)
    (output / 'index.html').write_text(page, encoding='utf-8')
    print(output / 'index.html')


if __name__ == '__main__':
    main()
