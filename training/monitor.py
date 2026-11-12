"""Render a portable observability dashboard and CSVs from atomic run metrics."""
import argparse
import csv
import html
import hashlib
import json
from pathlib import Path

from .observability import fit_signals


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def atomic_text(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(value, encoding='utf-8')
    temp.replace(path)


def csv_export(path, rows):
    if not rows:
        return
    keys = sorted({k for row in rows for k, v in row.items() if not isinstance(v, (dict, list))})
    temp = path.with_suffix('.csv.tmp')
    with temp.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def render(run):
    run = Path(run)
    train = [read(p) for p in sorted(run.glob('metrics-*.json'))]
    validation = [read(p) for p in sorted(run.glob('validation-*.json')) if p.stem[11:].isdigit()]
    probes = []
    for row in validation:
        if not row.get('train_probe'):
            continue
        m, t = row['metrics'], row['train_probe']['all']
        probes.append(dict(step=row['step'], train=t['combined'], familiar=m['familiar']['combined'],
            unseen=m['unseen']['combined'], train_first=t['first'], train_residual=t['residual'],
            familiar_first=m['familiar']['first'], familiar_residual=m['familiar']['residual'],
            unseen_first=m['unseen']['first'], unseen_residual=m['unseen']['residual'],
            familiar_gap=row['familiar_generalization_gap'], unseen_gap=row['unseen_generalization_gap'],
            best_step=row['best_step'], bad_checks=row['bad_checks'], epoch_fraction=row['epoch_fraction']))
        for name, values in (row.get('adapter_movement') or {}).items():
            probes[-1]['movement_'+name] = values['relative_delta_l2']
            probes[-1]['unchanged_'+name] = values['unchanged_fraction']
        for name, values in (row.get('overfit_guard') or {}).get('groups', {}).items():
            probes[-1]['overfit_checks_'+name] = values['bad_checks']
    for row in train:
        row['reserved_gib'] = row['peak_reserved']/2**30
        row['allocated_gib'] = row['peak_allocated']/2**30 if 'peak_allocated' in row else None
        for name, value in (row.get('component_gradients') or {}).items():
            row['gradient_'+name] = value
    csv_export(run / 'training-metrics.csv', train)
    csv_export(run / 'fit-probes.csv', probes)
    signals = fit_signals(probes)
    latest = probes[-1] if probes else {}
    result = read(run / 'result.json') if (run / 'result.json').exists() else None
    status = 'Training complete' if result else 'Run in progress or interrupted; refresh to check updates'
    guard = validation[-1].get('overfit_guard') if validation else None
    if guard:
        if guard['stopped']:
            status = 'Stopped: sustained overfit signal in ' + ', '.join(guard['triggered_groups'])
        for name, group in guard['groups'].items():
            signals.append(f"{name.title()} overfit guard: {group['bad_checks']} consecutive checks; "
                           f"validation change {group['latest_validation']-group['validation']:+.5f}, "
                           f"paired training change {group['latest_train']-group['train']:+.5f}.")
    summary = dict(status=status, latest_probe=latest, signals=signals,
                   latest_training=train[-1] if train else None, result=result, overfit_guard=guard)
    speech = read(run/'speech-diagnostics.json') if (run/'speech-diagnostics.json').exists() else None
    if speech:
        summary['speech_summary'] = speech['summary']
    atomic_text(run / 'observability-summary.json', json.dumps(summary, indent=2, allow_nan=False))
    descriptions = [
        ('Comparable evaluation loss', 'probes', ['train','familiar','unseen']),
        ('First-codebook cross entropy', 'probes', ['train_first','familiar_first','unseen_first']),
        ('Residual-codebook cross entropy', 'probes', ['train_residual','familiar_residual','unseen_residual']),
        ('Validation minus matched training loss', 'probes', ['familiar_gap','unseen_gap']),
        ('Training objective (mixed conditioning, dropout on)', 'train', ['window_loss']),
        ('Pre-clipping gradient norms', 'train', ['mean_gradient_norm','gradient_talker_decoder','gradient_residual_predictor','gradient_text_projection']),
        ('Fraction of updates clipped to norm 1', 'train', ['clipping_fraction']),
        ('Learning rate', 'train', ['lr']),
        ('Training codec frames per second (excludes evaluation)', 'train', ['codec_frames_per_second']),
        ('Peak GPU memory (GiB)', 'train', ['reserved_gib','allocated_gib']),
        ('Relative adapter movement since previous probe', 'probes', ['movement_talker_decoder','movement_text_projection','movement_residual_predictor']),
        ('Adapter fraction unchanged since previous probe', 'probes', ['unchanged_talker_decoder','unchanged_text_projection','unchanged_residual_predictor']),
    ]
    payload = json.dumps(dict(train=train, probes=probes, charts=descriptions), allow_nan=False).replace('<','\\u003c')
    overview = ''
    if latest:
        overview = f"Step {latest['step']} · epoch {latest['epoch_fraction']:.1%} · best loss checkpoint {latest['best_step']}"
    table = ''
    if validation:
        m = validation[-1]['metrics']
        if validation[-1].get('train_probe'):
            m = dict(m, train_matched=validation[-1]['train_probe']['all'])
        table = '<table><tr><th>Group</th><th>Clips</th><th>Speakers</th><th>Token-weighted loss</th><th>Speaker mean loss</th></tr>'
        for name in ('train_matched','familiar','unseen'):
            if name in m:
                s = m[name]
                table += f"<tr><td>{name}</td><td>{s['clips']}</td><td>{s['speakers']}</td><td>{s['combined']:.5f}</td><td>{s['speaker_macro_loss']:.5f}</td></tr>"
        table += '</table>'
    if speech:
        table += '<h2>Generated speech: paired development evaluation</h2><table><tr><th>Seed</th><th>Baseline CER</th><th>Candidate CER</th><th>Baseline WER</th><th>Candidate WER</th></tr>'
        for s in speech['per_seed']:
            a,b = s['baseline'],s['candidate']
            table += f"<tr><td>{s['seed']}</td><td>{a['pooled_cer']:.2%}</td><td>{b['pooled_cer']:.2%}</td><td>{a['pooled_wer']:.2%}</td><td>{b['pooled_wer']:.2%}</td></tr>"
        s = speech['summary']
        table += f"</table><p>WER improved / tied / worsened on {s['improved_recordings']} / {s['unchanged_recordings']} / {s['worsened_recordings']} recordings. WER ≥ 50% on {s['baseline_high_error_recordings']} baseline versus {s['candidate_high_error_recordings']} candidate recordings. These are review flags; no recordings are excluded.</p>"
        table += '<p><a href="speech-diagnostics.json">Per-speaker results and largest generation regressions</a>. ASR does not measure naturalness or voice identity.</p>'
    page = '''<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="60">
<title>LoRA training observability</title><style>
body{font:16px system-ui;margin:30px auto;padding:0 22px;max-width:1500px;background:#f4f6fa;color:#182239}h1{margin-bottom:4px}p{line-height:1.5}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(450px,1fr));gap:18px}.card{background:white;border:1px solid #dbe2ec;border-radius:12px;padding:16px}svg{width:100%;height:auto}.legend{font-size:12px;display:flex;flex-wrap:wrap;gap:12px}.note{color:#506078}table{border-collapse:collapse;margin:20px 0;width:100%}td,th{text-align:left;padding:10px;border-bottom:1px solid #dbe2ec}input,button{font:inherit;padding:6px}a{color:#2359b9}li{margin:8px 0}@media(max-width:520px){.grid{display:block}.card{margin:12px 0}}
</style><h1>LoRA training observability</h1><p>RUN_NAME</p><strong>STATUS</strong><p>OVERVIEW</p>
<ul>SIGNALS</ul><p class="note">Fixed train/validation probes use evaluation mode, no dropout, and identical non-streaming ICL conditioning. The training probe matches familiar speakers and target lengths. Unseen speakers still differ in distribution. These trends do not prove underfitting or overfitting; confirm with generated speech. Loss = first-codebook CE + 0.3 × residual CE.</p>
TABLE
<p><label>Show steps from <input id="start" type="number" min="0" value="0" step="100"></label> <button onclick="draw()">Update charts</button> · Hover points for values. Page refreshes every 60 seconds.</p>
<p><a href="training-metrics.csv">Training CSV</a> · <a href="fit-probes.csv">Probe CSV</a> · <a href="observability-summary.json">Summary JSON</a> · <a href="best.json">Selected checkpoint</a></p>
<div class="grid" id="charts"></div>
<script>
const data=PAYLOAD,colors=['#1965c8','#d47712','#159475','#ad43a8','#bc3d46'];
const ns='http://www.w3.org/2000/svg';function el(name,attrs={},text){const e=document.createElementNS(ns,name);for(const[k,v]of Object.entries(attrs))e.setAttribute(k,v);if(text!==undefined)e.textContent=text;return e}
function draw(){const root=document.getElementById('charts');root.replaceChildren();const start=Number(document.getElementById('start').value)||0;try{localStorage.setItem('probe-start-'+location.pathname,String(start))}catch(e){}
for(const[title,source,keys]of data.charts){const rows=data[source].filter(r=>r.step>=start),card=document.createElement('section');card.className='card';const h=document.createElement('h3');h.textContent=title;card.append(h);root.append(card);
const values=rows.flatMap(r=>keys.map(k=>r[k]).filter(v=>typeof v==='number'&&Number.isFinite(v)));if(!values.length){card.append('Waiting for data');continue}
const W=640,H=270,L=70,R=20,T=20,B=40;let lo=Math.min(...values),hi=Math.max(...values);const pad=(hi-lo||Math.abs(hi)*.1||1)*.08;lo-=pad;hi+=pad;const xmin=Math.min(...rows.map(r=>r.step)),xmax=Math.max(...rows.map(r=>r.step));const x=v=>L+(v-xmin)/(xmax-xmin||1)*(W-L-R),y=v=>T+(hi-v)/(hi-lo)*(H-T-B);const svg=el('svg',{viewBox:`0 0 ${W} ${H}`,role:'img','aria-label':title});card.append(svg);
for(let i=0;i<5;i++){const v=lo+(hi-lo)*i/4,Y=y(v);svg.append(el('line',{x1:L,x2:W-R,y1:Y,y2:Y,stroke:'#e3e8f0'}));svg.append(el('text',{x:L-8,y:Y+4,'text-anchor':'end',fill:'#617087','font-size':12},Math.abs(v)<.001?v.toExponential(1):v.toFixed(3)))}
svg.append(el('text',{x:L,y:H-10,fill:'#617087','font-size':12},String(xmin)));svg.append(el('text',{x:W-R,y:H-10,'text-anchor':'end',fill:'#617087','font-size':12},String(xmax)));const legend=document.createElement('div');legend.className='legend';card.append(legend);
keys.forEach((key,i)=>{const points=rows.filter(r=>typeof r[key]==='number'&&Number.isFinite(r[key]));const color=colors[i%colors.length];svg.append(el('polyline',{points:points.map(r=>`${x(r.step)},${y(r[key])}`).join(' '),fill:'none',stroke:color,'stroke-width':2}));for(const r of points){const dot=el('circle',{cx:x(r.step),cy:y(r[key]),r:3,fill:color});dot.append(el('title',{},`${key}: ${r[key]} at step ${r.step}`));svg.append(dot)}const label=document.createElement('span');label.style.color=color;label.textContent=key;legend.append(label)})}}
try{document.getElementById('start').value=localStorage.getItem('probe-start-'+location.pathname)||0}catch(e){}draw();
</script>'''
    replacements = {'RUN_NAME':html.escape(run.name), 'STATUS':html.escape(status),
                    'OVERVIEW':html.escape(overview), 'SIGNALS':''.join('<li>'+html.escape(s)+'</li>' for s in signals),
                    'TABLE':table}
    for key, value in replacements.items():
        page = page.replace(key, value)
    page = page.replace('PAYLOAD', payload)
    if result:
        page = page.replace('<meta http-equiv="refresh" content="60">','')
        page = page.replace('Page refreshes every 60 seconds.','Training complete; this report is a saved snapshot.')
    atomic_text(run / 'observability.html', page)
    atomic_text(run/'report-provenance.json',json.dumps(dict(renderer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        training_source_preserved_in='source/',includes_speech_diagnostics=bool(speech)),indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    from .policy import external
    print(json.dumps(render(external(args.run)), indent=2))


if __name__ == '__main__':
    main()
