"""Audited inference-strength search; never resumes or labels it as training."""
import argparse
import copy
import html
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from . import targeted_voice as pilot
from . import reference_protected as protected


def scaled_adapter(state, strength):
    import math
    if not math.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError('Strength must be finite and in [0, 1]')
    if len(state) != 4 or sum('.lora_B.' in k for k in state) != 2:
        raise ValueError('Expected exactly two projection LoRA A/B pairs')
    # Scale B once, giving strength * BA (not strength squared).
    return {k: v.detach().cpu().clone() * (strength if '.lora_B.' in k else 1)
            for k, v in state.items()}


def goal_screen(summary):
    import math
    if set(summary) != {'custom', 'development'}:
        raise ValueError('Both evaluation groups are required')
    return all(math.isfinite(s['pooled_cer']) and math.isfinite(s['speaker_macro_cosine'])
        and 0 <= s['pooled_cer'] < 0.03 and 0.984 < s['speaker_macro_cosine'] <= 1
        for s in summary.values())


def verify_protocol(run):
    protocol = pilot.read(run/'protocol.json')
    parent = Path(protocol['parent_run'])
    if pilot.digest(Path(__file__)) != protocol['driver_sha256']:
        raise ValueError('Search source changed')
    if pilot.digest(parent/'protocol.json') != protocol['parent_protocol_sha256']:
        raise ValueError('Parent protocol changed')
    original = pilot.read(parent/'protocol.json')
    if original['source'] != protected.source_hashes():
        raise ValueError('Parent implementation changed')
    if pilot.digest(parent/'checkpoint-000044.pt') != protocol['parent_checkpoint_sha256']:
        raise ValueError('Parent checkpoint changed')
    return protocol, original


def setup(parent, run):
    import torch
    pilot.external(run).mkdir(exist_ok=False)
    original = pilot.read(parent/'protocol.json')
    if pilot.read(parent/'status.json')['status'] != 'complete':
        raise ValueError('Parent is not terminal')
    if original['source'] != protected.source_hashes():
        raise ValueError('Parent source changed')
    checkpoint = parent/'checkpoint-000044.pt'
    receipt = pilot.read(parent/'checkpoint-000044.json')
    if pilot.digest(checkpoint) != receipt['sha256']:
        raise ValueError('Parent checkpoint changed')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if saved['protocol_sha256'] != pilot.digest(parent/'protocol.json'):
        raise ValueError('Parent checkpoint protocol mismatch')
    protocol = copy.deepcopy(original)
    protocol['config']['output'] = str(run)
    protocol.update(parent_run=str(parent), parent_protocol_sha256=pilot.digest(parent/'protocol.json'),
        parent_checkpoint_sha256=receipt['sha256'], driver_sha256=pilot.digest(Path(__file__)),
        variants={'25':0.25, '50':0.5, '75':0.75},
        goal='Both development and custom: pooled CER < 0.03 and speaker-macro cosine > 0.984',
        scope='Fixed two-seed screening panel; any pass requires additional fixed-example confirmation',
        interpretation='Variant IDs are adapter percentages, not training steps. Frozen backbone and decoding unchanged.')
    pilot.write(run/'protocol.json', protocol)
    shutil.copy2(Path(__file__), run/'source.py')
    baseline = pilot.read(parent/'eval-000000/results.json')
    for record in baseline['records']:
        if pilot.digest(parent/'eval-000000'/record['wav']) != record['audio_sha256']:
            raise ValueError('Reused baseline audio changed')
    baseline['cache_provenance'] = dict(path=str(parent/'eval-000000/results.json'),
        sha256=pilot.digest(parent/'eval-000000/results.json'), regenerated=False,
        note='Exact unchanged halfway baseline; original per-record provenance retained')
    (run/'eval-000000').mkdir()
    pilot.write(run/'eval-000000/results.json', baseline)
    for key, strength in protocol['variants'].items():
        variant = int(key)
        path = pilot.checkpoint_path(run, variant)
        pilot.save(path, dict(adapter=scaled_adapter(saved['adapter'], strength),
            frozen_backbone_sha256=saved['frozen_backbone_sha256'],
            protocol_sha256=pilot.digest(run/'protocol.json'),
            kind='inference_variant_without_optimizer', source_training_step=44,
            source_checkpoint_sha256=receipt['sha256'], strength=strength))
        pilot.write(run/f'checkpoint-{variant:06d}.json', dict(sha256=pilot.digest(path),
            kind='inference_variant', source_training_step=44, strength=strength))
    pilot.write(run/'status.json', dict(status='prepared'))
    return protocol


def render(run, reports, status):
    protocol, _ = verify_protocol(run)
    parent = Path(protocol['parent_run'])
    page = '<!doctype html><html><meta charset="utf-8"><title>Voice and pronunciation goal</title>'
    if status == 'running':
        page += '<meta http-equiv="refresh" content="60">'
    page += '<style>body{font:17px system-ui;background:#111821;color:#e7eef8;max-width:1100px;margin:40px auto;padding:20px}td,th{padding:12px;text-align:left}a{color:#91caff}audio{width:100%}</style>'
    page += '<h1>Adapter strength comparison</h1><p>'+html.escape(status)+'</p><p>Goal: CER below 3% and internal speaker cosine above 0.984 in BOTH groups. Same texts, reference recordings, decoding and two seeds. Scores are screening evidence, not proof of perceived voice preservation.</p>'
    page += '<table><tr><th>Strength</th><th>Dev CER / WER</th><th>Dev cosine</th><th>Custom CER / WER</th><th>Custom cosine</th><th>Thresholds</th></tr>'
    all_reports = [(0, pilot.read(run/'eval-000000/results.json'))] + reports + [(100, pilot.read(parent/'eval-000044/results.json'))]
    for strength, report in all_reports:
        d, c = report['summary']['development'], report['summary']['custom']
        passed = goal_screen(report['summary'])
        page += f'<tr><td>{strength}%</td><td>{100*d["pooled_cer"]:.2f}% / {100*d["pooled_wer"]:.2f}%</td><td>{d["speaker_macro_cosine"]:.6f}</td><td>{100*c["pooled_cer"]:.2f}% / {100*c["pooled_wer"]:.2f}%</td><td>{c["speaker_macro_cosine"]:.6f}</td><td>{"Screen pass; confirmation needed" if passed else "Not met"}</td></tr>'
    page += '</table><h2>Fixed listening example</h2><p>Custom example 001, seed 2026, for every strength.</p>'
    for strength, report in all_reports:
        directory = parent/f'eval-{0 if strength == 0 else 44:06d}' if strength in (0,100) else run/f'eval-{strength:06d}'
        record = next(r for r in report['records'] if r['case']=='custom-001' and r['seed']==2026)
        from urllib.parse import quote
        url = quote(os.path.relpath(directory/record['wav'], run).replace('\\','/'), safe='/.:')
        page += f'<h3>{strength}% adapter</h3><audio controls preload="none" src="{url}"></audio>'
    page += '<p><a href="protocol.json">Protocol</a> · <a href="results.json">All completed results</a></p></html>'
    (run/'report.html').write_text(page, encoding='utf-8')


def run_search(parent, run, resume):
    protocol = verify_protocol(run)[0] if resume else setup(parent, run)
    reports = []
    try:
        render(run, reports, 'running')
        for key in protocol['variants']:
            variant = int(key)
            result_path = run/f'eval-{variant:06d}/results.json'
            if not result_path.exists():
                pilot.write(run/'status.json', dict(status='running', phase='speech', strength=variant, updated=time.time()))
                with (run/f'variant-{variant}.log').open('a', encoding='utf-8') as log:
                    result = subprocess.run([sys.executable, '-m', 'training.voice_strength_search',
                        '--parent', str(parent), '--output', str(run), '--variant', key],
                        cwd=Path(__file__).resolve().parents[1], stdout=log, stderr=subprocess.STDOUT)
                if result.returncode:
                    raise RuntimeError(f'Variant {variant} failed')
            report = pilot.read(result_path)
            reports.append((variant, report))
            pilot.write(run/'results.json', dict(results=[dict(strength=v, **r) for v,r in reports],
                thresholds_passed=[v for v,r in reports if goal_screen(r['summary'])],
                goal_achieved=False, reason='Additional fixed-example confirmation required even for a screen pass'))
            render(run, reports, 'running')
            print({'strength':variant,'summary':report['summary'],'screen_pass':goal_screen(report['summary'])}, flush=True)
        pilot.write(run/'status.json', dict(status='complete', variants=[v for v,r in reports], updated=time.time()))
        render(run, reports, 'complete')
    except BaseException as error:
        pilot.write(run/'status.json', dict(status='failed', error=repr(error), updated=time.time()))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--variant', type=int)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    parent, run = args.parent.resolve(), args.output.resolve()
    original = pilot.read(parent/'protocol.json')
    cfg = original['config']
    os.environ['HF_HUB_OFFLINE'] = os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['NUMBA_CACHE_DIR'] = cfg['numba_cache']
    os.environ['TEMP'] = os.environ['TMP'] = cfg['tmp']
    import torch
    torch.set_num_threads(4)
    if args.variant is None:
        run_search(parent, run, args.resume)
    else:
        protocol, original = verify_protocol(run)
        if str(args.variant) not in protocol['variants']:
            raise ValueError('Unregistered strength')
        protected.configure_helpers()
        pilot.load_model = lambda _: protected.load_model(original)
        pilot.load_adapter = lambda talker, state: protected.audited_load_adapter(talker, state,
            run/f'routing-{args.variant}.json')
        pilot.speech(protocol, args.variant)


if __name__ == '__main__':
    main()
