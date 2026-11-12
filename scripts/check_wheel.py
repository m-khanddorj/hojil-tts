"""Check a built wheel offline, outside the checkout and without installing it.

Only standard-library functionality and dependency-free package imports are
exercised. This is not a training, model, or retained-experiment smoke test.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile


REPO = Path(__file__).resolve().parents[1]
REQUIRED = {
    'data/__init__.py', 'data/__main__.py', 'data/cli.py',
    'data/playground.html', 'data/youtube.html',
    'training/evaluate.py', 'training/reference_masked_supervision_evaluation.py',
    'scripts/score_latest_audio.py',
}

CHILD = r'''
import contextlib
import importlib
import importlib.metadata
import importlib.resources
import io
import json
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root))
loaded = []
for name in ('data.cli', 'training.evaluate',
             'training.reference_masked_supervision_evaluation',
             'scripts.score_latest_audio'):
    module = importlib.import_module(name)
    if not Path(module.__file__).resolve().is_relative_to(root):
        raise RuntimeError('Import escaped unpacked wheel: ' + name)
    loaded.append(name)

from scripts.score_latest_audio import score
if score('ӨӨ, үү!', 'өө үү', True)['cer'] != 0:
    raise RuntimeError('Packaged Unicode scorer failed')
if score('abc', 'adc', True) != dict(cer=1/3, wer=1.0,
        character_errors=1, characters=3, word_errors=1, words=1):
    raise RuntimeError('Packaged edit-count scorer failed')

distribution = importlib.metadata.distribution('hojil-data')
entries = {e.name: e.value for e in distribution.entry_points if e.group == 'console_scripts'}
expected = {'hojil-data': 'data.cli:main', 'hojil-intake': 'data.youtube:main',
            'hojil-playground': 'data.playground:main'}
if entries != expected:
    raise RuntimeError('Wheel console entry points differ')
entry = next(e for e in distribution.entry_points if e.name == 'hojil-data')
sys.argv = ['hojil-data', '--help']
help_output = io.StringIO()
with contextlib.redirect_stdout(help_output):
    try:
        entry.load()()
    except SystemExit as exc:
        if exc.code not in (None, 0):
            raise
if 'usage:' not in help_output.getvalue().lower():
    raise RuntimeError('Packaged data CLI help was not rendered')

assets = []
for name in ('playground.html', 'youtube.html'):
    value = importlib.resources.files('data').joinpath(name).read_text(encoding='utf-8')
    if '<html' not in value.lower():
        raise RuntimeError('Missing/invalid packaged HTML: ' + name)
    assets.append(name)
for name in ('torch', 'qwen_tts', 'transformers', 'peft', 'numpy'):
    if name in sys.modules:
        raise RuntimeError('Unexpected optional/model dependency imported: ' + name)
print(json.dumps(dict(status='passed', isolated=True, site_packages_enabled=False,
    imports=loaded, console_entry_points=entries, data_cli_help=True,
    data_assets=assets, scorer=True, models_loaded=False)))
'''


def check(wheel, temp_root=None):
    wheel = Path(wheel).resolve()
    if wheel.is_dir():
        candidates = sorted(wheel.glob('*.whl'))
        if len(candidates) != 1:
            raise ValueError(f'Expected exactly one wheel in {wheel}; found {len(candidates)}')
        wheel = candidates[0]
    if not wheel.is_file() or wheel.suffix != '.whl':
        raise ValueError('Provide an existing wheel or a directory containing exactly one wheel')
    with tempfile.TemporaryDirectory(prefix='hojil-wheel-smoke-', dir=temp_root) as folder:
        work = Path(folder).resolve()
        if work.is_relative_to(REPO):
            raise ValueError('Wheel smoke temporary directory must be outside the checkout; set --temp-root')
        unpacked = work / 'wheel'
        unpacked.mkdir()
        with zipfile.ZipFile(wheel) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)):
                raise ValueError('Duplicate wheel archive entries')
            missing = REQUIRED - set(names)
            if missing:
                raise ValueError(f'Wheel is missing required files: {sorted(missing)}')
            for name in names:
                if not (unpacked / name).resolve().is_relative_to(unpacked):
                    raise ValueError('Wheel archive member escapes extraction directory')
            archive.extractall(unpacked)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1',
                   TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1')
        result = subprocess.run([sys.executable, '-I', '-S', '-c', CHILD, str(unpacked)],
                                cwd=work, env=env, text=True, encoding='utf-8',
                                capture_output=True, timeout=60)
        if result.returncode:
            raise RuntimeError(f'Isolated wheel check failed ({result.returncode}):\n{result.stderr}{result.stdout}')
        report = json.loads(result.stdout)
        report['wheel'] = str(wheel)
        report['wheel_sha256'] = hashlib.sha256(wheel.read_bytes()).hexdigest()
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('wheel', type=Path, help='Wheel file or directory containing exactly one wheel')
    parser.add_argument('--temp-root', type=Path, help='Optional existing temporary parent outside the checkout')
    args = parser.parse_args()
    print(json.dumps(check(args.wheel, args.temp_root), indent=2))


if __name__ == '__main__':
    main()
