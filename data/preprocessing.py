"""Optional quality-gated denoising and edge-only energy or neural trimming."""
from array import array
import math
import re
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

from .audio import ffmpeg_executable, inspect


DEFAULTS = {
    'denoise': {'enabled': False, 'backend': 'afftdn', 'reduction_db': 6.0, 'noise_floor_db': -50.0,
                'executable': None, 'executable_sha256': None, 'model': None, 'model_sha256': None},
    'trim': {'enabled': False, 'threshold_dbfs': -45.0, 'window_ms': 20.0,
             'min_silence_ms': 200.0, 'padding_ms': 150.0,
             'max_removed_duration_fraction': .4, 'max_removed_energy_fraction': .1},
    'vad': {'enabled': False, 'model': None, 'model_sha256': None, 'threshold': .5,
            'min_speech_ms': 250.0, 'min_silence_ms': 500.0, 'padding_ms': 250.0,
            'max_removed_duration_fraction': .4, 'max_removed_energy_fraction': .1},
    'quality': {'enabled': False, 'model': None, 'model_sha256': None,
                'min_overall_gain': .05, 'min_background_gain': .05, 'max_speech_drop': .1,
                'clean_background_score': 4.0},
}


class NoActiveAudio(ValueError):
    """No analysis window exceeds the configured trimming threshold."""


def processing_options(value=None):
    if value is None:
        value = {}
    if not isinstance(value, dict) or set(value) - DEFAULTS.keys():
        raise ValueError('Unknown audio_processing section')
    result = {}
    bounds = {'reduction_db': (0.01, 24), 'noise_floor_db': (-80, -20),
              'threshold_dbfs': (-80, -20), 'window_ms': (5, 100),
              'min_silence_ms': (0, 5000), 'padding_ms': (0, 1000),
              'threshold': (.05, .95), 'min_speech_ms': (32, 2000),
              'min_overall_gain': (0, 2), 'min_background_gain': (0, 2),
              'max_speech_drop': (0, 1), 'clean_background_score': (1, 5),
              'max_removed_duration_fraction': (0, 1), 'max_removed_energy_fraction': (0, 1)}
    for name, defaults in DEFAULTS.items():
        overrides = value.get(name, {})
        if not isinstance(overrides, dict) or set(overrides) - defaults.keys():
            raise ValueError(f'Unknown audio_processing.{name} option')
        options = defaults | overrides
        if type(options['enabled']) is not bool:
            raise ValueError(f'audio_processing.{name}.enabled must be boolean')
        for key, number in options.items():
            if key == 'enabled':
                continue
            if key in ('model', 'executable'):
                if number is not None and (not isinstance(number, str) or not number.strip()):
                    raise ValueError(f'{key} must be a local path')
                continue
            if key.endswith('_sha256'):
                if number is not None and (not isinstance(number, str) or not re.fullmatch('[0-9a-f]{64}', number)):
                    raise ValueError(f'{key} must be a SHA-256 hex digest')
                continue
            if key == 'backend':
                if number not in ('afftdn', 'deepfilternet'):
                    raise ValueError('denoise.backend must be afftdn or deepfilternet')
                continue
            lo, hi = bounds[key]
            if type(number) not in (float, int) or not math.isfinite(number) or not lo <= number <= hi:
                raise ValueError(f'audio_processing.{name}.{key} must be between {lo} and {hi}')
        result[name] = options
    for name in ('vad', 'quality', 'denoise'):
        cfg = result[name]
        if cfg['enabled'] and (name != 'denoise' or cfg['backend'] == 'deepfilternet'):
            required = ['model', 'model_sha256']
            if name == 'denoise':
                required += ['executable', 'executable_sha256']
            if any(not cfg[key] for key in required):
                raise ValueError(f'audio_processing.{name} requires local files and their SHA-256 hashes')
    return result


def check_backend(options):
    """Fail once before preparation if the requested filter is unavailable."""
    if any(options[key]['enabled'] for key in ('denoise', 'vad', 'quality')):
        ffmpeg_executable()
    if options['denoise']['enabled'] and options['denoise']['backend'] == 'afftdn':
        p = subprocess.run([ffmpeg_executable(), '-hide_banner', '-filters'],
                           capture_output=True, text=True, check=True)
        if not any('afftdn' in line.split() for line in p.stdout.splitlines()):
            raise RuntimeError('Denoising requires an ffmpeg build with the afftdn filter')
    if options['denoise']['enabled'] and options['denoise']['backend'] == 'deepfilternet':
        from .neural import verified_file
        for key in ('executable', 'model'):
            verified_file(options['denoise'][key], options['denoise'][key + '_sha256'])
    if options['vad']['enabled'] or options['quality']['enabled']:
        from .neural import NeuralTools
        return NeuralTools(options)


def trim_edges(src, dst, options, bounds=None):
    with wave.open(str(src), 'rb') as f:
        rate = f.getframerate()
        if (f.getnchannels(), f.getsampwidth(), rate) != (1, 2, 24000):
            raise ValueError('Trimming requires canonical mono PCM16 at 24000 Hz')
        raw = f.readframes(f.getnframes())
    samples = array('h', raw)
    if sys.byteorder != 'little':
        samples.byteswap()
    first, last = bounds if bounds is not None else (None, None)
    if bounds is None:
        window = round(rate * options['window_ms'] / 1000)
        threshold_squared = (32768 * 10 ** (options['threshold_dbfs'] / 20)) ** 2
        for start in range(0, len(samples), window):
            block = samples[start:start + window]
            if sum(x*x for x in block) / len(block) >= threshold_squared:
                if first is None:
                    first = start
                last = start + len(block)
    if first is None:
        raise NoActiveAudio('No audio above the trimming energy threshold')
    padding = round(rate * options['padding_ms'] / 1000)
    minimum = round(rate * options['min_silence_ms'] / 1000)
    start = max(0, first - padding) if first >= minimum else 0
    stop = min(len(samples), last + padding) if len(samples) - last >= minimum else len(samples)
    proposed = dict(leading_trim_samples=start, trailing_trim_samples=len(samples)-stop)
    duration_fraction = (start + len(samples) - stop) / len(samples)
    energy_fraction = None
    decision = 'applied' if start or stop != len(samples) else 'unchanged'
    if duration_fraction > options.get('max_removed_duration_fraction', .4):
        total_energy = sum(x*x for x in samples)
        removed_energy = sum(x*x for x in samples[:start]) + sum(x*x for x in samples[stop:])
        energy_fraction = removed_energy / total_energy if total_energy else 0.0
        if energy_fraction > options.get('max_removed_energy_fraction', .1):
            # VAD absence is not proof of silence. Preserve the selected full
            # waveform when a large proposed cut would remove meaningful energy.
            start, stop = 0, len(samples)
            decision = 'reverted_edge_energy'
    with wave.open(str(dst), 'wb') as f:
        f.setparams((1, 2, rate, 0, 'NONE', 'not compressed'))
        f.writeframes(raw[start*2:stop*2])
    return {'trim_decision': decision, 'proposed_trim': proposed,
            'proposed_removed_duration_fraction': duration_fraction,
            'proposed_removed_energy_fraction': energy_fraction,
            'leading_trim_samples': start, 'trailing_trim_samples': len(samples) - stop,
            'leading_trim_seconds': start / rate, 'trailing_trim_seconds': (len(samples) - stop) / rate}


def process(path, options, neural=None):
    """Replace only a generated WAV; retain pre-processing metrics and exact offsets."""
    path = Path(path)
    before = inspect(path)
    audit = {'version': 3, 'order': ['assess_original', 'denoise_candidate', 'quality_selection', 'edge_trim'], 'settings': options,
             'before': before, 'denoiser': options['denoise']['backend'] if options['denoise']['enabled'] else None,
             'review_flags': [],
             'leading_trim_samples': 0, 'trailing_trim_samples': 0,
             'leading_trim_seconds': 0.0, 'trailing_trim_seconds': 0.0}
    if not any(section['enabled'] for section in options.values()):
        return audit
    if neural is None and (options['vad']['enabled'] or options['quality']['enabled']):
        neural = check_backend(options)
    if neural:
        audit['models'] = neural.provenance
    original_score = neural.score(path) if options['quality']['enabled'] else None
    original_vad = neural.detect(path) if options['vad']['enabled'] else None
    if original_score:
        audit['quality_before'] = original_score
    if original_vad:
        audit['vad_before'] = original_vad
    with tempfile.TemporaryDirectory(prefix='processing-', dir=path.parent) as temp:
        current = path
        clean_enough = original_score and original_score['background'] >= options['quality']['clean_background_score']
        audit['denoise_decision'] = 'disabled'
        if options['denoise']['enabled'] and clean_enough:
            audit['denoise_decision'] = 'skipped_clean_background'
        if options['denoise']['enabled'] and not clean_enough:
            denoised = Path(temp) / 'denoised.wav'
            cfg = options['denoise']
            filter_spec = f"afftdn=nr={cfg['reduction_db']}:nf={cfg['noise_floor_db']}:tn=0"
            try:
                if cfg['backend'] == 'deepfilternet':
                    input48 = Path(temp) / 'input48.wav'
                    subprocess.run([ffmpeg_executable(), '-nostdin', '-v', 'error', '-i', str(path),
                                    '-ac', '1', '-ar', '48000', '-c:a', 'pcm_s16le', str(input48)],
                                   capture_output=True, check=True)
                    output48 = Path(temp) / 'deepfilter'
                    subprocess.run([str(Path(cfg['executable']).resolve()), '-m', str(Path(cfg['model']).resolve()),
                                    '-D', '-a', str(cfg['reduction_db']), '-o', str(output48), str(input48)],
                                   capture_output=True, check=True)
                    outputs = list(output48.glob('*.wav'))
                    if len(outputs) != 1:
                        raise RuntimeError('Expected one DeepFilterNet output')
                    subprocess.run([ffmpeg_executable(), '-nostdin', '-v', 'error', '-i', str(outputs[0]),
                                    '-ar', '24000', '-ac', '1', '-c:a', 'pcm_s16le', str(denoised)],
                                   capture_output=True, check=True)
                else:
                    subprocess.run([ffmpeg_executable(), '-nostdin', '-v', 'error', '-i', str(current),
                                '-af', filter_spec, '-ac', '1', '-ar', '24000', '-c:a', 'pcm_s16le',
                                '-map_metadata', '-1', '-n', str(denoised)],
                               capture_output=True, check=True)
            except subprocess.CalledProcessError as exc:
                raise RuntimeError('Denoising failed: ' + exc.stderr.decode(errors='replace')[:1000]) from exc
            if abs(inspect(denoised)['duration'] - before['duration']) > .05:
                raise RuntimeError('Denoiser changed clip length by more than 50 ms')
            current = denoised
            audit['denoise_decision'] = 'applied_without_quality_gate'
            if original_score:
                from .neural import enhancement_wins
                score = neural.score(denoised)
                audit['quality_candidate'] = score
                if enhancement_wins(original_score, score, options['quality']):
                    audit['denoise_decision'] = 'kept_quality_gain'
                else:
                    current = path
                    audit['denoise_decision'] = 'reverted_quality_gate'
            if original_vad and current != path:
                candidate_vad = neural.detect(current)
                audit['vad_candidate'] = candidate_vad
                if candidate_vad['speech_seconds'] < original_vad['speech_seconds'] * .9:
                    current = path
                    audit['denoise_decision'] = 'reverted_speech_loss'
                    audit['review_flags'].append('vad_speech_loss_candidate')
        if options['vad']['enabled']:
            selected_vad = original_vad if current == path else audit.get('vad_candidate') or neural.detect(current)
            audit['vad_selected'] = selected_vad
            if not selected_vad['segments']:
                error = NoActiveAudio('Silero detected no speech segment')
                error.audit = audit
                raise error
            trimmed = Path(temp) / 'trimmed.wav'
            first = math.floor(selected_vad['segments'][0]['start'] * 1.5)
            last = math.ceil(selected_vad['segments'][-1]['end'] * 1.5)
            audit.update(trim_edges(current, trimmed, options['vad'], (first, last)))
            audit['trim_backend'] = 'silero-vad'
            current = trimmed
        elif options['trim']['enabled']:
            trimmed = Path(temp) / 'trimmed.wav'
            audit.update(trim_edges(current, trimmed, options['trim']))
            audit['trim_backend'] = 'energy'
            current = trimmed
        if original_score:
            audit['quality_after'] = neural.score(current)
            if audit['quality_after']['overall'] < 3 or audit['quality_after']['speech'] < 3:
                audit['review_flags'].append('low_predicted_quality')
        if audit.get('trim_decision') == 'reverted_edge_energy':
            audit['review_flags'].append('unsafe_edge_trim_reverted')
        if audit['leading_trim_seconds'] + audit['trailing_trim_seconds'] > before['duration'] * .4:
            audit['review_flags'].append('large_edge_trim')
        inspect(current)
        if current != path:
            current.replace(path)
    return audit
