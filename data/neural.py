"""Offline, checksummed ONNX speech detection and DNSMOS P.835 scoring.

Inference contracts follow Silero VAD (MIT) and Microsoft's DNSMOS reference
implementation (CC BY 4.0). Models and their notices are provisioned separately.
No model or package is downloaded by this module.
"""
import math
import subprocess
from pathlib import Path

from .audio import ffmpeg_executable
from .io import digest


def verified_file(path, sha256):
    p = Path(path).expanduser().resolve()
    if not p.is_file() or digest(p) != sha256:
        raise ValueError(f'Cleaning model/executable checksum mismatch: {p}')
    return p


def session(path):
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError('Install the neural extra for Silero VAD / DNSMOS') from exc
    settings = ort.SessionOptions()
    settings.inter_op_num_threads = settings.intra_op_num_threads = 1
    return ort.InferenceSession(str(path), settings, providers=['CPUExecutionProvider'])


def audio16(path):
    import numpy as np
    p = subprocess.run([ffmpeg_executable(), '-nostdin', '-v', 'error', '-i', str(path),
                        '-ac', '1', '-ar', '16000', '-f', 'f32le', '-'],
                       capture_output=True, check=True)
    audio = np.frombuffer(p.stdout, dtype='<f4').copy()
    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError('Empty or nonfinite analysis audio')
    return audio


def speech_segments(probabilities, length, threshold=.5, min_speech_ms=250, min_silence_ms=500):
    """Hysteresis at 32 ms/frame, returning unpadded 16 kHz intervals."""
    start, pending_end = None, None
    result = []
    minimum = min_speech_ms * 16
    quiet = min_silence_ms * 16
    for i, probability in enumerate(probabilities):
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError('Invalid VAD probability')
        offset = i * 512
        if probability >= threshold:
            pending_end = None
            if start is None:
                start = offset
        elif start is not None and probability < max(.01, threshold - .15):
            if pending_end is None:
                pending_end = offset
            if offset - pending_end >= quiet:
                if pending_end - start >= minimum:
                    result.append({'start': start, 'end': min(length, pending_end)})
                start, pending_end = None, None
    if start is not None and length - start >= minimum:
        result.append({'start': start, 'end': length})
    return result


def enhancement_wins(before, candidate, cfg):
    return (candidate['overall'] - before['overall'] >= cfg['min_overall_gain']
            and candidate['background'] - before['background'] >= cfg['min_background_gain']
            and candidate['speech'] >= before['speech'] - cfg['max_speech_drop'])


class NeuralTools:
    def __init__(self, options):
        self.options = options
        self.vad = self.mos = None
        self.provenance = {}
        for key in ('vad', 'quality'):
            cfg = options[key]
            if cfg['enabled']:
                p = verified_file(cfg['model'], cfg['model_sha256'])
                loaded = session(p)
                expected = {'input', 'state', 'sr'} if key == 'vad' else {'input_1'}
                if {item.name for item in loaded.get_inputs()} != expected:
                    raise ValueError(f'Unsupported {key} model input contract')
                if key == 'vad':
                    self.vad = loaded
                else:
                    self.mos = loaded
                import onnxruntime
                self.provenance[key] = dict(model=str(p), sha256=cfg['model_sha256'],
                                            runtime='onnxruntime', runtime_version=onnxruntime.__version__)

    def detect(self, path):
        import numpy as np
        audio = audio16(path)
        state = np.zeros((2, 1, 128), dtype=np.float32)
        context = np.zeros((1, 64), dtype=np.float32)
        probabilities = []
        for offset in range(0, len(audio), 512):
            chunk = audio[offset:offset+512]
            chunk = np.pad(chunk, (0, 512-len(chunk)))[None, :]
            frame = np.concatenate((context, chunk), axis=1)
            prediction, state = self.vad.run(None, {'input': frame, 'state': state,
                                                   'sr': np.array(16000, dtype=np.int64)})
            probabilities.append(float(prediction.reshape(-1)[0]))
            context = chunk[:, -64:]
        cfg = self.options['vad']
        segments = speech_segments(probabilities, len(audio), cfg['threshold'],
                                   cfg['min_speech_ms'], cfg['min_silence_ms'])
        return dict(segments=segments, analysis_rate=16000,
                    speech_seconds=sum(s['end']-s['start'] for s in segments)/16000,
                    duration=len(audio)/16000)

    def score(self, path):
        """Regular P.835 SIG/BAK/OVRL; intentionally no P.808 score."""
        import numpy as np
        audio = audio16(path)
        original_length = len(audio)
        window = 144160  # Microsoft's 9.01-second input contract.
        if len(audio) < window:
            audio = np.tile(audio, math.ceil(window/len(audio)))
        scores = []
        for offset in range(0, len(audio)-window+1, 16000):
            raw = self.mos.run(None, {'input_1': audio[offset:offset+window][None, :].astype('float32')})[0]
            sig, bak, overall = map(float, raw.reshape(-1))
            scores.append([-.08397278*sig**2 + 1.22083953*sig + .0052439,
                           -.13166888*bak**2 + 1.60915514*bak - .39604546,
                           -.06766283*overall**2 + 1.11546468*overall + .04602535])
        values = np.mean(scores, axis=0)
        if not np.isfinite(values).all():
            raise RuntimeError('DNSMOS returned nonfinite scores')
        return dict(speech=float(values[0]), background=float(values[1]), overall=float(values[2]),
                    windows=len(scores), repeated_short_clip=original_length < window)
