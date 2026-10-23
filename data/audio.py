"""Deterministic PCM analysis; ffmpeg is required only for conversion."""
from array import array
import hashlib
import math
import shutil
import subprocess
import sys
import wave
from pathlib import Path


def inspect(path):
    with wave.open(str(path), 'rb') as f:
        channels, width, rate, frames = f.getnchannels(), f.getsampwidth(), f.getframerate(), f.getnframes()
        if channels != 1 or width != 2 or rate != 24000:
            raise ValueError('Expected mono PCM16 WAV at 24000 Hz')
        raw = f.readframes(frames)
    if len(raw) != frames * 2 or not frames:
        raise ValueError('Empty or truncated WAV')
    samples = array('h', raw)
    if sys.byteorder != 'little':
        samples.byteswap()
    rms = math.sqrt(sum(x*x for x in samples) / len(samples)) / 32768
    return dict(duration=frames / rate, sample_rate=rate, samples=frames,
                pcm_sha256=hashlib.sha256(raw).hexdigest(),
                rms_dbfs=20 * math.log10(max(rms, 1e-12)),
                clipping_fraction=sum(abs(x) >= 32760 for x in samples) / frames,
                silence_fraction=sum(abs(x) < 328 for x in samples) / frames)


def ffmpeg_executable():
    path = shutil.which('ffmpeg')
    if path:
        return path
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        raise RuntimeError('Install the local audio extra or provide ffmpeg on PATH') from None


def convert(src, dst):
    """No denoising, trimming, or loudness changes. Preserve originals elsewhere."""
    try:
        with wave.open(str(src), 'rb') as f:
            native = dict(native_sample_rate=f.getframerate(), native_channels=f.getnchannels())
            direct = f.getnchannels() == 1 and f.getsampwidth() == 2 and f.getframerate() == 24000
    except (wave.Error, EOFError):
        native, direct = {}, False
    if direct:
        shutil.copyfile(src, dst)
        return native
    executable = ffmpeg_executable()
    if not native:
        if shutil.which('ffprobe'):
            import json
            p = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'a:0', '-show_entries',
                                'stream=sample_rate,channels', '-of', 'json', str(src)],
                               check=True, capture_output=True, text=True)
            streams = json.loads(p.stdout)['streams']
            if not streams:
                raise ValueError('No audio stream')
            native = dict(native_sample_rate=int(streams[0]['sample_rate']), native_channels=streams[0]['channels'])
        else:
            try:
                import soundfile
            except ImportError:
                raise RuntimeError('Install the local audio extra or ffprobe for compressed-audio metadata') from None
            try:
                info = soundfile.info(str(src))
            except soundfile.LibsndfileError as exc:
                raise ValueError('Audio metadata decode failed') from exc
            native = dict(native_sample_rate=info.samplerate, native_channels=info.channels)
    subprocess.run([executable, '-nostdin', '-v', 'error', '-i', str(src), '-map', '0:a:0',
                    '-ac', '1', '-ar', '24000', '-c:a', 'pcm_s16le', '-map_metadata', '-1',
                    '-n', str(dst)], check=True, capture_output=True)
    return native
