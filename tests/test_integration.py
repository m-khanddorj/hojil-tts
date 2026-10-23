"""Optional real-codec/Parquet fixtures. Install .[audio,parquet] to run."""
import importlib.util
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from test_pipeline import wav
from data.audio import convert, ffmpeg_executable, inspect
from data.io import read_rows, write_rows
from data.pipeline import prepare
from data.sources import import_parquet
from data.adaptation import export_adaptation

try:
    ffmpeg_executable()
    HAS_FFMPEG = True
except RuntimeError:
    HAS_FFMPEG = False
HAS_METADATA = bool(shutil.which('ffprobe') or importlib.util.find_spec('soundfile'))
HAS_EXTRAS = bool(importlib.util.find_spec('pyarrow') and HAS_FFMPEG and HAS_METADATA)


@unittest.skipUnless(HAS_EXTRAS, 'Install audio and parquet extras for actual integration tests')
class RealIntegration(unittest.TestCase):
    def test_stereo_mp3_conversion_and_parquet_end_to_end(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            original = root / 'original.wav'
            wav(original, rate=16000, duration=1.25)
            mp3 = root / 'stereo.mp3'
            subprocess.run([ffmpeg_executable(), '-v', 'error', '-i', str(original),
                            '-ac', '2', '-c:a', 'libmp3lame', str(mp3)], check=True)
            converted = root / 'converted.wav'
            native = convert(mp3, converted)
            self.assertEqual(native, {'native_sample_rate': 16000, 'native_channels': 2})
            info = inspect(converted)
            self.assertAlmostEqual(info['duration'], 1.25, delta=.03)
            self.assertEqual(info['sample_rate'], 24000)
            self.assertGreater(info['rms_dbfs'], -20)
            source = dict(id='mbfixture', kind='mbspeech', revision='a'*40,
                          license='generated-fixture', source_url='local:generated',
                          rights_clearance='synthetic-test-fixture', path=str(root / 'imported'))
            parquet = root / 'fixture.parquet'
            pq.write_table(pa.Table.from_pylist([{
                'audio': {'bytes': mp3.read_bytes(), 'path': 'stereo.mp3'},
                'sentence_orig': 'Өнөөдөр сайхан өдөр.', 'sentence_norm': 'өнөөдөр сайхан өдөр',
                'split': 'train', 'speaker_id': 'fixture-voice',
            }]), parquet)
            import_parquet(source, [parquet], root / 'imported')
            report = prepare({'sources': [source], 'splits': {'test': 0, 'validation': 0}}, root / 'prepared')
            self.assertEqual(report['accepted'], 1)
            self.assertEqual(report['rejected'], 0)
            record = next(read_rows(root / 'prepared/manifest.jsonl'))
            self.assertEqual(record['native_channels'], 2)
            self.assertEqual(record['native_sample_rate'], 16000)
            self.assertEqual(record['pcm_sha256'], info['pcm_sha256'])
            export_adaptation(root / 'prepared/manifest.jsonl', root / 'adaptation')
            self.assertEqual(next(read_rows(root / 'adaptation/train.jsonl'))['text'], 'Өнөөдөр сайхан өдөр.')
            self.assertIn('ffmpeg version', report['ffmpeg'])

    def test_mono_resampling_and_corrupt_compressed_audio_rejection(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            wav(root / 'mono.wav', rate=16000)
            native = convert(root / 'mono.wav', root / 'output.wav')
            self.assertEqual(native['native_sample_rate'], 16000)
            self.assertEqual(inspect(root / 'output.wav')['sample_rate'], 24000)
            (root / 'bad.mp3').write_bytes(b'corrupted')
            write_rows(root / 'metadata.jsonl', [dict(id='bad', audio='bad.mp3', text='Сайн байна уу?', split='train')])
            source = dict(id='fixture', kind='manifest', revision='v1', license='generated', source_url='local:fixture', path=str(root))
            report = prepare({'sources': [source]}, root / 'prepared')
            self.assertEqual(report['reasons'], {'audio_decode': 1})
