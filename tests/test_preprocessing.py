import math
import random
import shutil
import struct
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from data.audio import inspect
from data.io import read_rows, write_rows
from data.pipeline import prepare, split_records
from data.preprocessing import process, processing_options
from data.neural import enhancement_wins, speech_segments
from data.transcripts import character_error_rate, transcript_flags


def write_pcm(path, samples):
    with wave.open(str(path), 'wb') as f:
        f.setparams((1, 2, 24000, 0, 'NONE', 'not compressed'))
        f.writeframes(struct.pack('<' + 'h' * len(samples), *samples))


def tone(seconds):
    return [round(6000 * math.sin(2 * math.pi * 220 * i / 24000)) for i in range(round(seconds * 24000))]


class PreprocessingTests(unittest.TestCase):
    def test_vad_hysteresis_preserves_short_pauses_and_clamps_end(self):
        segments = speech_segments([.01]*10 + [.9]*10 + [.1]*5 + [.9]*10 + [.01]*20, 55*512)
        self.assertEqual(segments, [{'start': 5120, 'end': 35*512}])
        self.assertEqual(speech_segments([.9]*10, 4900), [{'start': 0, 'end': 4900}])
        self.assertEqual(speech_segments([.01]*20, 10240), [])
        self.assertEqual(speech_segments([.9]*2 + [.01]*20, 11264), [])

    def test_quality_gate_protects_speech_even_with_cleaner_background(self):
        cfg = processing_options()['quality']
        before = dict(speech=4, background=2, overall=3)
        self.assertTrue(enhancement_wins(before, dict(speech=4, background=3, overall=3.3), cfg))
        self.assertFalse(enhancement_wins(before, dict(speech=3, background=4, overall=3.5), cfg))
        self.assertFalse(enhancement_wins(before, dict(speech=4, background=2, overall=3), cfg))

    def test_transcript_comparison_preserves_mongolian_letter_distinctions(self):
        self.assertEqual(character_error_rate('Өдөр, үүл!', 'өдөр үүл'), 0)
        self.assertGreater(character_error_rate('Өдөр', 'Одор'), 0)
        self.assertEqual(character_error_rate('Сайн', ''), 1)
        flags = transcript_flags('Сайн уу', 'word '*20, [{'start': 0, 'end': 30}], 3)
        self.assertIn('asr_unreliable_repetition', flags)
        self.assertNotIn('possible_transcript_mismatch', flags)
        self.assertEqual(transcript_flags('Сайн уу', 'Өдөр', [], 3), ['possible_transcript_mismatch'])

    def test_trim_preserves_padding_interior_pause_and_source(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            original = root / 'original.wav'
            samples = [0]*12000 + tone(.6) + [0]*9600 + tone(.6) + [0]*14400
            write_pcm(original, samples)
            target = root / 'processed.wav'
            shutil.copyfile(original, target)
            audit = process(target, processing_options({'trim': {'enabled': True}}))
            self.assertEqual(audit['leading_trim_samples'], 8400)
            self.assertEqual(audit['trailing_trim_samples'], 10800)
            with wave.open(str(target), 'rb') as f:
                actual = f.readframes(f.getnframes())
            self.assertEqual(actual, struct.pack('<' + 'h' * (len(samples)-19200), *samples[8400:-10800]))
            self.assertEqual(inspect(original)['samples'], len(samples))
            self.assertEqual(audit['before']['samples'], len(samples))

    def test_short_edge_silence_is_retained_and_disabled_is_identical(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / 'a.wav'
            write_pcm(p, [0]*2400 + tone(1) + [0]*2400)
            original = p.read_bytes()
            process(p, processing_options())
            self.assertEqual(p.read_bytes(), original)
            audit = process(p, processing_options({'trim': {'enabled': True}}))
            self.assertEqual(audit['leading_trim_samples'], 0)
            self.assertEqual(audit['trailing_trim_samples'], 0)

    def test_invalid_processing_fails_before_output_creation(self):
        invalid = [{'other': {}}, {'trim': {'threshold_dbfs': float('nan')}},
                   {'trim': {'enabled': 'yes'}}, {'trim': {'padding_ms': -1}},
                   {'denoise': {'reduction_db': True}}, {'denoise': {'reduction_db': 100}},
                   {'trim': {'window_ms': 0}}, {'denoise': {'model': 12}}]
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'out'
            for value in invalid:
                with self.subTest(value=value), self.assertRaises(ValueError):
                    prepare({'sources': [], 'audio_processing': value}, output)
                self.assertFalse(output.exists())

    def test_silence_rejection_and_pipeline_provenance(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_pcm(root/'speech.wav', [0]*12000 + tone(1.2) + [0]*12000)
            write_pcm(root/'silent.wav', [0]*24000)
            write_rows(root/'metadata.jsonl', [dict(id=name, audio=name+'.wav', text='Сайн уу')
                                               for name in ('speech', 'silent')])
            source = dict(id='fixture', kind='manifest', revision='v1', license='generated',
                          source_url='local:fixture', path=str(root))
            report = prepare({'sources': [source], 'audio_processing': {'trim': {'enabled': True}}}, root/'out')
            self.assertEqual(report['reasons'], {'no_active_audio': 1})
            self.assertEqual(report['unmeasured_rejections'], 0)
            row = next(read_rows(root/'out/manifest.jsonl'))
            self.assertEqual(row['audio_processing']['before']['duration'], 2.2)
            self.assertEqual(row['duration'], 1.5)
            self.assertNotEqual(row['pcm_sha256'], row['original_pcm_sha256'])
            self.assertEqual(inspect(root/'speech.wav')['duration'], 2.2)

    def test_original_pcm_still_connects_heldout_after_processing(self):
        rows = [dict(id='a', text='Сайн', pcm_sha256='processed-a', original_pcm_sha256='same',
                     speaker_id=None, source_split='train'),
                dict(id='b', text='Өдөр', pcm_sha256='processed-b', original_pcm_sha256='same',
                     speaker_id=None, source_split='test')]
        accepted, rejected = split_records(rows, 2026, {'validation': .05, 'test': .05})
        self.assertEqual([r['id'] for r in accepted], ['b'])
        self.assertEqual(rejected[0]['rejection'], 'heldout_leakage')

    def test_missing_filter_is_a_run_error(self):
        with patch('data.preprocessing.subprocess.run') as run:
            from data.preprocessing import check_backend
            run.return_value.stdout = 'other filters'
            with patch('data.preprocessing.ffmpeg_executable', return_value='ffmpeg'):
                with self.assertRaisesRegex(RuntimeError, 'afftdn'):
                    check_backend(processing_options({'denoise': {'enabled': True}}))

    def test_tampered_neural_model_fails_before_creating_run(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'model.onnx').write_bytes(b'not the pinned model')
            with patch('data.preprocessing.ffmpeg_executable', return_value='ffmpeg'), \
                    self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                prepare({'sources': [], 'audio_processing': {'vad': {
                    'enabled': True, 'model': str(root/'model.onnx'), 'model_sha256': '0'*64}}}, root/'out')
            self.assertFalse((root/'out').exists())


try:
    from data.audio import ffmpeg_executable
    ffmpeg_executable()
    HAS_FFMPEG = True
except RuntimeError:
    HAS_FFMPEG = False


@unittest.skipUnless(HAS_FFMPEG, 'Install audio extra or ffmpeg for actual denoising')
class DenoisingIntegration(unittest.TestCase):
    def test_quality_gate_restores_original_bytes_when_speech_quality_drops(self):
        class Scores:
            provenance = {}

            def score(self, path):
                return dict(speech=2 if Path(path).name == 'denoised.wav' else 4,
                            background=3 if Path(path).name == 'denoised.wav' else 2,
                            overall=3.5 if Path(path).name == 'denoised.wav' else 3)

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'speech.wav'
            write_pcm(path, tone(1.2))
            original = path.read_bytes()
            options = processing_options({'denoise': {'enabled': True}, 'quality': {
                'enabled': True, 'model': 'unused', 'model_sha256': '0'*64}})
            audit = process(path, options, Scores())
            self.assertEqual(audit['denoise_decision'], 'reverted_quality_gate')
            self.assertEqual(path.read_bytes(), original)

    def test_actual_noise_reduction_and_combined_trimming(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rng = random.Random(42)
            samples = [rng.randint(-300, 300) for _ in range(96000)]
            speech = tone(1.2)
            for i, value in enumerate(speech):
                samples[36000+i] += value
            p = root/'noisy.wav'
            write_pcm(p, samples)
            audit = process(p, processing_options({'denoise': {'enabled': True, 'reduction_db': 12,
                                                                          'noise_floor_db': -40}}))
            with wave.open(str(p), 'rb') as f:
                raw = f.readframes(f.getnframes())
            actual = struct.unpack('<' + 'h'*(len(raw)//2), raw)
            rms = lambda values: math.sqrt(sum(v*v for v in values) / len(values))
            self.assertLess(rms(actual[12000:24000]), rms(samples[12000:24000]) * .8)
            self.assertGreater(rms(actual[40000:60000]), 3000)
            self.assertAlmostEqual(inspect(p)['duration'], 4, delta=.05)
            self.assertEqual(audit['denoiser'], 'afftdn')
            write_pcm(root/'combined.wav', [0]*12000 + tone(1.2) + [0]*12000)
            audit = process(root/'combined.wav', processing_options(
                {'denoise': {'enabled': True}, 'trim': {'enabled': True}}))
            self.assertGreater(audit['leading_trim_samples'], 0)
            self.assertGreater(audit['trailing_trim_samples'], 0)
            self.assertGreater(inspect(root/'combined.wav')['duration'], 1.2)
