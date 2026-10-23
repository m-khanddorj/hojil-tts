from pathlib import Path
import importlib.util
import math
import struct
import tempfile
import unittest

from data.source_audit import classify_sources, read_source_wav
from data.preprocessing import process, processing_options, trim_edges
from data.audio import inspect
from data.io import digest, read_rows, write_rows
from data.pipeline import DEFAULT_FILTERS, finalize, prepare, validated_config
from data.quality_revision import repair_trim
from test_preprocessing import write_pcm, tone


def feature(ident, duration=10, span=9, low_band=False):
    return dict(id=ident, text='Same sentence', retained=True, duration=duration,
                active_span_seconds=span, source_sha256='source-hash',
                rolloff={'0.95': 500 if low_band else 2500, '0.99': 900 if low_band else 6000},
                band_fraction={'4000': .999 if low_band else .94},
                quality_before={'speech': 2.3 if low_band else 3.5},
                vad={'speech_seconds': 2 if low_band else 8})


class SourceAuditTests(unittest.TestCase):
    def test_same_sentence_multiple_cues_and_uncertain_quarantine(self):
        reference = feature('normal')
        suspect = feature('suspect', duration=22, span=20, low_band=True)
        result = classify_sources([reference, suspect])
        self.assertEqual([r['id'] for r in result['paired']], ['suspect'])
        self.assertEqual(result['paired'][0]['reference_ids'], ['normal'])
        self.assertFalse(result['unresolved'])
        result = classify_sources([suspect])
        self.assertFalse(result['paired'])
        self.assertEqual(result['unresolved'][0]['reason'], 'source_slowdown_unresolved')

    def test_long_duration_or_low_bandwidth_alone_is_insufficient(self):
        reference = feature('normal')
        for candidate in [feature('slow-natural', duration=22, span=20),
                          feature('filtered-normal-rate', low_band=True)]:
            result = classify_sources([reference, candidate])
            self.assertFalse(result['paired'])
            self.assertFalse(result['unresolved'])
        already_rejected = feature('rejected', duration=22, span=20, low_band=True)
        already_rejected['retained'] = False
        self.assertFalse(classify_sources([reference, already_rejected])['paired'])

    @unittest.skipUnless(importlib.util.find_spec('numpy'), 'source WAV audit requires numpy')
    def test_native_float_wav_clock_is_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'float.wav'
            payload = struct.pack('<1600f', *[.2*math.sin(2*math.pi*200*i/16000) for i in range(1600)])
            fmt = struct.pack('<HHIIHH', 3, 1, 16000, 64000, 4, 32)
            body = b'WAVEfmt '+struct.pack('<I', len(fmt))+fmt+b'data'+struct.pack('<I', len(payload))+payload
            path.write_bytes(b'RIFF'+struct.pack('<I', len(body))+body)
            samples, sha = read_source_wav(path)
            self.assertEqual(len(samples), 1600)
            self.assertAlmostEqual(max(samples), .2, places=6)
            self.assertEqual(len(sha), 64)

    def test_large_trim_with_sustained_energy_is_reverted(self):
        with tempfile.TemporaryDirectory() as temp:
            original, target = Path(temp)/'original.wav', Path(temp)/'trimmed.wav'
            write_pcm(original, tone(4))
            audit = trim_edges(original, target, processing_options()['vad'], (12000, 36000))
            self.assertEqual(audit['trim_decision'], 'reverted_edge_energy')
            self.assertGreater(audit['proposed_removed_energy_fraction'], .1)
            self.assertEqual(audit['leading_trim_samples'], 0)
            self.assertEqual(audit['trailing_trim_samples'], 0)
            self.assertEqual(original.read_bytes(), target.read_bytes())

    def test_large_silent_padding_trim_still_applies(self):
        with tempfile.TemporaryDirectory() as temp:
            original, target = Path(temp)/'original.wav', Path(temp)/'trimmed.wav'
            write_pcm(original, [0]*24000 + tone(1) + [0]*24000)
            audit = trim_edges(original, target, processing_options()['vad'], (24000, 48000))
            self.assertEqual(audit['trim_decision'], 'applied')
            self.assertEqual(audit['proposed_removed_energy_fraction'], 0)
            self.assertEqual(audit['leading_trim_samples'], 18000)
            self.assertEqual(audit['trailing_trim_samples'], 18000)

    def test_missed_vad_speech_remains_in_output_and_is_flagged(self):
        class MissedSpeech:
            provenance = {}
            def detect(self, path):
                return dict(segments=[dict(start=8000, end=24000)], analysis_rate=16000,
                            speech_seconds=1, duration=4)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'source.wav'
            write_pcm(path, tone(4))
            original = path.read_bytes()
            options = processing_options({'vad': dict(enabled=True, model='fixture', model_sha256='0'*64)})
            audit = process(path, options, MissedSpeech())
            self.assertIn('unsafe_edge_trim_reverted', audit['review_flags'])
            self.assertNotIn('large_edge_trim', audit['review_flags'])
            self.assertEqual(path.read_bytes(), original)

    def test_repaired_waveform_is_rechecked_against_duration_filter(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'audio').mkdir()
            original, old = root/'original.wav', root/'old.wav'
            samples = tone(4)
            write_pcm(original, samples)
            write_pcm(old, samples[6000:42000])
            before = inspect(original)
            settings = processing_options({'vad': dict(enabled=True, model='fixture', model_sha256='0'*64)})
            row = dict(id='fixture:cut', audio=str(old), original_audio=str(original),
                original_audio_sha256=digest(original), original_pcm_sha256=before['pcm_sha256'],
                audio_processing=dict(before=before, settings=settings, trim_backend='silero-vad',
                    denoise_decision='disabled', leading_trim_seconds=.25, trailing_trim_seconds=2.25,
                    leading_trim_samples=6000, trailing_trim_samples=54000, review_flags=['large_edge_trim'],
                    vad_selected=dict(segments=[dict(start=8000, end=24000)])))
            row.update(inspect(old))
            approved, approved_action = repair_trim(dict(row, human_review={'decision': 'keep'}), root, DEFAULT_FILTERS)
            self.assertEqual(approved_action, 'human_approved')
            self.assertEqual(approved['pcm_sha256'], row['pcm_sha256'])
            repaired, action = repair_trim(row, root, DEFAULT_FILTERS | {'max_duration': 3})
            self.assertEqual(action, 'trim_reverted')
            self.assertEqual(repaired['duration'], 4)
            self.assertEqual(repaired['rejection'], 'duration')
            self.assertIn('unsafe_edge_trim_reverted', repaired['audio_processing']['review_flags'])
            self.assertEqual(repaired['pcm_sha256'], before['pcm_sha256'])

    def test_review_finalization_preserves_partitions_and_repairs_references(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_pcm(root/'a.wav', tone(1.1))
            write_pcm(root/'b.wav', [-x for x in tone(1.1)])
            write_rows(root/'metadata.jsonl', [
                dict(id='a', audio='a.wav', text='Сайн уу?', speaker_id='same'),
                dict(id='b', audio='b.wav', text='Өдөр сайхан.', speaker_id='same')])
            cfg = dict(sources=[dict(id='fixture', kind='manifest', path=str(root), revision='fixture',
                       license='generated', source_url='local:fixture')], splits=dict(validation=0, test=0))
            prepare(cfg, root/'parent')
            rows = list(read_rows(root/'parent'/'manifest.jsonl'))
            rows[0]['split'] = 'validation'
            out = root/'reviewed'
            (out/'audio').mkdir(parents=True)
            processing, filters = validated_config(cfg)
            finalize(cfg, out, rows, [], filters, processing, preserve_assigned_splits=True)
            final = list(read_rows(out/'manifest.jsonl'))
            self.assertEqual([r['split'] for r in final], ['validation', 'train'])
            self.assertTrue(all(r['ref_audio'] == r['audio'] for r in final))
