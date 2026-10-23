import hashlib
from pathlib import Path
import tempfile
import unittest
import wave

from data.pipeline import split_records, validated_config
from data.repartition import overlap_audit, pcm_hash, restore
from data.io import digest


class RepartitionTests(unittest.TestCase):
    def test_rebuilt_splits_keep_all_rows_and_isolate_shared_sentences(self):
        rows = [dict(id=str(i), text=f'Sentence {i // 2}', pcm_sha256=str(i),
                     original_pcm_sha256='original' + str(i), speaker_id='shared-speaker',
                     source='fixture', source_split='test' if i % 3 else 'train') for i in range(200)]
        accepted, rejected = split_records(rows, 2026, {'validation': .1, 'test': .1}, False, False)
        self.assertEqual(len(accepted), len(rows))
        self.assertFalse(rejected)
        audit = overlap_audit(accepted)
        self.assertEqual(audit, dict(text_keys_across_splits=0, audio_keys_across_splits=0,
                                    original_audio_keys_across_splits=0, speaker_keys_across_splits=1))
        again, _ = split_records(list(reversed(rows)), 2026, {'validation': .1, 'test': .1}, False, False)
        self.assertEqual({r['id']: r['split'] for r in accepted}, {r['id']: r['split'] for r in again})

    def test_policy_flags_must_be_booleans(self):
        for name in ('speaker_disjoint', 'preserve_source_splits'):
            with self.assertRaises(ValueError):
                validated_config({name: 'false'})

    def test_reconstruct_saved_trim_and_fail_on_changed_pcm(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            original = root / 'original.wav'
            raw = b'\x01\x00' * 24000
            with wave.open(str(original), 'wb') as audio:
                audio.setparams((1, 2, 24000, 0, 'NONE', 'not compressed'))
                audio.writeframes(raw)
            trimmed = raw[200:len(raw) - 400]
            row = dict(id='clip', original_audio=str(original), audio=str(root / 'missing.wav'),
                       original_audio_sha256=digest(original), original_pcm_sha256=pcm_hash(original),
                       pcm_sha256=hashlib.sha256(trimmed).hexdigest(),
                       audio_processing=dict(denoise_decision='reverted_quality_gate',
                                             leading_trim_samples=100, trailing_trim_samples=200))
            target = root / 'restored.wav'
            self.assertEqual(restore(row, target), 'reconstructed')
            self.assertEqual(pcm_hash(target), row['pcm_sha256'])
            self.assertEqual(restore(row, target), 'reused')
            row['pcm_sha256'] = 'invalid'
            with self.assertRaises(ValueError):
                restore(row, root / 'bad.wav')
            self.assertFalse((root / 'bad.wav').exists())
