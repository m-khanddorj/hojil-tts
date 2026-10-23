import json
import math
import struct
import sys
import tarfile
import tempfile
import types
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from data.adaptation import export_adaptation
from data.cli import main
from data.io import digest, extract, external, read_rows, write_json, write_rows
from data.pipeline import normalize, prepare, split_records
from data.qwen import encode, export, validate_codes
from data.sources import acquire, records, import_parquet


def wav(path, frequency=220, duration=1.1, amplitude=8000, rate=24000):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), 'wb') as f:
        f.setparams((1, 2, rate, 0, 'NONE', 'not compressed'))
        f.writeframes(b''.join(struct.pack('<h', int(amplitude * math.sin(2 * math.pi * frequency * i / rate)))
                             for i in range(int(duration * rate))))


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.raw = self.root / 'raw'
        self.raw.mkdir()
        self.source = dict(id='fixture', kind='manifest', revision='fixture-v1', license='generated-test',
                           source_url='local:synthetic-test', path=str(self.raw))

    def tearDown(self):
        self.tmp.cleanup()

    def config(self, rows, **extra):
        write_rows(self.raw / 'metadata.jsonl', rows)
        return dict(sources=[self.source], splits={'validation': 0, 'test': 0}, **extra)

    def row(self, ident, text='Сайн байна уу?', speaker='one', split='train', freq=220, **audio):
        path = self.raw / f'{ident}.wav'
        wav(path, frequency=freq, **audio)
        return dict(id=ident, audio=path.name, text=text, speaker_id=speaker, split=split)

    def test_fixture_cli_end_to_end_and_repeatability(self):
        rows = [self.row('a'), self.row('b', 'Өнөөдөр сайхан өдөр.', freq=330),
                self.row('c', 'Үүл харагдана.', speaker=None, split='test', freq=440),
                self.row('d', '  ', freq=550), self.row('e', 'Чимээгүй.', freq=660, amplitude=0)]
        cfg = self.config(rows)
        path = self.root / 'config.json'
        write_json(path, cfg)
        out = self.root / 'prepared'
        main(['prepare', '--config', str(path), '--output', str(out)])
        report = json.loads((out / 'report.json').read_text())
        self.assertEqual((report['accepted'], report['rejected']), (3, 2))
        self.assertEqual(report['reasons'], {'empty_text': 1, 'quiet': 1})
        self.assertAlmostEqual(report['accepted_hours'], 3.3/3600)
        accepted = list(read_rows(out / 'manifest.jsonl'))
        self.assertIsNone(next(r for r in accepted if r['id'].endswith(':c'))['ref_audio'])
        out2 = self.root / 'repeat'
        prepare(cfg, out2)
        self.assertEqual([(r['id'], r['split'], r['pcm_sha256']) for r in accepted],
                         [(r['id'], r['split'], r['pcm_sha256']) for r in read_rows(out2 / 'manifest.jsonl')])
        export(out / 'manifest.jsonl', 'fixture:one', self.root / 'qwen')
        exported = list(read_rows(self.root / 'qwen/train_raw.jsonl'))
        self.assertEqual(len(exported), 2)
        self.assertEqual(set(exported[0]), {'audio', 'text', 'ref_audio'})
        self.assertEqual(exported[0]['ref_audio'], exported[1]['ref_audio'])
        self.assertTrue(Path(exported[0]['audio']).is_absolute())
        with self.assertRaises(ValueError):
            export(out / 'manifest.jsonl', 'unknown', self.root / 'bad-export')
        self.assertEqual(len(list((out / 'audio').glob('*.wav'))), 3)

    def test_holdout_duplicate_text_and_speaker_transitive(self):
        rows = [self.row('test', split='test', speaker='A'),
                self.row('train', text='САЙН БАЙНА УУ!', speaker='B', freq=330),
                self.row('other', text='Өөр өгүүлбэр.', speaker='B', freq=440)]
        out = self.root / 'out'
        report = prepare(self.config(rows), out)
        self.assertEqual(report['accepted'], 1)
        self.assertEqual(report['reasons'], {'heldout_leakage': 2})
        self.assertEqual(next(read_rows(out / 'manifest.jsonl'))['split'], 'test')
        self.assertEqual(len(list((out / 'audio').glob('*.wav'))), 3)

    def test_pcm_duplicates_prefer_test(self):
        rows = [self.row('train'), self.row('test', split='test')]
        report = prepare(self.config(rows), self.root / 'out')
        self.assertEqual(report['accepted'], 1)
        self.assertEqual(report['rejected'], 1)
        self.assertEqual(next(read_rows(self.root / 'out/manifest.jsonl'))['source_split'], 'test')

    def test_quarantine_and_withheld(self):
        rows = [self.row('a')]
        self.source.update(kind='mbspeech', revision='a'*40)
        report = prepare(self.config(rows), self.root / 'out')
        self.assertEqual(report['reasons'], {'rights_quarantine': 1})
        with self.assertRaises(ValueError):
            acquire(self.source, self.root / 'download')

    def test_quality_and_text_preservation(self):
        self.assertEqual(normalize('  Өө\tҮү  12, test! '), 'Өө Үү 12, test!')
        rows = [self.row('short', duration=.1), self.row('english', text='Hello world', freq=330),
                self.row('control', text='Сайн\x00байна', freq=440),
                self.row('token', text='Сайн <|im_end|>', freq=550)]
        report = prepare(self.config(rows), self.root / 'out')
        self.assertEqual(report['accepted'], 0)
        self.assertEqual(set(report['reasons']), {'duration', 'insufficient_cyrillic', 'control_character', 'model_control_token'})

    def test_missing_decoder_fails_run(self):
        row = self.row('a', rate=16000)
        with patch('data.audio.ffmpeg_executable', side_effect=RuntimeError('missing')), self.assertRaises(RuntimeError):
            prepare(self.config([row]), self.root / 'out')

    def test_safe_archives_and_external_output(self):
        archive = self.root / 'bad.tar'
        with tarfile.open(archive, 'w') as f:
            member = tarfile.TarInfo('../escape')
            f.addfile(member)
        with self.assertRaises(ValueError):
            extract(archive, self.root / 'extract')
        repo = self.root / 'repo'
        (repo / '.git').mkdir(parents=True)
        with self.assertRaises(ValueError):
            external(repo / 'audio')

    def test_common_voice_reserves_official_splits(self):
        self.source.update(kind='common_voice', revision='27.0')
        header = 'client_id\tpath\tsentence\n'
        (self.raw / 'test.tsv').write_text(header + 'a\ta.wav\tСайн уу\n', encoding='utf-8')
        (self.raw / 'dev.tsv').write_text(header + 'b\tb.wav\tӨдөр\n', encoding='utf-8')
        (self.raw / 'validated.tsv').write_text(header + 'a\ta.wav\tСайн уу\n' + 'c\tc.wav\tҮүл\n', encoding='utf-8')
        result = list(records(self.source))
        self.assertEqual([(r['original_clip_id'], r['source_split']) for r in result],
                         [('a.wav', 'test'), ('b.wav', 'validation'), ('c.wav', 'train')])

    def test_fleurs_schema_and_unknown_speaker(self):
        self.source.update(kind='fleurs', revision='a'*40)
        for split in ('train', 'dev', 'test'):
            p = self.raw / 'audio' / split / split / 'file.wav'
            wav(p)
            (self.raw / f'{split}.tsv').write_text('1\tfile.wav\tСайн уу?\tсайн уу\tс а й н\t26400\tMALE\n', encoding='utf-8')
        result = list(records(self.source))
        self.assertEqual([r['source_split'] for r in result], ['train', 'validation', 'test'])
        self.assertTrue(all(r['speaker_id'] is None for r in result))
        self.assertEqual(result[0]['original_text'], 'Сайн уу?')

    def test_seeded_groups_are_order_independent(self):
        rows = [dict(id=str(i), text=f'Өгүүлбэр {i}', pcm_sha256=str(i), speaker_id=str(i//2),
                     source='s', source_split='train') for i in range(40)]
        a, _ = split_records(rows, 4, {'test': .2, 'validation': .2})
        b, _ = split_records(list(reversed(rows)), 4, {'test': .2, 'validation': .2})
        self.assertEqual({r['id']: r['split'] for r in a}, {r['id']: r['split'] for r in b})
        self.assertEqual({r['split'] for r in a}, {'train', 'validation', 'test'})
        for i in range(20):
            self.assertEqual(len({r['split'] for r in a if r['speaker_id'] == str(i)}), 1)

    def test_fleurs_literal_quotes_do_not_merge_or_escape_records(self):
        self.source.update(kind='fleurs', revision='a'*40)
        for split in ('train', 'dev', 'test'):
            for name in ('one.wav', 'two.wav'):
                wav(self.raw/'audio'/split/name)
            (self.raw/f'{split}.tsv').write_text(
                '1\tone.wav\t"Сайн уу?\tсайн уу\tс а й н\t26400\tMALE\n'
                '2\ttwo.wav\tТэр "Өдөр" гэв.\tтэр өдөр гэв\tт э р\t26400\tFEMALE\n',
                encoding='utf-8')
        result = list(records(self.source))
        self.assertEqual(len(result), 6)
        self.assertEqual(result[0]['original_text'], '"Сайн уу?')
        self.assertEqual(result[1]['original_text'], 'Тэр "Өдөр" гэв.')
    def test_audio_duplicate_does_not_hide_speaker_leakage(self):
        rows = [self.row('test', split='test', speaker='A'),
                self.row('duplicate', text='Өөр үг.', speaker='B'),
                self.row('same-speaker', text='Шинэ өгүүлбэр.', speaker='B', freq=440)]
        report = prepare(self.config(rows), self.root / 'out')
        self.assertEqual(report['accepted'], 1)
        self.assertEqual(report['reasons'], {'heldout_leakage': 2})

    def test_bad_audio_and_invalid_filters(self):
        row = self.row('bad')
        (self.raw / row['audio']).write_bytes(b'not audio')
        with patch('data.pipeline.convert', side_effect=wave.Error('bad')):
            report = prepare(self.config([row]), self.root / 'out')
        self.assertEqual(report['reasons'], {'audio_decode': 1})
        with self.assertRaises(ValueError):
            prepare(self.config([], filters={'max_duration': float('nan')}), self.root / 'invalid')


    def test_local_archive_acquisition_checksum(self):
        source = self.source | {'kind': 'common_voice', 'revision': '27.0'}
        (self.raw / 'note.txt').write_text('fixture')
        archive = self.root / 'source.tar.gz'
        with tarfile.open(archive, 'w:gz') as f:
            f.add(self.raw / 'note.txt', arcname='mn/note.txt')
        source.update(archive=str(archive), archive_sha256=digest(archive))
        acquire(source, self.root / 'acquired')
        self.assertEqual((self.root / 'acquired/fixture/mn/note.txt').read_text(), 'fixture')
        self.assertTrue((self.root / 'acquired/fixture/acquisition.json').is_file())
        with self.assertRaises(ValueError):
            acquire(source | {'archive_sha256': 'wrong'}, self.root / 'bad-acquisition')

    def test_fleurs_download_scope_mock(self):
        source = self.source | {'kind': 'fleurs', 'revision': 'a'*40}
        calls = []
        def fake_download(url, target, expected=None):
            calls.append(url)
            target.parent.mkdir(parents=True, exist_ok=True)
            if str(target).endswith('.tar.gz'):
                with tarfile.open(target, 'w:gz') as f:
                    item = tarfile.TarInfo('file.wav')
                    f.addfile(item)
            else:
                target.write_text('metadata')
        with patch('data.sources.download', side_effect=fake_download):
            acquire(source, self.root / 'acquired')
        self.assertEqual(len(calls), 6)
        self.assertTrue(all('/data/mn_mn/' in url and '/'+('a'*40)+'/' in url for url in calls))
        log = json.loads((self.root / 'acquired/fixture/acquisition.json').read_text())
        self.assertEqual(len(log['files']), 6)

    def test_mbspeech_parquet_interchange_mock(self):
        source = self.source | {'kind': 'mbspeech', 'revision': 'a'*40,
                                'rights_clearance': 'test-evidence-not-real-data'}
        audio = self.row('sample')
        parquet = self.root / 'input.parquet'
        parquet.write_bytes(b'fixture-placeholder')
        data = [{'audio': {'bytes': (self.raw / audio['audio']).read_bytes()},
                 'sentence': 'Сайн байна уу?', 'split': 'test'}]
        class ParquetFile:
            def __init__(self, path): pass
            def iter_batches(self, batch_size):
                yield types.SimpleNamespace(to_pylist=lambda: data)
        package = types.ModuleType('pyarrow')
        module = types.ModuleType('pyarrow.parquet')
        module.ParquetFile = ParquetFile
        package.parquet = module
        with patch.dict(sys.modules, {'pyarrow': package, 'pyarrow.parquet': module}):
            import_parquet(source, [parquet], self.root / 'imported')
        result = list(records(source | {'path': str(self.root / 'imported')}))
        self.assertEqual(len(result), 1)
        self.assertIsNone(result[0]['speaker_id'])
        self.assertEqual(result[0]['source_split'], 'test')
        self.assertEqual(result[0]['source_metadata']['parquet_sha256'], digest(parquet))

    def test_download_rejects_credentials_and_signed_inputs(self):
        from data.io import download
        for url in ('http://example.com/data', 'https://u:p@example.com/data',
                    'https://example.com/data?token=secret'):
            with self.assertRaises(ValueError):
                download(url, self.root / 'download')
        import io
        payload = io.BytesIO(b'short')
        payload.headers = {'Content-Length': '100'}
        with patch('data.io.urllib.request.urlopen', return_value=payload):
            with self.assertRaisesRegex(ValueError, 'Content-Length mismatch'):
                download('https://example.com/data', self.root / 'download')
        self.assertFalse((self.root / 'download').exists())
        self.assertFalse((self.root / 'download.partial').exists())

    def test_official_validation_test_conflict_reported(self):
        rows = [self.row('test', speaker='a', split='test'),
                self.row('dev', speaker='b', split='validation', freq=330)]
        report = prepare(self.config(rows), self.root / 'out')
        self.assertEqual(report['accepted'], 1)
        self.assertEqual(report['reasons'], {'heldout_leakage': 1})
        rejection = next(read_rows(self.root / 'out/rejected.jsonl'))
        self.assertEqual(rejection['source_split'], 'validation')


    def test_primary_multispeaker_adaptation_export_preserves_unknowns(self):
        rows = [self.row('a', speaker='one'),
                self.row('b', text='Өөр дуу хоолой.', speaker='two', freq=330),
                self.row('c', text='Үүлтэй өдөр.', speaker=None, split='test', freq=440)]
        prepare(self.config(rows), self.root / 'prepared')
        output = self.root / 'adaptation'
        main(['export-adaptation', '--manifest', str(self.root / 'prepared/manifest.jsonl'),
              '--output', str(output)])
        train = list(read_rows(output / 'train.jsonl'))
        test = list(read_rows(output / 'test.jsonl'))
        self.assertEqual({r['speaker_id'] for r in train}, {'fixture:one', 'fixture:two'})
        self.assertEqual(len(train) + len(test), 3)
        self.assertIsNone(test[0]['speaker_id'])
        self.assertIsNone(test[0]['ref_audio'])
        self.assertEqual(test[0]['language'], 'mn-MN')
        contract = json.loads((output / 'contract.json').read_text())
        self.assertEqual(contract['unknown_speaker_rows'], 1)
        self.assertEqual(contract['trainer_status'], 'integration-required-not-stock-qwen-sft')
        self.assertTrue(all(r['ref_audio'] == r['audio'] for r in train))
        wav(Path(train[0]['audio']), frequency=999)
        with self.assertRaises(ValueError):
            export_adaptation(self.root / 'prepared/manifest.jsonl', self.root / 'tampered')


    def test_acquisition_resumes_completed_files_and_detects_tampering(self):
        source = self.source | {'kind': 'fleurs', 'revision': 'a'*40}
        calls = []
        interrupted = [False]
        def fake_download(url, target, expected=None):
            calls.append(url)
            if len(calls) == 3 and not interrupted[0]:
                interrupted[0] = True
                raise OSError('network interruption')
            target.parent.mkdir(parents=True, exist_ok=True)
            if str(target).endswith('.tar.gz'):
                with tarfile.open(target, 'w:gz') as f:
                    f.addfile(tarfile.TarInfo('clip.wav'))
            else:
                target.write_text('metadata')
        out = self.root / 'acquired'
        with patch('data.sources.download', side_effect=fake_download):
            with self.assertRaises(OSError):
                acquire(source, out)
            self.assertFalse((out / source['id']).exists())
            acquire(source, out)
        self.assertEqual(len(calls), 7)  # six files plus interrupted request, no repeated successes
        cache = out / '.acquisition-cache' / source['id']
        (cache / 'train.tsv').write_text('tampered')
        import shutil
        shutil.rmtree(out / source['id'])
        with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
            acquire(source, out)

    def test_archive_links_duplicates_and_collisions_rejected_before_writing(self):
        for mode in ('duplicate', 'symlink', 'hardlink', 'absolute', 'collision'):
            archive = self.root / f'{mode}.tar'
            with tarfile.open(archive, 'w') as f:
                member = tarfile.TarInfo('/absolute' if mode == 'absolute' else 'file')
                if mode in ('symlink', 'hardlink'):
                    member.type = tarfile.SYMTYPE if mode == 'symlink' else tarfile.LNKTYPE
                    member.linkname = '../escape'
                f.addfile(member)
                if mode in ('duplicate', 'collision'):
                    f.addfile(tarfile.TarInfo('file' if mode == 'duplicate' else 'file/child'))
            dest = self.root / ('out-' + mode)
            with self.assertRaises(ValueError):
                extract(archive, dest)
            self.assertFalse(dest.exists())

    def test_code_schema(self):
        row = dict(audio='/a.wav', text='Сайн', ref_audio='/b.wav', audio_codes=[[1]*16])
        validate_codes(row)
        for codes in ([], [[1]*15], [[True]*16], [[-1]*16]):
            with self.assertRaises(ValueError):
                validate_codes(row | {'audio_codes': codes})

    def test_encoder_bounded_batches_and_resume(self):
        rows = [self.row(str(i), freq=200+i*100) for i in range(3)]
        prepare(self.config(rows), self.root / 'out')
        export(self.root / 'out/manifest.jsonl', 'fixture:one', self.root / 'qwen')
        model = self.root / 'model'
        model.mkdir()
        (model / 'config.json').write_text('{}')
        class Code:
            def cpu(self): return self
            def tolist(self): return [[1]*16]
        class Tokenizer:
            calls = 0
            @classmethod
            def from_pretrained(cls, *a, **kw): return cls()
            def encode(self, audios):
                self.__class__.calls += 1
                if self.calls == 2: raise RuntimeError('simulated interruption')
                return types.SimpleNamespace(audio_codes=[Code() for _ in audios])
        args = (self.root / 'qwen/train_raw.jsonl', self.root / 'coded.jsonl', model)
        with patch.dict(sys.modules, {'qwen_tts': types.SimpleNamespace(Qwen3TTSTokenizer=Tokenizer)}):
            with self.assertRaises(RuntimeError): encode(*args)
            encode(*args)
        self.assertEqual(len(list(read_rows(args[1]))), 3)
        self.assertEqual(Tokenizer.calls, 4)


if __name__ == '__main__':
    unittest.main()
