import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from data.bulk import atomic_json, prepare_bulk
from data.pipeline import prepare, prepare_record
from data.io import read_rows, write_rows
from test_pipeline import wav


class BulkTests(unittest.TestCase):
    def test_atomic_progress_retries_transient_file_lock(self):
        target = self.root/'status.json'
        real_replace = Path.replace
        attempts = []
        def locked_once(src, dst):
            attempts.append(1)
            if len(attempts) == 1:
                raise PermissionError('file is open for reading')
            return real_replace(src, dst)
        with patch.object(Path, 'replace', locked_once), patch('data.bulk.time.sleep'):
            atomic_json(target, {'stage': 'complete'})
        self.assertEqual(len(attempts), 2)
        self.assertEqual(json.loads(target.read_text()), {'stage': 'complete'})

    def test_atomic_progress_does_not_mask_permanent_permission_failure(self):
        target = self.root/'status.json'
        target.write_text('{"stage":"old"}')
        with patch.object(Path, 'replace', side_effect=PermissionError('denied')), \
                patch('data.bulk.time.sleep'), self.assertRaises(PermissionError):
            atomic_json(target, {'stage': 'new'})
        self.assertEqual(json.loads(target.read_text()), {'stage': 'old'})

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.raw = self.root/'raw'
        self.raw.mkdir()
        rows = []
        for i in range(6):
            wav(self.raw/f'{i}.wav', frequency=220+i*100)
            rows.append(dict(id=str(i), audio=f'{i}.wav', text=f'Өдөр {i}',
                             split='test' if i == 0 else 'train', speaker_id='one' if i < 2 else str(i)))
        write_rows(self.raw/'metadata.jsonl', rows)
        self.cfg = dict(sources=[dict(id='fixture', kind='manifest', revision='v1', license='generated',
                                    source_url='local:generated', path=str(self.raw))])

    def tearDown(self):
        self.tmp.cleanup()

    def test_parallel_matches_serial_with_global_speaker_holdout(self):
        serial = prepare(self.cfg, self.root/'serial')
        parallel = prepare_bulk(self.cfg, self.root/'parallel', workers=2)
        self.assertEqual(serial['reasons'], parallel['reasons'])
        def signature(path):
            return [(r['id'], r['split'], r['pcm_sha256']) for r in read_rows(path/'manifest.jsonl')]
        self.assertEqual(signature(self.root/'serial'), signature(self.root/'parallel'))
        self.assertEqual(prepare_bulk(self.cfg, self.root/'parallel', workers=2, resume=True)['accepted'], parallel['accepted'])
        self.assertEqual(json.loads((self.root/'parallel/progress.json').read_text())['status'], 'complete')

    def test_interruption_reuses_checkpoints_and_detects_tampering(self):
        calls = 0
        def interrupted(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 3:
                raise RuntimeError('simulated interruption')
            return prepare_record(*args, **kwargs)
        out = self.root/'interrupted'
        with patch('data.bulk.prepare_record', side_effect=interrupted), self.assertRaises(RuntimeError):
            prepare_bulk(self.cfg, out, workers=1)
        self.assertTrue(list((out/'checkpoints').glob('*.json')))
        with patch('data.bulk.prepare_record', wraps=prepare_record) as process:
            report = prepare_bulk(self.cfg, out, workers=2, resume=True)
        self.assertLess(process.call_count, 6)
        self.assertGreater(report['accepted'], 0)
        row = next(read_rows(out/'manifest.jsonl'))
        Path(row['audio']).write_bytes(b'tampered')
        with self.assertRaises((ValueError, EOFError)):
            prepare_bulk(self.cfg, out, workers=2, resume=True)

    def test_resume_config_change_is_rejected(self):
        out = self.root/'out'
        prepare_bulk(self.cfg, out, workers=2)
        with self.assertRaisesRegex(ValueError, 'identical'):
            prepare_bulk(dict(self.cfg, seed=7), out, workers=2, resume=True)
