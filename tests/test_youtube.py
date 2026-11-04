import json
import math
import tempfile
import unittest
import wave
from array import array
from pathlib import Path
from unittest.mock import patch

from data.youtube import Intake, captions, segment, video_url
from data.io import read_rows


class YoutubeTests(unittest.TestCase):
    def test_video_urls(self):
        expected = 'https://www.youtube.com/watch?v=abcdefghijk'
        for value in [expected + '&list=anything', 'https://youtu.be/abcdefghijk?t=3',
                      'https://www.youtube.com/shorts/abcdefghijk']:
            self.assertEqual(video_url(value), expected)
        for value in ['http://youtu.be/abcdefghijk', 'https://youtube.com.evil/watch?v=abcdefghijk',
                      'https://www.youtube.com/playlist?list=x', 'https://user@youtube.com/watch?v=abcdefghijk']:
            with self.assertRaises(ValueError):
                video_url(value)

    def test_caption_parser(self):
        cues = captions({'events': [{'tStartMs': 2000, 'dDurationMs': 1200,
                                     'segs': [{'utf8': 'Сайн '}, {'utf8': 'байна уу &amp; та'}]}]})
        self.assertEqual(cues[0]['text'], 'Сайн байна уу & та')
        self.assertEqual(cues[0]['start'], 2)
        with self.assertRaises(ValueError):
            captions({'events': []})

    def test_automatic_captions_never_downloaded(self):
        import sys
        import types
        from data.youtube import acquire
        class YDL:
            def __init__(self, options): pass
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def extract_info(self, *args, **kwargs):
                return dict(duration=10, subtitles={}, automatic_captions={'mn': [{'ext': 'json3'}]})
            def process_ie_result(self, *args, **kwargs):
                raise AssertionError('Must not download automatic captions')
        fake = types.ModuleType('yt_dlp')
        fake.YoutubeDL = YDL
        fake.version = types.ModuleType('yt_dlp.version')
        with patch.dict(sys.modules, {'yt_dlp': fake, 'yt_dlp.version': fake.version}), \
             patch('data.youtube.ffmpeg_executable', return_value='ffmpeg'), tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, 'No uploaded mn'):
                acquire('https://www.youtube.com/watch?v=abcdefghijk', 'mn', Path(tmp), lambda _: None)

    def test_segment_and_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / ('a' * 32)
            folder.mkdir()
            source = root / 'source.wav'
            with wave.open(str(source), 'wb') as f:
                f.setparams((1, 2, 24000, 0, 'NONE', 'not compressed'))
                f.writeframes(array('h', [int(5000 * math.sin(i * .06)) for i in range(24000*5)]).tobytes())
            def decode(command, **kwargs):
                import shutil
                shutil.copyfile(source, command[-1])
            cues = [dict(start=0, end=2, text='Сайн байна уу.'),
                    dict(start=2, end=4, text='Өнөөдөр сайхан өдөр байна.'),
                    dict(start=4, end=4.2, text='Тийм.')]
            with patch('data.youtube.ffmpeg_executable', return_value='ffmpeg'), \
                 patch('data.youtube.subprocess.run', side_effect=decode):
                clips = segment(source, cues, folder)
            self.assertNotIn('rejection', clips[0])
            self.assertIn('rejection', clips[2])
            intake = Intake(root)
            identity = folder.name
            intake.jobs[identity] = dict(id=identity, state='review', clips=clips,
                                        url='https://www.youtube.com/watch?v=abcdefghijk', clearance='Own recording',
                                        provenance={'video_id': 'abcdefghijk'})
            with self.assertRaisesRegex(ValueError, 'Approve'):
                intake.export(identity)
            intake.review(identity, dict(id='00000', text=clips[0]['text'], speaker_id='speaker1', approved=True))
            output = Path(intake.export(identity))
            rows = list(read_rows(output / 'manifest.jsonl'))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]['speaker_id'], 'youtube:speaker1')
            self.assertTrue(rows[0]['source_metadata']['reviewed'])
            self.assertTrue((output / 'report.json').is_file())
            intake.pool.shutdown()

    def test_acquisition_keeps_extraction_session(self):
        import sys
        import types
        from data.youtube import acquire
        instances = []
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            class YDL:
                def __init__(self, options):
                    instances.append(self)
                    self.options = options
                    self.extracted = False
                def __enter__(self): return self
                def __exit__(self, *args): pass
                def extract_info(self, url, download):
                    self.extracted = True
                    assert download is False
                    assert self.options['format'] == 'bestaudio/best'
                    assert self.options['writeautomaticsub'] is False
                    return dict(id='abcdefghijk', duration=5, subtitles={'mn': [
                        {'ext': 'json3', 'url': 'https://example.test/captions'},
                        {'ext': 'json3', 'url': 'https://example.test/captions?tlang=mn'}]})
                def process_ie_result(self, info, download):
                    assert self.extracted, 'Media transfer must reuse extraction session'
                    assert download is True
                    assert len(info['subtitles']['mn']) == 1
                    (folder / 'original.webm').write_bytes(b'fixture')
                    (folder / 'original.mn.json3').write_text(json.dumps({'events': [
                        {'tStartMs': 0, 'dDurationMs': 2000, 'segs': [{'utf8': 'Сайн байна уу'}]}]}))
                    return info
                def prepare_filename(self, info): return str(folder / 'original.webm')
            fake = types.ModuleType('yt_dlp')
            fake.YoutubeDL = YDL
            fake.version = types.ModuleType('yt_dlp.version')
            fake.version.__version__ = 'fixture'
            with patch.dict(sys.modules, {'yt_dlp': fake, 'yt_dlp.version': fake.version}), \
                 patch('data.youtube.ffmpeg_executable', return_value='ffmpeg'):
                media, cues, provenance = acquire('https://www.youtube.com/watch?v=abcdefghijk',
                                                  'mn', folder, lambda _: None)
            self.assertEqual(len(instances), 1)
            self.assertTrue(media.is_file())
            self.assertEqual(len(cues), 1)
            self.assertEqual(provenance['subtitle_type'], 'uploaded')

    def test_no_artifacts_in_repository(self):
        with self.assertRaisesRegex(ValueError, 'outside Git'):
            Intake(Path(__file__).resolve().parents[1] / 'downloads')

    def test_interrupted_jobs_marked_failed(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / ('b' * 32)
            folder.mkdir()
            (folder / 'job.json').write_text(json.dumps(dict(id=folder.name, state='working')))
            intake = Intake(tmp)
            self.assertEqual(intake.jobs[folder.name]['state'], 'failed')
            intake.pool.shutdown()


if __name__ == '__main__':
    unittest.main()
