import copy
import hashlib
from pathlib import Path
import tempfile
import unittest
import wave

from training import reference_masked_supervision_broad_data as d


def pair(identity,reference,speaker='s'):
    return dict(id=identity,ref_id=reference,speaker_id=speaker,audio=identity+'.wav',ref_audio=reference+'.wav',
        text='text '+identity,ref_text='text '+reference,pcm_sha256='pcm'+identity,original_pcm_sha256='rawpcm'+identity,
        original_audio_sha256='rawaudio'+identity,original_clip_id=identity,split='train',duration=1.,sample_rate=24000,
        source='cv27',source_family='common_voice',revision='27.0',license='CC0-1.0; preserve separate access terms',language='mn-MN')


class BroadDataTests(unittest.TestCase):
    def test_unicode_text_guard_normalization(self):
        self.assertEqual(d.text_key(' ӨӨ,\tҮҮ! '),d.text_key('өө үү'))
        self.assertEqual(d.text_key('E\u0301'),d.text_key('é'))

    def test_text_exclusion_applies_both_pair_sides(self):
        a,b=pair('a','b'),pair('b','a');canonical={'a':a,'b':b}
        for text in (a['text'],b['text']):
            self.assertTrue(d.conflicts(a,canonical,{d.text_key(text)},d.blocked_identity([])))
        self.assertFalse(d.conflicts(a,canonical,{'different'},d.blocked_identity([])))

    def test_all_identity_fields_block_both_sides(self):
        a,b=pair('a','b'),pair('b','a');canonical={'a':a,'b':b}
        for side in (a,b):
            for key in d.IDENTITIES:
                with self.subTest(side=side['id'],key=key):
                    self.assertTrue(d.conflicts(a,canonical,set(),{key:{side[key]}}))

    def test_whole_speaker_exclusion(self):
        rows=[pair('a','b','blocked'),pair('b','a','blocked'),pair('c','e','safe'),pair('e','c','safe')]
        result,counts=d.eligible_pool(rows,{r['id']:r for r in rows},{'blocked'},[])
        self.assertEqual([r['id'] for r in result],['c','e']);self.assertEqual(counts['after_speaker_exclusions'],2)

    def test_reference_closure_cascades(self):
        result,rounds,removed=d.reference_closure([dict(id='a',ref_id='b'),dict(id='b',ref_id='missing'),dict(id='c',ref_id='e'),dict(id='e',ref_id='c')])
        self.assertEqual({r['id'] for r in result},{'c','e'});self.assertEqual(rounds,[3,2]);self.assertEqual(set(removed),{'a','b'})

    def test_canonical_and_reference_tampering_rejected(self):
        a,b=pair('a','b'),pair('b','a');canonical=copy.deepcopy({'a':a,'b':b})
        for field,value in [('text','changed'),('ref_text','changed'),('pcm_sha256','changed')]:
            altered=dict(a,**{field:value})
            with self.assertRaises(ValueError):d.eligible_pool([altered,b],canonical,set(),[])

    def test_distinct_same_speaker_contract(self):
        a,b=pair('a','b'),pair('b','a');d.validate_pair(a,b)
        for field,value in [('speaker_id','other'),('split','test'),('text',a['text']),('pcm_sha256',a['pcm_sha256']),('original_audio_sha256',a['original_audio_sha256'])]:
            with self.assertRaises(ValueError):d.validate_pair(a,dict(b,**{field:value}))

    def test_deterministic256_schedule_and_probe_subset(self):
        pool=[]
        for i in range(256):pool.extend([pair(f'a{i}',f'b{i}',f's{i}'),pair(f'b{i}',f'a{i}',f's{i}')])
        selected,plan=d.selected_plan(pool);self.assertEqual((selected,plan),d.selected_plan(list(reversed(pool))))
        self.assertEqual(len(plan['fitting_feature_ids']),512);self.assertEqual(len(plan['updates']),32)
        self.assertEqual(len(plan['fitting_probe_pair_ids']),16)
        self.assertTrue(set(plan['fitting_probe_pair_ids'])<=set(plan['selected_fitting_pair_ids']))
        index={r['id']:r for r in pool}
        for i,u in enumerate(plan['updates'],1):
            self.assertEqual(len({index[x]['speaker_id'] for x in u['pair_ids']}),8)
            self.assertEqual(u['modes'],['full','half'] if i%2 else ['half','full'])
            self.assertEqual((u['pair_weight'],u['mode_weight']),(.125,.5))

    def test_old_narrow_population_cannot_claim256_speakers(self):
        pool=[]
        for i in range(47):pool.extend([pair(f'a{i}',f'b{i}',f's{i}'),pair(f'b{i}',f'a{i}',f's{i}')])
        with self.assertRaises(ValueError):d.selected_plan(pool)

    def make_wav(self,folder,channels=1,rate=24000,width=2):
        path=Path(folder)/'synthetic.wav';raw=b'\0'*(24*channels*width)
        with wave.open(str(path),'wb') as f:f.setnchannels(channels);f.setsampwidth(width);f.setframerate(rate);f.writeframes(raw)
        return path,dict(sample_rate=rate,samples=24,duration=24/rate,pcm_sha256=hashlib.sha256(raw).hexdigest())

    def test_selected_wav_rehash_and_pcm_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            path,row=self.make_wav(tmp);receipt=d.audit_wav(path,d.digest(path),row);self.assertEqual(receipt['samples'],24)
            with self.assertRaises(ValueError):d.audit_wav(path,'changed',row)
            with self.assertRaises(ValueError):d.audit_wav(path,d.digest(path),dict(row,pcm_sha256='changed'))

    def test_wav_header_samples_and_duration(self):
        with tempfile.TemporaryDirectory() as tmp:
            for args in ({'channels':2},{'rate':16000},{'width':1}):
                path,row=self.make_wav(tmp,**args)
                with self.assertRaises(ValueError):d.audit_wav(path,d.digest(path),row)
            path,row=self.make_wav(tmp)
            for key,value in [('samples',25),('duration',float('nan')),('duration',10.)]:
                with self.assertRaises(ValueError):d.audit_wav(path,d.digest(path),dict(row,**{key:value}))

    def test_path_scope_and_source_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(d.audio_path(Path(tmp),'safe.wav').parent,Path(tmp).resolve())
            with self.assertRaises(ValueError):d.audio_path(Path(tmp),'../outside.wav')
        row=pair('a','b');d.source_contract(row)
        for key,value in [('source','unknown'),('split','test'),('language','en'),('license','unverified')]:
            with self.assertRaises(ValueError):d.source_contract(dict(row,**{key:value}))


if __name__=='__main__':unittest.main()
