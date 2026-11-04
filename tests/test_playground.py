import tempfile,unittest,json,threading,urllib.request,urllib.error,io,base64,importlib.util
from pathlib import Path
from http.server import ThreadingHTTPServer
from unittest.mock import patch
from data.playground import Playground,make_handler
class PlaygroundTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.p=Path(self.tmp.name);profile=self.p/'profile.json';profile.write_text(json.dumps({'reference':{'id':'default-ref','text':'Reference','local_audio':'unused','sha256':'unused'}}))
  self.app=Playground(self.p/'model',profile,self.p/'versions')
  self.server=ThreadingHTTPServer(('127.0.0.1',0),make_handler(self.app));self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start();self.url='http://127.0.0.1:'+str(self.server.server_port)
 def tearDown(self):
  self.server.shutdown();self.server.server_close();self.app.pool.shutdown(wait=True);self.tmp.cleanup()
 def test_submission_busy_and_seed(self):
  with patch.object(self.app.pool,'submit'):
   r=self.app.submit('Сайн байна уу.',2028);self.assertEqual(r['seed'],2028)
   self.assertEqual(self.app.history()[0]['text'],'Сайн байна уу.')
   with self.assertRaises(ValueError):self.app.submit('Again',2029)
 def test_invalid_input(self):
  for text,seed in [('',1),('x'*351,1),('x',-1),('x',True),('x',2**31)]:
   with self.assertRaises(ValueError):self.app.submit(text,seed)
 def test_post_requires_token_and_local_origin(self):
  for headers in [{},{'X-Playground-Token':self.app.token,'Origin':'https://example.com'}]:
   req=urllib.request.Request(self.url+'/api/generate',data=b'{}',headers=headers)
   with self.assertRaises(urllib.error.HTTPError) as e:urllib.request.urlopen(req)
   self.assertEqual(e.exception.code,403)
 def test_audio_path_traversal_denied(self):
  with self.assertRaises(urllib.error.HTTPError) as e:urllib.request.urlopen(self.url+'/audio/../profile.json')
  self.assertEqual(e.exception.code,404)
 @unittest.skipUnless(importlib.util.find_spec('numpy') and importlib.util.find_spec('soundfile'), 'Requires audio dependencies')
 def test_reference_validation_and_retry_pinning(self):
  import numpy as np,soundfile as sf
  stream=io.BytesIO();sf.write(stream,.1*np.sin(np.arange(4*24000)*.03),24000,format='WAV',subtype='PCM_16')
  data={'audio':base64.b64encode(stream.getvalue()).decode(),'transcript':'Test recording','label':'Test voice'}
  ref=self.app.upload_reference(data)
  self.assertEqual(ref['seconds'],4)
  with patch.object(self.app.pool,'submit'):
   first=self.app.submit('Hello',1,reference_id=ref['selection_id']);self.app.busy=False
   retry=self.app.submit('Hello',2,parent=first['id'],reference_id='default')
  self.assertEqual(first['reference'],retry['reference'])
  self.assertEqual(retry['reference']['text'],'Test recording')
  self.assertEqual(self.app.reference_list()[0]['label'],'Test voice')
  for invalid in [dict(data,transcript=''),dict(data,audio='not-base64')]:
   with self.assertRaises(ValueError):self.app.upload_reference(invalid)
if __name__=='__main__':unittest.main()
