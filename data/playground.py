"""Local-only CPU playground. Model/audio/history stay outside Git."""
import argparse,hashlib,json,secrets,threading,time,uuid,os,base64,io
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlsplit
from concurrent.futures import ThreadPoolExecutor

def write_json(path,value):
 temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');temp.replace(path)

class Playground:
 def __init__(self,model,profile,output):
  self.model_path=model;self.profile=json.loads(profile.read_text(encoding='utf-8'));self.output=output
  output.mkdir(parents=True,exist_ok=True);self.lock=threading.Lock();self.pool=ThreadPoolExecutor(max_workers=1)
  self.references=output.parent/'references';self.references.mkdir(exist_ok=True)
  self.wrapper=None;self.busy=False;self.stage='Ready';self.token=secrets.token_urlsafe(32)
 def history(self):
  return sorted([json.loads(p.read_text(encoding='utf-8')) for p in self.output.glob('*.json')],key=lambda r:r['created'],reverse=True)
 def reference(self,identity='default'):
  if identity=='default':return dict(self.profile['reference'],label=self.profile['reference'].get('label','Alternate 2'),selection_id='default')
  if not isinstance(identity,str) or len(identity)!=32 or any(c not in '0123456789abcdef' for c in identity):raise ValueError('Invalid reference')
  return json.loads((self.references/(identity+'.json')).read_text(encoding='utf-8'))
 def reference_list(self):
  return [{'selection_id':r['selection_id'],'label':r['label'],'text':r['text'],'seconds':r['seconds']} for r in (json.loads(p.read_text(encoding='utf-8')) for p in self.references.glob('*.json'))]
 def upload_reference(self,data):
  import soundfile as sf,numpy as np
  transcript=data.get('transcript','');encoded=data.get('audio','');label=data.get('label','Uploaded voice')
  if not isinstance(transcript,str) or not 1<=len(transcript.strip())<=500:raise ValueError('Enter the exact reference transcript (1–500 characters).')
  if not isinstance(encoded,str) or len(encoded)>21*1024*1024:raise ValueError('Reference file must be at most 15 MB.')
  if not isinstance(label,str):raise ValueError('Invalid voice name')
  try:
   raw=base64.b64decode(encoded,validate=True)
   if len(raw)>15*1024*1024:raise ValueError('File too large')
   info=sf.info(io.BytesIO(raw))
   if not 3<=info.duration<=30 or not 8000<=info.samplerate<=192000 or info.channels not in (1,2):raise ValueError('Use a 3–30 second mono/stereo recording at 8–192 kHz.')
   wav,sr=sf.read(io.BytesIO(raw),dtype='float32',always_2d=True)
  except Exception as e:raise ValueError('Cannot use that recording. Upload a valid 3–30 second WAV, FLAC, OGG or MP3 file (up to 15 MB).') from e
  if not np.isfinite(wav).all():raise ValueError('Reference contains invalid audio samples.')
  wav=wav.mean(axis=1)
  if np.max(np.abs(wav))<1e-5:raise ValueError('The recording is silent. Choose audible speech.')
  identity=uuid.uuid4().hex;path=self.references/(identity+'.wav');sf.write(path,wav,sr,subtype='PCM_16')
  r={'selection_id':identity,'id':identity,'label':label.strip()[:80] or 'Uploaded voice','text':transcript.strip(),'local_audio':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'source_sha256':hashlib.sha256(raw).hexdigest(),'seconds':len(wav)/sr}
  write_json(self.references/(identity+'.json'),r)
  return {k:r[k] for k in ('selection_id','label','text','seconds')}
 def submit(self,text,seed,parent=None,reference_id='default'):
  if not isinstance(text,str):raise ValueError('Text must be a string')
  text=text.strip()
  if not text or len(text)>350:raise ValueError('Enter1–350characters. Start with one sentence.')
  if not isinstance(seed,int) or isinstance(seed,bool) or not 0<=seed<2**31:raise ValueError('Seed must be an integer from0to2147483647.')
  ref=self.reference(reference_id)
  if parent:
   if not isinstance(parent,str) or len(parent)!=32 or any(c not in '0123456789abcdef' for c in parent):raise ValueError('Invalid parent version')
   previous=json.loads((self.output/(parent+'.json')).read_text(encoding='utf-8'))
   if previous['text']!=text:raise ValueError('Retry must keep the original text.')
   ref=previous.get('reference',self.reference())
  with self.lock:
   if self.busy:raise ValueError('A version is already generating. Wait for it to finish.')
   self.busy=True
  identity=uuid.uuid4().hex
  record={'id':identity,'text':text,'seed':seed,'parent':parent,'reference':ref,'created':time.time(),'status':'queued','feedback':{'rating':'','notes':''}}
  write_json(self.output/(identity+'.json'),record)
  self.pool.submit(self.generate,record);return record
 def generate(self,r):
  path=self.output/(r['id']+'.json');started=time.time()
  try:
   os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
   import torch,numpy as np,soundfile as sf
   torch.set_num_threads(max(1,min(4,(os.cpu_count() or 4)//2)))
   if self.wrapper is None:
    self.stage='Loading model into memory';r['status']='loading';write_json(path,r)
    if not (self.model_path/'LOCAL_VERIFIED.json').exists():raise RuntimeError('The verified model download is not complete.')
    from qwen_tts import Qwen3TTSModel
    self.wrapper=Qwen3TTSModel.from_pretrained(str(self.model_path),device_map='cpu',torch_dtype=torch.float32,attn_implementation='sdpa')
    self.wrapper.model.eval()
   self.stage='Generating speech on CPU';r['status']='generating';write_json(path,r)
   ref=r['reference']
   if hashlib.sha256(Path(ref['local_audio']).read_bytes()).hexdigest()!=ref['sha256']:raise RuntimeError('Voice reference hash mismatch')
   torch.manual_seed(r['seed']);np.random.seed(r['seed'])
   with torch.inference_mode():
    waves,sr=self.wrapper.generate_voice_clone(text=[r['text']],language=['Auto'],ref_audio=[ref['local_audio']],ref_text=[ref['text']],x_vector_only_mode=False,non_streaming_mode=True,**self.profile['generation'])
   wav=np.asarray(waves[0]);assert len(wav)>0 and np.isfinite(wav).all()
   audio=self.output/(r['id']+'.wav');sf.write(audio,wav,sr,subtype='PCM_16')
   r.update(status='complete',seconds=len(wav)/sr,elapsed_seconds=time.time()-started,audio_sha256=hashlib.sha256(audio.read_bytes()).hexdigest(),profile=self.profile['profile'] if ref['selection_id']=='default' else 'custom-reference-full-sentence',device='cpu',precision='FP32',generation=self.profile['generation'])
   self.stage='Ready'
  except Exception as e:
   r.update(status='failed',error=str(e)[:500]);self.stage='Generation failed — see the version for details'
  finally:
   write_json(path,r)
   with self.lock:self.busy=False
 def feedback(self,identity,data):
  if not isinstance(identity,str) or len(identity)!=32 or any(c not in '0123456789abcdef' for c in identity):raise ValueError('Invalid version')
  path=self.output/(identity+'.json');r=json.loads(path.read_text(encoding='utf-8'))
  if r['status']!='complete':raise ValueError('Wait for generation to finish')
  rating=data.get('rating','');notes=data.get('notes','')
  if rating not in ('','Correct','Minor mistakes','Major mistakes') or not isinstance(notes,str) or len(notes)>5000:raise ValueError('Invalid feedback')
  r['feedback']={'rating':rating,'notes':notes};write_json(path,r);return r

def make_handler(app):
 class Handler(BaseHTTPRequestHandler):
  def log_message(self,*args):pass
  def send(self,status,body,kind='application/json'):
   if kind=='application/json':body=json.dumps(body,ensure_ascii=False).encode()
   self.send_response(status);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(body)));self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff');self.end_headers();self.wfile.write(body)
  def valid_host(self):return self.headers.get('Host') in (f'127.0.0.1:{self.server.server_port}',f'localhost:{self.server.server_port}')
  def do_GET(self):
   if not self.valid_host():return self.send(403,{'error':'Local access only'})
   path=urlsplit(self.path).path
   if path=='/':
    page=Path(__file__).with_name('playground.html').read_text(encoding='utf-8').replace('__TOKEN__',app.token);return self.send(200,page.encode(),'text/html; charset=utf-8')
   if path=='/api/status':return self.send(200,{'application':'hojil-playground','busy':app.busy,'stage':app.stage,'model_ready':(app.model_path/'LOCAL_VERIFIED.json').exists(),'versions':app.history()})
   if path=='/api/export':return self.send(200,{'schema':'hojil-playground-feedback-v1','versions':app.history()})
   if path=='/api/references':return self.send(200,{'references':app.reference_list()})
   if path.startswith('/reference-audio/'):
    identity=path.removeprefix('/reference-audio/')
    try:r=app.reference(identity)
    except (ValueError,FileNotFoundError):return self.send(404,{})
    return self.send(200,Path(r['local_audio']).read_bytes(),'audio/wav')
   if path.startswith('/audio/'):
    name=path.removeprefix('/audio/')
    if len(name)!=36 or not name.endswith('.wav') or any(c not in '0123456789abcdef' for c in name[:-4]):return self.send(404,{})
    p=app.output/name
    if p.exists():return self.send(200,p.read_bytes(),'audio/wav')
   return self.send(404,{})
  def do_POST(self):
   if not self.valid_host() or self.headers.get('X-Playground-Token')!=app.token:return self.send(403,{'error':'Reload the local playground.'})
   origin=self.headers.get('Origin')
   if origin and origin not in (f'http://127.0.0.1:{self.server.server_port}',f'http://localhost:{self.server.server_port}'):return self.send(403,{'error':'Local access only'})
   try:
    size=int(self.headers.get('Content-Length','0'))
    path=urlsplit(self.path).path
    limit=22*1024*1024 if path=='/api/references' else 20000
    if not 0<size<=limit:raise ValueError('Request is too large.')
    data=json.loads(self.rfile.read(size))
    if not isinstance(data,dict):raise ValueError('Expected a JSON object.')
    if path=='/api/references':return self.send(201,app.upload_reference(data))
    if path=='/api/generate':
     seed=data.get('seed');seed=secrets.randbelow(2**31) if seed is None else seed
     return self.send(202,app.submit(data.get('text',''),seed,data.get('parent'),data.get('reference_id','default')))
    if path=='/api/feedback':return self.send(200,app.feedback(data.get('id'),data))
    if path=='/api/stop':
     if app.busy:raise ValueError('Wait for generation to finish before closing.')
     self.send(200,{'stopped':True});threading.Thread(target=self.server.shutdown,daemon=True).start();return
    return self.send(404,{})
   except (ValueError,TypeError,FileNotFoundError) as e:return self.send(400,{'error':str(e)})
 return Handler

def main():
 p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--profile',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--port',type=int,default=8766);a=p.parse_args()
 root=Path(__file__).resolve().parents[1]
 for path in [a.model,a.output]:
  if path.resolve().is_relative_to(root):p.error('Model and generated audio must be outside the repository.')
 app=Playground(a.model,a.profile,a.output)
 # Recover interrupted versions honestly after a restart.
 for r in app.history():
  if r['status'] in ('queued','loading','generating'):
   r.update(status='failed',error='Interrupted by a previous app shutdown. Generate another version.');write_json(a.output/(r['id']+'.json'),r)
 server=ThreadingHTTPServer(('127.0.0.1',a.port),make_handler(app))
 print(f'Playground: http://127.0.0.1:{a.port}',flush=True)
 server.serve_forever()
 server.server_close();app.pool.shutdown(wait=True)
if __name__=='__main__':main()



