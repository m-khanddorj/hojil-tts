"""Score saved Alternate 2 development recordings with pinned FP32 BuzzASR."""
import argparse,hashlib,json,time,unicodedata,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
FULL=ROOT.parent
ART=FULL.parent
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def distance(a,b):
 row=list(range(len(b)+1))
 for i,x in enumerate(a,1):
  nxt=[i]
  for j,y in enumerate(b,1):nxt.append(min(row[j]+1,nxt[-1]+1,row[j-1]+(x!=y)))
  row=nxt
 return row[-1]
def score(a,b,strip):
 def norm(s):
  s=unicodedata.normalize('NFC',s).casefold()
  if strip:s=''.join(c for c in s if not unicodedata.category(c).startswith('P'))
  return ' '.join(s.split())
 a,b=norm(a),norm(b);ce=distance(a,b);we=distance(a.split(),b.split())
 return {'cer':ce/max(1,len(a)),'wer':we/max(1,len(a.split())),'character_errors':ce,'characters':len(a),'word_errors':we,'words':len(a.split())}
def main():
 global ROOT,FULL,ART
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--full-training',type=Path,required=True,help='External experiment folder containing completed recordings and prior ASR evidence')
 parser.add_argument('--output',type=Path,required=True)
 args=parser.parse_args();FULL=args.full_training;ART=FULL.parent;ROOT=args.output
 if ROOT.resolve().is_relative_to(Path(__file__).resolve().parents[1]):parser.error('Keep generated evidence outside Git')
 ROOT.mkdir(parents=True,exist_ok=True)
 os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
 import torch,torchaudio,soundfile as sf
 from transformers import WhisperForConditionalGeneration,WhisperProcessor
 torch.set_num_threads(4);torch.manual_seed(2026)
 modeldir=ART/'buzzasr-model'
 manifest=read(FULL/'post-evaluation/completed/buzzasr-files.json')
 print(json.dumps({'stage':'verifying-model','manifest_type':type(manifest).__name__}),flush=True)
 for name,entry in manifest.items():
  assert sha(modeldir/name)==entry['sha256'],name
 tasks=[]
 for folder in ['alternate2-validation','seed-recheck']:
  for r in read(FULL/folder/'completed/results.json'):
   if folder=='alternate2-validation' and r['chunked']:continue
   if folder=='seed-recheck' and r['seed']==2026:continue
   path=FULL/folder/'completed/audio'/r['file'];assert sha(path)==r['sha256']
   tasks.append({'case_id':r['id'],'group':'primary20' if folder=='alternate2-validation' else 'selected_retry_'+str(r['seed']),
     'reference':' '.join(r['texts']),'seed':r['seed'],'audio':str(path),'audio_sha256':r['sha256']})
 assert len(tasks)==28
 write(ROOT/'plan.json',tasks)
 signature={'model':'BuzzASR/mongolian','revision':'185f1bf1f562ad4b1bdbfc71811fb165a9235416','device':'cpu','dtype':'float32','torch':torch.__version__,
 'decoding':dict(num_beams=1,no_repeat_ngram_size=3,repetition_penalty=1.2,do_sample=False),'audio_verified':True,'model_verified':True}
 if (ROOT/'run.json').exists():assert read(ROOT/'run.json')==signature
 write(ROOT/'run.json',signature)
 model=WhisperForConditionalGeneration.from_pretrained(modeldir,torch_dtype=torch.float32,local_files_only=True).eval()
 processor=WhisperProcessor.from_pretrained(modeldir,local_files_only=True)
 def transcribe(p):
  audio,sr=sf.read(p,dtype='float32');assert audio.ndim==1 and 0<len(audio)/sr<=30
  wav=torchaudio.functional.resample(torch.from_numpy(audio),sr,16000).numpy()
  batch=processor(wav,sampling_rate=16000,return_tensors='pt',return_attention_mask=True)
  with torch.inference_mode():ids=model.generate(batch.input_features,attention_mask=batch.attention_mask,**signature['decoding'])
  assert ids.shape[-1]<model.generation_config.max_length
  return processor.batch_decode(ids,skip_special_tokens=True)[0].strip()
 prior=[json.loads(s) for s in (FULL/'post-evaluation/completed/buzzasr-gpu/transcriptions.jsonl').read_text(encoding='utf-8').splitlines()]
 parity=[]
 for r in prior[:3]:
  p=FULL/'post-evaluation/completed/generation/audio'/(r['case_id']+'.wav');assert sha(p)==r['audio_sha256']
  hyp=transcribe(p);parity.append({'case_id':r['case_id'],'cpu':hyp,'prior_gpu':r['hypothesis'],'exact_match':hyp==r['hypothesis'],'audio_sha256':r['audio_sha256']})
 write(ROOT/'parity.json',{'checks':parity,'passed':all(r['exact_match'] for r in parity)})
 print(json.dumps({'stage':'parity','passed':all(r['exact_match'] for r in parity)}),flush=True)
 records=[json.loads(s) for s in (ROOT/'transcriptions.jsonl').read_text(encoding='utf-8').splitlines()] if (ROOT/'transcriptions.jsonl').exists() else []
 done={r['case_id']:r for r in records}
 for task in tasks:
  if task['case_id'] in done:
   assert done[task['case_id']]['audio_sha256']==task['audio_sha256'];continue
  start=time.monotonic();hyp=transcribe(Path(task['audio']))
  record={**task,'hypothesis':hyp,'strict':score(task['reference'],hyp,False),'without_punctuation':score(task['reference'],hyp,True),'elapsed':time.monotonic()-start}
  with (ROOT/'transcriptions.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(record,ensure_ascii=False)+'\n')
  records.append(record)
  print(json.dumps({'completed':len(records),'total':len(tasks),'seconds':record['elapsed']}),flush=True)
 summary={}
 for group in sorted(set(r['group'] for r in records)):
  rows=[r for r in records if r['group']==group];summary[group]={'n':len(rows)}
  for kind in ['strict','without_punctuation']:
   summary[group][kind]={'mean_cer':sum(r[kind]['cer'] for r in rows)/len(rows),'mean_wer':sum(r[kind]['wer'] for r in rows)/len(rows),
    'pooled_cer':sum(r[kind]['character_errors'] for r in rows)/sum(r[kind]['characters'] for r in rows),
    'pooled_wer':sum(r[kind]['word_errors'] for r in rows)/sum(r[kind]['words'] for r in rows)}
 write(ROOT/'summary.json',summary);print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
