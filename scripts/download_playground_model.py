"""Download the pinned private inference export without exposing credentials."""
import argparse,json,hashlib,shutil,os
from pathlib import Path
from huggingface_hub import snapshot_download
def main():
 a=argparse.ArgumentParser();a.add_argument('--receipt',type=Path,required=True);a.add_argument('--env-file',type=Path,required=True);a.add_argument('--output',type=Path,required=True);x=a.parse_args()
 receipt=json.loads(x.receipt.read_text(encoding='utf-8-sig'))
 if x.output.resolve().is_relative_to(Path(__file__).resolve().parents[1]):raise ValueError('Store model weights outside the repository')
 token=os.environ.get('HF_TOKEN')
 for line in x.env_file.read_text(encoding='utf-8-sig').splitlines():
  k,sep,v=line.partition('=')
  if sep and k.strip()=='HF_TOKEN':token=v.strip().strip('"\x27')
 if not token:raise RuntimeError('HF_TOKEN is missing')
 files=receipt['verified_files'];need=sum(v['bytes'] for v in files.values())
 x.output.mkdir(parents=True,exist_ok=True)
 have=sum((x.output/n).stat().st_size for n in files if (x.output/n).exists())
 if shutil.disk_usage(x.output).free<need-have+3*2**30:raise RuntimeError('Need model space plus3GiB free headroom')
 snapshot_download(receipt['repo'],revision=receipt['revision'],token=token,local_dir=x.output,allow_patterns=list(files),max_workers=2)
 for n,v in files.items():
  f=x.output/n;assert f.stat().st_size==v['bytes'],n
  h=hashlib.sha256()
  with f.open('rb') as s:
   for block in iter(lambda:s.read(8*1024*1024),b''):h.update(block)
  assert h.hexdigest()==v['sha256'],n
 (x.output/'LOCAL_VERIFIED.json').write_text(json.dumps({'repo':receipt['repo'],'revision':receipt['revision'],'verified_files':len(files)},indent=2))
 print('Pinned inference model downloaded and hash-verified.',flush=True)
if __name__=='__main__':main()
