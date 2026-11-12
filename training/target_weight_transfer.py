"""Transfer the full trained text mapping only to target transcript positions.

This is a controlled component-weight transfer, not an optimizer update or a
claim of discovering a speaker circuit by reading weights. Reference/control
projections and every other original parameter retain the halfway values.
"""
import argparse
import copy
import os
from pathlib import Path
import shutil
from types import MethodType

from .engine import digest,read,save,write
from .policy import external
from . import targeted_voice as pilot
from . import reference_protected as protected
from .voice_strength_search import goal_screen


def install(model, full_weights, strength=1.0):
    import torch
    import torch.nn.functional as F
    if strength not in (0.0,1.0):
        raise ValueError('Only the registered no-op control and full-target variant are supported')
    projection=model.talker.text_projection
    if not hasattr(projection,'_target_route') or hasattr(projection,'transferred_fc1'):
        raise ValueError('Requires fresh protected target routing')
    for i in (1,2):
        layer=getattr(projection,f'linear_fc{i}').base_layer
        target=full_weights[f'linear_fc{i}.weight']
        if target.shape!=layer.weight.shape or target.dtype!=torch.float32 or not torch.isfinite(target).all():
            raise ValueError('Invalid transferred weight')
        value=layer.weight.detach().float().lerp(target.to(layer.weight.device),strength)
        projection.register_parameter(f'transferred_fc{i}',torch.nn.Parameter(value,requires_grad=False))

    def forward(self,x):
        a,b=self.linear_fc1.base_layer,self.linear_fc2.base_layer
        baseline=b(self.act_fn(a(x)))
        span=self._target_route['span']
        if span is None:return baseline
        start,stop=span
        if x.ndim!=3 or x.shape[1]!=stop:raise ValueError('Unexpected protected text layout')
        full=F.linear(self.act_fn(F.linear(x,self.transferred_fc1,a.bias)),self.transferred_fc2,b.bias)
        index=torch.arange(stop,device=x.device)
        return torch.where(((index>=start)&(index<stop))[None,:,None],full,baseline)

    projection.forward=MethodType(forward,projection)
    # This experiment has no optimizer and no trainable weights.
    model.requires_grad_(False)
    return model


def load(protocol,strength=1.0):
    import torch
    from safetensors import safe_open
    protected.configure_helpers()
    parent=read(Path(protocol['parent'])/'protocol.json')
    if digest(Path(protocol['parent'])/'protocol.json')!=protocol['parent_protocol_sha256']:
        raise ValueError('Parent protocol changed')
    wrapper=protected.load_model(parent)
    initial_path=Path(protocol['parent'])/'checkpoint-000000.pt'
    if (digest(initial_path)!=protocol['initial_checkpoint_sha256']
            or digest(initial_path)!=read(initial_path.with_suffix('.json'))['sha256']):
        raise ValueError('Initial adapter checkpoint changed')
    initial=torch.load(initial_path,map_location='cpu',weights_only=True)
    pilot.load_adapter(wrapper.model.talker,initial['adapter'])
    adapted=Path(parent['config']['adapted'])/'model.safetensors'
    if digest(adapted)!=parent['adapted_sha256']:raise ValueError('Trained source changed')
    with safe_open(adapted,framework='pt',device='cpu') as tensors:
        weights={f'linear_fc{i}.weight':tensors.get_tensor(f'talker.text_projection.linear_fc{i}.weight').float()
            for i in (1,2)}
        for i in (1,2):
            key=f'talker.text_projection.linear_fc{i}.bias'
            layer=getattr(wrapper.model.talker.text_projection,f'linear_fc{i}').base_layer
            if layer.bias is not None and not torch.equal(layer.bias.detach(),tensors.get_tensor(key).float()):
                raise ValueError('Unexpected changed text-projection bias')
    install(wrapper.model,weights,strength)
    return wrapper


def verify(run):
    p=read(run/'protocol.json')
    if p['transfer_source_sha256']!=digest(Path(__file__)):
        raise ValueError('Weight-transfer implementation changed')
    if p['strength']!=1 or p['kind']!='inference_weight_transfer_without_optimizer':
        raise ValueError('Unregistered variant')
    return p


def prepare(parent,run):
    p=copy.deepcopy(read(parent/'protocol.json'))
    initial=parent/'checkpoint-000000.pt'
    if digest(initial)!=read(initial.with_suffix('.json'))['sha256']:
        raise ValueError('Initial adapter changed')
    run=external(run.resolve());run.mkdir(exist_ok=False)
    p['config']['output']=str(run)
    p.update(parent=str(parent),parent_protocol_sha256=digest(parent/'protocol.json'),
        transfer_source_sha256=digest(Path(__file__)),strength=1,kind='inference_weight_transfer_without_optimizer',
        variant_id=101,initial_checkpoint_sha256=digest(initial),
        intervention='Two text-projection matrices restored to full epoch-2 values only at target transcript positions; reference/control projections and all other original parameters remain halfway',
        rationale='Global full text mapping previously reduced custom CER but affected voice; test the missing protected-target control',
        decision='Require the same two-group CER/cosine goal; no automatic promotion. ID 101 identifies an inference variant, not training steps.')
    write(run/'protocol.json',p)
    shutil.copy2(Path(__file__),run/'source.py')
    (run/'eval-000000').mkdir()
    source=parent/'eval-000000/results.json'
    baseline=read(source)
    baseline['reused_baseline']=dict(source=str(source),source_sha256=digest(source))
    write(run/'eval-000000/results.json',baseline)
    for r in baseline['records']:
        wav=source.parent/r['wav']
        if digest(wav)!=r['audio_sha256']:raise ValueError('Baseline waveform changed')
        shutil.copy2(wav,run/'eval-000000'/r['wav'])
    write(run/'status.json',dict(status='prepared',optimizer_updates=0,goal_achieved=False))


def parity(run):
    import numpy as np
    import torch
    import soundfile as sf
    p=verify(run)
    # Use the wrapper generation path; install(0) must reproduce accepted audio.
    wrapper=load(p,0)
    wrapper.model.eval()
    case=next(c for c in p['cases'] if c['case']=='custom-001')
    if digest(case['reference'])!=case['reference_sha256']:raise ValueError('Reference changed')
    with torch.inference_mode():
        prompt=wrapper.create_voice_clone_prompt(ref_audio=case['reference'],ref_text=case['reference_text'],x_vector_only_mode=False)
        torch.manual_seed(2026);np.random.seed(2026)
        waves,sr=wrapper.generate_voice_clone(text=case['text'],language='Auto',voice_clone_prompt=prompt,
            non_streaming_mode=True,**p['generation'])
    out=run/'no-op-control.wav';sf.write(out,waves[0],sr,subtype='PCM_16')
    if digest(out)!=p['waveform_baseline_sha256']:raise ValueError('No-op transfer does not reproduce halfway audio')
    write(run/'parity.json',dict(exact_waveform_match=True,sha256=digest(out),strength=0,
        protocol_sha256=digest(run/'protocol.json')))


def evaluate(run):
    import torch
    p=verify(run)
    control=read(run/'parity.json')
    if (not control['exact_waveform_match'] or control['protocol_sha256']!=digest(run/'protocol.json')
            or control['sha256']!=p['waveform_baseline_sha256']
            or digest(run/'no-op-control.wav')!=control['sha256']):
        raise ValueError('No-op control must pass first')
    path=run/'checkpoint-000101.pt'
    if not path.exists():
        wrapper=load(p)
        initial=torch.load(Path(p['parent'])/'checkpoint-000000.pt',map_location='cpu',weights_only=True)
        save(path,dict(adapter=initial['adapter'],frozen_backbone_sha256=pilot.frozen_digest(wrapper.model),
            protocol_sha256=digest(run/'protocol.json'),kind=p['kind'],optimizer_updates=0))
        write(path.with_suffix('.json'),dict(sha256=digest(path),kind=p['kind']))
        del wrapper,initial
        import gc
        gc.collect()
    pilot.load_model=lambda ignored:load(p)
    pilot.load_adapter=lambda talker,state:protected.audited_load_adapter(talker,state,run/'routing.json')
    write(run/'status.json',dict(status='evaluating',variant_id=101,optimizer_updates=0))
    pilot.speech(p,101)
    result=read(run/'eval-000101/results.json')
    write(run/'results.json',dict(kind=p['kind'],summary=result['summary'],goal_screen=goal_screen(result['summary']),
        goal_achieved=False,optimizer_updates=0,global_model_selection_changed=False))
    write(run/'status.json',dict(status='complete',goal_achieved=False,optimizer_updates=0))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--phase',choices=['prepare','parity','evaluate'],required=True)
    args=parser.parse_args()
    if args.phase=='prepare':
        prepare(args.parent,args.output);return
    p=verify(args.output);cfg=p['config']
    os.environ['NUMBA_CACHE_DIR']=cfg['numba_cache']
    os.environ['TEMP']=os.environ['TMP']=cfg['tmp']
    os.environ['HF_HUB_OFFLINE']=os.environ['TRANSFORMERS_OFFLINE']='1'
    import torch
    torch.set_num_threads(4)
    if args.phase=='parity':parity(args.output)
    else:evaluate(args.output)


if __name__=='__main__':
    main()
