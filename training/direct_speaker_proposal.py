"""Prepare speaker-only candidates from verified fitting-data voice gradients.

Candidate weights are proposals, not selected models. Actual matched fitting
speech must pass the registered gate before a complete development/custom screen.
"""
import argparse,time
from pathlib import Path
import torch
from .engine import read,write,save,digest,check_optimizer
from . import direct_voice_training as direct
from . import middle_attention_adapter as attention
from .direct_voice_objective import constrained_direction
from .voice_gradient_geometry import capture_gradients

SETTINGS=dict(learning_rate=.003,scales=[2.**(-i) for i in range(8)],
    maximum_speaker_input_change=.0025,anchor_tolerance=1e-6,
    maximum_fitting_speech_trials=4,maximum_clip_cosine_drop=.005,
    maximum_added_character_errors_per_clip=1)


def speaker_state(model):
    return {k:v.detach().cpu().clone() for k,v in model.speaker_preference_adapter.state_dict().items()}


def restore_speaker(model,state):
    current=model.speaker_preference_adapter.state_dict()
    if set(current)!=set(state):raise ValueError('Speaker state coverage differs')
    with torch.no_grad():
        for name,value in current.items():
            saved=state[name]
            if saved.dtype!=torch.float32 or value.shape!=saved.shape or not torch.isfinite(saved).all():
                raise ValueError('Invalid FP32 speaker state')
            value.copy_(saved)


def sources():
    result=direct.sources()
    for name in ('direct_speaker_proposal.py','voice_gradient_geometry.py'):
        result[name]=digest(Path(__file__).with_name(name))
    result['test_training_direct_speaker_proposal.py']=digest(Path(__file__).parent.parent/'tests/test_training_direct_speaker_proposal.py')
    return result


def prepare(run,out):
    if out.exists():raise ValueError('Never overwrite registered candidate proposals')
    if read(run.parent/'direct-voice-pipeline-status.json')['status']!='complete':
        raise ValueError('Original native training must finish first')
    pilot=direct.verify(run);geometry=run.parent/'voice-gradient-geometry-probe'
    proof=read(geometry/'results.json');registration=read(geometry/'protocol.json')
    if proof['status']!='verified_read_only_gradient_geometry' or proof['weights_updated']:
        raise ValueError('Native geometry proof missing')
    if proof['protocol_sha256']!=digest(geometry/'protocol.json'):
        raise ValueError('Geometry registration changed')
    for path,expected in registration['inputs'].items():
        if digest(path)!=expected:raise ValueError('Geometry input changed')
    result=proof['results'][0]
    if (result['step']!=1 or not result['all_model_weights_unchanged']
            or not result['attention_gradient_matches_original_update']):
        raise ValueError('Wrong native gradient policy')
    path=geometry/result['gradient_path']
    if digest(path)!=result['gradient_sha256']:raise ValueError('Native gradient changed')
    dependencies=direct.sources()
    for name,expected in registration['source'].items():
        if name in dependencies:actual=dependencies[name]
        elif name=='external_driver':actual=digest(run.parent/'probe-voice-gradient-geometry.py')
        elif name.startswith(('training/','tests/')):actual=digest(Path(__file__).parent.parent/name)
        else:raise ValueError('Unknown proof dependency')
        if actual!=expected:raise ValueError('Native proof dependency changed')
    tests=run.parent/'direct-speaker-proposal-tests.log';log=tests.read_text(encoding='utf-8-sig')
    if 'Ran 2 tests' not in log or '\nOK\n' not in log or 'skipped' in log.lower():
        raise ValueError('Native speaker proposal unit checks missing')
    from .policy import external
    out=external(out.resolve());out.mkdir()
    write(out/'protocol.json',dict(source=sources(),runtime=direct.initial.collection.runtime(),
        initial_run=str(run),initial_protocol_sha256=digest(run/'protocol.json'),
        initial_checkpoint_sha256=digest(run/'checkpoint-000000.pt'),
        initial_screen_sha256=digest(run/'eval-000000/results.json'),
        initial_audit_sha256=digest(run/'verification-000000.json'),
        geometry_results_sha256=digest(geometry/'results.json'),gradient_path=str(path),gradient_sha256=digest(path),
        fitting_rollout=str(run/'rollout-000001'),fitting_results_sha256=digest(run/'rollout-000001/results.json'),
        tests=str(tests),tests_sha256=digest(tests),settings=SETTINGS,parameters=32768,trained_component='existing rank16 speaker-input adapter only',
        frozen_components='all original acoustic weights and existing middle-attention update4 LoRA',
        selection='Try feasible scales in descending order, at most four actual fitting-speech trials; never select a scale from development/custom scores',
        fitting_gate='All16 matched cases/seeds terminate; pooled CER no higher; mean original-reference cosine strictly higher; no clip loses more than .005 cosine or adds more than one character error',
        evaluation='Only the first fitting-gate pass receives the complete26 unchanged CPU screen; a joint screen pass requires all114 confirmation recordings',
        goal_achieved=False,global_model_selection_changed=False))
    return out


def verify(out):
    p=read(out/'protocol.json');run=Path(p['initial_run'])
    if p['source']!=sources() or p['runtime']!=direct.initial.collection.runtime() or p['settings']!=SETTINGS:
        raise ValueError('Registered source/runtime/settings changed')
    for path,key in [(run/'protocol.json','initial_protocol_sha256'),(run/'checkpoint-000000.pt','initial_checkpoint_sha256'),
                     (run/'eval-000000/results.json','initial_screen_sha256'),(run/'verification-000000.json','initial_audit_sha256'),
                     (Path(p['gradient_path']),'gradient_sha256'),(Path(p['fitting_rollout'])/'results.json','fitting_results_sha256'),
                     (run.parent/'voice-gradient-geometry-probe/results.json','geometry_results_sha256'),
                     (Path(p['tests']),'tests_sha256')]:
        if digest(path)!=p[key]:raise ValueError('Registered input changed')
    return p


def proposals(out):
    registered=verify(out);run=Path(registered['initial_run'])
    p,wrapper,initial=direct.load(run,0,'cuda',True);model=wrapper.model
    initial_speaker=speaker_state(model)
    model.requires_grad_(False);model.speaker_preference_adapter.requires_grad_(True)
    named=[(n,v) for n,v in model.named_parameters() if v.requires_grad];params=[v for n,v in named]
    if len(params)!=2 or sum(v.numel() for v in params)!=registered['parameters']:raise ValueError('Wrong speaker parameter coverage')
    cached=torch.load(registered['gradient_path'],map_location='cpu',weights_only=True)
    voice=torch.cat([cached['voice'][n].flatten() for n,v in named]).to('cuda')
    model.zero_grad(set_to_none=True)
    before=direct.mean_nll(model,p,p['anchors'],backward=True)
    anchor_state=capture_gradients(model)
    for n,v in named:torch.testing.assert_close(anchor_state[n],cached['anchor'][n],atol=1e-7,rtol=2e-4)
    anchor=torch.cat([anchor_state[n].flatten() for n,v in named]).to('cuda')
    model.zero_grad(set_to_none=True)
    references=torch.stack([direct.trajectory(Path(p['pool']),r)['ref_xvec'].flatten() for r in p['anchors']]).to('cuda')
    with torch.no_grad():before_inputs=model.speaker_preference_adapter(references).clone()
    optimizer=torch.optim.AdamW(params,lr=SETTINGS['learning_rate'],weight_decay=0.,fused=True)
    offset=0
    for v in params:v.grad=voice[offset:offset+v.numel()].view_as(v).clone();offset+=v.numel()
    old=torch.cat([v.detach().flatten().clone() for v in params]);optimizer.step();check_optimizer(optimizer)
    proposed=torch.cat([v.detach().flatten() for v in params])-old
    direction,projection=constrained_direction(proposed,voice,anchor)
    rows=[]
    for index,scale in enumerate(SETTINGS['scales']):
        offset=0
        with torch.no_grad():
            for v in params:
                v.copy_((old[offset:offset+v.numel()]+scale*direction[offset:offset+v.numel()]).view_as(v));offset+=v.numel()
            after=direct.mean_nll(model,p,p['anchors'])
            inputs=model.speaker_preference_adapter(references)
            changes=(inputs-before_inputs).norm(dim=-1)/references.norm(dim=-1)
            total=(inputs-references).norm(dim=-1)/references.norm(dim=-1)
        actual=torch.cat([v.detach().flatten() for v in params])-old
        predicted=float(actual.double().dot(voice.double()))
        feasible=bool(after<=before+SETTINGS['anchor_tolerance'] and changes.max()<=SETTINGS['maximum_speaker_input_change']
                      and total.max()<=model.speaker_preference_adapter.radius_fraction+1e-6 and predicted<0)
        row=dict(index=index,scale=scale,feasible=feasible,anchor_nll_before=before,anchor_nll_after=after,
            speaker_input_change_max=float(changes.max()),speaker_input_change_mean=float(changes.mean()),
            total_speaker_displacement_max=float(total.max()),predicted_voice_change=predicted,
            actual_parameter_update_norm=float(actual.norm()))
        if feasible:
            path=out/f'candidate-{index:02d}.pt'
            save(path,dict(speaker=speaker_state(model),index=index,scale=scale,protocol_sha256=digest(out/'protocol.json')))
            row.update(path=path.name,sha256=digest(path))
        rows.append(row);write(out/f'proposal-{index:02d}.json',row);print(row,flush=True)
        if attention.frozen_digest(model)!=initial['frozen_backbone_sha256']:raise ValueError('Original acoustic weights changed')
        for name,m in attention.modules(model).items():
            for key in ('lora_A','lora_B'):
                if not torch.equal(getattr(m,key).detach().cpu(),initial['adapter'][name+'.'+key]):
                    raise ValueError('Attention adapter changed')
    restore_speaker(model,initial_speaker)
    model.speaker_preference_adapter.requires_grad_(False)
    for m in attention.modules(model).values():m.lora_A.requires_grad_(True);m.lora_B.requires_grad_(True)
    direct.unchanged(model,initial);verify(out)
    write(out/'results.json',dict(status='speaker_proposals_prepared',proposals=rows,projection=projection,
        attention_and_original_weights_unchanged=True,initial_model_restored=True,
        voice_gradient_norm=float(voice.norm()),anchor_gradient_norm=float(anchor.norm()),
        training_speech_gate_pending=True,goal_achieved=False,global_model_selection_changed=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--initial-run',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=prepare(args.initial_run,args.output);proposals(out)


if __name__=='__main__':main()
