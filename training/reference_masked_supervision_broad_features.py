"""Conditional fresh512 fitting features; old calibration56 stays separate.

Preparation is model independent. Native execution needs a closed failed
base-K/V screen; a qualifying screen takes original114 priority. Old overlap
assets are comparisons only: every new tensor must be freshly extracted.
"""
from pathlib import Path
import os

from . import reference_masked_supervision_features as old
from . import reference_masked_supervision_broad_data as data

read,digest,write,require,fingerprint=old.read,old.digest,old.write_new,old.require,old.fingerprint
DATA_SHA='661f5fa536e01d198f757c007ee0c64b232029f009f813d301e486ebe2165bcc'
AUDIT_SHA='d6af64860dc1e93e9568fabd3471890894ebcd347dbe302bc6a126b54ecb299e'
REVIEW_SHA='a648df706a07e50b6835bdd1550710cbaa7194c65f29547ceb4a4fc3efbf8c7d'
SCREEN_SHA='ca108055567fe14bc34ea28fcccf76b2654f01ce33c83c284c148c20a3d2cb84'
SETTINGS=dict(device='cpu',dtype='float32',attention='sdpa',cpu_threads=4,matmul_tf32=False,cudnn_tf32=False,
    seed=20261111,total_assets=512,overlap_controls=52,codebooks=16,speaker_size=1024,
    fresh_all_assets=True,old_assets_used_as_outputs=False,calibration_assets_in_new_extraction=0)
FLAGS=dict(training_registered=False,optimizer_updates=0,goal_achieved=False,global_model_selection_changed=False)


def closed(receipt,session,bindings):
    require(receipt.get('status')=='closed' and type(receipt.get('exit_code')) is int and receipt['exit_code']==0
        and type(receipt.get('original_session_id')) is int and receipt['original_session_id']==session
        and all(receipt.get(k)==v for k,v in bindings.items()),'Authoritative successful original process closure required')


def require_feature_priority(qualified):
    require(type(qualified) is bool and qualified is False,
        'Qualifying screen must receive original114 confirmation before any feature-stage alternative')


def scope(assets,plan):
    ids=[r['id'] for r in assets];fitting=set(plan['selected_fitting_pair_ids']);probes=plan['fitting_probe_pair_ids']
    require(len(ids)==len(set(ids))==512 and sorted(ids)==plan['fitting_feature_ids']
        and len(fitting)==256 and len(probes)==len(set(probes))==16 and set(probes)<=fitting
        and not set(ids)&set(plan['calibration_feature_source']['ids'])
        and '__custom_reference__' not in ids and plan['previous_cache_overlap_is_readiness'] is False
        and len(plan['previous_cache_overlap_ids'])==52 and set(plan['previous_cache_overlap_ids'])<=set(ids),
        'Exact512 fitting-only assets and actual scheduled probes required')


def screen_evidence(root):
    from . import reference_masked_supervision_base_kv as screen
    from .reference_masked_supervision_evaluation import validate_records
    from .evaluate import ASR_REV
    run=root/'masked-supervision-base-kv-v1';_,p=screen.verify(run,SCREEN_SHA)
    generated=screen.native_evidence(run,SCREEN_SHA,p);scored=read(run/'scored.json');result=read(run/'results.json')
    require(scored['protocol_sha256']==SCREEN_SHA and scored['native_verification_sha256']==digest(run/'native-verification.json'),
        'Complete screen scoring binding required')
    summary=validate_records(p['cases'],scored['records']);by={(r['case'],r['seed']):r for r in scored['records']}
    revision=read(p['asr_manifest'])['revision']
    require(revision==ASR_REV,'Pinned recognizer revision changed')
    for r in generated:
        s=by[r['case'],r['seed']]
        require(set(s)==set(r)|{'hypothesis','score','asr_revision','decoding'} and all(s[k]==v for k,v in r.items())
            and s['decoding']==p['decoding'] and s['asr_revision']==revision,'Screen scoring changed native records')
    require(result['status']=='verified' and result['recordings']==80 and result['summary']==summary and result['truncated']==0
        and result['protocol_sha256']==SCREEN_SHA and result['generated_sha256']==digest(run/'generated.json')
        and result['scored_sha256']==digest(run/'scored.json') and result['native_verification_sha256']==digest(run/'native-verification.json')
        and result['calibration_qualified'] is screen.eligible(summary,0) and all(result.get(k)==v for k,v in screen.FLAGS.items()),
        'Complete native/scalar screen required')
    require_feature_priority(result['calibration_qualified'])
    status_path=root/'masked-supervision-base-kv-v1-status.json';status=read(status_path)
    driver=root/'run-masked-supervision-base-kv-v1.py';closure_path=root/'masked-supervision-base-kv-v1-process-closed.json'
    closed(read(closure_path),14712,dict(protocol_sha256=SCREEN_SHA,driver_sha256=digest(driver),
        terminal_status_sha256=digest(status_path),results_sha256=digest(run/'results.json')))
    phases={'controls-base','controls-student','donors','generate','audit-donors','audit','score','scalar-audit'}
    require(status['status']=='complete' and status['protocol_sha256']==SCREEN_SHA
        and status['results_sha256']==digest(run/'results.json') and set(status['logs'])==phases,'Screen phase history incomplete')
    evidence=dict(p['evidence']);evidence.update(read(run/'native-verification.json')['artifacts'])
    for q in [run/n for n in ('protocol.json','results.json','generated.json','scored.json','native-verification.json','donor-audit.json','donors.json','controls.json','base-controls.json')]+[driver,status_path,closure_path]:
        evidence[str(q)]=digest(q)
    for item in status['logs'].values():
        require(type(item['exit_code']) is int and item['exit_code']==0 and digest(item['path'])==item['sha256'],'Screen phase failed/changed')
        evidence[item['path']]=item['sha256']
    return evidence


def resolve(root):
    root=Path(root).resolve();require(root==data.ROOT.resolve(),'Registered external root required')
    run=root/'reference-masked-supervision-broad-data-v1';p=read(run/'protocol.json');a=read(run/'audit.json');plan=read(run/'plan.json')
    require(digest(run/'protocol.json')==DATA_SHA and digest(run/'audit.json')==AUDIT_SHA
        and a['status']=='verified_metadata_and_selected512_wavs' and a['protocol_sha256']==DATA_SHA
        and a['selected_wavs_rehashed']==512 and a['fitting_probes_subset_verified']==16
        and a['two_sided_holdout_conflicts']==0 and a['historical_test30_sha256']==data.TEST_SHA,
        'Fixed audited broader-data preparation required')
    driver=root/'prepare-reference-masked-supervision-broad-data-v1.py';closure_path=root/'reference-masked-supervision-broad-data-v1-process-closed.json'
    closed(read(closure_path),51327,dict(protocol_sha256=DATA_SHA,audit_sha256=AUDIT_SHA,driver_sha256=digest(driver)))
    review=root/'masked-supervision-broad-data-independent-audit-v1.json';require(digest(review)==REVIEW_SHA,'Independent512 WAV review changed')
    evidence=dict(p['evidence']);evidence.update(p['sources'])
    for name,value in p['manifests'].items():evidence[str(run/name)]=value
    for q in (run/'protocol.json',run/'audit.json',closure_path,review,driver):evidence[str(q)]=digest(q)
    old.bound_files(evidence)
    assets=data.rows(run/'selected-assets.jsonl');scope(assets,plan);wav=read(run/'selected-wav-audit.json')['assets']
    records=[]
    for r in assets:
        require(r['id'] in wav and wav[r['id']]['audio_sha256']==r['audio_sha256'],'Selected WAV audit binding differs')
        header={k:wav[r['id']][k] for k in ('sample_rate','channels','samples','sample_width')}
        require(old.wav_header(r['audio'])==header and digest(r['audio'])==r['audio_sha256'],'Selected real fitting audio changed')
        records.append(dict(r,wav_header=header,kind='fresh_fitting_utterance'))
        evidence[r['audio']]=r['audio_sha256']
    source=plan['calibration_feature_source'];prior=Path(source['run']);prior_p=read(prior/'protocol.json');prior_a=read(prior/'verification.json');prior_x=read(prior/'extraction.json')
    require(digest(prior/'protocol.json')==data.FEATURE_PROTOCOL_SHA and digest(prior/'verification.json')==data.FEATURE_AUDIT_SHA,
        'Original native feature reference changed')
    previous={r['id']:r for r in prior_p['records']};overlap=plan['previous_cache_overlap_ids']
    for identity in overlap:
        descriptor=previous[identity];new=next(r for r in records if r['id']==identity)
        require(new['text']==descriptor['text'] and new['audio_sha256']==descriptor['audio_sha256'],'Overlap asset identity differs')
        folder=prior/'assets'/old.asset_key(identity);receipt=read(folder/'receipt.json')
        require(digest(folder/'receipt.json')==prior_x['asset_receipts'][identity]
            and receipt['feature_sha256']==prior_a['feature_sha256'][identity]==digest(folder/'features.pt'),
            'Old overlap control feature changed')
        evidence[str(folder/'receipt.json')]=digest(folder/'receipt.json');evidence[str(folder/'features.pt')]=receipt['feature_sha256']
    evidence.update(screen_evidence(root))
    return records,plan,prior_p,evidence


def sources(driver,test_log):
    files=(Path(__file__),Path(__file__).with_name('reference_masked_supervision_features.py'),
        Path(__file__).with_name('reference_masked_supervision_broad_data.py'),Path(__file__).with_name('reference_masked_supervision_base_kv.py'),
        Path(__file__).parents[1]/'tests/test_reference_masked_supervision_broad_features.py',Path(driver),Path(test_log))
    return {str(q.resolve()):digest(q) for q in files}


def prepare(root,run,driver,test_log):
    run=old.external(run);require(run.parent==Path(root).resolve() and run.name=='reference-masked-supervision-broad-features-v1'
        and not run.exists(),'New external fitting-feature directory required')
    records,plan,prior,evidence=resolve(root)
    log=Path(test_log).read_text(encoding='utf-8-sig')
    require('Ran 8 tests in ' in log and log.strip().endswith('OK') and 'skipped' not in log.lower() and 'FAILED' not in log,'Eight passing focused tests required')
    runtime=old.runtime();require(runtime==prior['runtime'],'Native extraction runtime differs from exact-overlap reference')
    models=old.file_tree(prior['base']);require(models==prior['model_files'],'Original native base assets changed')
    run.mkdir();write(run/'protocol.json',dict(kind='fresh512_fitting_features_no_training',root=str(Path(root).resolve()),records=records,
        settings=SETTINGS,plan=plan,data_protocol_sha256=DATA_SHA,data_audit_sha256=AUDIT_SHA,base=prior['base'],model_files=models,
        old_feature_run=plan['calibration_feature_source']['run'],old_feature_protocol_sha256=data.FEATURE_PROTOCOL_SHA,
        overlap_ids=plan['previous_cache_overlap_ids'],calibration_features_separate=plan['calibration_feature_source'],
        evidence=evidence,sources=sources(driver,test_log),driver=str(Path(driver).resolve()),test_log=str(Path(test_log).resolve()),runtime=runtime,**FLAGS))


def verify(run,protocol_sha):
    run=old.external(run);require(digest(run/'protocol.json')==protocol_sha,'Registered feature protocol changed')
    p=read(run/'protocol.json');records,plan,prior,evidence=resolve(p['root'])
    require(run.parent==Path(p['root']).resolve() and run.name=='reference-masked-supervision-broad-features-v1','Feature output location changed')
    require(p['records']==records and p['plan']==plan and p['evidence']==evidence and p['settings']==SETTINGS
        and p['data_protocol_sha256']==DATA_SHA and p['data_audit_sha256']==AUDIT_SHA
        and p['base']==prior['base'] and p['model_files']==prior['model_files']
        and p['runtime']==old.runtime()==prior['runtime'] and p['sources']==sources(p['driver'],p['test_log'])
        and p['old_feature_run']==plan['calibration_feature_source']['run'] and p['old_feature_protocol_sha256']==data.FEATURE_PROTOCOL_SHA
        and p['overlap_ids']==plan['previous_cache_overlap_ids'] and p['calibration_features_separate']==plan['calibration_feature_source']
        and all(p.get(k)==v for k,v in FLAGS.items()),'Feature scope/provenance/runtime changed')
    old.bound_files(p['sources']);old.bound_files(p['model_files']);return p


def compare_overlap(fresh,previous):
    import torch
    require(fresh['id']==previous['id'] and fresh['audio_sha256']==previous['audio_sha256'],'Overlap identity differs')
    for name in old.TENSORS:
        a,b=fresh[name],previous[name]
        require(isinstance(a,torch.Tensor) and isinstance(b,torch.Tensor) and a.dtype==b.dtype
            and a.shape==b.shape and a.device.type==b.device.type=='cpu' and torch.equal(a,b),
            'Fresh/old exact tensor comparison failed: '+name)
    return {name:old.tensor_sha256(fresh[name]) for name in old.TENSORS}


def old_controls(p):
    previous=Path(p['old_feature_run']);prior=read(previous/'protocol.json');index={r['id']:r for r in prior['records']}
    # Only52 fitting overlap controls; never load old calibration tensors here.
    return {i:old.read_asset(previous,index[i],p['old_feature_protocol_sha256'])[0] for i in p['overlap_ids']}


def native_tensors(wrapper):
    return {f'{label}.{kind}.{name}':value
        for label,module in [('native',wrapper.model),('codec',wrapper.model.speech_tokenizer.model)]
        for kind,items in [('parameter',module.named_parameters()),('buffer',module.named_buffers())]
        for name,value in items}


def full_state(wrapper):
    # named_buffers includes nonpersistent buffers omitted by state_dict().
    return {name:old.tensor_sha256(value) for name,value in native_tensors(wrapper).items()}


def same_registry(wrapper,initial):
    current=native_tensors(wrapper)
    return current.keys()==initial.keys() and all(current[k] is v for k,v in initial.items())


def extract(run,protocol_sha):
    import torch
    from qwen_tts import Qwen3TTSModel
    run=Path(run);p=verify(run,protocol_sha);require(not (run/'assets').exists() and not (run/'extraction.json').exists(),'No implicit extraction restart')
    old.configure_numeric();assets=run/'assets';assets.mkdir()
    wrapper=Qwen3TTSModel.from_pretrained(p['base'],device_map='cpu',torch_dtype=torch.float32,attn_implementation='sdpa',local_files_only=True)
    modules=(wrapper.model,wrapper.model.speech_tokenizer.model)
    for module in modules:module.requires_grad_(False).eval()
    require(not hasattr(wrapper.model,'speaker_preference_adapter') and all(v.device.type=='cpu' and (not v.is_floating_point() or v.dtype==torch.float32)
        for m in modules for v in m.state_dict().values()),'Original unadapted native CPUFP32 required')
    identity=old.native_component_identity(wrapper);prior_identity=read(Path(p['old_feature_run'])/'extraction.json')['native_components']
    require(identity==prior_identity,'Original native codec/speaker encoder identity differs')
    before=full_state(wrapper);registry=native_tensors(wrapper);versions=[(v,v._version) for v in registry.values()]
    controls=old_controls(p);receipts={};comparisons={}
    for number,descriptor in enumerate(p['records'],1):
        require(digest(descriptor['audio'])==descriptor['audio_sha256'] and old.wav_header(descriptor['audio'])==descriptor['wav_header'],'Fitting WAV changed')
        with torch.inference_mode():prompt=wrapper.create_voice_clone_prompt(ref_audio=descriptor['audio'],ref_text=descriptor['text'],x_vector_only_mode=False)
        require(len(prompt)==1 and prompt[0].icl_mode and not prompt[0].x_vector_only_mode and prompt[0].ref_text==descriptor['text']
            and prompt[0].ref_spk_embedding.dtype==torch.float32 and prompt[0].ref_code.dtype==torch.long,'Native prompt mode/dtype differs')
        value=dict(old.native_ids(wrapper.processor,descriptor['text']),codes=prompt[0].ref_code.detach().cpu().clone(),
            speaker=prompt[0].ref_spk_embedding.detach().cpu().flatten().clone(),id=descriptor['id'],audio_sha256=descriptor['audio_sha256'],
            protocol_sha256=protocol_sha,descriptor_sha256=fingerprint(descriptor))
        tensors=old.validate_feature(value,descriptor,protocol_sha)
        if descriptor['id'] in controls:comparisons[descriptor['id']]=compare_overlap(value,controls[descriptor['id']])
        require(same_registry(wrapper,registry) and all(v._version==version and v.grad is None and not v.requires_grad for v,version in versions),'Native extraction mutated parameters/buffers or gradients')
        folder=assets/old.asset_key(descriptor['id']);folder.mkdir()
        with (folder/'features.pt').open('xb') as stream:torch.save(value,stream);stream.flush();os.fsync(stream.fileno())
        receipt=dict(id=descriptor['id'],protocol_sha256=protocol_sha,descriptor_sha256=fingerprint(descriptor),
            feature_sha256=digest(folder/'features.pt'),tensors=tensors,native_components=fingerprint(identity),
            fresh_native_extraction=True,old_feature_reused_as_output=False,original_unadapted_speaker=True,
            full_wav_samples=descriptor['wav_header']['samples'])
        write(folder/'receipt.json',receipt);receipts[descriptor['id']]=digest(folder/'receipt.json')
        print(f'fresh fitting features {number}/512; overlap controls checked {len(comparisons)}/52',flush=True)
    require(set(comparisons)==set(p['overlap_ids']) and full_state(wrapper)==before
        and old.native_component_identity(wrapper)==identity and same_registry(wrapper,registry)
        and all(v._version==version and v.grad is None for v,version in versions),
        'Incomplete overlap controls or native state changed')
    verify(run,protocol_sha)
    write(run/'extraction.json',dict(protocol_sha256=protocol_sha,records=512,asset_receipts=receipts,native_components=identity,
        before_after_full_state_sha256=fingerprint(before),all_parameters_buffers_unchanged=True,all_gradients_absent=True,
        overlap_tensor_comparisons=comparisons,overlap_comparisons=52,all512_fresh=True,cached_features_used_as_outputs=False,**FLAGS))


def extraction_evidence(run,p,sha):
    e=read(run/'extraction.json');expected={r['id'] for r in p['records']}
    require(e['protocol_sha256']==sha and e['records']==512 and set(e['asset_receipts'])==expected
        and e['all512_fresh'] is True and e['cached_features_used_as_outputs'] is False
        and e['all_parameters_buffers_unchanged'] is e['all_gradients_absent'] is True
        and set(e['overlap_tensor_comparisons'])==set(p['overlap_ids']) and e['overlap_comparisons']==52
        and e['native_components']==read(Path(p['old_feature_run'])/'extraction.json')['native_components']
        and all(e.get(k)==v for k,v in FLAGS.items()),'Complete fresh512 native extraction required')
    return e


def audit(run,protocol_sha):
    import torch
    run=Path(run);p=verify(run,protocol_sha);e=extraction_evidence(run,p,protocol_sha)
    require(not (run/'verification.json').exists(),'No feature-audit overwrite');old.configure_numeric()
    processor=old.load_native_processor(p['base']);controls=old_controls(p);hashes={};comparisons={}
    for descriptor in p['records']:
        identity=descriptor['id'];folder=run/'assets'/old.asset_key(identity)
        require(digest(folder/'receipt.json')==e['asset_receipts'][identity],'Extracted receipt changed')
        value,r=old.read_asset(run,descriptor,protocol_sha)
        require(r['fresh_native_extraction'] is True and r['old_feature_reused_as_output'] is False
            and r['original_unadapted_speaker'] is True and r['native_components']==fingerprint(e['native_components'])
            and r['full_wav_samples']==descriptor['wav_header']['samples']
            and digest(descriptor['audio'])==descriptor['audio_sha256'] and old.wav_header(descriptor['audio'])==descriptor['wav_header'],
            'Fresh native asset/source provenance differs')
        ids=old.native_ids(processor,descriptor['text'])
        require(all(torch.equal(value[k],ids[k]) for k in ids),'Independent native role-framed IDs differ')
        if identity in controls:comparisons[identity]=compare_overlap(value,controls[identity])
        hashes[identity]=r['feature_sha256']
    require(comparisons==e['overlap_tensor_comparisons'] and len(hashes)==512,'Independent52 overlap comparisons differ')
    verify(run,protocol_sha)
    write(run/'verification.json',dict(status='verified_fresh512_fitting_features',protocol_sha256=protocol_sha,
        extraction_sha256=digest(run/'extraction.json'),records=512,feature_sha256=hashes,native_ids_verified=512,
        all_tensor_contracts_verified=True,source_wavs_rehashed=512,exact_overlap_tensor_comparisons=52,
        all512_fresh=True,calibration_assets_extracted=0,full_codec_speaker_reextraction_in_audit=False,**FLAGS))


def verify_complete(run,protocol_sha=None):
    run=Path(run);sha=protocol_sha or digest(run/'protocol.json');p=verify(run,sha);e=extraction_evidence(run,p,sha);a=read(run/'verification.json')
    require(a['status']=='verified_fresh512_fitting_features' and a['protocol_sha256']==sha
        and a['extraction_sha256']==digest(run/'extraction.json') and a['records']==a['native_ids_verified']==a['source_wavs_rehashed']==512
        and a['all_tensor_contracts_verified'] is a['all512_fresh'] is True and a['exact_overlap_tensor_comparisons']==52
        and a['calibration_assets_extracted']==0 and set(a['feature_sha256'])==set(e['asset_receipts'])
        and all(a.get(k)==v for k,v in FLAGS.items()),'Complete independently audited fitting features required')
    return p


def load_features(run):
    run=Path(run);sha=digest(run/'protocol.json');p=verify_complete(run,sha);a=read(run/'verification.json');e=read(run/'extraction.json');values={}
    for descriptor in p['records']:
        v,r=old.read_asset(run,descriptor,sha);identity=descriptor['id']
        require(r['feature_sha256']==a['feature_sha256'][identity]
            and digest(run/'assets'/old.asset_key(identity)/'receipt.json')==e['asset_receipts'][identity],'Audited fitting feature changed')
        values[identity]=v
    return p,values


def assert_native_components(wrapper,run):
    verify_complete(run);require(old.native_component_identity(wrapper)==read(Path(run)/'extraction.json')['native_components'],
        'Training policy changed native codec/speaker encoder')


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=('prepare','extract','audit'))
    parser.add_argument('--root',type=Path,default=data.ROOT);parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--driver',type=Path);parser.add_argument('--test-log',type=Path);parser.add_argument('--protocol-sha');a=parser.parse_args()
    if a.phase=='prepare':prepare(a.root,a.run,a.driver,a.test_log);return
    require(a.protocol_sha,'Explicit registered protocol hash required')
    {'extract':extract,'audit':audit}[a.phase](a.run,a.protocol_sha)


if __name__=='__main__':main()
