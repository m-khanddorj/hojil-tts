"""Pure, independently auditable broader-speaker data preparation.

No torch, model, native feature extraction or optimizer imports. Original
datasets/plans remain immutable. Only selected fitting WAVs are physically
audited; the complete candidate pool is metadata evidence, not a native audit.
"""
from collections import Counter,defaultdict
import hashlib
import json
import math
from pathlib import Path
import unicodedata
import wave

from .reference_masked_supervision_plan import SETTINGS,fitting_schedule,fitting_probe

ROOT=Path('C:/Users/Khan/.codex/visualizations/2026/11/09/01a1bf77-0b17-7391-9781-f859f1fbb0b1')
DATA_SHA='89b0819de650369fb03556fd190a4b7129e58bfc32ebdfd5886e09addb98e62d'
PLAN_SHA='a4cefa5a77ca97696048df57a31f023c5a0441f88a626811e46edd13470c0c34'
TEST_SHA='187c8dd03b560e90e62594e9ecc6d452f3985d3c241d64efadb2d1fb6b54e112'
PANEL_SHA='f77c6f247d87e2132a21ae0eafda04f64f614ef44d6acd275e3f7ee8832b6041'
FEATURE_PROTOCOL_SHA='394d668d34cf01bf9e777952d2dc0c202cb1702d05c1bc038724c87a142d42d6'
FEATURE_AUDIT_SHA='4b36bb802beaa26346976e994f5c4b0f62ac08695181de2bfd99033282d7caad'
TRAINING_SHA='89463810f5f9fc5d75d3eabf16144e30872aa1488b0605418f4f604abee12651'
SPEECH_SHA='07c12b076ee08f81c2a409a300a02232d04a99a88c1fb1b6bcdcdb11248d318a'
IDENTITIES=('id','audio','pcm_sha256','original_pcm_sha256','original_audio_sha256','original_clip_id')
FLAGS=dict(native_models_imported=False,native_features_extracted=False,native_features_ready=False,
    training_registered=False,optimizer_updates=0,goal_achieved=False,global_model_selection_changed=False)


def require(value,message):
    if not value:raise ValueError(message)


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def rows(path):
    with Path(path).open(encoding='utf-8-sig') as stream:return [json.loads(line) for line in stream if line.strip()]


def write(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)


def write_rows(path,records):
    with Path(path).open('x',encoding='utf-8') as stream:
        for row in records:stream.write(json.dumps(row,ensure_ascii=False,sort_keys=True,allow_nan=False)+'\n')


def text_key(text):return ''.join(c for c in unicodedata.normalize('NFC',text).casefold() if c.isalnum())


def blocked_identity(records):return {k:{r[k] for r in records if r.get(k)} for k in IDENTITIES}


def conflicts(pair,canonical,texts,identities):
    return any(text_key(r['text']) in texts or any(r.get(k) in values for k,values in identities.items() if r.get(k))
               for r in (pair,canonical[pair['ref_id']]))


def reference_closure(records):
    removed=[];rounds=[]
    while True:
        ids={r['id'] for r in records};drop={r['id'] for r in records if r['ref_id'] not in ids}
        if not drop:return records,rounds,removed
        removed.extend(sorted(drop));records=[r for r in records if r['id'] not in drop];rounds.append(len(records))


def validate_pair(target,reference):
    require(target['id']!=reference['id'] and target['speaker_id']==reference['speaker_id']
        and target['speaker_id'] and target['split']==reference['split']=='train'
        and target['audio']!=reference['audio'] and text_key(target['text'])!=text_key(reference['text'])
        and all(target.get(k) and reference.get(k) and target[k]!=reference[k]
                for k in ('pcm_sha256','original_pcm_sha256','original_audio_sha256')),
        'Distinct original-train same-speaker target/reference contract failed')


def eligible_pool(train,canonical,blocked_speakers,guards):
    require(len({r['id'] for r in train})==len(train),'Duplicate original training ID')
    initial=[r for r in train if r['speaker_id'] not in blocked_speakers]
    require(all(r['ref_id'] in canonical for r in initial),'Missing canonical reference')
    retained=[r for r in initial if not any(conflicts(r,canonical,g['texts'],g['identities']) for g in guards)]
    result,closure,dropped=reference_closure(retained);result.sort(key=lambda r:r['id'])
    for r in result:
        require(r['id'] in canonical and all(r[k]==canonical[r['id']][k] for k in
            (*IDENTITIES,'text','speaker_id','split','duration','sample_rate','source','source_family','revision','license')),
            'Training row differs from canonical source')
        ref=canonical[r['ref_id']];validate_pair(r,ref)
        require(r['ref_audio']==ref['audio'] and r['ref_text']==ref['text'],'Immutable reference mapping changed')
    return result,dict(after_speaker_exclusions=len(initial),after_identity_exclusions=len(retained),
        reference_closure_counts=closure,reference_closure_removed_ids=dropped)


def selected_plan(pool):
    updates=fitting_schedule(pool);indexed={r['id']:r for r in pool}
    visits=[i for update in updates for i in update['pair_ids']];selected=[indexed[i] for i in sorted(set(visits))]
    probe=fitting_probe(selected)
    require(len(visits)==len(selected)==256 and len({r['speaker_id'] for r in selected})==256,
        'Exactly256 different scheduled pairs/speakers required')
    require(len(probe)==16 and set(probe)<=set(visits),'Fitting probes must be a subset of actual scheduled pairs')
    feature_ids=sorted({r[k] for r in selected for k in ('id','ref_id')})
    require(len(feature_ids)==512,'Exactly512 selected fitting endpoints required')
    return selected,dict(settings=dict(SETTINGS),updates=updates,fitting_probe_pair_ids=probe,
        selected_fitting_pair_ids=[r['id'] for r in selected],fitting_feature_ids=feature_ids,
        fitting_feature_policy='Freshly extract all512 selected fitting endpoints in a separate future native job; previous-cache overlap is metadata only')


def audio_path(dataset,relative):
    path=(dataset/relative).resolve()
    require(path.is_relative_to(dataset.resolve()),'Audio path escapes registered dataset')
    return path


def audit_wav(path,expected_sha,canonical):
    require(Path(path).is_file() and digest(path)==expected_sha,'Selected cleaned WAV hash differs')
    with wave.open(str(path),'rb') as stream:
        sr,ch,width,n,compression=stream.getframerate(),stream.getnchannels(),stream.getsampwidth(),stream.getnframes(),stream.getcomptype()
        raw=stream.readframes(n)
    require(sr==canonical['sample_rate']==24000 and ch==1 and width==2 and compression=='NONE' and n>0
        and len(raw)==n*2 and canonical['samples']==n and math.isfinite(canonical['duration'])
        and abs(n/sr-canonical['duration'])<=1/sr
        and hashlib.sha256(raw).hexdigest()==canonical['pcm_sha256'],'Selected WAV PCM/header/duration differs')
    return dict(audio_sha256=expected_sha,pcm_sha256=canonical['pcm_sha256'],sample_rate=sr,
        channels=ch,sample_width=width,samples=n,duration=n/sr)


def source_contract(row):
    require(row['source']=='cv27' and row['source_family']=='common_voice' and str(row['revision'])=='27.0'
        and row['split']=='train' and row['language']=='mn-MN' and row['sample_rate']==24000
        and row['license']=='CC0-1.0; preserve separate access terms','Original source/format/licence differs')


def normalized_pair(row,canonical,dataset,inventory):
    ref=canonical[row['ref_id']]
    value=dict(id=row['id'],ref_id=ref['id'],speaker_id=row['speaker_id'],partition='fitting',original_split='train',
        text=row['text'],ref_text=ref['text'],audio=str(audio_path(dataset,row['audio'])),ref_audio=str(audio_path(dataset,ref['audio'])),
        audio_sha256=inventory[row['audio']]['sha256'],ref_audio_sha256=inventory[ref['audio']]['sha256'],
        duration=row['duration'],ref_duration=ref['duration'],sample_rate=24000,source='cv27',source_family='common_voice',
        revision='27.0',license=row['license'],reference_policy='distinct same-speaker real utterance; original train split; immutable pairing')
    for name in ('pcm_sha256','original_pcm_sha256','original_audio_sha256','original_clip_id'):
        value[name]=row[name];value['ref_'+name]=ref[name]
    return value


def reconstruct(root):
    """Pure canonical reconstruction; no media/tensor loads or output writes."""
    root=Path(root).resolve();require(root==ROOT.resolve(),'Registered external artifact root required')
    dp=root/'reference-masked-supervision-data-v1';old=read(dp/'protocol.json');old_plan=root/'reference-masked-supervision-plan-v1.json'
    require(digest(dp/'protocol.json')==DATA_SHA and digest(old_plan)==PLAN_SHA,'Original data/plan changed')
    plan0=read(old_plan);require(plan0['settings']==SETTINGS,'Comparison training settings changed')
    evidence=dict(old['inputs']);evidence.update(old['source_sha256'])
    evidence.update({str(dp/'protocol.json'):DATA_SHA,str(dp/'audit.json'):plan0['data_audit_sha256'],str(old_plan):PLAN_SHA})
    for name,value in old['manifests'].items():evidence[str(dp/name)]=value
    for path,value in evidence.items():require(digest(path)==value,'Bound original data evidence changed')
    manifest=next(Path(k) for k in old['inputs'] if k.endswith('manifest.jsonl'))
    cohort=next(Path(k) for k in old['inputs'] if Path(k).parent.name=='cohorts' and Path(k).name=='train.jsonl')
    dataset=manifest.parent;inventory_path=dataset/'dataset-inventory.json';inventory=read(inventory_path)['files']
    original=rows(cohort);canonical_rows=rows(manifest);canonical={r['id']:r for r in canonical_rows}
    require(len(original)==28001 and len({r['speaker_id'] for r in original})==435 and len(canonical)==len(canonical_rows),'Original source cohort differs')
    test_path=cohort.parent/'unseen-test.jsonl';require(digest(test_path)==TEST_SHA,'Previously inspected TEST30 changed')
    test=rows(test_path);require(len(test)==30 and len({r['speaker_id'] for r in test})==10,'Historical TEST30 scope differs')
    for r in test:
        require(r['id'] in canonical and r['ref_id'] in canonical and all(r[k]==canonical[r['id']][k] for k in (*IDENTITIES,'speaker_id','text'))
            and r['ref_text']==canonical[r['ref_id']]['text'] and r['ref_audio']==canonical[r['ref_id']]['audio'],'TEST30 canonical identity differs')
    panel_path=root/'joint-goal-confirmation/protocol.json';require(digest(panel_path)==PANEL_SHA,'Original114 changed');panel=read(panel_path)
    require(len(panel['cases'])==57 and panel['seeds']==[2026,2027]
        and Counter(c['group'] for c in panel['cases'])=={'development':44,'custom':13},'Original114 scope differs')
    cal=rows(dp/'calibration.jsonl');require(len(cal)==32 and len({r['speaker_id'] for r in cal})==16,'Original calibration differs')
    eval_ids={c[k] for c in panel['cases'] for k in ('row_id','text_source_row_id') if k in c};eval_audio={c['reference_sha256'] for c in panel['cases']}
    eval_rows=[r for r in canonical_rows if r['id'] in eval_ids or inventory[r['audio']]['sha256'] in eval_audio]
    cal_ids={r[k] for r in cal for k in ('id','ref_id')};test_ids={r[k] for r in test for k in ('id','ref_id')}
    guards=[dict(name='original114',texts={text_key(c[k]) for c in panel['cases'] for k in ('text','reference_text')},identities=blocked_identity(eval_rows)),
        dict(name='calibration',texts={text_key(r[k]) for r in cal for k in ('text','ref_text')},identities=blocked_identity([canonical[i] for i in cal_ids])),
        dict(name='historical_TEST30_previously_inspected',texts={text_key(r[k]) for r in test for k in ('text','ref_text')},identities=blocked_identity([canonical[i] for i in test_ids]))]
    speaker_groups=dict(calibration={r['speaker_id'] for r in cal},original114={c['speaker_id'] for c in panel['cases']},historical_TEST30={r['speaker_id'] for r in test})
    forbidden=set().union(*speaker_groups.values());pool,counts=eligible_pool(original,canonical,forbidden,guards)
    require(len(pool)==24369 and len({r['speaker_id'] for r in pool})==414
        and counts['after_speaker_exclusions']==26403 and counts['after_identity_exclusions']==25941,'Previously reviewed broader pool differs')
    for r in pool:source_contract(r);source_contract(canonical[r['ref_id']])
    selected,plan=selected_plan(pool);normal=[normalized_pair(r,canonical,dataset,inventory) for r in pool]
    index={r['id']:r for r in normal};selected_normal=[index[r['id']] for r in selected]
    feature=root/'reference-masked-supervision-features-v1';feature_p=read(feature/'protocol.json');feature_a=read(feature/'verification.json');feature_x=read(feature/'extraction.json')
    require(digest(feature/'protocol.json')==FEATURE_PROTOCOL_SHA and digest(feature/'verification.json')==FEATURE_AUDIT_SHA,
        'Previously audited feature protocol/audit identity changed')
    require(feature_a['verified'] is True and feature_a['records']==324 and feature_a['protocol_sha256']==digest(feature/'protocol.json')
        and feature_a['extraction_sha256']==digest(feature/'extraction.json') and set(feature_a['feature_sha256'])==set(feature_x['asset_receipts']),
        'Original calibration feature provenance differs')
    train=root/'reference-masked-supervision-v2';training=read(train/'protocol.json');speech=read(train/'speech-protocol.json')
    require(digest(train/'protocol.json')==TRAINING_SHA and digest(train/'speech-protocol.json')==SPEECH_SHA,
        'Historical v2 comparator or unchanged calibration protocol identity changed')
    require(training['plan_sha256']==PLAN_SHA and training['policy']['kind']=='centered16'
        and speech['training_protocol_sha256']==digest(train/'protocol.json')
        and speech['features_audit_sha256']==digest(feature/'verification.json'),'Frozen comparator/old calibration binding differs')
    custom_id='__custom_reference__';cal_feature_ids=sorted(cal_ids|{custom_id});old_assets={r['id']:r for r in feature_p['records']}
    require(len(cal_feature_ids)==56 and set(cal_feature_ids)<=set(old_assets),'Unchanged55real+custom calibration features required')
    for identity in cal_feature_ids:
        descriptor=old_assets[identity]
        directory=feature/'assets'/hashlib.sha256(identity.encode()).hexdigest()
        receipt_path=directory/'receipt.json';tensor_path=directory/'features.pt';receipt=read(receipt_path)
        descriptor_sha=hashlib.sha256(json.dumps(descriptor,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
        require(digest(receipt_path)==feature_x['asset_receipts'][identity]
            and receipt['id']==identity and receipt['protocol_sha256']==digest(feature/'protocol.json')
            and receipt['descriptor_sha256']==descriptor_sha
            and digest(tensor_path)==receipt['feature_sha256']==feature_a['feature_sha256'][identity],
            'Unchanged calibration feature/receipt byte binding failed')
        evidence[str(receipt_path)]=digest(receipt_path);evidence[str(tensor_path)]=receipt['feature_sha256']
        if identity!=custom_id:
            require(descriptor['text']==canonical[identity]['text'] and descriptor['audio_sha256']==inventory[canonical[identity]['audio']]['sha256'],
                'Calibration feature descriptor differs')
    overlap=sorted(set(plan['fitting_feature_ids'])&set(old_assets));require(len(overlap)==52,'Expected old-cache metadata overlap changed')
    assets=[]
    for identity in plan['fitting_feature_ids']:
        r=canonical[identity];source_contract(r);path=audio_path(dataset,r['audio'])
        assets.append(dict(id=identity,text=r['text'],audio=str(path),audio_sha256=inventory[r['audio']]['sha256'],
            speaker_id=r['speaker_id'],duration=r['duration'],sample_rate=r['sample_rate'],samples=r['samples'],pcm_sha256=r['pcm_sha256'],
            source=r['source'],source_family=r['source_family'],revision=str(r['revision']),license=r['license'],language=r['language'],split=r['split'],
            source_split=r['source_split'],original_audio_sha256=r['original_audio_sha256'],original_pcm_sha256=r['original_pcm_sha256'],
            original_clip_id=r['original_clip_id'],roles=sorted({role for role,field in [('target','id'),('reference','ref_id')] if any(x[field]==identity for x in selected)})))
    plan.update(kind='broader_speaker_data_only_controlled32',calibration_cases=plan0['calibration_cases'],custom_calibration=plan0['custom_calibration'],
        generation_seeds=plan0['generation_seeds'],calibration_recordings=80,
        calibration_feature_source=dict(run=str(feature),protocol_sha256=digest(feature/'protocol.json'),audit_sha256=digest(feature/'verification.json'),
            extraction_sha256=digest(feature/'extraction.json'),ids=cal_feature_ids,policy='Reuse unchanged old calibration features; not part of new512 fitting extraction'),
        previous_cache_overlap_ids=overlap,previous_cache_overlap_is_readiness=False,
        comparator=dict(train=str(train),protocol_sha256=digest(train/'protocol.json'),checkpoint0_sha256=digest(train/'checkpoint-000000.pt'),
            speech_protocol_sha256=digest(train/'speech-protocol.json'),frozen_policy=training['policy'],numeric_config=training['numeric_config'],
            generation=training['generation'],decoding=training['decoding']),
        no_base_kv_or_objective_change=True,**FLAGS)
    for q in (test_path,panel_path,feature/'protocol.json',feature/'verification.json',feature/'extraction.json',train/'protocol.json',
              train/'speech-protocol.json',train/'checkpoint-000000.pt'):
        evidence[str(q)]=digest(q)
    receipt=dict(pool_pairs=24369,pool_speakers=414,selected_pairs=256,selected_speakers=256,selected_assets=512,
        fitting_probe_pairs=16,probes_subset_of_scheduled=True,prior_feature_overlap=52,prior_cache_readiness_claim=False,
        test30_previously_inspected=True,whole_speaker_guards={k:sorted(v) for k,v in speaker_groups.items()},
        text_and_source_audio_guards=[g['name'] for g in guards],two_sided_guard_conflicts=0,
        counts=counts,full_pool_media_or_native_features_verified=False,**FLAGS)
    return dict(pool=normal,selected=selected_normal,assets=assets,plan=plan,metadata=receipt,evidence=evidence)


def prepare(root,out,driver,test_log):
    root,out=Path(root).resolve(),Path(out).resolve()
    require(out.parent==root and out.name=='reference-masked-supervision-broad-data-v1' and not out.exists(),'Fresh external data directory required')
    log=Path(test_log).read_text(encoding='utf-8-sig')
    require('Ran 12 tests in ' in log and log.strip().endswith('OK') and 'skipped' not in log.lower() and 'FAILED' not in log,'Twelve passing pure tests required')
    built=reconstruct(root);media={}
    for i,asset in enumerate(built['assets']):
        media[asset['id']]=audit_wav(asset['audio'],asset['audio_sha256'],asset)
        if (i+1)%64==0:print(f'selected WAV audit {i+1}/512',flush=True)
    evidence=built['evidence'];sources={str(q.resolve()):digest(q) for q in (Path(__file__),Path(__file__).with_name('reference_masked_supervision_plan.py'),
        Path(__file__).parents[1]/'tests/test_reference_masked_supervision_broad_data.py',Path(driver),Path(test_log))}
    # All checks finish before the first output directory is created.
    out.mkdir();write_rows(out/'pool.jsonl',built['pool']);write_rows(out/'selected.jsonl',built['selected'])
    write_rows(out/'selected-assets.jsonl',built['assets']);write(out/'plan.json',built['plan']);write(out/'selected-wav-audit.json',dict(assets=media,recordings=512,native_features_verified=False))
    protocol=dict(kind='broader_supervision_pure_preparation',root=str(root),metadata=built['metadata'],evidence=evidence,sources=sources,
        manifests={name:digest(out/name) for name in ('pool.jsonl','selected.jsonl','selected-assets.jsonl','plan.json','selected-wav-audit.json')},
        old_data_protocol_sha256=DATA_SHA,old_plan_sha256=PLAN_SHA,historical_test30_sha256=TEST_SHA,original114_sha256=PANEL_SHA,
        calibration_speech_panel_unchanged=True,independent_audit_pending=True,**FLAGS)
    write(out/'protocol.json',protocol);return protocol


def audit(out):
    out=Path(out).resolve();p=read(out/'protocol.json');require(not (out/'audit.json').exists(),'No audit overwrite')
    require(out.parent==ROOT.resolve() and out.name=='reference-masked-supervision-broad-data-v1'
        and p['root']==str(ROOT.resolve()) and p['historical_test30_sha256']==TEST_SHA and p['original114_sha256']==PANEL_SHA
        and p['old_data_protocol_sha256']==DATA_SHA and p['old_plan_sha256']==PLAN_SHA
        and p['calibration_speech_panel_unchanged'] is True and all(p.get(k)==v for k,v in FLAGS.items()),'Registered data scope differs')
    for group in ('evidence','sources'):
        for path,value in p[group].items():require(digest(path)==value,'Registered data source/evidence changed')
    for name,value in p['manifests'].items():require(digest(out/name)==value,'Prepared data manifest changed')
    rebuilt=reconstruct(p['root']);require(rebuilt['evidence']==p['evidence'] and rebuilt['metadata']==p['metadata'],'Independent population reconstruction differs')
    require(rows(out/'pool.jsonl')==rebuilt['pool'] and rows(out/'selected.jsonl')==rebuilt['selected']
        and rows(out/'selected-assets.jsonl')==rebuilt['assets'] and read(out/'plan.json')==rebuilt['plan'],'Independent schedule/asset reconstruction differs')
    observed={r['id']:audit_wav(r['audio'],r['audio_sha256'],r) for r in rebuilt['assets']}
    require(read(out/'selected-wav-audit.json')==dict(assets=observed,recordings=512,native_features_verified=False),'Independent512 WAV audit differs')
    write(out/'audit.json',dict(status='verified_metadata_and_selected512_wavs',protocol_sha256=digest(out/'protocol.json'),
        manifests=p['manifests'],pool_pairs=24369,pool_speakers=414,selected_pairs=256,selected_speakers=256,
        selected_wavs_rehashed=512,fitting_probes_subset_verified=16,historical_test30_sha256=TEST_SHA,
        two_sided_holdout_conflicts=0,full_pool_wavs_rehashed=False,**FLAGS))


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=('prepare','audit'))
    parser.add_argument('--root',type=Path,default=ROOT);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--driver',type=Path);parser.add_argument('--test-log',type=Path);a=parser.parse_args()
    if a.phase=='prepare':prepare(a.root,a.out,a.driver,a.test_log)
    else:audit(a.out)


if __name__=='__main__':main()
