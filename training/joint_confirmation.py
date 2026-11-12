"""Register broader joint-goal confirmation before preference results exist."""
import argparse
import copy
from pathlib import Path

from data.pipeline import text_key
from .engine import digest, read, rows, write
from .policy import external, fingerprint, validate_cohorts


def prepare(parent, output):
    p=read(parent/'protocol.json');backup=Path(p['config']['backup'])
    recovery=read(backup/'recovery-verification.json')['cohorts']
    for name in ('train','unseen-validation'):
        if digest(backup/f'cohorts/{name}.jsonl')!=recovery[name]['sha256']:
            raise ValueError('Cohort changed')
    inventory=read(backup/'dataset-v1/dataset-inventory.json')['files']
    if digest(backup/'dataset-v1/manifest.jsonl')!=inventory['manifest.jsonl']['sha256']:
        raise ValueError('Canonical manifest changed')
    train=rows(backup/'cohorts/train.jsonl');development=rows(backup/'cohorts/unseen-validation.jsonl')
    validate_cohorts(train,development,rows(backup/'dataset-v1/manifest.jsonl'))
    if len(development)!=44:
        raise ValueError('Unexpected development size')
    cases=copy.deepcopy(p['cases'])
    screened={c['row_id'] for c in cases if c['group']=='development'}
    extra=sorted((r for r in development if r['id'] not in screened),key=lambda r:r['id'])
    for i,r in enumerate(extra):
        reference=backup/'dataset-v1'/r['ref_audio']
        sha=inventory[r['ref_audio']]['sha256']
        if digest(reference)!=sha:raise ValueError('Development reference changed')
        cases.append(dict(group='development',case=f'dev-extra-{i:03d}',row_id=r['id'],
            text=r['text'],speaker_id=r['speaker_id'],reference=str(reference),
            reference_text=r['ref_text'],reference_sha256=sha))
    original=next(c for c in cases if c['group']=='custom')
    forbidden={text_key(c['text']) for c in cases if c['group']=='custom'}
    forbidden.add(text_key(original['reference_text']))
    extra_texts=sorted((r for r in development if text_key(r['text']) not in forbidden),
        key=lambda r:fingerprint(['joint-confirmation-v1',r['id']]))[:10]
    for i,r in enumerate(extra_texts):
        cases.append(dict(original,case=f'custom-extra-{i:03d}',text=r['text'],text_source_row_id=r['id']))
    if (len(cases)!=57 or len({(c['group'],c['case']) for c in cases})!=57
            or len({text_key(c['text']) for c in cases if c['group']=='custom'})!=13):
        raise ValueError('Confirmation panel is incomplete or duplicates custom text')
    if digest(original['reference'])!=original['reference_sha256']:
        raise ValueError('Custom reference changed')
    out=external(output.resolve());out.mkdir(exist_ok=False)
    write(out/'protocol.json',dict(parent_protocol_sha256=digest(parent/'protocol.json'),
        preparation_source_sha256=digest(Path(__file__)),cases=cases,seeds=p['seeds'],
        generation=p['generation'],decoding=p['decoding'],
        backend='Same CPU FP32 SDPA backend as screening; fixed ASR and unchanged internal speaker encoder',
        development_recordings=88,custom_recordings=26,
        criteria=dict(pooled_cer_strictly_below=0.03,speaker_macro_cosine_strictly_above=0.984,
            groups=['development','custom'],require_complete_panel=True),
        provenance='All 44 existing development targets, and 13 custom-reference texts (3 existing plus 10 hash-selected development transcripts). These are expanded development/personalization checks, not a new untouched test set.',
        promotion='Only consider a screen-passing candidate. Record every fixed seed, both group metrics, all audio and model hashes. No favorable-seed or best-of-N selection.',
        goal_achieved=False))
    print(dict(development_recordings=88,custom_recordings=26,registered=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    prepare(args.parent,args.output)
