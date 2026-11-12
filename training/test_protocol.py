"""Explicit reserved-test protocol, isolated from development evaluation caches."""
import copy
import hashlib
import json
from pathlib import Path

from data.pipeline import text_key
from .policy import validate_cohorts


def evaluation_folder(cohort, seed):
    if cohort not in ('unseen-validation', 'unseen-test'):
        raise ValueError('Unsupported evaluation cohort')
    prefix = 'speech-validation' if cohort == 'unseen-validation' else 'speech-unseen-test'
    return prefix if seed == 2026 else f'{prefix}-seed-{seed}'


def assert_development_disjoint(test, development, manifest):
    by_id = {row['id']: row for row in manifest}
    def used(rows):
        ids = {key for row in rows for key in (row['id'], row.get('ref_id')) if key}
        return [by_id[key] for key in ids]
    a, b = used(test), used(development)
    if {r['speaker_id'] for r in a} & {r['speaker_id'] for r in b}:
        raise ValueError('Test speakers overlap development speakers')
    for key in ('id', 'audio', 'pcm_sha256', 'original_pcm_sha256', 'text'):
        def values(rows):
            return {text_key(r[key]) if key == 'text' else r[key] for r in rows if r.get(key)}
        if values(a) & values(b):
            raise ValueError(f'Test/development leakage: {key}')


def build_test_plan(backup, template):
    backup = Path(backup)
    def read(path):
        return json.loads(path.read_text(encoding='utf-8-sig'))
    def rows(path):
        return [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    def sha(path):
        with path.open('rb') as stream:
            return hashlib.file_digest(stream, 'sha256').hexdigest()
    metadata = read(backup/'cohorts/cohorts.json')
    recovery = read(backup/'recovery-verification.json')['cohorts']
    cohorts = {}
    for name in ('train', 'unseen-test', 'unseen-validation', 'familiar-validation'):
        path = backup/f'cohorts/{name}.jsonl'
        if sha(path) != metadata['files'][name] or sha(path) != recovery[name]['sha256']:
            raise ValueError(f'Changed cohort: {name}')
        cohorts[name] = rows(path)
        if len(cohorts[name]) != metadata['counts'][name]:
            raise ValueError('Cohort row count differs')
    manifest_path = backup/'dataset-v1/manifest.jsonl'
    inventory = read(backup/'dataset-v1/dataset-inventory.json')['files']
    if sha(manifest_path) != inventory['manifest.jsonl']['sha256']:
        raise ValueError('Changed canonical manifest')
    manifest = rows(manifest_path)
    train, test = cohorts['train'], cohorts['unseen-test']
    validate_cohorts(train, test, manifest, heldout_split='test')
    test_speakers = {row['speaker_id'] for row in test}
    if test_speakers != set(metadata['speakers']['test']) or test_speakers & {r['speaker_id'] for r in train}:
        raise ValueError('Reserved test speaker membership differs or overlaps training')
    assert_development_disjoint(test, cohorts['unseen-validation']+cohorts['familiar-validation'], manifest)
    plan = {key: copy.deepcopy(value) for key, value in template.items() if key != 'cases'}
    plan.update(cohort='unseen-test', cohort_sha256=metadata['files']['unseen-test'],
        split_hashes=metadata['files'], unseen_speakers=len(test_speakers),
        manifest_sha256=sha(manifest_path), protocol='reserved-unseen-test-v1')
    fields = ('id', 'text', 'speaker_id', 'audio', 'ref_id', 'ref_audio', 'ref_text')
    plan['cases'] = [dict(case=f'{i:03d}', **{k: row[k] for k in fields}) for i, row in enumerate(test, 1)]
    return plan
