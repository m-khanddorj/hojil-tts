"""Versioned multi-speaker integration contract for a future Mongolian trainer.

This is deliberately not advertised as the stock single-speaker SFT interface.
Unknown identities remain usable corpus evidence, with unresolved conditioning.
"""
from collections import Counter
from pathlib import Path

from .audio import inspect
from .io import digest, fresh_dir, read_rows, write_json, write_rows


def export_adaptation(manifest, output):
    rows = list(read_rows(manifest))
    if not rows:
        raise ValueError('Cannot export an empty adaptation corpus')
    out_rows = []
    reference_owners = {(r['audio'], r['speaker_id'], r['split']) for r in rows}
    for row in rows:
        if row['split'] not in {'train', 'validation', 'test'}:
            raise ValueError('Unexpected canonical split')
        audio = str(Path(row['audio']).resolve())
        if inspect(audio)['pcm_sha256'] != row['pcm_sha256']:
            raise ValueError('Canonical audio changed since preparation')
        reference = row.get('ref_audio')
        speaker = row.get('speaker_id')
        if bool(speaker) != bool(reference):
            raise ValueError('Known speakers require references; unknown speakers must not borrow a voice')
        if reference:
            if (reference, speaker, row['split']) not in reference_owners:
                raise ValueError('Reference must belong to the same speaker and partition')
        out_rows.append(dict(schema_version=1, id=row['id'], audio=audio, text=row['text'],
                             language='mn-MN', speaker_id=speaker, ref_audio=reference,
                             conditioning_status='known-speaker' if speaker else 'unknown-speaker-needs-trainer-policy',
                             split=row['split'], duration=row['duration'], pcm_sha256=row['pcm_sha256'],
                             source=row['source'], source_family=row['source_family'],
                             revision=row['revision'], original_clip_id=row['original_clip_id']))
    out = fresh_dir(output)
    for split in ('train', 'validation', 'test'):
        write_rows(out / f'{split}.jsonl', [r for r in out_rows if r['split'] == split])
    write_json(out / 'contract.json', dict(
        schema='hojil-multispeaker-adaptation-v1', language='mn-MN', sample_rate=24000,
        manifest=str(Path(manifest).resolve()), manifest_sha256=digest(manifest),
        rows=len(out_rows), splits=dict(Counter(r['split'] for r in out_rows)),
        unknown_speaker_rows=sum(r['speaker_id'] is None for r in out_rows),
        references='same known speaker, same partition; null for unknown identity',
        audio_codes='not extracted; explicit optional model stage required',
        trainer_status='integration-required-not-stock-qwen-sft',
        required_trainer_decisions=['Evaluate Auto conditioning and text token coverage; no new language ID assumed',
                                   'Unknown-speaker conditioning without fabricated identities',
                                   'Multi-speaker reference batching and codec integration',
                                   'LoRA/parameter-efficient versus full tuning based on experiments'],
    ))
    return dict(rows=len(out_rows), unknown_speaker_rows=sum(r['speaker_id'] is None for r in out_rows))
