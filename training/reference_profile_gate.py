"""Matched speech checks for reference personalization, with split isolation."""
from .targeted_voice import aggregate


def compare(before,after,settings,group):
    count={'profile-fit':16,'profile-validation':8}[group]
    old={(r['case'],r['seed']):r for r in before}
    new={(r['case'],r['seed']):r for r in after}
    if len(before)!=count or len(after)!=count or len(old)!=count or set(old)!=set(new):
        raise ValueError('Incomplete matched profile partition')
    if any(r['group']!=group for r in before+after):
        raise ValueError('Profile partition contamination')
    for key,r in new.items():
        for field in ('case','seed','speaker_id','text','reference','reference_sha256','reference_text'):
            if r[field]!=old[key][field]:raise ValueError('Profile condition changed')
        if any(r['score'][k]!=old[key]['score'][k] for k in ('characters','words')):
            raise ValueError('Score denominators changed')
    baseline=aggregate(before)[group];candidate=aggregate(after)[group]
    worst=min(new[k]['cosine']-old[k]['cosine'] for k in old)
    added=max(new[k]['score']['character_errors']-old[k]['score']['character_errors'] for k in old)
    checks=dict(all_terminated=all(r['terminated'] for r in after),
        pooled_cer_preserved=candidate['pooled_cer']<=baseline['pooled_cer'],
        mean_voice_improved=candidate['speaker_macro_cosine']>baseline['speaker_macro_cosine'],
        worst_clip_voice_preserved=worst>=-settings['maximum_clip_cosine_drop'],
        no_large_new_pronunciation_error=added<=settings['maximum_added_character_errors_per_clip'])
    return dict(passed=all(checks.values()),checks=checks,before=baseline,after=candidate,partition=group,
        worst_clip_cosine_delta=worst,maximum_added_character_errors=added,
        interpretation='Training-only selection' if group=='profile-fit' else
            'Reserved-text generalization check after fitting selection; not used to choose a scale')
