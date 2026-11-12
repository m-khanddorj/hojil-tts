"""Predeclared actual-speech fitting gate; never consumes evaluation speakers."""
from .targeted_voice import aggregate


def fitting_gate(before,after,settings):
    def keyed(records):
        result={(r['case'],r['seed']):r for r in records}
        if len(result)!=len(records):raise ValueError('Duplicate fitting condition')
        return result
    old,new=keyed(before),keyed(after)
    if len(old)!=16 or set(old)!=set(new):raise ValueError('Incomplete matched fitting panel')
    for key,r in new.items():
        for field in ('case','seed','speaker_id','group','text','reference','reference_sha256','reference_text'):
            if r[field]!=old[key][field]:raise ValueError('Fitting condition changed')
        if r['group']!='fitting':raise ValueError('No evaluation-condition fitting')
        if (r['score']['characters']!=old[key]['score']['characters']
                or r['score']['words']!=old[key]['score']['words']):raise ValueError('Score denominators differ')
    baseline=aggregate(before)['fitting'];candidate=aggregate(after)['fitting']
    worst=min(new[k]['cosine']-old[k]['cosine'] for k in old)
    added=max(new[k]['score']['character_errors']-old[k]['score']['character_errors'] for k in old)
    checks=dict(all_terminated=all(r['terminated'] for r in after),
        pooled_cer_preserved=candidate['pooled_cer']<=baseline['pooled_cer'],
        mean_voice_improved=candidate['speaker_macro_cosine']>baseline['speaker_macro_cosine'],
        worst_clip_voice_preserved=worst>=-settings['maximum_clip_cosine_drop'],
        no_large_new_pronunciation_error=added<=settings['maximum_added_character_errors_per_clip'])
    return dict(passed=all(checks.values()),checks=checks,before=baseline,after=candidate,
        worst_clip_cosine_delta=worst,maximum_added_character_errors=added,
        interpretation='Training-only candidate selection, not evidence of generalization or goal completion')
