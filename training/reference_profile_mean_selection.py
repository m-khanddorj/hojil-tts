"""Prospective alternative selection aligned with the group-mean objective.

Used only in a separate experiment after the original four-trial run ends.
Per-clip damage remains reported; it is not an additional hard requirement.
No reserved or screening observation may enter this selector.
"""
import math
from .reference_profile_gate import compare


def select(baseline,trials,settings,expected_indices):
    if [r['index'] for r in trials]!=expected_indices or len(set(expected_indices))!=len(expected_indices):
        raise ValueError('All registered fitting trials in order are required')
    results=[]
    for trial in trials:
        audit=trial['audit']
        if (audit['status']!='verified' or audit['recordings']!=16 or audit['candidate_index']!=trial['index']
                or not audit['all_weights_unchanged'] or audit['maximum_cosine_recomputation_error']>2e-6):
            raise ValueError('Complete independent fitting audit required')
        gate=compare(baseline,trial['records'],settings,'profile-fit')
        after=gate['after']
        if audit['summary']!={'profile-fit':after}:raise ValueError('Audited fitting summary differs')
        eligible=(gate['checks']['all_terminated'] and gate['checks']['pooled_cer_preserved']
            and gate['checks']['mean_voice_improved'] and math.isfinite(after['pooled_cer'])
            and math.isfinite(after['speaker_macro_cosine']))
        results.append(dict(index=trial['index'],eligible=eligible,gate=gate))
    eligible=[r for r in results if r['eligible']]
    under_cer=[r for r in eligible if r['gate']['after']['pooled_cer']<.03]
    if under_cer:
        chosen=min(under_cer,key=lambda r:(-r['gate']['after']['speaker_macro_cosine'],
            r['gate']['after']['pooled_cer'],r['index']))
    elif eligible:
        chosen=min(eligible,key=lambda r:(r['gate']['after']['pooled_cer'],
            -r['gate']['after']['speaker_macro_cosine'],r['index']))
    else:chosen=None
    return dict(selected=None if chosen is None else chosen['index'],trials=results,
        basis='Fitting-only: require nonincreasing CER and increasing mean cosine; among CER<3% choose highest cosine, otherwise lowest CER',
        per_clip_checks_retained_as_diagnostics=True,validation_used=False,goal_achieved=False)
