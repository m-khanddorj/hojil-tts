"""Joint pronunciation/voice preference filtering for training-only rollouts.

Do not turn the held-out evaluation panel into preference-training examples.
Only model-generated versus model-generated pairs are supported here.
"""
import math


def joint_pairs(candidates, min_cosine_margin=0.001):
    if len(candidates) < 2:
        return []
    if not math.isfinite(min_cosine_margin) or min_cosine_margin <= 0:
        raise ValueError('A positive voice-only preference margin is required')
    keys = ('text', 'reference_sha256', 'speaker_id', 'case')
    first = candidates[0]
    if len({r['sample_id'] for r in candidates}) != len(candidates):
        raise ValueError('Duplicate candidate samples')
    for row in candidates:
        if row['partition'] != 'train' or row['origin'] != 'model_generated':
            raise ValueError('Only generated training rollouts may form preference pairs')
        if any(row[key] != first[key] for key in keys):
            raise ValueError('Preference conditions differ')
        score = row['score']
        if (score['characters'] <= 0 or score['character_errors'] < 0
                or score['characters'] != first['score']['characters']
                or not math.isfinite(row['cosine']) or not -1 <= row['cosine'] <= 1):
            raise ValueError('Invalid candidate metrics')
        # Same text and normalization imply identical character denominators.
        if not math.isclose(score['cer'], score['character_errors']/score['characters'], abs_tol=1e-12):
            raise ValueError('CER disagrees with error counts')
    pairs = []
    for chosen in candidates:
        for rejected in candidates:
            fewer = rejected['score']['character_errors'] - chosen['score']['character_errors']
            voice_gain = chosen['cosine'] - rejected['cosine']
            if fewer < 0 or voice_gain < 0:
                continue
            # One corrected character is substantive pronunciation evidence;
            # voice-only pairs need a larger margin than floating-point jitter.
            if fewer == 0 and voice_gain < min_cosine_margin:
                continue
            pairs.append(dict(case=first['case'], speaker_id=first['speaker_id'],
                chosen=chosen['sample_id'], rejected=rejected['sample_id'],
                character_error_reduction=fewer,
                cer_reduction=fewer/first['score']['characters'], cosine_gain=voice_gain))
    return pairs
