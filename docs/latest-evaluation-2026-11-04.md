# Alternate 2 ASR evaluation — November 4, 2026

The current full-sentence configuration scored **2.08% CER and 5.83% WER** as
mean per-recording BuzzASR error rates without punctuation, on 20 development
sentences. This measures recognizer agreement with text, not human error rates.

## Primary measurement

These are the original 20 full-sentence recordings, all at seed 2026, using the
final step-5253 model and Alternate 2 reference with its matching transcript.
Temperature 0.9, top_p 1, top_k 50, max_new_tokens 320; talker and subtalker
sampling enabled; audio-plus-transcript conditioning; nonstreaming; batch 1.
Audio was generated on GPU FP32 with TF32 disabled. No speech was regenerated
for this scoring pass, and no retry or best-of-N selection enters the primary score.

| Aggregation | CER | WER |
|---|---:|---:|
| Mean per recording, punctuation removed | **2.08%** | **5.83%** |
| Pooled edit counts, punctuation removed | **2.23%** | **6.40%** |
| Mean per recording, punctuation retained | 3.75% | 17.50% |
| Pooled edit counts, punctuation retained | 3.87% | 17.73% |

Normalization is Unicode NFC, case-folding and whitespace collapse. The
punctuation-free variant removes Unicode punctuation categories. Character
distance includes the remaining spaces; word distance splits on whitespace.
Mean rates weight each recording equally; pooled rates divide total edits by
total reference characters or words.

## Recognizer and comparability

Pinned `BuzzASR/mongolian` revision
`185f1bf1f562ad4b1bdbfc71811fb165a9235416`, all local model-file SHA256 hashes
verified against the earlier evaluation manifest. Scoring used CPU FP32,
PyTorch 2.8.0+cpu, four threads, 16 kHz resampling, greedy decoding,
no-repeat-ngram size 3 and repetition penalty 1.2, matching the prior evaluator.
All 28 scored WAV hashes matched their generation records. Three fixed earlier
GPU-scored clips reproduced their exact transcripts on CPU. This small parity
check supports comparability; it is not proof of parity on every possible clip.

Existing GPU pods lacked host capacity, so this pass used local CPU and incurred
no GPU compute charge. The demo server stayed stopped.

The older 5.62% CER / 11.12% WER combined multiple conditioning modes across 44
targets. It is a historical measurement, not a matched comparison with these
20 sentences and this single reference. The current result is below 3% on this
development sample; it does not establish a population CER below 3% or 1%.
Reserved test data remain unused. Possible ASR training-data overlap and
repeated development selection still limit conclusions.

## Selected-failure retries — separate from the primary score

| Seed | Selected recordings | Mean CER without punctuation | Mean WER without punctuation |
|---|---:|---:|---:|
| 2028 | 4 | 4.14% | 12.91% |
| 2029 | 4 | 5.00% | 15.47% |

All eight retries were rated correct by the listener. These ASR differences
are not confirmed speech errors. The four texts were selected because the
originals failed listening review; their scores cannot be compared directly
with the entire 20-sentence mean to rank seeds. They do not establish a new
default seed or justify automatic lowest-CER selection.

## Reproduction and evidence

`scripts/score_latest_audio.py` reads the saved experiment layout, checks model
and audio hashes, performs the parity check, and saves transcripts, edit counts
and both aggregations. It requires the same external artifacts and the pinned
BuzzASR snapshot in the sibling `buzzasr-model` directory; it does not download
audio or models, and no private recordings are included in Git.

```powershell
python scripts/score_latest_audio.py --full-training C:/external/experiments/full-training --output C:/external/experiments/full-training/latest-asr
```

Local evidence is in `full-training/latest-asr`: `plan.json`, `run.json`,
`parity.json`, `transcriptions.jsonl` and `summary.json`. The primary sample is
`alternate2-validation/completed`; selected retries are `seed-recheck/completed`.
Human findings are documented in [the listening report](listening-findings-2026-11-04.md).
