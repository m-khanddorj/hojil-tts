# Listening findings — November 4, 2026

## Working default

The user selected Alternate2 reference audio with its matching transcript, full-sentence
generation, and nonstreaming output. The verified profile is stored outside Git as
full-training/inference-defaults.json in the local experiment artifacts.
The tested settings are temperature0.9, top_p1.0, top_k50, max_new_tokens320,
talker and subtalker sampling enabled, FP32, TF32 disabled, batch size1.
A seed records a generation attempt; none is established as universally best.

## What the listener established

An initial audit of18selected difficult clips found5correct,9minor mistakes and4major
mistakes. Some ASR scores disagreed sharply with listening: one clip rated correct had
37.63% recognizer CER. ASR differences alone must not be treated as confirmed speech errors.

A targeted reference/chunking comparison covered5failure cases but only2unique texts.
Alternate2 with chunks was rated correct in5/5cases; Alternate2 full sentences in4/5.
This was candidate selection, not a general accuracy benchmark.

The subsequent20distinct development sentences gave these results:

| Outcome | Full sentence | Manual phrase chunks |
|---|---:|---:|
| Correct, after word-level review |16/20|16/20|
| Minor mistakes |4/20|3/20|
| Major mistakes |0/20|1/20|
| Natural |14/20|2/20|
| Correct and natural |13/20|2/20|

Sentence19 was initially rated minor mistakes, then confirmed all words correct on replay.
The original export remains preserved. Chunking used manually chosen boundaries and180ms
join silence without trimming or loudness normalization. The result argues against this
within-sentence chunking policy as a default; it does not rule out every chunking method.

Remaining full-sentence errors were:
- Sentence3: initial “Хоёр” sounded “хххоёр”; “хоёр жилийг” became “Хожлыг”, with “жилийг” missing.
- Sentence7: “говийн” sounded “Говын”.
- Sentence11: “охиндоо” sounded “Охондоо”.
- Sentence15: “уулаар” sounded “улаар”.

Repeating those4sentences with seeds2028and2029 produced8clips,all rated correct.
Nine explicit word checks were marked Fixed; three individual checks for sentence3seed2029
were left blank although that whole clip was rated Correct.
Seed2026replays exactly matched all4original audio hashes, supporting comparability.
The same model can produce correct versions of these words; these observed failures
are seed-sensitive. They do not yet justify retraining as the first intervention.

## Practical decision and limits

Use Alternate2/full sentences. Preserve versions and offer retry with a recorded new seed.
Do not automatically pick the lowest ASR score without validating that selection rule
against listening. Do not promote seed2028or2029 as globally superior from selected failures.

These findings come from one listener, development data, limited seeds and targeted selection.
The listening ratings alone do not establish population accuracy or a CER. A later
[ASR measurement](latest-evaluation-2026-11-04.md) of the 20 original full sentences
gave 2.08% mean CER and 5.83% mean WER without punctuation; this does not establish
production readiness. Naturalness
was not rescored on the seed retries. CPU inference may differ from the evaluated GPU runs.

## Evidence location

All audio, exact text, case IDs, hashes, plans and raw feedback remain outside Git under:
C:/Users/mkhan/.codex/visualizations/2026/10/26/01a176bb-11c3-7c12-82a3-14bc905add93/full-training/

Relevant folders: cer-sweep, reference-chunk-comparison, alternate2-validation, seed-recheck.
Each contains its saved reports and human feedback JSON. The private Hugging Face
[model card](https://huggingface.co/khanddorj/hojil-tts) was updated with these findings
on November 4, 2026, at revision `eb4a774f345d8bccf3fd8edf7dc3c738c8c0007a`.
The published card was downloaded and verified; repository privacy and all non-card
files are unchanged. The local publication receipt is
`full-training/huggingface-private/listening-card-update.json` in the artifact directory.
