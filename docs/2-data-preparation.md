# Data preparation

The local preparation of Common Voice 27 (CV27), Mongolian FLEURS, and MBSpeech
finished with **38,576 accepted clips, totaling 50.53 hours**. The latest ten
original/processed listening pairs were accepted by the user. This is a prepared
corpus with documented limitations, not a trained model or a guarantee that every
clip is suitable. No training was run in this change.

This document records the actual run through October 24, 2026. The
[source review](1-open-source-data-review.md) preserves the earlier research
snapshot; the [pipeline reference](data-pipeline.md) describes commands and
configuration. The [cleaning research](data-cleaning-research.md) and
[initial pilot](data-pilot-2026-10-22.md) explain the experimental background.

## Original inputs

| Source | Selected input rows | Original metadata hours | Original audio |
| --- | ---: | ---: | --- |
| CV27 Mongolian | 33,234 | 46.832 | MP3, mixed native sample rates |
| FLEURS Mongolian | 4,428 | 15.553 | Mono 16 kHz WAV, including IEEE float32 |
| MBSpeech | 3,846 | 6.629 | Mono 16 kHz WAV stored in two Parquet shards |
| Total | 41,508 | 69.013 | Paired recordings and supplied transcripts |

Original hours use CV's `clip_durations.tsv`, FLEURS' sample counts divided by
16,000, and MBSpeech's duration column. They include silence and reflect source
metadata rather than final decoded/trimmed durations. They are not the publisher
estimates in the earlier catalog, nor an estimate of unique speech content.

CV's archive contains 95,940 clips. The selected 33,234 are unique validated clips
plus official development/test IDs; unvalidated and invalidated material was not
queued. Those unselected recordings are outside the rejection counts below.
FLEURS includes all 3,074 train, 405 development, and 949 test rows. MBSpeech
includes both 1,923-row shards. Unknown speakers remain null; an assumed narrator
identity is not invented.

Source pins:

- CV27: user-supplied `1789493002208-cv-corpus-27.0-2026-09-11-mn.tar.gz`, SHA-256 `05c5a447be5136f08cfd7eb9b3c0c7edb17790c01dd5c627b02bb92bbeeda364`.
- FLEURS: `google/fleurs`, revision `70bb2e84b976b7e960aa89f1c648e09c59f894dd`.
- MBSpeech: `bilguun/mbspeech`, revision `5cdcc286f77419e3d0e3b60cb1852c99aaa91013`. The originally configured `btsee/mbspeech_mn` endpoint returned 401; the public original-family distribution was used, not a cleaned repackaging.

CV records CC0-1.0 with the source's access terms; FLEURS records CC-BY-4.0.
MBSpeech's reviewed card did not establish an explicit license. The user approved
full local preparation of all three sources. That approval is recorded as an
operational reference; it does not independently establish recording or text
rights, or permission to redistribute MBSpeech. The default example leaves that
source disabled until an explicit reference is supplied.

## Preparation strategy

The objective was to retain usable speech with its original transcript, reduce
background noise conservatively, and remove safe edge silence. Source files stay
immutable. The pipeline records source revision, original file hash, decoded
original PCM hash, final PCM hash, settings, model/runtime hashes, measurements,
and rejection or review reasons.

Text normalization applies Unicode NFC and whitespace normalization only.
It preserves case, punctuation, digits, spelling, and Mongolian letters such as
Ө and Ү. No automatic spelling correction, transcript replacement, or number
expansion was applied.

Audio becomes mono PCM16 WAV at 24 kHz. Resampling 16 kHz sources does not restore
missing high-frequency information. Among retained CV clips, native rates are
32 kHz (23,258), 48 kHz (9,076), and 44.1 kHz (24). Retained FLEURS and MBSpeech
clips originate at 16 kHz.

### Gated enhancement and speech-aware edge trimming

1. Measure the original decoded signal, including clipping, and score speech,
   background, and overall quality using DNSMOS P.835 predictions.
2. Skip denoising when the predicted background score is at least 4. Otherwise
   try DeepFilterNet3 with a 6 dB attenuation cap, compensated model delay, and no
   strong postfilter. It works at 48 kHz before returning to 24 kHz.
3. Keep the candidate only when predicted overall and background scores each
   improve by at least 0.05, speech score drops by no more than 0.1, and VAD speech
   duration loses no more than 10%. Otherwise use the pre-enhancement waveform.
   A denoiser duration discrepancy above 50 ms fails processing.
4. Use Silero VAD at 16 kHz for the first/last speech boundaries: threshold 0.5,
   minimum speech 250 ms, minimum quiet interval 500 ms, and 250 ms padding.
   Preserve everything between those boundaries, including internal pauses.
5. Guard against unsafe edge cuts: if a proposed trim removes more than 40% of
   the duration and more than 10% of signal energy, retain the entire selected
   waveform and flag `unsafe_edge_trim_reverted`. Reapply the normal hard filters.

UVR was not used. The code also supports FFmpeg `afftdn` and an energy-based edge
trimmer. The portable example uses the FFT backend; the actual full run used
DeepFilterNet3 with local, hash-pinned assets. Backend choice is explicit.

In the initial 60-clip CV/FLEURS comparison, all three profiles accepted 50 clips.
DeepFilterNet retained 16 enhancement candidates versus 10 for FFT, with mean
predicted overall-score gains of 0.110 and 0.019 respectively. These are model
predictions on a small diagnostic sample, not measured human MOS or evidence of
training improvement. See the pilot for its separate MBSpeech diagnostic.

The actual models were Silero VAD v6.2.3 (revision
`5cd7945676eb32225748052e2e6a0580e4686a08`), DNSMOS revision
`591184a9fcb2cbdec02520fed81a32bbbf9d73ff`, and DeepFilterNet3 model revision
`978576aa8400552a4ce9730838c635aa30db5e61` with native executable v0.5.6.
Their acquisition records and file hashes are retained with the external run;
no weights or executable binaries are committed.

### Hard filters and advisory flags

| Check | Actual threshold/action |
| --- | --- |
| Final clip duration | Keep 1–20 seconds |
| RMS level | At least -45 dBFS |
| Clipped samples | At most 1%, checked before and after enhancement |
| Sample-level silence | At most 80% |
| Transcript length | At most 500 characters |
| Cyrillic letter fraction | At least 60% of letters |
| Missing speech, empty/invalid text | Reject with a recorded reason |
| Predicted speech or overall score below 3 | Advisory review flag |
| More than 40% edge trimming | Advisory review flag |
| Enhancement loses too much VAD speech | Revert enhancement and flag |

DNSMOS scores are not calibrated measurements of Mongolian intelligibility. A
low prediction alone does not reject a clip. Flags identify further listening
work; an unflagged clip is not a certification either.

## What was corrected during the full run

FLEURS TSV parsing was changed to preserve literal quotation marks rather than
treating them as CSV quoting. The earlier parser merged 347 rows; the full input
inventory now contains all 4,428 rows. Source sample counts match the native WAV
clocks, including float32 files.

The first finalization retained only 8,185 clips (15.25 hours). The main loss was
**split policy**, not a finding that all other audio was poor. Connecting shared
sentences, exact audio, sessions, and speakers while preserving official holdouts
formed large connected groups; 30,480 rows were excluded as `heldout_leakage`.

The corpus was rebuilt with `speaker_disjoint: false` and
`preserve_source_splits: false`, seed 2026, and target validation/test fractions of
5% each. Hash-based connected groups still keep normalized sentences, original
PCM, final PCM, and known sessions together. Known speakers may occur in multiple
partitions. The original `source_split` remains provenance, but official test
recordings may now enter training. These splits are **not** official benchmark
or unseen-speaker evaluation splits; do not reuse the original official test set
as an independent evaluation set for this corpus.

This recovered 38,665 clips (50.74 hours) before the later listening corrections.
Some old split-excluded generated WAVs had been removed. Recovery replayed only
the saved enhancement selection and exact trim offsets from immutable originals,
and required an exact match to the saved PCM hash. It did not rerun VAD or quality
inference. New finalization retains split/deduplication-excluded WAVs to permit
future policy revisions.

### Slowed FLEURS recording and trim repair

The first ten listening pairs received nine approvals and one rejection. For
`fleurs:train/4317282147642166313.wav`, the user heard slow speech in both original
and processed audio; processing retained only part of the utterance. The source
was 21 seconds, while the old processed clip was only 2.772 seconds. A healthy
same-sentence reference was 10.98 seconds; the suspect's active span was 2.36
times the reference's active span. The VAD boundary was therefore insufficient
protection for this source anomaly.

All 4,428 FLEURS source WAVs were screened. The screening combines severe low
bandwidth (99% rolloff at or below 2 kHz, 95% at or below 1 kHz, at least 99.5%
energy below 4 kHz) with low original VAD coverage or predicted speech quality.
Where a healthy same-sentence reference exists, the suspect must also have an
active span at least 1.6 times the median reference span. Spectral features use
1,024-sample Hann windows, 512-sample hops, and the louder half of DC-centered
frames; active spans use a -20 dB relative energy threshold.

The revision quarantined **71 recordings**: the one human-confirmed rejection,
15 additional suspected slowed sources with reference evidence, and 55 severe
unresolved anomalies without sufficient reference evidence. These are not 71
confirmed speed defects. Naturally narrow-band recordings can resemble this
pattern, and the uncertain exclusions are reversible. Audio was not sped up and
transcripts were not shortened to match partial speech.

The revision also checked prior large trims and added the energy guard described
above. Of 34 unsafe trims reverted to full selected waveforms, 16 survived the
hard filters; 16 exceeded the duration limit and two failed silence checks.
The nine previously approved processed clips were protected by their exact PCM
hashes. Existing split assignments were retained and references rebuilt within
their partitions.

The second review supplied ten fresh pairs, including two repaired trims. The
user accepted all ten. Across both rounds this is 19 distinct accepted examples
and one rejected example. Selection was diagnostic and stratified, not random
enough to estimate the whole corpus's error rate.

## Final results

| Source | Input clips | Accepted clips | Accepted hours | Rejected clips |
| --- | ---: | ---: | ---: | ---: |
| CV27 | 33,234 | 32,358 | 37.844 | 876 |
| FLEURS | 4,428 | 2,386 | 6.132 | 2,042 |
| MBSpeech | 3,846 | 3,832 | 6.557 | 14 |
| Total | 41,508 | **38,576** | **50.533** | **2,932** |

| Partition | Clips | Hours |
| --- | ---: | ---: |
| Train | 34,744 | 45.526 |
| Validation | 1,985 | 2.644 |
| Test | 1,847 | 2.363 |

Rejection reasons are mutually exclusive final reasons: silence 1,321; quiet
1,291; duration 142; no speech detected 81; unresolved source slowdown/anomaly 55;
text too long 16; suspected slowed source 15; control character 7; no letters 2;
insufficient Cyrillic 1; human rejection 1. These sum to 2,932. Trimming also
removed 594.25 minutes of edge audio from retained clips; the change in hours
therefore includes both exclusions and trimming, not just rejected recordings.

The final split audit found zero cross-partition overlaps in normalized text,
original PCM hashes, or cleaned PCM hashes. There are 245 known speaker keys
shared across partitions, intentionally permitted by the revised policy, and
6,218 rows with unknown speaker identity.

| Advisory flag among accepted clips | Clips | Minutes |
| --- | ---: | ---: |
| Low predicted quality | 21,268 | 1,547.21 |
| Large edge trim | 1,091 | 81.73 |
| Enhancement candidate lost VAD speech; reverted | 165 | 12.88 |
| Unsafe edge trim; reverted | 16 | 2.83 |
| Any flag, unique clips | **21,585** | **1,565.03** |
| No flags | 16,991 | 1,466.97 |

Flag categories overlap; their rows cannot be added together. About 26.08 hours
have at least one flag. These remain accepted under the agreed policy and are
an outstanding listening-review workload, not confirmed defective audio.

Among retained clips, enhancement was kept for 10,059, reverted by the quality
gate for 16,821, reverted for speech loss for 165, and skipped for predicted-clean
backgrounds for 11,531. These add to 38,576.

## Artifacts and reproduction

Preparation code is directly under [`data/`](../data/README.md), with no nested
`hojil_data` package. From the repository root, `python -m data --help` lists the
main commands. Install `.[audio,parquet,neural]` for codec, Parquet, and local
neural processing dependencies. The [pipeline reference](data-pipeline.md)
documents external configurations, model acquisition records, and commands:

```bash
python -m data prepare --config /external/run/config.json --output /external/run/cleaned --workers 6
python -m data.repartition --prepared /external/run/cleaned --output /external/run/rebalanced --workers 6
python -m data.quality_revision --prepared /external/run/rebalanced --output /external/run/reviewed --review /external/run/human-review.json --workers 6
```

Use fresh output directories. Parallel preparation records checkpoints and
supports `--resume` with input/configuration/implementation identity checks.
Recorded checkpoints from an older implementation should be used with that
implementation; moving the package changes implementation identity. Repartition
and quality revision preserve provenance and verify source/output hashes.

Local storage root: `C:\Users\mkhan\Documents\data\tts`.
The initial run is `full-cleaning-2026-10-22`, its split revision is
`rebalanced-2026-10-23`, and the **current final run is
`quality-reviewed-2026-10-24`**. The final manifest SHA-256 is
`cba8ae7dc77c22e6b05d718fb93b54c4ad20db83be1fb4f29c1778ff1efa853c`.
Originals, source/model acquisition records, reports, rejection/review manifests,
source-screening evidence, and adaptation export remain outside Git.

The current revision references unchanged WAVs in its parent rebalanced run and
stores corrected WAVs in its own `audio/` directory. Copying only the revision
folder is insufficient. Preserve both runs and resolve all manifest audio and
reference paths when making a portable bundle. The JSONL currently contains
Windows absolute paths; it is not directly portable to RunPod. A future transfer
should materialize the referenced assets and rewrite/verify those paths. No
cloud compute, dataset upload, or training was performed here.

The ignored worktree folder `local-audio-example/` contains the generated HTML
report (`preparation-report.html`) and latest listening page
(`review-10-051470c0/index.html`) for local review. These are machine-local
artifacts and are deliberately absent from the PR. This document carries the
reviewable strategy and results in Git.

## Validation and remaining limits

The test suite covers synthetic audio, real codec/Parquet fixtures when optional
dependencies are present, provenance and hash checks, leakage prevention,
checkpoint recovery, quality-gate decisions, trim repair, source-anomaly
classification, and exports. CI checks the core and optional-dependency profiles
on Python 3.11 and 3.12 without acquiring datasets or model weights. Real-corpus
results above are separate local execution evidence, not reproduced in CI.

This preparation does not establish full transcript/audio alignment, detect all
overlapping speakers/music, perform perceptual near-duplicate matching, or certify
natural tempo. An offline nine-sample ASR diagnostic produced unreliable
repetition, non-Cyrillic output, or invalid timestamps, so its hypotheses were
not used to rewrite or reject supplied transcripts. Model/runtime causes were
not isolated. The source-slowdown screen is a conservative, dataset-specific
quarantine heuristic rather than a universal speed detector. Further native
listening, especially for flagged clips, and rights verification remain relevant
before distribution or downstream use. Actual training is outside this PR.
