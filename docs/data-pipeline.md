# Khalkha language-adaptation data pipeline

The primary deliverable is a **multi-speaker canonical corpus**, not a claim that
Qwen3-TTS already supports Mongolian. The optional official single-speaker export
is for later voice refinement. Adding a new language still needs trainer/model
integration and native-listener evaluation. See the
[source review](1-open-source-data-review.md) and
[actual preparation strategy and results](2-data-preparation.md).

## Requirements and architecture

Run from the repository with Python 3.11+; the core CLI uses only the
standard library. Optional tests skip when their dependencies are absent.
Modules live directly under `data/`. No installation is necessary for `python3 -m data`.
Optional packaging: `python3 -m pip install -e .` enables `hojil-data`.

- `sources.py`: explicit acquisition, CV27/FLEURS adapters, quarantined MBSpeech
  local manifests and optional Parquet import.
- `audio.py`: PCM analysis; ffmpeg/ffprobe conversion to mono, PCM16, 24 kHz WAV.
- `preprocessing.py` and `neural.py`: gated FFT/DeepFilterNet enhancement,
  DNSMOS scoring, VAD/energy edge trimming, and unsafe-trim guards.
- `pipeline.py`: conservative text/audio validation, leakage groups, splits,
  references, canonical JSONL, rejection and duration reports.
- `adaptation.py`: primary versioned multi-speaker export with explicit integration contract.
- `qwen.py`: optional official single-speaker export and explicit offline code extraction.
- `io.py`: artifact hashes, safe extraction, credential-free public downloads.
- `bulk.py`: parallel processing with resumable, identity-checked checkpoints.
- `repartition.py`: split-policy revisions and verified recovery of cleaned PCM.
- `source_audit.py` and `quality_revision.py`: FLEURS anomaly quarantine and
  review-driven trim repairs without changing existing partition assignments.
- `transcripts.py`: optional offline ASR audit, without replacing source text.

For real datasets, either provide ffmpeg/ffprobe on PATH or install the optional
project-local audio dependencies in an isolated virtual environment:

```bash
python3 -m venv /external/work/venv
/external/work/venv/bin/python -m pip install -e '.[audio,parquet]'
```

The `audio` extra uses the ffmpeg binary bundled in `imageio-ffmpeg` and SoundFile
for native metadata when ffprobe is absent. No system-wide executable installation
is necessary. CV MP3 and FLEURS 16 kHz WAV conversion use these explicit dependencies;
the CLI itself never installs them. The 24 kHz PCM fixture suite works without them.
The optional local MBSpeech Parquet importer requires `pip install -e '.[parquet]'`.
PyArrow is not imported by the core commands.

All audio, manifests, acquisition logs, reports, and model artifacts must live
**outside any Git checkout**. The CLI refuses output paths inside Git. Source
files are never edited. Existing output directories are never overwritten.
Use a fresh preparation run directory; failed preparation can leave partial artifacts
for inspection. Acquisition uses temporary extraction and atomic publication, so
a failed extraction never publishes a partial source tree.
No command launches training or purchases compute. `prepare` has no network access.

## 1. Configure sources

Copy [configs/example.json](../configs/example.json) to an external work directory.
Replace absolute paths. Relative input paths, if used, resolve from the current
working directory; absolute paths are recommended. Keep this edited config with
the resulting run. The example enables CV27 and FLEURS, and disables MBSpeech
pending clearance. Three planned source families do **not** imply three available,
validated or licensed training corpora.

### Common Voice v27

Obtain the official Mongolian archive yourself through Mozilla Data Collective,
using your authorized access and reviewing its terms. This tool neither accepts
terms nor bypasses access gates. Set `archive`, `archive_sha256` (from
`shasum -a 256 /path/to/archive.tar.gz`), and the eventual extracted locale `path`.
The archive digest pins its bytes; `revision` must be `27.0`.

```bash
python3 -m data acquire --config /external/work/config.json \
  --source cv27 --output /external/work/raw
```

Extraction creates `/external/work/raw/cv27/`; set `path` to the directory containing
`validated.tsv`, `dev.tsv`, `test.tsv`, and `clips/` (usually below the release root).
Only validated clips plus official dev/test clips are considered. `other.tsv` and
`invalidated.tsv` are not ingested. Dev/test reservations take precedence over
rows repeated in `validated.tsv`. Original `client_id` is retained as a
source-qualified pseudonymous speaker ID; it is not used to identify real people.

If an authorized archive is available at a public HTTPS URL, `download_url`,
`archive_sha256`, and `authorized_download: true` can replace `archive`. This mode
supports no tokens, cookies, auth headers, credentials, URL query strings, or term
acceptance. Gated/signed access must be completed outside the tool and supplied
as a local archive. Preserve Mozilla access terms separately from its license tag.

### FLEURS `mn_mn`

The example pins the verified Google repository revision
`70bb2e84b976b7e960aa89f1c648e09c59f894dd`. Acquisition downloads **only** Mongolian
train/dev/test TSVs and their three audio archives, not all languages:

```bash
python3 -m data acquire --config /external/work/config.json \
  --source fleurs --output /external/work/raw
```

Set `path` to `/external/work/raw/fleurs`. Acquisition logs downloaded SHA-256
hashes. Completed downloads are cached in `.acquisition-cache/<source-id>/` with
an atomic journal; re-running an interrupted acquisition rehashes and reuses those
files. Changed configuration or corrupted cached content fails closed. Transfers
interrupted mid-file restart that file (there is no HTTP byte-range resume).
Content-Length, when supplied, is checked, and provided archive checksums are
verified before extraction. A crash after download completion but before journaling
can leave an unjournaled cached file: inspect/remove that file before retrying;
it is never silently trusted. Completed source destinations are not overwritten. Revisions must be full Git SHAs. The seven-column, headerless TSV adapter
uses the original punctuated transcript (column 3), retains normalized transcript,
sentence ID and gender as metadata, and preserves official partitions. Gender
and sentence ID are **not speaker IDs**. Speaker identity and reference audio are
null because the source does not supply reliable speaker IDs.

### MBSpeech: explicit clearance and local interchange

MBSpeech is disabled by default. `rights_clearance` must contain a meaningful
reference to evidence covering the recording and text before acquisition/import
is allowed. This field records the operator's clearance; the tool cannot establish
legal permission. A license label on a redistribution is insufficient.

The catalog's `btsee/mbspeech_mn` metadata endpoint returned HTTP 401 during this
implementation. Its current remote layout/revision was therefore not verified.
No attempt was made to bypass access. Supply a full verified source commit SHA
and a checksummed authorized archive, or locally acquired Parquet files. Do not
substitute the similarly named cleaned `mbspeech-mn` without updating provenance.

For local Parquet distributions with Hugging Face `audio.bytes` and
`original_sentence`, `sentence_orig` (the `bilguun/mbspeech` schema), `sentence`, or `text` columns:

```bash
python3 -m data import-mbspeech --config /external/work/config.json \
  --source mbspeech --output /external/work/raw/mbspeech \
  /external/downloads/part-00000.parquet
```

This writes embedded audio bytes without re-encoding and a `metadata.jsonl` with
Parquet file hashes and row numbers. A `.wav` filename is not proof of codec;
ffmpeg detects and converts the contents if needed. Split and speaker fields are
preserved if present. Unknown speakers remain unknown. This optional adapter is
tested with real locally generated PyArrow Parquet plus embedded MP3 bytes, as well
as mock batches; no real MBSpeech download was made.

Alternatively provide a local `metadata.jsonl` with one record per utterance:

```json
{"id":"original-clip-id","audio":"audio/clip.wav","text":"Сайн байна уу?","split":"unspecified","speaker_id":null,"session_id":"reviewed-session-if-known"}
```

Paths must stay inside the source directory. Supported splits are `train`,
`validation`, `test`, `unspecified`, and `withheld` (always excluded). A reviewed
single-narrator identity can be added explicitly; the code does not invent one.
Set `enabled: true` only when ready. Without `rights_clearance`, preparation
records `rights_quarantine` rejections without decoding audio. Archive extraction
rejects links, devices, path traversal, existing files, and more than 100 GiB of
uncompressed member sizes.

## 2. Prepare the canonical corpus

```bash
python3 -m data prepare --config /external/work/config.json \
  --output /external/work/runs/001
```

For full corpora, `--workers 6` enables parallel cleaning and per-clip checkpoints.
Use the same command with `--resume` after interruption. Config, source metadata,
input rows and implementation must match; reused source/output audio hashes are
checked. Split grouping and deduplication happen globally after cleaning. Worker
count can change on resume. `progress.json` records live counts/status; these
preliminary counts precede global leakage and duplicate rejection. Completed runs
are verified when resumed. Keep the checkpoint directory with the run.

The local overnight run is in
`C:\Users\mkhan\Documents\data\tts\full-cleaning-2026-10-22`. It has a frozen
`implementation/`, persistent runtime, `full-config.json`, `run-full.ps1` launcher,
`job-status.json`, logs, and `prepared/progress.json`. The launcher runs independently
of the chat and requests system wakefulness only for its lifetime. Source records
retain the user's explicit authorization for local MBSpeech preparation and the
fact that underlying rights were not independently verified. Completion writes
`completion-report.json` and an adaptation export. Re-running this launcher resumes
the cleaning stage; check existing export artifacts before retrying an interrupted
export. No training or publication is launched.

Outputs:

- `manifest.jsonl`: all accepted multi-speaker records, sorted by stable ID.
- `train.jsonl`, `validation.jsonl`, `test.jsonl`: canonical split views.
- `audio/`: mono PCM16, 24 kHz WAV, one conversion from original source.
- `rejected.jsonl`: complete provenance and rejection reason; no rejected output audio.
- `report.json`: counts/hours by source and split, rejection reasons, measured
  rejected hours, unmeasured rejection count, unknown speaker count, effective
  filters, configuration, input metadata hashes, manifest hash and runtime versions.

Canonical fields include source family/URL/revision/license/clearance, original
clip ID, original and normalized text, original path and file hash, native sample
rate/channels, canonical PCM hash, duration/sample count, quality measurements,
original and assigned split, source-qualified speaker ID or null, `language:
"mn-MN"`, and a same-speaker reference within the same split where possible.
`language` is descriptive metadata, **not** a registered Qwen codec/language ID.
References are deterministic (first accepted ID per known speaker and partition).
Evaluation audio never supplies a training reference. Unknown speakers get null
references, not another person's voice. No source-wide default voice is inferred.

Text normalization performs only Unicode NFC and whitespace folding. It preserves
Ө/Ү, case, punctuation, numbers and spelling. Control characters, model control
tokens, empty text and text without letters are rejected. The configurable
Cyrillic fraction is only a screen, not a dialect classifier. Number expansion,
abbreviation expansion and pronunciation correction require reviewed language
rules and are deliberately not guessed.

Default acoustic filters: 1–20 seconds, RMS at least -45 dBFS, no more than 1%
near-full-scale samples, no more than 80% samples below -40 dBFS. These sample-level
silence/clipping heuristics are **not** VAD, SNR estimation, alignment checks or MOS.
Optional denoising and trimming are described below. There is no loudness normalization or augmentation. Resampling
16 kHz audio to 24 kHz does not restore bandwidth. Tune filters using native-listener
review and inspect rejection reports before scaling up.

## 3. Duplicates and split guarantees

Exact duplicates are detected on decoded canonical PCM, so equivalent containers
can be matched. There is no perceptual matching across lossy re-encodings, speed
changes or edited recordings. Preserve source family/clip identifiers for manual
cross-distribution audits.

Connected groups combine punctuation/case-insensitive text, exact audio,
known speaker (default), and known source session. Relationships are built **before**
deduplication so duplicate removal cannot hide a speaker link to held-out audio.
Official test takes priority, then official validation. Training rows linked to
an official holdout are rejected as `heldout_leakage`, never moved into that holdout.
If official validation/test themselves overlap, the conflicting validation rows
are excluded and reported, while their original raw source partition is untouched.
Exact duplicates inside the retained partition retain one representative.

Remaining groups receive seeded hash-based train/validation/test assignments.
Fractions are targets, not exact quotas. Results are order-independent for a fixed
input/config; adding records can change connected groups and assignments. Keep
run manifests immutable. Dense speaker/text connections may form huge components
and eliminate much training data: the report is the basis for revising corpus
selection, not an invitation to leak evaluation examples.

Known speakers are disjoint by default. Unknown identities cannot provide a
speaker-disjoint guarantee; reports say so. `speaker_disjoint: false` is an explicit
alternative for within-speaker voice refinement; text, exact audio and session
separation still apply. A single-speaker corpus with disjoint-speaker splitting
can naturally land entirely in one partition. Zero-sized splits are reported;
inspect them rather than assuming there is a valid evaluation set.

For a training corpus that does not need to reproduce upstream benchmark splits,
set `preserve_source_splits: false` and `speaker_disjoint: false` explicitly.
This rebuilds seeded partitions by text/audio/session groups while retaining the
original `source_split` as provenance. It permits familiar speakers across splits
and must not be described as unseen-speaker evaluation or an official benchmark.
Never combine this corpus with the old official test set for evaluation.

`python -m data.repartition --prepared OLD --output NEW --workers 6`
rebuilds this policy from cleaning checkpoints in a separate output directory.
It preserves initial quality rejections and checks cross-split normalized text,
cleaned PCM and original PCM overlap before export. If old finalization removed
a cleaned file, recovery verifies the raw source, replays only the saved selected
enhancement and exact trimming offsets, and requires the resulting PCM hash to
match the original checkpoint. Quality scoring and VAD inference are not repeated.
New finalizations retain cleaned files excluded by splitting or deduplication.

### Source review and safe trim revision

The local ten-clip listening review exposed a slowed FLEURS original whose VAD
trim retained only part of the spoken sentence. Its source and processed versions
were both judged slowed by the reviewer. A large proposed trim is now reverted
when it removes more than 40% of duration **and** more than 10% of the selected
waveform's energy. The complete selected waveform is kept, flagged
`unsafe_edge_trim_reverted`, and subjected to the normal duration/quietness/
clipping/silence filters. This safeguard does not certify transcript alignment.
Both limits are configurable in `trim` and `vad` as
`max_removed_duration_fraction` and `max_removed_energy_fraction`.

Create a reviewed revision without re-running DNSMOS/VAD inference:

```powershell
python -m data.quality_revision --prepared C:\data\rebalanced `
  --output C:\data\quality-reviewed --review C:\reviews\human-review.json --workers 6
```

This requires the audio/neural optional dependencies, immutable originals, local
enhancement files for any selected denoising replay, and the parent manifest.
It scans every original FLEURS file using the pinned source configuration,
supports native PCM16/IEEE-float32 WAVs without changing their clocks, and records
source hashes. Severe spectral anomalies require multiple cues; pitch or gender
alone is never an exclusion rule. Same-sentence active-duration comparisons yield
`suspected_slowed_source`; severe cases without a suitable comparison are
quarantined as `source_slowdown_unresolved`. Only listener-rejected samples carry
`human_rejected`. These are reversible preparation exclusions, not an assertion
that all detected recordings have a proven speed defect. Complete features,
thresholds, candidate evidence and the submitted review are preserved in the run.

The diagnosis used 99% spectral rolloff <=2 kHz, 95% rolloff <=1 kHz, >=99.5% energy
below 4 kHz, and weak original speech evidence (VAD coverage <50% or predicted
speech score <2.8). Suitable same-sentence references require 99% rolloff >=4 kHz,
95% rolloff >=1.5 kHz, predicted speech score >=3, and active span >=2 seconds.
Paired candidates additionally need active span >=1.6 times the reference median.
These engineering thresholds are not a calibrated multilingual slowdown detector.
Spectra use 1024-sample Hann windows with 512-sample hops, aggregated over the
louder half of DC-centered frames. Active span uses frame energy at least 1% of
the maximum. The method can miss slowed recordings or quarantine other severe
channel problems; evidence remains available for human restoration decisions.

Only previously large trims need waveform reconstruction. Replay first verifies
the old cleaned PCM hash, applies the guard, and records any new PCM hash. Human
Keep decisions tied to the parent PCM hash retain their approved waveform.
Existing partitions are preserved and speaker references are regenerated within
each partition. The revision references unchanged WAVs in the parent run and
stores corrected WAVs in its own `audio/`; copying the revision folder alone does
not produce a standalone audio bundle. The original run and sources are retained.
No training is performed. Transient Windows locks on progress-file replacement
use bounded retries while retaining atomic publication; permanent errors still
fail explicitly.

## 4. Language-adaptation integration contract

Export the primary multi-speaker adaptation interface:

```bash
python3 -m data export-adaptation \
  --manifest /external/work/runs/001/manifest.jsonl \
  --output /external/work/adaptation/001
```

This retains **every** accepted row across train/validation/test, including unknown
speakers. It verifies PCM hashes and reference ownership, writes a versioned row
schema, and emits `contract.json` identifying missing trainer integration. Rows
include `schema_version`, ID, absolute audio, text, language, nullable speaker and
reference, conditioning status, split, duration, PCM hash and source revision.
It does not fabricate Qwen language IDs or pretend null references are supported
by stock SFT. The canonical manifest remains the full provenance authority.

Consume this contract for the **new-language** training work. The trainer
must evaluate conditioning (including an Auto-based approach, without assuming a
new language ID is needed), unknown-speaker handling,
speaker/reference batching, whether token/embedding updates are needed, and optimization strategy.
LoRA or another parameter-efficient method may be explored; this data contract does
not assume full-parameter training is necessary. It must not map all unknown
identities to one speaker. Validate tokenizer text
coverage and pronunciation for Khalkha Cyrillic and build a native-listener test
set. `mn-MN` alone does not add a supported language. This repository does not
implement or claim to validate a new multilingual trainer.

Canonical rows with known references can be encoded by the explicit tokenizer
stage below, but reference-null rows need a trainer policy before that stage.
Do not throw away FLEURS simply because the official voice-refinement trainer
cannot consume its unknown speakers; retain it in the canonical corpus.

## 5. Optional official single-speaker Qwen export

Verified against QwenLM/Qwen3-TTS commit
[`022e286b98fbec7e1e916cb940cdf532cd9f488e`](https://github.com/QwenLM/Qwen3-TTS/tree/022e286b98fbec7e1e916cb940cdf532cd9f488e/finetuning),
including `README.md`, `prepare_data.py`, `dataset.py` and `sft_12hz.py`.
The upstream public interface currently supports single-speaker Base-model SFT.
It is not a complete Mongolian language-adaptation workflow.

```bash
python3 -m data export-qwen \
  --manifest /external/work/runs/001/manifest.jsonl \
  --speaker cv27:REVIEWED_CLIENT_ID --output /external/work/qwen/voice-001
```

This selects exactly one known speaker with training data and writes
`train_raw.jsonl`, `validation_raw.jsonl`, `test_raw.jsonl`, plus an export audit.
Each row has **absolute** `audio`, `text`, and `ref_audio`. All rows use the same
reference from this speaker's training partition. No `ref_text` is required.
24 kHz reference WAVs are mandatory: upstream loads them without resampling and
asserts that rate. Unknown or heldout-only speakers cannot be exported.

### Explicit optional audio-code extraction

Separately provision a compatible `qwen-tts`/PyTorch environment and an already
obtained local Qwen3-TTS-Tokenizer-12Hz snapshot. Pin that environment and retain
its package lock. The core pipeline does not install it or download weights.

```bash
python3 -m data encode-qwen \
  --input /external/work/qwen/voice-001/train_raw.jsonl \
  --output /external/work/qwen/voice-001/train_with_codes.jsonl \
  --tokenizer-model /external/models/Qwen3-TTS-Tokenizer-12Hz \
  --device cpu --batch-size 1
python3 -m data check-codes \
  --input /external/work/qwen/voice-001/train_with_codes.jsonl
```

This loads a **local** tokenizer with Hugging Face offline modes enabled. It runs
only when explicitly invoked; it is compute work, not part of `prepare`.
Choose a device actually supported by the installed stack. Codes have shape
`T × 16` integer entries and are added without changing audio/text/reference fields.
Unlike upstream's batch-32, all-in-memory script, this wrapper processes bounded
batches and checkpoints after each batch. Re-run the same command after an
interruption: input SHA, source audio/reference hashes, tokenizer file hashes and device must match. A failed
batch can be recomputed; a partial uncommitted tail is truncated. Completed outputs
are never overwritten. Keep the checkpoint as provenance. `check-codes` validates
structure, not codec vocabulary bounds or model quality.

The upstream trainer's `--train_jsonl` can consume the resulting single-speaker
file. Its implementation hardcodes one speaker embedding/ID, BF16 and
FlashAttention 2; it also uses a local `init_model_path` for checkpoint copying.
The `language` field is read but unused. These constraints require deliberate
trainer work for Mongolian language adaptation and for a Mac/MPS path. No trainer
was launched or model compatibility/performance measured here.

## Verification and limits

```bash
python3 -m unittest discover -s tests -v
python3 -m data --help
```

Tests generate tiny synthetic PCM fixtures in temporary directories outside Git.
They exercise acquisition with mocked network downloads, source schemas,
quarantine, filters, hashes, leakage, reproducible splits, canonical and Qwen
exports, and interrupted/resumed code extraction with a mocked tokenizer.
Neither synthetic fixtures nor mocks establish speech quality or tokenizer model
compatibility. Real PyArrow read/write, mono resampling, stereo MP3 downmix/conversion, corrupted
audio rejection and end-to-end adaptation export passed using locally generated
fixtures. The current suite passed all 53 tests with zero skips on Python 3.12.12,
using PyArrow 26.0.0, imageio-ffmpeg 0.6.0, NumPy 2.5.4, and a local FFprobe.
The optional dependency snapshot is in `requirements-verification.txt`; it is not
needed for the core CLI. Without extras, codec, NumPy source-reader, and denoising
integration tests skip explicitly. CI exercises core and optional-dependency
profiles on Python 3.11 and 3.12 without corpus or model downloads.
The [initial pilot](data-pilot-2026-10-22.md) is historical; the subsequent full
three-source run, review corrections, and current results are recorded in
[Data preparation](2-data-preparation.md). Actual tokenizer inference remains
untested and requires the corresponding model environment. Neither these tests
nor the small listening review certify every clip or downstream model quality.

## Optional noise reduction and silence trimming

The example config enables quality-gated denoising and Silero VAD trimming. It
requires the optional `neural` extra and the explicitly provisioned local models
documented in [the cleaning research](data-cleaning-research.md). Existing configs that omit
`audio_processing` preserve the previous unprocessed behavior. Each stage can be
disabled independently. This smaller configuration uses the energy-based fallback
and FFmpeg without neural models:

```json
"audio_processing": {
  "denoise": {"enabled": true, "reduction_db": 6.0, "noise_floor_db": -50.0},
  "trim": {"enabled": true, "threshold_dbfs": -45.0, "window_ms": 20.0,
           "min_silence_ms": 200.0, "padding_ms": 150.0}
}
```

Processing order is original decode/resample → noise reduction → edge trimming →
quality filters. Originals are never modified; use a fresh output run to compare
against an earlier unprocessed run. Conversion/resampling happens once; later
stages operate on the generated mono PCM16 24 kHz WAV.

Denoising uses FFmpeg's [`afftdn` FFT filter](https://ffmpeg.org/ffmpeg-filters.html#afftdn),
with fixed noise floor and noise tracking disabled. This is lightweight stationary
noise reduction, not UVR vocal/music separation. It downloads no models and needs
no GPU. The requested reduction is a filter setting, not a measured improvement in
SNR. Background voices, music, and changing noise may remain. Filtering can alter
speech timbre; listen to before/after samples before processing an entire corpus.
Missing FFmpeg/filter support fails the run rather than silently skipping denoising.
The alternate `deepfilternet` backend uses a local DeepFilterNet3 executable/model,
with `reduction_db` interpreted as its attenuation limit. It compensates model
delay and uses the model's required 48 kHz working rate before returning to 24 kHz;
these additional rate conversions do not restore lost bandwidth.

Trimming identifies the first and last RMS windows at or above `threshold_dbfs`.
Only edge quiet runs at least `min_silence_ms` long are eligible. It retains up to
`padding_ms` from each detected edge; internal pauses remain byte-for-byte intact
after the denoising stage. This is energy screening, not speech VAD or alignment.
Clips with no active window are rejected as `no_active_audio`. Quiet speech may
need a lower threshold or more padding to retain initial/final consonants.

Each canonical row records `audio_processing` settings, stage order, pre-processing
quality metrics/duration/hash, and exact leading/trailing trimmed sample counts and
seconds. `original_pcm_sha256` identifies decoded canonical PCM before cleaning;
`pcm_sha256` identifies the final output. Leakage grouping uses both hashes so
processing cannot hide an original exact duplicate. Reports retain effective
processing settings; normal quality filters use final audio. No-active rejections
retain the pre-processing measurements. The report's FFmpeg version records the
denoiser implementation version for the FFT backend. DeepFilterNet executable and
model SHA-256 values pin that backend. MBSpeech rights quarantine still precedes decoding.

When `vad.enabled` is true, its speech-aware boundary finder replaces the energy
trimmer. It retains everything between the first and last detected speech segments,
including internal pauses. No speech is rejected as `no_speech_detected`. With
`quality.enabled`, the denoiser is skipped for predicted-clean backgrounds, and its
candidate is kept only if predicted overall/background quality improve without an
excessive speech-quality drop. A greater than 10% loss of VAD speech duration also
reverts enhancement. This is a heuristic safeguard, not proof of intelligibility.
DNSMOS is an adapted regular P.835 SIG/BAK/OVRL scorer, not P.808 or measured SNR;
short clips repeat to cover its 9.01-second input window. Effective settings and
model/runtime hashes are recorded. Quality scores below 3 and trims exceeding 40%
of input duration create review flags, not automatic quality-score rejections.

`review.jsonl` records before/after metadata and rejection/review status for every
row. `report.json` includes denoising decisions, removed edge duration, and review
flag counts. Source clipping is checked before and after cleaning so attenuation
cannot hide a damaged source recording.

### Optional offline transcript audit

```bash
python3 -m data audit-transcripts --manifest /external/run/manifest.jsonl \
  --model-dir /external/models/faster-whisper-small --output /external/run/asr-review
```

Install the `asr` extra for faster-whisper, or use a native CPU executable:

```bash
python3 -m data audit-transcripts --manifest /external/run/manifest.jsonl \
  --model-dir /external/models/whisper-cpp --backend whisper-cpp \
  --executable /external/models/whisper-cpp/bin/whisper-cli --output /external/run/asr-review
```

The model directory requires `acquisition.json` with `revision` and `files`, each
containing `file` and `sha256`. Faster-whisper requires hashed `config.json`,
`model.bin`, `tokenizer.json`, and `vocabulary.txt`; native Whisper requires a
`model_file` entry and hashes for its executable and supporting libraries.
All model loading is local. Audio hashes are checked before inference. Mongolian
ASR hypotheses and CER are saved separately; supplied transcripts are never
replaced or automatically rejected. Repetition, non-Cyrillic output, empty output,
or out-of-range timestamps flag an unreliable recognizer instead of labeling the
source transcript wrong. ASR timestamps are not forced alignment of source text.
