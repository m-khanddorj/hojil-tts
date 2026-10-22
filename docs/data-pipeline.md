# Khalkha language-adaptation data pipeline

The primary deliverable is a **multi-speaker canonical corpus**, not a claim that
Qwen3-TTS already supports Mongolian. The optional official single-speaker export
is for later voice refinement. Adding a new language still needs trainer/model
integration and native-listener evaluation. Existing dataset research remains in
[data/README.md](../data/README.md).

## Requirements and architecture

Run from the repository with Python 3.11+; the core CLI and tests use only the
standard library. No installation is necessary for `python3 -m hojil_data`.
Optional packaging: `python3 -m pip install -e .` enables `hojil-data`.

- `sources.py`: explicit acquisition, CV27/FLEURS adapters, quarantined MBSpeech
  local manifests and optional Parquet import.
- `audio.py`: PCM analysis; ffmpeg/ffprobe conversion to mono, PCM16, 24 kHz WAV.
- `pipeline.py`: conservative text/audio validation, leakage groups, splits,
  references, canonical JSONL, rejection and duration reports.
- `adaptation.py`: primary versioned multi-speaker export with explicit integration contract.
- `qwen.py`: optional official single-speaker export and explicit offline code extraction.
- `io.py`: artifact hashes, safe extraction, credential-free public downloads.

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
python3 -m hojil_data acquire --config /external/work/config.json \
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
python3 -m hojil_data acquire --config /external/work/config.json \
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
`original_sentence`, `sentence`, or `text` columns:

```bash
python3 -m hojil_data import-mbspeech --config /external/work/config.json \
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
python3 -m hojil_data prepare --config /external/work/config.json \
  --output /external/work/runs/001
```

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
There is no denoising, trimming, loudness normalization or augmentation. Resampling
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

## 4. Language-adaptation integration contract

Export the primary multi-speaker adaptation interface:

```bash
python3 -m hojil_data export-adaptation \
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
python3 -m hojil_data export-qwen \
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
python3 -m hojil_data encode-qwen \
  --input /external/work/qwen/voice-001/train_raw.jsonl \
  --output /external/work/qwen/voice-001/train_with_codes.jsonl \
  --tokenizer-model /external/models/Qwen3-TTS-Tokenizer-12Hz \
  --device cpu --batch-size 1
python3 -m hojil_data check-codes \
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
python3 -m hojil_data --help
```

Tests generate tiny synthetic PCM fixtures in temporary directories outside Git.
They exercise acquisition with mocked network downloads, source schemas,
quarantine, filters, hashes, leakage, reproducible splits, canonical and Qwen
exports, and interrupted/resumed code extraction with a mocked tokenizer.
Neither synthetic fixtures nor mocks establish speech quality or tokenizer model
compatibility. Real PyArrow read/write, mono resampling, stereo MP3 downmix/conversion, corrupted
audio rejection and end-to-end adaptation export passed using locally generated
fixtures. The suite ran 24 tests with zero failures/skips in an isolated Python 3.15
environment (PyArrow 26.0.0, imageio-ffmpeg 0.6.0, SoundFile 0.14.0). Dependency pins
are in `requirements-verification.txt`. Without extras, the two real integration
tests explicitly skip while the standard-library tests remain runnable.
Real remote corpus downloads and actual tokenizer inference remain untested and
require the corresponding access/model environment and a subsequent small pilot. Do not call any planned dataset
hour total “usable training hours” until the resulting report and manual sample
audit are available.
