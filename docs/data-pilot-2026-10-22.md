# Real-data pilot: 2026-10-22

Historical pilot snapshot. Later user approval, full-corpus processing, listening
corrections, and current totals are recorded in
[Data preparation](2-data-preparation.md).

Preparation and adaptation export passed on a small real CV27/FLEURS sample.
MBSpeech audio/schema checks passed, but training preparation remains quarantined
without a recording/text rights-clearance reference. No training was run.

| Source | Sample | Result |
| --- | --- | --- |
| Common Voice 27.0 | 30 clips: 10 test, 10 dev, 10 other validated | 29 accepted; 1 quiet |
| FLEURS mn_mn | 30 clips: 10 per official partition | 18 accepted; 7 silence, 4 quiet, 1 duration |
| MBSpeech | First 30 rows of the first bilguun Parquet shard | All decoded from 16 kHz to mono PCM16 24 kHz; all transcripts passed text screening; all 30 correctly quarantined by preparation |

Samples use the first eligible metadata rows, not a random or representative
quality estimate. The combined accepted corpus contains 47 clips / 5.868 minutes,
with 16 train, 16 validation, and 15 test rows. Adaptation export retained all 47,
including 18 FLEURS rows with unknown speakers and null references. No listening
or transcript/alignment audit was performed. MBSpeech's real Parquet importer was
not run through its clearance gate; the codec and schema were inspected directly,
and importer compatibility was checked with a generated Parquet integration test.

## Sources and local artifacts

Data root: `C:\Users\mkhan\Documents\data\tts` (outside Git).

- CV27 archive SHA-256:
  `05c5a447be5136f08cfd7eb9b3c0c7edb17790c01dd5c627b02bb92bbeeda364`.
- `fleurs/`: full Mongolian acquisition pinned to
  `70bb2e84b976b7e960aa89f1c648e09c59f894dd`. Corrected full inventory: 3,074 train,
  405 dev, and 949 test rows. The initial inventory undercounted train by 347
  because CSV quote handling merged literal TSV records. Full-run preflight
  caught this; the adapter now uses `QUOTE_NONE`. All 4,428 rows have audio.
- `mbspeech/`: full public `bilguun/mbspeech`, pinned to
  `5cdcc286f77419e3d0e3b60cb1852c99aaa91013`, with two Parquet shards of 1,923
  rows each. This is the original-family distribution listed in the project catalog.
  The configured `btsee/mbspeech_mn` returned HTTP 401. No credentials were used,
  and the cleaned `btsee/mbspeech-mn` was not substituted. The downloaded card has
  no explicit license; training clearance remains unset.
- Each downloaded corpus has an `acquisition.json` recording revision and SHA-256
  hashes. MBSpeech files were rehashed after download. FLEURS archives were
  rehashed before final extraction; original downloads remain in `.acquisition-cache`.
- `verification-2026-10-22/`: sampled source files, `config.json`, canonical
  `prepared/` reports/audio/manifests, `adaptation/`, `mbspeech-config.json`,
  `mbspeech-quarantine/`, `mbspeech-diagnostics.json`, and `verification-summary.json`.

## Fixes and environment

The real MBSpeech schema uses `sentence_orig`; the importer now recognizes it
before normalized/fallback text. Its real Parquet fixture verifies preservation of
original punctuation. Cyrillic test TSVs now explicitly use UTF-8 on Windows.
All 24 tests passed, including actual MP3 conversion and Parquet import/export.

Verification used Python 3.12.12, imageio-ffmpeg 0.6.0, PyArrow 26.0.0, and the
temporary FFprobe fallback. Windows application control blocked `_ctypes`, making
SoundFile unusable despite installation; providing FFprobe avoided that dependency.
The temporary environment is `%TEMP%\hojil-data-pilot-venv`; FFprobe is in
`%TEMP%\hojil-ffprobe`. Prepend that directory to PATH for compressed input.

FLEURS acquisition downloaded and extracted successfully but Windows denied its
final staging-directory rename. Final placement was completed by extracting the
rehashed cached archives directly into `fleurs/`. The acquisition command's atomic
publication step therefore remains an unresolved Windows environment limitation.

## RunPod data workflow

Prepare and review locally first. Keep raw archives locally and upload the accepted
canonical corpus, reports, config, and source/checksum records as an immutable run.
Use checksummed tar archives for transfer and unpack them once on a persistent
network volume. Preserve splits rather than splitting again in a training loader.

RunPod network volumes survive Pod termination. Its S3-compatible API supports
uploading without a running Pod in supported datacenters. Select storage compatible
with the intended GPU location before uploading. See
[network volumes](https://docs.runpod.io/storage/network-volumes) and
[S3 transfers](https://docs.runpod.io/storage/s3-api).

Suggested Pod layout, assuming the volume is mounted at `/workspace`:

```text
/workspace/hojil/data/runs/<run-id>/
/workspace/hojil/models/
/workspace/hojil/checkpoints/<experiment>/
```

Current canonical/export manifests contain absolute Windows audio/reference paths.
Copying these files alone does not make them usable on Linux. A relocation step
must map accepted `audio` and `ref_audio` paths to the unpacked run, preserve source
provenance fields, verify PCM hashes, and regenerate the adaptation export on the
Pod. This relocation feature is not implemented yet. Alternatively, transfer raw
sources and run preparation on a CPU environment using the final Linux paths.

Run tokenizer encoding on the GPU only after selecting the trainer's conditioning
policy. The repository currently implements preparation/export, not a Mongolian
language-adaptation trainer. Keep checkpoints on persistent storage and back them
up separately; keep datasets, weights, credentials, and generated runs outside Git.
