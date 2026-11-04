# YouTube speech intake

The local HTML page is served by `python -m data.youtube --output /external/youtube`.
Install the `youtube` optional dependency group first. This adds yt-dlp and bundled
FFmpeg; no model weights are required. On Windows use `.venv/Scripts/python.exe`.
Open `http://127.0.0.1:8765`. The server binds only to loopback and validates Host
and a per-session token on writes. This is a single-user local tool, not a public
web service. Keep it running while jobs download and process.

Paste up to 20 individual YouTube URLs and enter the exact uploaded subtitle
language (`mn` by default). Record a reference to the recording/text permission
or license evidence. The field preserves your evidence reference; it does not
establish permission. Download only content you are authorized to acquire and use.
No browser cookies, sign-in credentials, automatic captions or translated tracks
are used. Uploaded tracks are not proof that a human wrote every word; spot-check
their contents. Missing uploaded tracks fail before media download.

Audio-only acquisition is the default. Select **Keep video too** to retain a
combined video/audio format (up to 720p when available). Videos over four hours
and live streams are rejected. Downloads are capped at 4 GiB when their size is
known; unknown-size transfers may exceed this. Queued jobs run one at a time.
Errors are shown per video; other queued videos continue. Interrupted jobs become
failed after restart; submit again to create a fresh run. There is no job deletion,
cancellation or automatic retry of failed jobs in this first version.

## Processing and review

Original media and JSON3 subtitles are retained with SHA-256 hashes, source URL,
video title/channel, subtitle language/type and downloader version. Audio is
decoded to mono PCM16 at 24 kHz. Each subtitle cue becomes a candidate clip.
Overlaps, out-of-range timestamps, text failing the repository's Cyrillic checks,
and cues outside 1–20 seconds are rejected. Long/short cues are not automatically
merged or split because that would require new transcript alignment.

Optional mild stationary noise reduction uses the existing FFmpeg backend.
Conservative energy-based trimming affects edges only and retains its existing
unsafe-trim guards. Clipping is checked before and after processing. Quiet,
mostly silent and too-short output is rejected. Audio metrics, original PCM hash,
processing settings and trim audit are retained. These checks do not detect all
music, overlapping speakers or transcription errors. There is no speaker diarization,
word-level forced alignment, automatic spelling correction or music separation.

Expand **Listen & review clips**. Listen, correct the transcript, optionally supply
a stable speaker label, check the review box, and click **Save review** for each
clip. Reuse a label across videos only when you have verified the same speaker;
leave uncertain identities blank. A channel or video is not treated as a speaker.
If cue boundaries are incorrect, leave the clip unapproved; boundary editing is
not included. Unsaved edits are not exported.

## Export contract

**Export approved clips** takes only saved approvals and invokes the repository's
existing canonical preparation pipeline. An export contains an interchange
`metadata.jsonl`, `config.json`, a copy of selected audio, and a `prepared/`
directory with canonical `manifest.jsonl`, rejection/review reports and WAVs.
Existing corpus files are never modified. Preparation can reject additional rows
or deduplicate identical audio; inspect its report. Every export gets a fresh path.

The interchange config uses source ID `youtube`, stable reviewed speaker labels,
and video IDs as sessions. These allow a later combined preparation run to group
related clips and prevent split leakage. **Before training, combine this source
with your existing sources in one preparation run** so text/audio/speaker/session
overlap is checked globally. For multiple YouTube exports, collect their approved
interchange rows under one external source root, copy audio below that root and
update relative audio paths. Use a single `youtube` manifest source in the combined
config: the pipeline disallows duplicate source IDs. Preserve row metadata and
record all clearance references in that source. Do not concatenate independently
split canonical manifests.
`prepared/` is the staging interface for the existing adaptation export; there
is no separate database connection or automatic training launch here.

The server stores state in `job.json` beside each video's artifacts and restores
reviewable jobs on restart. Preserve the whole job directory for provenance and
review history. Export paths are local absolute paths, not portable downloads.

Downloader options follow the [yt-dlp documentation](https://github.com/yt-dlp/yt-dlp#subtitle-options).
YouTube changes may require upgrading yt-dlp or provisioning a supported JS runtime.
The server enables Node when found on PATH; otherwise yt-dlp's runtime default applies.
No network download is exercised by the unit tests; they cover URL validation,
manual-track gating, caption parsing, segmentation, review and canonical export
using generated fixtures and mocked acquisition.
