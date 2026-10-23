# Cleaning Khalkha TTS data: evidence and implementation

This document records the research and initial decisions. See
[Data preparation](2-data-preparation.md) for the full run, later trim safeguards,
FLEURS source-anomaly quarantine, and final measured results.

Research date: 2026-10-22. Goal: prepare CV27, FLEURS mn_mn, and original-family
MBSpeech for multi-speaker Khalkha language adaptation. There is no established
universally best cleaner for these three corpora. The practical choice here is
conservative speech-aware trimming, selective enhancement, original-audio quality
checks, transcript review, and preserved provenance/splits. Confidence is high in
the workflow components, moderate in transferring enhancement research to this
task, and low in any uncalibrated numerical threshold for Mongolian.

## What the evidence supports

### 1. Select and verify audio; do not assume enhancement always helps

A Common Voice Catalan study combines enhancement and NISQA filtering before TTS
training. A low-resource Arabic study also reports benefits while explicitly
investigating nonlinear enhancement distortion. These results motivate an
enhance-and-select workflow; they do not establish a best model or threshold for
Mongolian. [Catalan study](https://arxiv.org/abs/2410.13357),
[Arabic study](https://arxiv.org/abs/2309.10795).

Implementation: retain unmodified sources, hash decoded audio before cleaning,
create an enhancement candidate, and retain it only when the quality gate passes.
Known clipping in the source still causes rejection even if the enhanced waveform
looks less clipped. Raw and cleaned exact PCM hashes both contribute to leakage
groups. Every row has a processing audit and appears in `review.jsonl`.

### 2. Use speech detection for boundaries and preserve sentence prosody

Silero offers a small multilingual VAD with local ONNX inference at 8/16 kHz. Its
reference implementation uses speech-probability hysteresis, minimum speech/quiet
durations, and boundary padding. Those controls address a different problem from
simple amplitude thresholds. Language coverage is a publisher claim, not a
Khalkha accuracy measurement. [Silero documentation](https://github.com/snakers4/silero-vad),
[reference inference and boundary logic](https://github.com/snakers4/silero-vad/blob/v6.2.3/src/silero_vad/utils_vad.py).

Implementation: a separate 16 kHz analysis copy feeds Silero; the training waveform
remains 24 kHz. Use threshold 0.5, at least 250 ms speech, 500 ms quiet separation,
and 250 ms edge padding. Keep the entire interval between first and last speech;
never concatenate speech islands or delete internal pauses. The chosen defaults
are conservative engineering starting points, not values validated for Mongolian.
An energy trimmer remains available without neural dependencies. No speech creates
an explicit rejection, with the original measurements retained for review.

### 3. Compare speech denoisers and preserve speech, not only quiet backgrounds

DeepFilterNet is a speech-enhancement model with published efficiency/quality
results and an offline executable. Its 48 kHz input requirement and compensated
model delay must be respected. FFmpeg `afftdn` is a lightweight FFT noise-reduction
alternative. UVR is a vocal/music source-separation application, so it is a separate
candidate for confirmed music contamination rather than a universal speech cleaner.
[DeepFilterNet paper](https://arxiv.org/abs/2305.08227),
[official usage](https://github.com/Rikorose/DeepFilterNet#usage),
[FFmpeg filter](https://ffmpeg.org/ffmpeg-filters.html#afftdn),
[UVR project](https://github.com/Anjok07/ultimatevocalremovergui).

Implementation: both FFT and DeepFilterNet3 backends are available. Start at 6 dB
FFT reduction or a 6 dB DeepFilterNet attenuation cap, without a strong postfilter.
DeepFilterNet uses a checksummed executable and explicit model, `--compensate-delay`,
and 48 kHz working audio. That backend incurs additional resampling; no bandwidth
recovery is claimed. A duration change beyond 50 ms aborts the run. A greater than
10% decline in detected speech duration reverts the candidate. VAD duration is a
coarse safeguard and can miss phoneme loss or falsely disagree.

RNNoise is another lightweight candidate; its model catalog distinguishes general
sound, human voice, and speech targets. It was reviewed but not added as a third
neural backend because the pilot already compares an FFT and a speech-enhancement
model. [RNNoise model catalog](https://github.com/GregorR/rnnoise-models).

### 4. Use quality predictors as review aids and relative gates

DNSMOS P.835 predicts speech, background, and overall quality separately. That is
useful when a quieter output also damages speech. MOS-predictor research documents
generalization problems with unseen conditions, so a universal absolute MOS cutoff
would be unjustified here. Human listening remains necessary.
[DNSMOS P.835 paper](https://arxiv.org/abs/2110.01763),
[official reference](https://github.com/microsoft/DNS-Challenge/blob/master/DNSMOS/dnsmos_local.py),
[generalization study](https://arxiv.org/abs/2110.02635).

Implementation: use the regular P.835 model and published calibration polynomials.
This adapted scorer repeats clips shorter than 9.01 seconds the minimum whole
number of times needed, then averages valid one-second-hop windows. The reference
script instead repeatedly doubles short clips; short-clip scores need not match
its outputs exactly. No P.808 score, true SNR, or measured human MOS is claimed.

Skip denoising when predicted background quality is at least 4. Otherwise require
overall and background gains of at least 0.05 and speech-score loss at most 0.1.
These are adjustable relative heuristics. Compare before and candidate at the same
duration; record post-trim quality separately. Final overall/speech scores below 3
and removal of over 40% of input duration create review flags, not MOS-based
automatic rejections. The predictor can still prefer an audibly worse output.

### 5. Verify audio/text correspondence without silently rewriting Mongolian

NVIDIA's TTS data pipeline treats alignment, diarization/overlap, audio quality,
and bandwidth as separate stages. Denoising alone does not solve bad transcripts
or multiple speakers. Whisper includes a Mongolian language code, but support
does not establish sufficient accuracy for automatic relabeling.
[NeMo TTS pipeline](https://github.com/NVIDIA/NeMo-speech-data-processor/blob/main/dataset_configs/tts/ytc/config.yaml),
[Whisper language registry](https://github.com/openai/whisper/blob/main/whisper/tokenizer.py).

Implementation: `audit-transcripts` runs a separately provisioned local model and
records hypotheses, timestamps, and character error rate. Comparison ignores case,
punctuation and spacing while preserving Cyrillic letter distinctions, including
Ө/О and Ү/У. CER above 0.3 is a review flag only. Empty, repetitive, non-Cyrillic,
or out-of-range-timestamp hypotheses flag ASR unreliability instead. Neither source
text nor training membership is changed by the audit. ASR timestamps are not
forced alignment to the source transcript.

Faster-whisper supports local models and CPU int8 inference; whisper.cpp provides
a native CPU alternative. The local Windows application-control policy blocks
faster-whisper's `_ctypes` dependency, so the pilot uses official whisper.cpp with
a pinned multilingual Whisper-small Q5 model. This is a feasibility check, not a
validated Mongolian transcript judge. [faster-whisper](https://github.com/SYSTRAN/faster-whisper),
[whisper.cpp](https://github.com/ggml-org/whisper.cpp).

Full source-text forced alignment and overlapping-speaker detection are not
implemented or claimed. Before automating those filters, validate an appropriate
Mongolian recognizer/aligner against native-listener labels. The same applies to
automatic spelling, number expansion, dereverberation, aggressive equalization,
and loudness processing. This pilot does not provide evidence to apply those
transforms across all three datasets.

## Configuration and local artifacts

`configs/example.json` contains the speech-aware profile with model paths to replace.
Existing configurations with no `audio_processing` preserve the old behavior.
Install the optional `audio,neural` extras; `asr` is needed only for faster-whisper.
Preparation never downloads dependencies or model weights.

Local model folder: `C:\Users\mkhan\Documents\data\tts\models\cleaning`.
Its `acquisition.json` records URLs/revisions/hashes, and license notices are
preserved alongside the files. Silero is pinned to v6.2.3 commit
`5cd7945676eb32225748052e2e6a0580e4686a08`; DNSMOS to
`591184a9fcb2cbdec02520fed81a32bbbf9d73ff`; DeepFilterNet to release v0.5.6 and
model commit `978576aa8400552a4ce9730838c635aa30db5e61`.

To use DeepFilterNet, set `denoise.backend` to `deepfilternet` and supply local
`executable`, `executable_sha256`, `model`, and `model_sha256`. Its working model is
`DeepFilterNet3_onnx.tar.gz`. The other settings are shared with the FFT profile.
The ASR model/executable and their hashes are in the separate `models/whisper-cpp`
folder. Model files, datasets, audio and generated reports remain outside Git.

Pilot artifacts are in
`C:\Users\mkhan\Documents\data\tts\verification-2026-10-22\research-cleaning`:
separate configs/runs for `vad-only`, `afftdn`, and `deepfilternet`, plus adaptation
exports, MBSpeech diagnosis, and transcript-review artifacts. The original
unprocessed pilot and simpler energy-trimming pilot remain separate.

## Applied pilot results

The comparison used the same 30 CV27 and 30 FLEURS clips for all three profiles.
All profiles retained 29 CV27 and 21 FLEURS clips; the other ten failed quietness
(5) or silence (5) filters. The older untrimmed run retained 47 of these 60 clips.
Accepted counts alone do not rank denoisers.

| Profile | Skipped predicted-clean | Enhancement retained | Reverted by quality gate | Reverted by VAD speech loss |
| --- | ---: | ---: | ---: | ---: |
| VAD trimming, no denoiser | — | — | — | — |
| FFT, 6 dB, gated | 24 | 10 | 24 | 2 |
| DeepFilterNet3, 6 dB cap, gated | 24 | 16 | 17 | 3 |

Across the 36 clips with denoising candidates, mean predicted overall-quality
change was +0.019 for FFT and +0.110 for DeepFilterNet3, before candidate selection
and trimming. These are predictor scores on a tiny sample, not human MOS gains.
DeepFilterNet3 with the gates is the preferred local pilot candidate because more
enhancements passed the same gates and its candidate score gain was larger.
The portable example retains the lighter FFT backend with the same neural gates;
the ready-to-run local DeepFilterNet settings are in `deepfilternet-config.json`.
No universal model superiority or downstream TTS gain is established.

All 30 MBSpeech diagnostic clips passed model execution. DeepFilterNet was skipped
for 18, retained for 4, and reverted by the quality gate for 8. No MBSpeech
training export was created; the rights quarantine is unchanged.

The native Whisper-small Q5 pilot executed on nine clips, three per source. All
nine outputs had repetitive non-Cyrillic text and timestamps beyond the clips.
The completed results are therefore labeled **ASR unreliable**, not nine bad source
transcripts. This model/runtime combination failed the Mongolian audit pilot.
The cause has not been isolated; a model/runtime compatibility or inference issue
has not been ruled out. Do not enable CER-based automatic rejection. The audit
command is available for a better validated local model on RunPod; source-text
alignment remains unverified.

All 36 regression/integration tests passed in the local Python 3.12 environment,
including actual FFT denoising, exact interior-pause preservation, disabled-stage
identity, restoration of the original after a failed candidate, checksum failure,
and duplicate/holdout safeguards. Actual ONNX and DeepFilterNet execution were
checked by the separate real-data pilot. Results, configs, hashes and audio are
outside Git. A local `review.html` presents original and processed audio side by
side; `transcript-review.jsonl` distinguishes failed ASR from transcript concerns.

Before full-corpus processing, calibrate thresholds with native-listener labels
from training data. Keep raw held-out recordings available for evaluation and
avoid selecting thresholds from held-out model performance.

## Method and evidence limits

Eight targeted search queries were followed by primary papers, official model
implementations, licenses, and toolkit documentation. Firecrawl/Exa tools were
unavailable; the web tool was used. Sources were assessed for this project's
recording types rather than selected only by publication date. Decisions above
separate published findings from our engineering choices and unvalidated thresholds.

Samples are the first eligible metadata rows, not a representative random draw.
No native-listener evaluation, TTS training, or test of final synthesized speech
was performed. A higher accepted-clip count or predicted quality score alone is
not evidence that downstream TTS improved. MBSpeech remains quarantined for
training; processing its local copies is technical diagnosis only.
