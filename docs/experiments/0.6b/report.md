# 0.6B optimization experiment report

Living report, updated November 12, 2026. [Summary page](README.md).

This report covers the investigation and optimization of the 0.6B Mongolian TTS
model in this workstream. It separates completed experiments, implementation
diagnostics, interrupted attempts, and work that is only prepared. The existing
data pipeline, older 1.7B model, and historical demonstration are background;
their results are not attributed to a newly trained 0.6B candidate.

The evidence appendices are part of this report. They provide the individual
variants/checkpoints, recorded metrics, panel definitions, and source paths
behind the register below. Missing measurements are identified rather than
filled with estimates. The report will be updated when the active evaluations
finish and again before the authorized post-goal push.

### Contents

1. [Objective and evidence standard](#1-objective-and-evidence-standard)
2. [Experiment register](#2-report-structure-and-experiment-register)
3. [Current results and active experiment](#3-current-comparable-results-and-active-experiment)
4. [Engineering changes and observability](#4-engineering-changes-and-observability)
5. [Data quality and metric diagnostics](#5-data-quality-and-metric-diagnostics)
6. [Listening examples](#6-listening-examples-and-improvement-claims)
7. [Evidence and reproduction](#7-evidence-and-reproduction-map)
8. [Maintenance and publication](#8-maintenance-reproduction-and-publication)

## 1. Objective and evidence standard

The user requested better Mongolian pronunciation without losing the supplied
reference voice, and explicitly confirmed that **CER below 3% and internal
cosine above 0.984 apply to both development and custom-reference groups**.

The registered original confirmation panel contains 57 text/reference cases
at seeds 2026 and 2027: 88 development recordings from ten speakers and 26
custom-reference recordings from one speaker. Completion requires all 114,
strict thresholds in both groups, no truncated generation, independently
verified original-reference cosines, unchanged evaluation metadata, and complete
artifact/native-conditioning checks. A positive verdict also requires the
owning process to have actually finished successfully.

The original panel protocol SHA-256 is
`f77c6f247d87e2132a21ae0eafda04f64f614ef44d6acd275e3f7ee8832b6041`.
It has been inspected repeatedly. Results on it are evidence for the requested
fixed-panel objective, not an unbiased claim about arbitrary unseen text.

### How to read the measurements

| Measurement | Meaning | Important limit |
|---|---|---|
| Pooled CER | Total character edits divided by total normalized reference characters | Different from averaging clip CER; normalization and panel must match |
| Pooled WER | Total word edits divided by total normalized reference words | Reported even though WER is not one of the two final threshold criteria |
| Speaker-macro cosine | Average within each speaker, then average speakers | Qwen's internal original-reference embedding similarity; not a human identity percentage |
| Per-seed/per-speaker scores | Distribution of behavior beneath the pooled result | A favorable seed or clip cannot replace complete-panel scoring |
| Teacher-forced loss | Accuracy of predicting registered codec targets under fixed contexts | Can improve while generated pronunciation or voice worsens |
| Gradient and adapter movement | Whether updates are finite, bounded, and actually change new parameters | Do not establish perceptual or ASR improvement |
| Native replay/artifact audit | Checks conditioning, codec/EOS alignment, provenance, and fixed weights | Verifies implementation and evidence integrity, not the quality target by itself |

Later native evaluations use the pinned BuzzASR/Mongolian revision
`185f1bf1f562ad4b1bdbfc71811fb165a9235416` and registered decoding. The scoring
normalization uses Unicode NFC, case folding, punctuation removal, and collapsed
whitespace; CER retains normalized spaces. Historical tables retain their own
explicit mean/pooled definitions and scopes. They are not silently converted
into the current original114 benchmark.

## 2. Report structure and experiment register

Each experiment is documented using the same questions:

1. **Purpose:** which observed problem or hypothesis motivated it?
2. **Changes:** what was changed, trained, or held fixed?
3. **Setup:** which data, reference policy, checkpoint, panel, seeds, and scoring settings were used?
4. **Results:** what actually completed, with measured CER/WER/cosine or the appropriate diagnostic?
5. **Decision:** retained, rejected, stopped, superseded, or still pending?
6. **Limitations:** what does the evidence not establish?
7. **Evidence:** where are the implementation, saved outputs, audits, and detailed notes?

The appendices below contain the complete variant and checkpoint tables. The
register summarizes families rather than treating every setup retry or saved
directory as a successful new experiment.
Some experiments address more than one question and are cross-referenced in
multiple appendices; those references do not represent additional executions.

| Family | Work performed | Detailed results and decisions |
|---|---|---|
| Original 0.6B audit | Established the actual LoRA recipe, precision behavior, and checkpoint-selection assumptions | [Training evidence](evidence/training.md), [original audit](../../0.6b-training-analysis.md) |
| Optimization/precision | Controlled LoRA-related comparisons, FP32 parameters/moments, and export/merge investigation | [Training evidence](evidence/training.md), [precision investigation](../../0.6b-merge-precision.md) |
| Epoch continuation | Fresh FP32 epoch, optimizer-preserving continuation, and further epochs until regression evidence | [Training evidence](evidence/training.md), [continuation record](../../0.6b-until-overfit.md) |
| Test and voice investigation | Compared checkpoints on development/test scopes and investigated base-versus-adapted reference voice | [Training evidence](evidence/training.md), [voice investigation](../../0.6b-voice-reference-analysis.md) |
| Targeted pronunciation adaptation | Frozen halfway backbone, text-only adaptation, and reference-protected text routing | [Training evidence](evidence/training.md), [protected training](../../0.6b-reference-protected-training.md) |
| Joint preferences | Generated-trajectory preference learning with pronunciation and voice gates | [Alignment evidence](evidence/alignment.md), [joint pilot](../../0.6b-joint-preference-training.md) |
| Speaker conditioning | Per-reference adaptation, broader speaker preferences, refreshed trajectories, and centered speaker updates | [Alignment evidence](evidence/alignment.md) |
| Reference routing | Reference-prefix, acoustic-only, and speaker-plus-acoustic routing ablations | [Alignment evidence](evidence/alignment.md) |
| Middle attention | Bounded Q/V LoRA with the earlier speaker adapter frozen | [Alignment evidence](evidence/alignment.md), [attention experiment](../../0.6b-middle-attention-preferences.md) |
| Quality-aware preferences | Filtered quality comparisons and gradient-calibrated supervised-loss weighting | [Alignment evidence](evidence/alignment.md) |
| Sampling | Controlled residual-code sampling temperature comparisons | [Alignment evidence](evidence/alignment.md), [sampling record](../../0.6b-residual-sampling-calibration.md) |
| Direct voice gradients | Differentiable codec feasibility, memory-separated policy gradients, and bounded direct-voice candidates | [Voice/supervision evidence](evidence/voice-and-supervision.md) |
| Reference profiles | Reference-specific conditioning profiles, selection, and joint preference adaptation | [Voice/supervision evidence](evidence/voice-and-supervision.md) |
| Weight-space work | Actual subspace/transfer diagnostics and transfer trials, distinguished from unimplemented WSP proposals | [Voice/supervision evidence](evidence/voice-and-supervision.md) |
| PAD and full-panel routing | Compared prefix/continuation PAD policies and checked the complete original114 panel | [Voice/supervision evidence](evidence/voice-and-supervision.md), [full-panel findings](../../0.6b-attention-pad-full-panel-findings.md) |
| Real paired supervision | Fresh native features, masked real-speech LoRA, numerical replay diagnosis, and the completed 32-update v2 pilot | [Voice/supervision evidence](evidence/voice-and-supervision.md), [current protocol notes](../../0.6b-reference-masked-supervision.md) |
| Observability and reporting | Fixed probes, independent score audits, source/artifact binding, data diagnostics, and verified listening-example sharing | Sections 4–7 below and the evidence appendices |

## 3. Current comparable results and active experiment

### Original114 comparisons

These two policies were evaluated on the same complete original panel. The
earlier centered-step-16/full-PAD policy was selected as the frozen starting
point for the new supervised work. This is a practical selection based on the
remaining joint-goal gaps, not a claim of dominance in every individual metric.

| Policy | Group | Recordings | CER | WER | Cosine | Verdict |
|---|---|---:|---:|---:|---:|---|
| Centered-step-16/full-PAD | Development | 88 | 4.222615% | 10.730594% | 0.984226523 | CER fails |
| Centered-step-16/full-PAD | Custom | 26 | 3.279632% | 8.270677% | 0.983555874 | Both fail |
| Attention-step-4/full-PAD | Development | 88 | 4.399293% | 11.986301% | 0.984458801 | CER fails |
| Attention-step-4/full-PAD | Custom | 26 | 4.085155% | 11.278195% | 0.982749196 | Both fail |

The attention/PAD experiment improved custom scores relative to its own
half-PAD control but did not replace the earlier reference-preserving result.
All development recordings reproduced its baseline exactly, so that
custom-only change could not repair development pronunciation. Its complete
audited failure directly motivated a shared adapter capable of changing
development outputs.

### Real-speech supervision v1 and v2

**Purpose.** Test real, distinct, same-speaker reference/target supervision after
generated-preference and routing experiments failed to meet the complete goal.

**Changes.** Train only new rank-4 Q/V LoRA adapters in main-decoder layers
10–17. Preserve the selected starting policy, old adapters, speaker encoder,
and codec. Exclude reference/control rows from direct new-adapter updates while
retaining indirect gradients through those contexts. Replay every real pair in
both full-reference and half-reference modes, with separate prefix and
continuation PAD behavior matching native inference.

**Setup.** The audited eligible population contains 2,718 real pairs from 47
fitting speakers. The bounded schedule has 256 pair visits, with 204 distinct
fitting/probe pairs requiring fresh features; 32 calibration cases from 16
different speakers are excluded from new fitting. The feature audit covers 324
fresh native assets including the supplied reference. These calibration
speakers are not claimed to be unseen to historical base-model training.

The registered recipe uses FP32 parameters/forward computation/optimizer
moments, constant learning rate 2e-5, no weight decay or dropout, gradient norm
cap 1, eight distinct-speaker pairs per update, and equal weighting of both
reference modes. The per-pair objective is main-codebook NLL/(T+1) plus
0.3 times residual-codebook NLL/(15T). Checkpoints are 0, 8, 16, and 32.

**V1 outcome.** Stopped before optimizer updates because cached native and
batched teacher residual logits exceeded an overly strict elementwise
comparison. Both historical waveform controls reproduced exactly. A separate
diagnostic found exact conditioning, maximum residual-logit discrepancy
0.000200272, tiny probability differences, and no residual argmax disagreement
in its 1,080 comparisons. This supported a numerical explanation, not permission
to ignore the failed run or modify its registered files.

**V2 validation.** A new registration fixed probability-sensitive and raw-logit
bounds before new production checks. Five fitting-only probes across both
reference modes and zero/nonzero new-adapter states produced 20 controls. All
passed, both historical waveform controls reproduced exactly, CUDA memory
preflight passed, and original/new parameter states were restored exactly.
These are implementation results, not speech-quality improvements.

**V2 calibration baseline.** All 80 recordings completed and passed the
independent audit, with zero truncation and zero cosine recomputation error:

| Group | Recordings | CER | WER | Cosine |
|---|---:|---:|---:|---:|
| Development, 16 speakers | 64 | 3.905519% | 10.182371% | 0.982569867 |
| Custom | 16 | 1.773836% | 3.623188% | 0.984323889 |

These measurements are on a different panel from original114. They must not be
used to claim the original goal has passed or to rank a checkpoint across
incompatible test scopes.

**Checkpoint 8.** Eight updates completed. The fixed fitting objective moved
from 3.394105673 to 3.393595599 and calibration objective from 3.426517192 to
3.425955288. Checkpoint 8 subsequently passed its complete native/artifact and
independent cosine audit, with no truncated recordings:

| Checkpoint 8, calibration80 | CER | WER | Cosine |
|---|---:|---:|---:|
| Development | 3.858746% | 10.486322% | 0.982577998 |
| Custom | 0.997783% | 3.623188% | 0.983990882 |

Development character errors decreased from 167 to 165 out of 4,276 characters,
while word errors increased from 67 to 69 out of 658 words. Custom character
errors decreased from 16 to 9 out of 902 characters, with word errors unchanged
at 5/138. Only two custom ASR outputs changed: one recording improved and one
worsened. Seventeen development and four custom waveforms remained exactly
identical to baseline. These descriptive paired diagnostics preserve every
case and seed; they are not estimates of statistical significance.

The custom cosine decrease crosses below the strict 0.984 threshold. The
checkpoint is neither calibration-qualified nor an improvement on all four
goal metrics. No new-best example or global promotion was triggered. Audit SHA:
`39a6abe9e7c634ab0ae09db82fd413e05fda3796ac50603d54fe73809260b115`.

A separate source/checkpoint review found no evident stalled optimizer or extra
loss-normalization defect: all 32 new factor tensors have finite nonzero Adam
moments at step 8, both factors changed, and no update was gradient-clipped.
Maximum effective projection movement is only 0.027755%. About 95.6% of the
combined calibration teacher-loss reduction comes from half-reference mode;
the full-reference mode used for development improves by only 0.000049323.
Only 64 distinct pairs have been visited, including three of the sixteen fixed
fitting probes. That probe is therefore mostly a fitting-population diagnostic
at this stage, not loss on just-optimized examples. Two small decreases in the
fixed losses do not establish convergence, underfitting, or freedom from overfit.

**Checkpoint 16.** The complete 80-recording audit passed with no truncation.
Quality regressed despite further fixed-loss decreases (fitting 3.392970063;
calibration 3.425355233):

| Checkpoint 16, calibration80 | CER | WER | Cosine |
|---|---:|---:|---:|
| Development | 4.747428% | 12.613982% | 0.982527697 |
| Custom | 2.771619% | 10.144928% | 0.983500279 |

Development has 203/4,276 character edits and 83/658 word edits; custom has
25/902 and 14/138 respectively. Both baseline and checkpoint 8 are better on
all four group CER/cosine measures. Seven development and one custom waveform
remain identical to baseline. This is a measured generated-speech regression;
the two-mode teacher-loss decrease is not a reason to promote it or assume
that more training must help. The audit SHA is
`a06384f2d7ec0f967711ad9f4190456f52e1380889596a9ee9eb833c5d477b55`.

**Checkpoint 32 and terminal decision.** All 80 recordings passed the audit,
with no truncation, finite independently recomputed cosines and zero
recomputation error:

| Checkpoint 32, calibration80 | CER | WER | Cosine |
|---|---:|---:|---:|
| Development | 4.607109% | 12.158055% | 0.982479587 |
| Custom | 0.443459% | 2.173913% | 0.984215401 |

Against baseline, development edits rose from 167 to 197/4,276 characters and
67 to 80/658 words. Custom edits fell from 16 to 4/902 characters and 5 to
3/138 words. Twelve of sixteen custom ASR outputs are unchanged; two recordings
improved CER and two worsened. Development has 17 better, 26 equal and 21 worse
CER recordings; 21 ASR outputs and one WAV are unchanged. Custom cosine
recovers above the threshold after step16 but remains below baseline. These
paired observations are descriptive, not significance tests.

The final paired-error review finds that 11 of the 12 net custom character
edits removed come from one case at seed 2027. Five development texts contribute
103/197 final character edits, rising from 67 at baseline; the remaining 27 texts
improve from 100 to 94. Those five therefore account for the net development
regression. This concentration does not justify removing cases or fitting on
these evaluation texts.

Fixed teacher objectives continued down to fitting **3.391651064** and
calibration **3.423979275**. Generated development speech nevertheless remains
worse than baseline on CER, WER and cosine. The custom recovery prevents a
simple claim that all quality monotonically worsens with training; the loss
decrease establishes neither generalization nor the cause of the tradeoff.

Audit SHA: `9fd25e34176efe729c5c2ba5b949457b89fb96c14ba3c845eee0953de7bb0baf`.
The run ended normally with `maximum_updates_reached`; authoritative process
33720 closed with exit code 0. The complete audited history is 0/8/16/32.
No checkpoint passed both groups, no original114 confirmation ran, and no model
was promoted. The sharing helper reports `no_unshared_audited_improvement`.

The subsequent completed investigation measured the
main-CE, residual-CE and conditional teacher-KL gradients on fitting pairs only,
with zero optimizer updates. The fixed panel combines sixteen existing fitting
probes and fifteen previously flagged acoustic pairs, deduplicated to thirty.
Its enriched composition cannot estimate corpus-wide prevalence. This is a
mechanism diagnostic, not a chosen new loss or proof of better speech; native
execution completed successfully after the predecessor and natural-audio
control processes closed. An independent CPU audit of saved vectors passed. The
[new helper](../../../training/reference_masked_supervision_gradient_probe.py)
passed nine synthetic CPU tests and independent implementation review before
the native execution. On the fixed 16 fitting probes averaged across both modes,
main/residual CE gradients have cosine **+0.120890**; the 0.3-weighted residual
gradient norm is **0.080428** of the main norm, and the combined task/main
gradient cosine is **0.996889**. This gives no support for residual-gradient
dominance or removing residual CE on this panel. These are local raw gradients,
not Adam updates or a generated-speech guarantee. The registered broader-data pilot is tracked separately.
Independent audit SHA: `2bf2efa4fbbbff80af20aff5e479a43cd50b42a42bec3b53d1bacd2f122ac03e`.
The separate saved-vector cross-mode check finds positive fixed 16 full/half
aggregate cosines: main CE **0.822046243**, residual CE **0.889591924**, task
**0.814485713**. It gives no evidence of aggregate mode opposition at this
checkpoint and does not predict Adam updates or speech quality.
The
[literature follow-up](../../0.6b-joint-goal-literature.md#follow-up-after-masked-supervision-checkpoint-16--november-11-2026)
explains why published on-policy distillation is not a direct replacement here.

**Completed conditioning comparison: all-half route rejected.** The inference-only
2×2 held checkpoints 0/32, references and seeds 2026/2027 fixed. It audited all
128 new half-route development WAVs and reused 160 recordings: 128 full-route
development controls plus 16 unchanged custom recordings at each endpoint.
Custom evidence is shared, not newly generated or counted twice.

| Development64 | Character edits / 4,276 | CER | Word edits / 658 | WER | Cosine |
|---|---:|---:|---:|---:|---:|
| Step 0, full reference | 167 | 3.905519% | 67 | 10.182371% | 0.982569867 |
| Step 0, half reference | 248 | 5.799813% | 97 | 14.741641% | 0.981883879 |
| Step 32, full reference | 197 | 4.607109% | 80 | 12.158055% | 0.982479587 |
| Step 32, half reference | 222 | 5.191768% | 95 | 14.437690% | 0.981689340 |

Both new arms passed complete native audits with zero truncation and zero cosine
recomputation discrepancy. Independent scalar review matched all 256 development
hypotheses, edit counts, case/reference metadata and speaker-macro averages.
At step 32, switching full→half adds 25 character and 15 word edits; 22 recordings
improve CER, 20 tie and 22 worsen. Half routing is worse on every aggregate
metric at both checkpoints. From0→32 CER improves in half mode (248→222 errors)
but worsens in full mode (167→197); this fixed-panel interaction is not a general
causal explanation or evidence that half routing is preferable.

Original process 23429 closed with exit 0. No arm met both-group criteria, no new
best was selected, and original114 confirmation was not launched. Protocol SHA:
`faaca2f70c9b47dd3a3373c5de0605a8a6d2bd1efe0cf2c0f952cba50c024ee9`;
final results SHA: `e07ada42a68dbcca15e7acce2523b10b00ef33bde8e0b0eda9482059ab1cccc6`.
The external `masked-routing-final-findings-v1.md` preserves both endpoints,
paired seed/speaker counts and source hashes. The earlier arm0 note remains a
historical interim review. The fitting-only KV diagnostic completed with no waveform effect. A separate
original-base KV calibration80 screen completed and was rejected for development regression. The registered broader-data pilot is tracked separately.

**Completed KV diagnostic, fitting only.** Two fitting probes × two reference
routes × four arms produced 16 recordings (12 comparisons and four teacher
self-graft controls), at seed 2026. All four student32/teacher-KV-graft conditions
had exactly identical WAV hashes, original-reference cosine, ASR hypotheses and
CER/WER. Teacher identity controls passed; the structural audit independently
recomputed all 16 cosines with zero error. Original process 35473 closed with
exit 0. This is `diagnostic_no_waveform_effect` on four fitting conditions, not a
voice-quality improvement or generalization result. The donor was the already
adapted checkpoint 0, so this no-effect finding does not establish restoration
of voice differences inherited from earlier adaptation. Results SHA:
`91bfa85483a94ebab2c8e5e2fc108bfc6e22e366302e4b4a1a89b5ad53f0a266`.
Independent saved-artifact review also matched all 16 codec-group/EOS tensors
for the four student/graft and four teacher/self-graft pairs, and independently
recomputed all 16 ASR edit counts. Receipt SHA:
`3ff7b1defe1be2efc9837d0d943d110e85ac65e0e6099b498f565f9e9569856d`
(`masked-kv-final-findings-v1.json`). This peer check did not rerun native models,
ASR or embedding extraction.

The original-base KV calibration80 follow-up completed and failed the joint criteria;
no original114 confirmation or new best followed.

The original-base KV screen was launched as session **14712**, with protocol
`ca108055567fe14bc34ea28fcccf76b2654f01ce33c83c284c148c20a3d2cb84`.
Its sequential stages completed: **8 controls → 40 original-base prefill donors
→ 80 calibration recordings**, followed by native and scalar audits. Original
session 14712 closed with exit 0; quality criteria failed despite valid artifacts. The [launch receipt](C:/Users/Khan/.codex/visualizations/2026/11/09/01a1bf77-0b17-7391-9781-f859f1fbb0b1/masked-supervision-base-kv-v1-launch-handle.json) records a live handle, not process closure.

A separate [all-half-route original114 helper](../../../training/reference_masked_supervision_routing_confirmation.py)
is prepared and independently reviewed, with eight pure tests passing; it is
not registered or launched. Eligibility was fixed before new-arm scores:
only checkpoint 32 may proceed, because checkpoint 0 preserves the known failed
original114 custom branch. Both ablation arms must first pass their complete
audits, checkpoint 32 must meet both groups' strict calibration80 thresholds,
and authoritative process 23429 exit 0 must be recorded. Confirmation then requires
114 freshly generated recordings, complete native audits and its own successful
process closure before completion. The older confirmation gate is unchanged;
neither preparation nor calibration alone establishes the goal. The completed
checkpoint 32 routing failure prevents this conditional confirmation from running.

## 4. Engineering changes and observability

| Area | Implemented capability | Representative code |
|---|---|---|
| Training/continuation | Explicit LoRA training, optimizer-preserving continuation, fixed recipe/configuration checks | [engine](../../../training/engine.py), [continuation](../../../training/continuation.py), [optimization](../../../training/optimization.py) |
| Sequence correctness | Native conditioning and codec target alignment, full residual groups and EOS coverage | [sequence](../../../training/sequence.py), [masked supervision](../../../training/reference_masked_supervision.py) |
| Loss/health telemetry | Fixed fitting/validation probes, per-codebook losses, gradient norms/clipping, memory, elapsed time, update counters and adapter movement | [observability](../../../training/observability.py), [overfit](../../../training/overfit.py), [current trainer](../../../training/reference_masked_supervision_training_v2.py) |
| Speech measurement | CER/WER, source-preserving normalization, group/speaker/seed aggregation and generated speech audits | [evaluation](../../../training/evaluate.py), [scoring](../../../scripts/score_latest_audio.py), [current speech evaluator](../../../training/reference_masked_supervision_speech_v2.py) |
| Voice behavior | Speaker diagnostics, reference conditioning ablations, original-reference cosine and frozen-state checks | [speaker diagnostics](../../../training/speaker_diagnostics.py), [reference routing](../../../training/reference_prefix_routing.py) |
| Native feature provenance | Fresh FP32 reference speaker features, codec/ID contracts, asset hashes and source bindings | [feature pipeline](../../../training/reference_masked_supervision_features.py) |
| Numerical validation | Fixed raw/probability gates, numerical tie handling, rejection of meaningful logit/label/frame mismatches | [parity checks](../../../training/reference_masked_supervision_parity.py) |
| Complete-goal confirmation | Separate unchanged original114 runner, completed-parent gating and independent audio/native audits | [confirmation](../../../training/reference_masked_supervision_confirmation.py) |

The current telemetry deliberately separates fitting diagnostics from generated
quality. A loss decrease is not treated as proof of less underfitting, absence
of overfitting, or better pronunciation. The generated-speech stopping rule
checks both groups and stops after repeated Pareto regressions or the registered
maximum, while a qualifying positive checkpoint requests broader confirmation.

Active execution sources, test logs, protocols, and drivers are immutable once
a run starts. Fixes use a new registered version. The original process handle
is authoritative; a timeout or an old status timestamp does not justify a
duplicate job. Full native/model jobs are serialized to respect local memory
limits. Small read-only diagnostics and isolated tiny-model tests are distinct
from model training or native generation.

## 5. Data-quality and metric diagnostics

The cached natural-pair diagnostic measured speaker-macro cosine 0.984155 over
204 selected fitting pairs/47 labels and 0.986877 over 32 calibration pairs/16
labels. These are real-recording controls, not generated model results. No
natural paired target exists for the supplied custom reference in this pilot.

Fifteen fitting pairs scored below at least one representative reference with a
different supplied speaker label. A detailed follow-up found broader acoustic
differences within two labels, involving targets as well as references. The
flagged pairs account for 15/256 scheduled visits: three through update 8, six
through update 16, and fifteen through update 32. The 27 directly involved
stored WAVs showed no current saturation/near-clipping; preprocessing could
still conceal earlier artifacts.

This does not identify different people or prove incorrect speaker labels.
Fitting and calibration rank tests also have different numbers of alternative
labels. The bounded current pilot remains unchanged. A larger follow-up should
audit these recording groups rather than assume metadata alone establishes
consistent acoustic conditioning. No automatic pruning or relabeling occurred.

The initial artifact search found no matching natural-audio ASR scores for the
current 32 calibration targets. The older 44-recording control shares zero
target IDs/audio hashes and could not substitute. After the pilot closed, the
separate exact-32 control completed successfully with the same pinned recognizer,
decoding, WAVs and transcripts: **7/2,138 character edits (CER 0.327409%)** and
**4/329 word edits (WER 1.215805%)**, across 16 speakers and 157.948 seconds.
Its receipt is `verified_diagnostic`; process 69898 closed with exit code 0.
Verification SHA: `32b90e0c985842039ccfbdf7fa8743e6f206d0e1bd43aeb17e1a8750939f02ea`.
This provides an exact-panel recognizer control, not a custom-speaker paired
human control or an estimate to subtract from generated CER. No thresholds,
cases or training data changed. The detailed provenance is in the
[supervision note](../../0.6b-reference-masked-supervision.md#completed-natural-audio-asr-control).

A separate execution-coverage audit found that broad half-reference-prefix
routing with full continuation PAD has not been tested on generated development
speech. Earlier broad-prefix development tests retained half PAD; later full-PAD
tests enabled that routing only for the supplied reference. Prefix-specific
interpolation was tested only at strength 0.5. This identifies an untested
conditioning combination, not an improvement. Universal dispatch at unchanged
checkpoint zero would retain the known custom114 failure, so a development
gain alone could not satisfy the complete goal.

## 6. Listening examples and improvement claims

The user requested an example whenever a better result appears. The external
sharing helper uses only complete independently audited checkpoints, checks
audio hashes, and compares the same registered custom text at seed 2026.
It reports group-level metrics separately from the chosen clip's metrics.

A new calibration best requires a genuine improvement in the measured quality:
either crossing the two-group quality/termination requirements from a baseline
that failed them, or being no worse in all four group CER/cosine metrics and
strictly better in at least one. WER is reported but does not redefine the
requested criterion. A passing step-zero baseline is not falsely labeled a
quality failure merely because it is ineligible for training progression.

The helper drafts recommendations; it does not send messages or mark delivery.
Only audio actually shared in the conversation may enter a notification ledger.
If the fixed clip regresses while aggregate results improve, that tradeoff is
reported rather than replacing it with a favorable clip. Calibration evidence
does not automatically promote the global model.

## 7. Evidence and reproduction map

Human-readable notes and source code belong in the repository. Recorded audio,
datasets, weights, raw generated reports, run logs, feature caches, and listening
pages stay outside Git. The primary local artifact root for this workstream is:

`C:/Users/Khan/.codex/visualizations/2026/11/09/01a1bf77-0b17-7391-9781-f859f1fbb0b1`

| Current evidence | Location beneath that root |
|---|---|
| Original114 panel | `joint-goal-confirmation/protocol.json` |
| Earlier full-panel policy | `reference-prefix-generation-pad-confirmation/` |
| Later attention/PAD full-panel comparison | `reference-prefix-lora-pad/` |
| Fresh paired data and schedule | `reference-masked-supervision-data-v1/`, `reference-masked-supervision-plan-v1.json` |
| Fresh native feature audit | `reference-masked-supervision-features-v1/verification.json` |
| Preserved failed attempt | `reference-masked-supervision-v1/`, `reference-masked-supervision-v1-status.json` |
| Numerical diagnosis | `masked-native-replay-diagnosis-v1.json` |
| Completed training and all checkpoint evidence | `reference-masked-supervision-v2/`, including `verification-000032.json` |
| Parent orchestration and status | `run-reference-masked-supervision-v2.py`, `reference-masked-supervision-v2-status.json` |
| Independently checked dashboard | `reference-masked-supervision-v2-observability.html` and `.json` |
| Natural-pair diagnosis | `masked-natural-voice-controls-v1.json`, `masked-fitting-voice-outliers-v1.json` and interpretation note |
| Listening recommendation drafts | `reference-masked-supervision-improvement.json` and `.md` |
| Paired checkpoint diagnostics | `analyze-masked-checkpoint-changes-v1.py`, `masked-checkpoint-changes-v1.json` |
| Update-scale/normalization review | `masked-supervision-update-scale-review-v1.md` |
| Completed natural-audio ASR control | `masked-calibration-natural-asr-v1/{protocol,results,verification}.json`; historical preparation: `masked-calibration-natural-asr-preparation-v1.md`; driver: `run-masked-calibration-natural-asr-v1.py` |
| Completed gradient diagnostic | `masked-supervision-gradient-probe-v1/results.json`, `masked-gradient-probe-independent-audit-v1.json`; historical rationale: `masked-supervision-next-decision-review-v1.md`; driver: `run-masked-supervision-gradient-probe-v1.py` |
| Completed fitting-only KV diagnostic | `masked-supervision-kv-probe-v1/{results,structural-verification,scored}.json`, `masked-supervision-kv-probe-v1-process-closed.json`; four exact student/graft waveform matches |
| Final paired errors and cross-mode gradient geometry | `masked-supervision-final-error-review-v1.md`, `masked-gradient-cross-mode-v1.json` |
| Prepared all-half-route original114 confirmation | `masked-supervision-routing-confirmation-preparation-v1.md`, `run-masked-supervision-routing-confirmation-v1.py`; unregistered and unlaunched |
| Reference-routing coverage audit | `reference-prefix-pad-development-coverage-v1.md` |
| Completed routing comparison | `masked-routing-final-findings-v1.md`, `masked-supervision-routing-ablation-v1/{results,verification-000000,verification-000032}.json`; historical interim note: `masked-routing-arm0-findings-v1.md` |

Current registered training protocol SHA-256:
`89463810f5f9fc5d75d3eabf16144e30872aa1488b0605418f4f604abee12651`.
Baseline80 audit SHA-256:
`c6d8e5e7390a10b50eb391513998d727afc392d86bf90d6500bbf4eaf140c72b`.

The appendices preserve older evidence paths, exact table values and missing
measurements. Local artifact paths are evidence locations, not promises that
private weights or recordings will be downloadable from GitHub. Reproduction
requires the corresponding authorized assets and registered environment.

## 8. Maintenance, reproduction, and publication

The user authorized pushing after the goal is reached and requested continued
ownership of code quality and documentation. No push is performed or claimed
before the complete goal audit succeeds.

The publication work will include:

1. Finalize the summary, complete experiment tables, chosen policy/configuration, and known limitations with the actual final results.
2. Review the code changes, distinguish supported entry points from historical experimental runners, and document environment/data/model prerequisites.
3. Run the checks appropriate to the final changes and inspect the staged tree for unintended files, credentials, audio, datasets, weights, and generated artifacts.
4. Preserve historical execution evidence; do not silently rewrite code or logs needed to reproduce past results.
5. Verify that the authenticated GitHub account is **m-khanddorj**, then push the reviewed code and documentation to the project remote. Do not use another account or change global authentication to bypass a mismatch.

The current worktree contains uncommitted experiment code and documentation;
their presence is not a published release. New results will update this report
at verified milestones. Future maintenance should preserve the same separation
between hypotheses, executed experiments, measured outcomes, and promoted
behavior.

Publication preparation now separates the
[data workflow](../../../.github/workflows/data-preparation.yml) from the
[CPU training workflow](../../../.github/workflows/training-cpu.yml). The former
retains its original core/audio matrix and all data tests. The latter installs
the training dependencies and explicitly excludes only two retained-artifact
methods; unexpected training skips and collection failures are errors.
Existing registered experiment sources/tests remain unchanged.

Seven new test-selector checks passed. Metadata-only training collection
selected exactly 274 unique methods, excluding the two retained-artifact
methods and collecting imported helper classes only in their defining module.
No training test methods or fixtures ran in this collection check.
The local data lane passed 69 of 71 tests;
two localhost tests hit sandbox socket restrictions. A subsequent unrestricted
localhost recheck passed all five playground tests, including those two,
without changing the tests. This is evidence from two local runs, not a claim
that a clean Linux training CI run has passed. The
[CI inventory](evidence/ci-portability.md) records the implementation, exact
coverage exclusions, logs, remaining wheel-packaging issue, and unrun checks.

### Fitting coverage: balanced pilot, narrow inherited population

The original **28,001 rows / 435 speakers** were narrowed by a deterministic
speaker-hash allocation for an earlier preference experiment: 64 speakers,
with 16 reserved for validation and 48 for fitting. Taking **all rows** from
those 48 speakers yielded 2,800 rows; exclusions and immutable-reference
closure left **2,718 pairs / 47 speakers**. This was neither the first 2,800
rows nor a random row sample. The completed pilot already balanced all 47
speakers: **256 visits to 201 unique pairs**, with 21 speakers visited six
times and 26 five times. The limitation was breadth and repeated small-speaker
queues, not failure to balance the existing speaker set.

A metadata-only broader candidate pool has **24,369 pairs / 414 speakers**.
A deterministic schedule for the same proposed 32-update budget is now prepared
and verified: **256 different pairs / 256 speakers**, with 512 target/reference
assets. All 16 fitting probes are within those 256 scheduled pairs. Its 52-asset overlap with the old feature cache is a coverage statistic,
not readiness evidence: all **512 fitting features are planned fresh**, while
the existing 55 real calibration assets and one custom reference stay unchanged.
Selected-WAV preparation and independent audit completed in session **51327**,
exit 0: all **512 selected WAVs** passed byte/hash, PCM, header and duration
checks, followed by rehash verification. The full 24,369-pair pool remains
metadata-only. All 512 fresh fitting features are now audited and complete;
session **45998** closed with exit 0. The broader-data-only driver completed checkpoint 16 calibration80 and has checkpoint 32 underway as session
**41410**. Checkpoint 16 is fully audited but does not qualify; no confirmation gate changed.

Sources: [coverage review](C:/Users/Khan/.codex/visualizations/2026/11/09/01a1bf77-0b17-7391-9781-f859f1fbb0b1/masked-supervision-broader-coverage-review-v1.md)
and [exact selection-origin reproduction](C:/Users/Khan/.codex/visualizations/2026/11/09/01a1bf77-0b17-7391-9781-f859f1fbb0b1/masked-supervision-selection-origin-addendum-v1.md).

The exclusion review also binds the **previously inspected TEST30** cohort
[manifest](C:/Users/Khan/.codex/visualizations/2026/11/09/01a1beb0-7b7b-7f73-993e-71ad9a8eefd3/hojil-17b-recovery/cohorts/unseen-test.jsonl),
SHA256 `187c8dd03b560e90e62594e9ecc6d452f3985d3c241d64efadb2d1fb6b54e112`.
It is not an untouched final test set; these metadata checks do not establish
physical audio or fresh-feature readiness for the proposed wider pool.
The [independent metadata review receipt](C:/Users/Khan/.codex/visualizations/2026/11/09/01a1bf77-0b17-7391-9781-f859f1fbb0b1/masked-supervision-broader-coverage-independent-review-v1.json)
records zero broader-pool speaker overlap and zero target/reference transcript
or source-identity conflicts with TEST30. It records a successful prior
stdout-only reconstruction; no standalone re-executable audit script is claimed.

Completed selected-data evidence: [protocol](C:/Users/Khan/.codex/visualizations/2026/11/09/01a1bf77-0b17-7391-9781-f859f1fbb0b1/reference-masked-supervision-broad-data-v1/protocol.json),
[audit](C:/Users/Khan/.codex/visualizations/2026/11/09/01a1bf77-0b17-7391-9781-f859f1fbb0b1/reference-masked-supervision-broad-data-v1/audit.json), and
[authoritative process closure](C:/Users/Khan/.codex/visualizations/2026/11/09/01a1bf77-0b17-7391-9781-f859f1fbb0b1/reference-masked-supervision-broad-data-v1-process-closed.json).
The audit SHA256 is `d6af64860dc1e93e9568fabd3471890894ebcd347dbe302bc6a126b54ecb299e`.

### Original-base KV: complete calibration80 rejection

The fixed centered16 checkpoint-0 student received original-base protected-prefix
K/V before native normalization/RoPE/cache. All 80 candidate recordings completed
with no truncation and zero independent cosine-recomputation error. The baseline
80 recordings were reused with exact provenance; this is not a new original114 run.

| Calibration group | Baseline CER | Base-KV CER | Baseline WER | Base-KV WER | Baseline cosine | Base-KV cosine |
|---|---:|---:|---:|---:|---:|---:|
| development | 3.905519% | 12.956034% | 10.182371% | 26.139818% | 0.982569867 | 0.980199310 |
| custom | 1.773836% | 2.549889% | 3.623188% | 8.695652% | 0.984323889 | 0.984688181 |

Development CER worsened on both seeds, 21 of 32 texts and 13 of 16 speakers.
No candidate WAV matched its baseline. Custom voice cosine increased, but
custom CER/WER also worsened. Reject this intervention: no new best, no better
listening example and no original114 qualification. The development failure
does not support prioritizing epoch 3 plus base-KV as the next fallback.

[Independent complete80 review](C:/Users/Khan/.codex/visualizations/2026/11/09/01a1bf77-0b17-7391-9781-f859f1fbb0b1/base-kv-final-independent-review-v1.json) independently rescored all
160 hypotheses and bound native artifacts and [process closure](C:/Users/Khan/.codex/visualizations/2026/11/09/01a1bf77-0b17-7391-9781-f859f1fbb0b1/masked-supervision-base-kv-v1-process-closed.json).
Results SHA256: `29af2c754dd340be50674d2aebb5ed30a6a484d87e31038a4ad82f7ca8145e61`.

### Natural reference similarity and the next data stage

For the exact32 real calibration targets/reference pairs, audited cached FP32
speaker vectors give recording mean and speaker macro **0.986877035**. Pair
minimum/median/maximum are **0.966414392 / 0.987650990 / 0.995668888**;
**22 pairs exceed 0.984 and 10 fall below it**. Twelve of 16 speaker means
exceed 0.984. The generated baseline development macro is **0.982569867**.
This descriptive natural-audio comparison is not a ceiling or justification
to change the threshold. No paired natural custom target exists.
[Detailed diagnostic](C:/Users/Khan/.codex/visualizations/2026/11/09/01a1bf77-0b17-7391-9781-f859f1fbb0b1/natural-calibration32-reference-cosine-v1.md)
includes every pair/speaker and checked file/tensor provenance. Independent
review confirmed scalar arithmetic and file hashes; it did not rerun tensor
cosines or native extraction.

Fresh512 feature extraction is registered under protocol `01247c5a5176b0938d57414c4fe248a24531ba83e0acea3b045127083065963e`
and completed as session **45998**, exit 0 ([closure receipt](C:/Users/Khan/.codex/visualizations/2026/11/09/01a1bf77-0b17-7391-9781-f859f1fbb0b1/reference-masked-supervision-broad-features-v1-process-closed.json)).
The [feature audit](C:/Users/Khan/.codex/visualizations/2026/11/09/01a1bf77-0b17-7391-9781-f859f1fbb0b1/reference-masked-supervision-broad-features-v1/verification.json) verified all 512 fresh
feature byte/tensor contracts, all 512 native token identities and source-WAV
rehashes, and 52 exact comparisons with prior tensors. The initial extraction
checked frozen model state; the audit did **not** re-extract every codec or
speaker feature. Zero calibration assets were extracted: the original 55 real
calibration assets plus custom reference remain separate and unchanged.
Audit SHA256: `b3ae2ac332ac82f263a1e63c29798993522464e903781324d99c9febdd71e98e`. A same-budget
**broader-data-only 32-update experiment** has completed checkpoint 16 calibration80 and has checkpoint 32 underway
as session **41410**. Eight updates and their complete speech audit are recorded below. No base-KV intervention is included.

The [broader pilot input registration](C:/Users/Khan/.codex/visualizations/2026/11/09/01a1bf77-0b17-7391-9781-f859f1fbb0b1/reference-masked-supervision-broad-training-v1-inputs.json) has SHA256
`fd8a504ab2567e4cecefc6a91e3eb91961329e31db1a501a1f9887a8b882ddec`
(preparation process 98833 exited 0). The root owns live driver session 41410.
The sequential stages are preflight, protocol preparation, initialization/native
controls and eligible baseline80 reuse, then bounded updates 8/16/32 with fresh
complete80 speech evaluation at each evaluated checkpoint. These are planned
stages, not completed measurements; existing stopping/confirmation gates remain.
The [independent saved-feature audit](C:/Users/Khan/.codex/visualizations/2026/11/09/01a1bf77-0b17-7391-9781-f859f1fbb0b1/broad-features-independent-audit-v1.json)
verified 512 saved CPU tensor contracts, file/tensor/source-WAV hashes and
52 × 4 exact prior-tensor comparisons. It did not re-extract native features.

The [launch receipt](C:/Users/Khan/.codex/visualizations/2026/11/09/01a1bf77-0b17-7391-9781-f859f1fbb0b1/reference-masked-supervision-broad-training-v1-launch.json) binds the actual live
session 41410. Memory preflight, training/speech preparation, initialization, native controls
and baseline reuse completed with exit 0. Checkpoint 8 training, generation, scoring and native audit completed; the next segment toward checkpoint 16 is running. The audited result is a tradeoff, with no new best.

The [completed memory preflight](C:/Users/Khan/.codex/visualizations/2026/11/09/01a1bf77-0b17-7391-9781-f859f1fbb0b1/reference-masked-supervision-broad-training-v1-memory.json) contains eight
measurements with finite fitting gradients, zero calibration backward calls and
zero optimizer updates. Peak reserved memory was **5,821,693,952 bytes (5.42 GiB)**.
All parameters were restored. This is memory/trainability validation, not a new
checkpoint or generated-speech quality result.

### Broader-data pilot: controls passed, checkpoint 8 audited

Original session **41410** completed the checkpoint-8 speech audit and has checkpoint 32 underway. Checkpoint 16 did not qualify for original114. [Native controls](C:/Users/Khan/.codex/visualizations/2026/11/09/01a1bf77-0b17-7391-9781-f859f1fbb0b1/reference-masked-supervision-broad-training-v1/native-verification.json)
passed two byte-exact historical WAV controls and 12 zero/nonzero × full/half
trajectory-parity checks. New weights and CPU RNG were restored. The
[baseline receipt](C:/Users/Khan/.codex/visualizations/2026/11/09/01a1bf77-0b17-7391-9781-f859f1fbb0b1/reference-masked-supervision-broad-training-v1/verification-000000.json) explicitly
records **0 fresh / 80 shared recordings**, with the original audited checkpoint-0
provenance; those scores are not a new independent repeat. All 32 calibration
pairs × two modes have exactly equal saved teacher observations and summary
versus old step 0 (objective **3.426517192274332**). This is a
[calibration-only comparison](C:/Users/Khan/.codex/visualizations/2026/11/09/01a1bf77-0b17-7391-9781-f859f1fbb0b1/broad-training-zero-teacher-comparison-v1.json);
no cross-run fitting-loss comparison or speech improvement follows.
Training protocol SHA256: `ab7c2e7da83a76e7da83a9765e015f778f0db4cc0ccf77cfefc1f7968d65f03d`.

### Conditional literature options, not selected experiments

If the completed broader-data pilot still improves teacher loss without adequate
speech, first consider a small fitting-only fresh-rollout diagnostic before
any new objective. [Qwen3-TTS post-training](https://arxiv.org/html/2601.15621v1),
[GLM-TTS joint rewards](https://arxiv.org/html/2512.14291v1) and
[GSPO](https://arxiv.org/html/2507.18071v2) motivate testing refreshed generated
trajectories, but do not supply a validated Mongolian 0.6B recipe. Require usable
within-context CER/voice variation, absolute quality support, and exact sampler
likelihood/replay controls. This would differ from the failed fixed-pool DPO
experiments; their failures still argue for a small test.

A lower-priority option is a verified pronunciation annotation frontend, only
after qualified Mongolian listening/reading labels confirm recurring lexical
errors. [UtterTune](https://arxiv.org/html/2508.09767v2) improved targeted
Japanese accent control but worsened general CER; it is not a voice-preservation
guarantee. No validated Mongolian G2P resource or new token/layout parity is
assumed. [Pronunciation-oriented RL](https://arxiv.org/html/2610.07575v1) also
warns that pronunciation rewards can encourage elongated outputs.

Neither option is implemented, selected or registered. Do not repeat the failed
base-KV intervention or add arbitrary KL/residual-loss changes. Same-context
teacher KL starts without an improvement signal; the
[reference-privileged OPSD setting](https://arxiv.org/html/2610.17838v1) differs
from this already reference-conditioned student. The unchanged calibration80
and original114 gates remain; a current-run pass takes confirmation priority.
[Full conditional review and limitations](C:/Users/Khan/.codex/visualizations/2026/11/09/01a1bf77-0b17-7391-9781-f859f1fbb0b1/pronunciation-speaker-literature-followup-2026-11-11-v1.md).

### Broader checkpoint 8: training diagnostics

Eight updates covered **64 distinct real pairs / 64 speakers**. Gradient norms
before clipping ranged **0.22009585798–0.27351564169**; there were
**0 clipped updates and 0 OOM retries**. The registered full/half route order
and weighting were unchanged. Checkpoint SHA256:
`5467984fbdc9db2c0b44ce21132346b3272a5a6c90cc260a383fae33dd94a6c7`.

| Matched calibration teacher objective | Step 0 | Step 8 |
|---|---:|---:|
| Earlier 47-speaker pilot | 3.426517192274332 | 3.4259552881121635 |
| Broader 256-speaker pilot | 3.426517192274332 | 3.4259027615189552 |

These compare the same 32 calibration pairs × two modes on the same CUDA FP32
teacher backend, with no calibration backward. The broader run's **own fixed16
fitting-probe objective** fell **3.57301127910614 → 3.5723811835050583**; only
**3 of 16 probes** have been seen in its schedule so far, with 13 unseen. Those
fitting probes differ from the old pilot and must not be compared across runs.
These small loss changes do not establish speech improvement or diagnose
underfitting/overfitting. The complete80 speech result below is a tradeoff; no new best or audio
example is selected. Stable [checkpoint receipt](C:/Users/Khan/.codex/visualizations/2026/11/09/01a1bf77-0b17-7391-9781-f859f1fbb0b1/reference-masked-supervision-broad-training-v1/checkpoint-000008.json),
[teacher0](C:/Users/Khan/.codex/visualizations/2026/11/09/01a1bf77-0b17-7391-9781-f859f1fbb0b1/reference-masked-supervision-broad-training-v1/teacher-000000.json) and
[teacher8](C:/Users/Khan/.codex/visualizations/2026/11/09/01a1bf77-0b17-7391-9781-f859f1fbb0b1/reference-masked-supervision-broad-training-v1/teacher-000008.json) bind this snapshot;
all eight update-file hashes are included in the visual-data diagnostic.

### Broader checkpoint 8: complete calibration80 result (November 12)

All **80 fresh recordings** passed the native/artifact audit: 80 native inputs
verified, zero truncations, and zero independent cosine-recomputation error.
This is the same calibration panel and two seeds as the earlier pilot, not
the original114 confirmation panel. Eight updates visited **64 distinct fitting
speakers of 256 planned**. Session **41410** has checkpoint 32 underway (latest audited quality: step 16).

| Calibration80 policy | Group | CER | WER | Original-reference macro cosine |
|---|---|---:|---:|---:|
| Shared baseline 0 | development (64) | 3.905519% | 10.182371% | 0.982569867 |
| Shared baseline 0 | custom (16) | 1.773836% | 3.623188% | 0.984323889 |
| Earlier pilot 8 | development (64) | 3.858746% | 10.486322% | 0.982577998 |
| Earlier pilot 8 | custom (16) | 0.997783% | 3.623188% | 0.983990882 |
| Broader pilot 8 | development (64) | 4.069224% | 11.246201% | 0.982655996 |
| Broader pilot 8 | custom (16) | 1.108647% | 3.623188% | 0.983968820 |

Against baseline, development errors rose **167 → 174 / 4,276 characters**
and **67 → 74 / 658 words**, while cosine improved slightly. Development's net
seven extra character errors combine 43 worse and 36 better edits across texts.
Custom character errors fell **16 → 10 / 902**, with words unchanged at **5 / 138**,
but cosine fell below 0.984. The custom edit change was **+1 in seed 2026 and −7
in seed 2027**. Fifteen of 64 development WAVs and five of 16 custom WAVs remained
identical to baseline. These paired changes are descriptive, not significance
or generalization claims.

The four-metric comparison is a **tradeoff versus earlier steps 0, 8 and 32**;
it dominates earlier step 16. Neither group passes the joint CER/voice gate.
**No new best, no new audio example, and no original114 qualification** follows.
Teacher-loss changes are not evidence of speech improvement or under/overfitting.

- [Native checkpoint-8 verification](C:/Users/Khan/.codex/visualizations/2026/11/09/01a1bf77-0b17-7391-9781-f859f1fbb0b1/reference-masked-supervision-broad-training-v1/verification-000008.json) — SHA256 `f4d88af93bf00662656e78faf0c77c6400e22d1f823414b73d73c2bc183ede1b`.
- [Complete paired analysis](C:/Users/Khan/.codex/visualizations/2026/11/09/01a1bf77-0b17-7391-9781-f859f1fbb0b1/broad-calibration-paired-step000008-v1.json) — SHA256 `0d99f34da0c3450b08744af47450ea67b639d382d24d03fe33e9cd0f1ea24a6f`.
- [Independent saved-checkpoint audit](C:/Users/Khan/.codex/visualizations/2026/11/09/01a1bf77-0b17-7391-9781-f859f1fbb0b1/broad-checkpoint8-independent-audit-v1.json) — SHA256 `9141c79135c5239ec4afd3d7c4656e9df32e444a2141555979e7ff1911eb27b8`.

The [conditional original114 helper readiness note](C:/Users/Khan/.codex/visualizations/2026/11/09/01a1bf77-0b17-7391-9781-f859f1fbb0b1/broad-confirmation-readiness-2026-11-12-v1.md)
records independent review and eight metadata/artifact tests. It is **prepared,
unregistered and currently ineligible**; it has not run confirmation. Eligibility
still requires a positive checkpoint passing both complete calibration groups
and the parent process's authoritative successful closure. The unchanged
original114 scope and separate complete native audit remain required.

### Broader checkpoint 16: complete calibration80 result (November 12)

All 80 fresh recordings passed native and artifact audit with **80 native inputs,
zero truncations and zero cosine-recomputation error**. The fixed CPU FP32
calibration80 scope, references, seeds and thresholds did not change. Session
**41410** has checkpoint 32 underway; latest audited quality is step 16.

| Broader pilot | Group | CER | WER | Original-reference macro cosine |
|---|---|---:|---:|---:|
| Shared 0 | development (64) | 3.905519% | 10.182371% | 0.982569867 |
| Shared 0 | custom (16) | 1.773836% | 3.623188% | 0.984323889 |
| Step 8 | development (64) | 4.069224% | 11.246201% | 0.982655996 |
| Step 8 | custom (16) | 1.108647% | 3.623188% | 0.983968820 |
| Step 16 | development (64) | 3.811974% | 10.486322% | 0.982400869 |
| Step 16 | custom (16) | 1.773836% | 5.072464% | 0.983665444 |

Compared with broader step 8, development improved **174 → 163 / 4,276
character edits** and **74 → 69 / 658 word edits**, but cosine fell by about
0.000255127. All 11 fewer development character edits came from seed 2026;
seed 2027 tied. Custom worsened **10 → 16 / 902 characters** and **5 → 7 / 138
words**, while cosine fell by about 0.000303376. All six extra custom character
edits came from one text (`masked-custom-calibration-002`). Twenty-eight of 80
WAVs matched broader step 8 exactly; their hypotheses, scores, cosines and
reference/settings metadata also matched in a separate scalar check.

At the **same 16-update budget**, earlier-pilot → broader-pilot character errors
were **203 → 163 development** and **25 → 16 custom**. Development cosine was
slightly lower (0.982527697 → 0.982400869); custom cosine was higher
(0.983500279 → 0.983665444). This is also a tradeoff. Across all earlier-pilot
checkpoints and broader step 8, no four-metric dominance or joint qualification
was established. **No new best, audio example or original114 run.** These are
descriptive matched-panel results, not a generalization or significance claim.

The 16 updates visited **128 distinct pairs / speakers of 256 planned**. Gradient
norms ranged **0.21403615177–0.28705722094**, with zero clipped updates and OOM
retries. Eight of 16 fixed fitting probes have been seen, eight remain unseen.
The broader run's own fixed fitting objective is **3.5716682970523834**, versus
3.57301127910614 at zero. Its matched calibration teacher objective is
**3.425279825925827**, versus 3.426517192274332 at zero and 3.4259027615189552
at eight. Earlier-pilot step16 calibration was **3.4253552332520485** on the
same 32 pairs × two modes/backend. Fitting panels differ across pilots; their
losses are not a cross-run comparison. Falling teacher loss is not speech
improvement or evidence establishing underfitting/overfitting.

- [Native audit](C:/Users/Khan/.codex/visualizations/2026/11/09/01a1bf77-0b17-7391-9781-f859f1fbb0b1/reference-masked-supervision-broad-training-v1/verification-000016.json) — SHA256 `6649019e5faaf004e58ea5de44fdac089b0533e1f572ff070ef77ffedd7f1b54`.
- [Paired analysis](C:/Users/Khan/.codex/visualizations/2026/11/09/01a1bf77-0b17-7391-9781-f859f1fbb0b1/broad-calibration-paired-step000016-v1.json) — SHA256 `ec897936a0ba64d37296bd20d0d1fc5863026593272d119fff89c7eb3dd29cd0`.
- [Identical-WAV consistency check](C:/Users/Khan/.codex/visualizations/2026/11/09/01a1bf77-0b17-7391-9781-f859f1fbb0b1/broad-calibration-step16-waveform-consistency-v1.json) — SHA256 `147707d24e8e52ca9b4fc4eda847314d678378b75d1effb40c7d842d6849271c`.
- [Teacher16](C:/Users/Khan/.codex/visualizations/2026/11/09/01a1bf77-0b17-7391-9781-f859f1fbb0b1/reference-masked-supervision-broad-training-v1/teacher-000016.json) — SHA256 `44ab0536d72171e44cbef6f9dcb71238dc0c0b91e6c108fe50cda298d38ca759`.

A [read-only fitting-GPU feasibility note](C:/Users/Khan/.codex/visualizations/2026/11/09/01a1bf77-0b17-7391-9781-f859f1fbb0b1/fitting-gpu-rollout-feasibility-2026-11-12-v1.md)
considers a separately registered CUDA FP32 fitting-rollout backend. It is
**not selected, implemented, registered or qualified**. Same seed does not imply
CPU/GPU output equivalence; no speed benchmark exists. Native versus instrumented
GPU controls, processed sampling-probability/replay checks and bounded memory
measurement would be required before use. CPU calibration80 and original114
remain unchanged. This is a conditional engineering option, not a new recipe.
