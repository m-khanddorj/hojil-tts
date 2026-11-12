# 0.6B experiment summary

**Status: the joint goal is not yet achieved.** Updated November 12, 2026,
through the broader-data pilot’s complete checkpoint-16 speech audit.

| Read this | What it contains |
|---|---|
| [Comprehensive report](report.md) | Purpose, experiment register, engineering changes, decisions, limitations, and publication plan |
| [Training evidence](evidence/training.md) | Original LoRA audit, precision, epochs, reserved-test comparisons, and early voice experiments |
| [Alignment evidence](evidence/alignment.md) | Preference learning, speaker adapters, routing, attention adaptation, sampling, and broader confirmation |
| [Voice and supervision evidence](evidence/voice-and-supervision.md) | Direct voice gradients, reference profiles, weight-space diagnostics, PAD experiments, and the current supervised pilot |
| [CI and portability evidence](evidence/ci-portability.md) | Local validation, portable checks, environment limits, and publication readiness |

## The target

On the **original fixed 114-recording panel**, both groups must independently
achieve pooled **CER <3%** and original-reference speaker-macro **cosine >0.984**:

- Development: 88 recordings, ten speakers.
- Your supplied reference voice: 26 recordings, one speaker.

Both fixed seeds, all original texts and references, and the registered scoring
settings remain included. Every recording must finish and pass its native
conditioning and artifact audit. This panel has been reused during development;
it is not an untouched final test set. Cosine is an internal model score, not a
percentage of identity match or a substitute for listening.

## What is established

The **0.6B model was trained with LoRA**, not full-parameter fine-tuning. A merged
inference export does not change that fact. The older 1.7B full-talker training
and its demonstration results are a separate model history.

The supervised pilot started from **centered-step-16/full-PAD**. Its original114
results are:

| Group | CER | WER | Cosine | Joint target |
|---|---:|---:|---:|---|
| Development | 4.2226% | 10.7306% | 0.984227 | Fails CER |
| Supplied reference | 3.2796% | 8.2707% | 0.983556 | Fails CER and cosine |

This starting point does not dominate every earlier metric. The
[full-panel comparison](../../0.6b-attention-pad-full-panel-findings.md) explains its selection.

## Latest completed experiment

**Reference-masked real-speech supervision v2** trained a new rank-4 attention
LoRA for 32 updates. The frozen starting policy and evaluation panels stayed
fixed. All four calibration checkpoints (0/8/16/32) passed their complete
80-recording integrity audits. The process closed successfully at its update
limit; no checkpoint met the joint quality criteria.

This **separate calibration80 panel** is not the original114 panel above:

| Group | Baseline → step 32 CER | Baseline → step 32 WER | Baseline → step 32 cosine |
|---|---:|---:|---:|
| Development: 64 recordings, 16 speakers | 3.9055% → 4.6071% | 10.1824% → 12.1581% | 0.982570 → 0.982480 |
| Supplied reference: 16 recordings | 1.7738% → 0.4435% | 3.6232% → 2.1739% | 0.984324 → 0.984215 |

Custom pronunciation improved and its final scores meet both calibration
thresholds; 11 of the 12 net character edits removed come from one case/seed.
Development worsened and fails both thresholds; custom cosine also
remains below baseline. **No new best was selected, and original114 was not
run for this pilot.** Custom recovery after step16 means this is not evidence
of uniform monotonic overfitting. [Full checkpoint history](report.md#3-current-comparable-results-and-active-experiment)
and [protocol/evidence](../../0.6b-reference-masked-supervision.md) retain all results.

The exact 32 natural development recordings were separately scored at CER
0.3274% and WER 1.2158%. This diagnostic does not supply custom-speaker ground
truth or change thresholds. A fitting-only gradient diagnostic also completed:
it found no aggregate residual-loss dominance on the fixed probe panel.
The completed routing comparison also failed: half-reference routing worsened
development CER/WER/cosine at both checkpoints 0/32. No new best or original114
confirmation followed. The fitting-only KV diagnostic also completed: all four student/graft conditions
produced identical audio. An original-base KV calibration80 screen completed and was rejected for development regression. No new best is claimed.

## Examples and publication

Playable examples will accompany verified improvements under the
[sharing rules](../../0.6b-result-sharing.md); calibration gains remain provisional.
Audio, data, weights and generated artifacts stay outside Git. After the full
goal is verified, publication review and checks precede the authorized push as
**m-khanddorj**. No push is claimed. [Maintenance plan](report.md#8-maintenance-reproduction-and-publication).

Fresh native features for all 512 fitting WAVs are audited and complete.
The broader-data-only pilot has audited checkpoint 16 (128 fitting speakers visited):
development CER **3.8120%**, cosine **0.982401**; custom CER **1.7738%**, cosine
**0.983665**. Development pronunciation improved versus step 8, but voice
similarity fell in both groups and custom pronunciation worsened. No joint
qualification or new best. Session 41410 has checkpoint 32 underway.
[Matched calibration comparison](report.md#broader-checkpoint-16-complete-calibration80-result-november-12).
