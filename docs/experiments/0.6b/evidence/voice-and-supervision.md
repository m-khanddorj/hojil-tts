# Evidence inventory: voice preservation, reference conditioning, and real-speech supervision

[Summary page](../README.md) · [Comprehensive report](../report.md)

**Jump to:** [Weight-space work](#1-weight-space-ideas-measured-transferred-and-deliberately-not-done) · [Direct voice optimization](#4-direct-generated-voice-attention-optimization-two-updates-then-stop) · [Reference profiles and DPO](#6-reference-specific-profile-local-descent-mean-follow-up-then-dpo) · [PAD and full-panel confirmation](#7-recurrent-full-pad-ablation-first-joint-small-screen-pass-then-full-panel-failure) · [Data and features](#9-real-speech-data-preparation-and-native-features) · [Current supervision](#10-reference-masked-pronunciation-lora-v1-numerical-stop-and-v2-current-evidence) · [Data-quality diagnosis](#11-new-natural-speech-consistency-diagnosis-a-future-data-audit-not-pruning) · [Evidence index](#14-compact-evidence-index)

This inventory covers the late experimental families, including unsuccessful variants and implementation-only probes. Updated **November 12, 2026**, through the complete masked-supervision v2 checkpoint-32 audit and successful terminal process closure. The exact-32 natural-audio ASR control and separately audited fitting-only gradient diagnostic also completed. The separately registered inference-only conditioning comparison completed and failed; the fitting-only KV diagnostic also completed without changing student waveforms. An original-base KV calibration80 screen completed and was rejected for development regression. The registered broader-data pilot is tracked separately.

**No experiment in this inventory has satisfied the complete original 114-recording goal.** The centered-step-16/full-generation-PAD policy passed the 26-recording screen, then failed the expanded confirmation. The next shared real-speech adapter improved calibration CER while losing custom-reference cosine at step 8, then regressed in both groups at step 16. Neither is a new best.

## How to read the results

- CER and WER are pooled edit counts divided by pooled reference-character/word counts, reported as percentages. They are not means of per-clip percentages. ASR text normalization uses NFC/casefold, punctuation removal, and normalized whitespace; spaces remain relevant to CER.
- “Cosine” means similarity to the **original, unadapted reference voice vector**, aggregated by speaker and then averaged across speakers. The one-custom-speaker groups reduce to a recording mean. This is an internal encoder metric, not a calibrated human identity probability or interchangeable with scores from other papers.
- The quality requirement is **CER strictly below 3% and cosine strictly above 0.984 in both groups**, with complete, terminated, audited output. WER is reported but is not an additional selection threshold. Rounded table values do not decide boundary cases.
- “Audited” denotes the relevant native provenance/conditioning/score verification receipt. A unit test, a lower fitting loss, a feasible parameter step, or a partial generation is not a speech-quality result. Later independent observers recompute counts and aggregates but rely on the bound native audit for speaker-encoder extraction.
- A screen pass is provisional. The original confirmation contains **44 development texts from ten speakers and 13 custom texts from one speaker, each at seeds 2026 and 2027: 88 + 26 = 114 recordings**. It has been inspected repeatedly; it is a reused confirmation panel, not an untouched final test set.
- Unless specified otherwise, deployment comparisons use CPU FP32, SDPA, four threads, both TF32 switches disabled, the same native sampler, and fixed references/seeds. Training collection can use CUDA FP32 talker/scorer plus the CPU codec; those backend-specific comparisons are kept separate.

### Panel key

| Name used below | Exact scope | Seeds | Purpose |
|---|---|---|---|
| Screen26 | Ten development texts/speakers × 2 = 20; three custom texts × 2 = 6 | 2026, 2027 | Reused small two-group screen |
| Original114 | 44 development texts/ten speakers × 2 = 88; 13 custom texts × 2 = 26 | 2026, 2027 | Required broader confirmation |
| Direct-speaker fit16 | Sixteen matched original fitting rollout conditions, 16 speakers | One fixed seed per case, 32517–32532 in registered order | Candidate selection on fitting speech; not Screen26 |
| Profile fit16 / reserved8 | Eight fresh unpaired texts × 2; four reserved texts × 2, all using the custom reference | 41201, 41202 | Reference-profile fitting and separate personalization validation |
| Profile-DPO pool64 | Sixteen further fresh texts × 4 draws | 41301–41304 | Preference collection only |
| Profile-DPO checkpoint50 | Sixteen training texts at first seed + reserved8 + Screen26 | 41301; 41201/41202; 2026/2027 | Fixed post-update probes, separate partitions |
| Masked calibration80 | 32 real cases/16 development speakers × 2 = 64; eight custom texts × 2 = 16 | 2026, 2027 | New adapter calibration; not Original114 |

Artifacts are outside Git under **`ARTIFACT_ROOT = C:/Users/Khan/.codex/visualizations/2026/11/09/01a1bf77-0b17-7391-9781-f859f1fbb0b1`**. Every artifact path below is relative to that root. These local paths identify the evidence; publishing this Markdown does not publish the audio, datasets, or weights. Repository links point to the associated experiment notes. Some notes contain historical “next” paragraphs; completed receipts and the chronology here take precedence over those stale planning sentences.

## 1. Weight-space ideas: measured, transferred, and deliberately not done

Source: [joint goal and literature review](../../../0.6b-joint-goal-literature.md), `weight-space-voice-diagnostic/results.json`, and `protected-target-weight-transfer-v2/{results,verification}.json`.

### Training-only speaker-subspace diagnostic

The concrete Weight-Space Programming-inspired hypothesis was that the learned text-output projection update might align unusually strongly with speaker-variation directions. The diagnostic used cached embeddings for **28,001 original training clips, 435 equally weighted speakers**, and `text_projection.linear_fc2`. It compared centered speaker subspaces with 64 deterministic random-subspace controls. No development/custom speech entered the basis and no weight was changed.

| Subspace rank | Speaker variation explained | Text-update energy in speaker subspace | Mean random-subspace energy |
|---:|---:|---:|---:|
| 1 | 20.1731% | 0.04484% | 0.09749% |
| 4 | 45.7064% | 0.30915% | 0.38875% |
| 8 | 58.4092% | 0.53520% | 0.74784% |
| 16 | 72.0066% | 1.08371% | 1.52169% |
| 32 | 83.4953% | 2.08340% | 3.25578% |

The tested projection was not unusually aligned with this simple speaker subspace. **The proposed subtraction/projection edit was not executed.** This rejects a particular geometric rationale, not all possible weight-space interventions. No CER/WER improvement can be attributed to this diagnostic.

### Target-only full-weight transfer

This separate zero-optimizer experiment restored the two text-projection matrices to full epoch-2 values **only at target transcript positions**. Reference/control projections and the remaining halfway backbone stayed unchanged. It was component transfer, not discovered speaker circuitry or a reproduction of the WSP language-model result. DPO step-16 changes had only 5.50%/10.92% of the corresponding halfway-to-full matrix-change norms and near-zero directional alignment, so this tested a genuinely different edit from simply enlarging that DPO update.

`protected-target-weight-transfer` is the first **prepared control snapshot**, not a second completed quality experiment. `protected-target-weight-transfer-v2` adds explicit checkpoint/control provenance and contains the actual complete Screen26 comparison. Its internal **variant 101 is an inference identifier, not 101 optimizer updates**.

| Policy, Screen26 | Development CER | Development WER | Development cosine | Custom CER | Custom WER | Custom cosine |
|---|---:|---:|---:|---:|---:|---:|
| Halfway parent, reused audited endpoint | 9.3110% | 23.5632% | 0.986513475 | 7.2430% | 23.4375% | 0.984960198 |
| Target-only full transfer, variant 101 | 7.5419% | 18.9655% | 0.986651519 | 13.5514% | 39.0625% | 0.984899938 |

Development pronunciation improved while custom pronunciation deteriorated sharply. Both candidate cosine aggregates exceeded 0.984, but the joint goal failed. The audit checked all 26 recordings, exact target matrices, unchanged reference/control projections, waveform/reference hashes and ASR counts; maximum re-extracted cosine discrepancy was `1.1920928955078125e-7`. There was no promotion.

Later reference-prefix routing and recurrent-PAD replacement were other **specific, frozen-weight inference interventions**, not evidence that a general WSP procedure learned Mongolian pronunciation. The PAD results and their failed broader confirmation are in sections 7–8.

## 2. Differentiable codec feasibility: a gradient path, not an improved model

Source: [differentiable codec feasibility](../../../0.6b-differentiable-codec-feasibility.md); `native-codec-gradient-probe/{protocol,results}.json`; implementation [differentiable_codec_probe.py](../../../../training/differentiable_codec_probe.py).

The implementation uses the frozen native 16-codebook embeddings/projections and an exact hard-code waveform forward pass with a zero-valued soft-gradient expression. It is a **biased straight-through surrogate on a fixed trajectory**, not a derivative of the discrete sampling algorithm.

The native probe used fitting trajectory `balanced-114-draw-1`, selected for length/termination rather than score: **92 frames, 7.36 seconds**. Rebuilding its original four-draw batch reproduced the PCM WAV exactly (`9add04d3c1578e4b958092edfcc271e36a7aa0056d7a5f3cbcc89236bce63c9c`). Re-extracted original-reference cosine was 0.9858881235 versus recorded 0.9858880639, a roughly `5.96e-8` discrepancy.

| Probe measurement | Result | Meaning |
|---|---:|---|
| Hard native waveform versus straight-through forward | Exact | The gradient construction did not alter forward audio |
| Generated-code logit gradient norm | 0.0005054121 | Finite, nonzero signal exists |
| Peak allocated GPU memory | 6,875,452,416 bytes, about 6.40 GiB | Feasible local codec-gradient path |
| Probe time | 2.24 seconds, excluding load/CPU reconstruction | Not an end-to-end training-throughput claim |
| Original codec/speaker parameter changes | Zero | Fixed scoring/decoding components |
| Optimizer updates | Zero | No trained candidate or new CER/WER result |

The floating GPU waveform loss, 0.01433873, came from a different floating/PCM/batch path and is not a gain over the stored CPU cosine. Reference logits can be mathematically sensitive, but the later policy objective holds reference codes constant. Three tiny actual-Qwen decoder tests preceded the native feasibility check.

## 3. Codec-to-policy replay: memory separation verified

Source: [codec policy replay](../../../0.6b-codec-policy-replay.md); `policy-codec-replay-probe/{protocol,collection,codec,replay,results}.json`.

The next probe connected that derivative to the existing rank-4 middle-attention Q/V LoRA, layers 10–17: **32 tensors, 163,840 trainable parameters**. Collection, frozen codec differentiation, and policy vector–Jacobian replay ran in separate processes. Reference probabilities stayed fixed one-hot; raw policy posteriors were not adjusted by generation temperature/top-k. Residual heads conditioned on the recorded earlier codebooks.

| Phase | Observed result |
|---|---|
| Collect, initial attention-4/speaker-8 policy | Full sequence likelihood −5105.680999755859 over 1,473 tokens; peak 3,764,584,448 bytes |
| Codec derivative | Exact hard GPU waveform; constant reference codes; peak 6,864,966,656 bytes |
| Policy replay | Maximum posterior discrepancy zero; all 32 LoRA tensors had finite nonzero gradients; peak 3,937,047,552 bytes |
| Gradient and state | Policy-gradient norm 0.0055274289; original acoustic/speaker/LoRA state unchanged; zero updates |

Main-codebook acoustic probability mass averaged 0.99942988, minimum 0.95303524. The acoustic-only soft distribution therefore is not the full EOS-inclusive distribution, although the separate raw likelihood includes EOS. Three small actual-Qwen tests checked full-graph versus separated gradients, length/EOS behavior, and mismatch rejection.

This probe reused the same historical, off-policy fitting trajectory. It proved a local replay mechanism and a memory strategy; it did not establish useful current-policy optimization, an unbiased sampler gradient, or speech quality. A coverage inventory at that point found 173 zero-ASR-CER fitting recordings/44 speakers, including 110/36 in cosine [0.96, 0.984), with 109 within the chosen frame bound. Those are pool diagnostics, not human-certified clean pronunciation or automatically selected training data.

## 4. Direct generated-voice attention optimization: two updates, then stop

Source: [direct voice pilot](../../../0.6b-direct-voice-pilot.md); `direct-voice-pilot/`, `direct-voice-pipeline-status.json`, `direct-voice-observability.json`.

Starting point: audited unfiltered attention update 4 plus speaker update 8. Only the existing 163,840 attention-LoRA parameters trained; the speaker adapter, acoustic backbone, codec, and scorer stayed frozen. The pilot allowed four updates, drawing 16 fresh current-policy fitting rollouts each time from a 36-speaker cycle. Thirty-six fixed fitting anchors constrained token NLL; 62 validation trajectories from 16 held-out preference-validation speakers supplied a separate NLL diagnostic.

AdamW used learning rate `2e-5`, zero weight decay, FP32 moments, and gradient clipping at 1. A proposal was projected against the mean-anchor-NLL direction, with a projected steepest-voice fallback and up to eight step halvings. Acceptance required measured anchor NLL no higher than the preceding value plus `1e-6`, actual rounded negative predicted voice change, and at most 1% effective displacement per matrix. **Those local checks did not guarantee speech preservation.**

| Checkpoint, complete Screen26 | Development CER | Development WER | Development cosine | Custom CER | Custom WER | Custom cosine |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 2.0484% | 4.5977% | 0.9850573 | 1.6355% | 4.6875% | 0.9790958 |
| 1 | 2.7002% | 5.7471% | 0.9850205 | 0.9346% | 3.1250% | 0.9792840 |
| 2 | 4.1899% | 9.7701% | 0.9850176 | 0.9346% | 4.6875% | 0.9789247 |

| Update diagnostic | Step 1 | Step 2 |
|---|---:|---:|
| Anchor NLL before → after | 3.649726524 → 3.649727018 | 3.649727018 → 3.649727321 |
| Validation NLL | 3.683837432 | 3.683849019 |
| Posterior replay error | 0 | 0 |
| Changed WAVs versus initialization | 11/26 | 19/26 |
| Speech-harm warnings | First | Second; stopped |

The first voice/anchor parameter-gradient cosine was −0.176906; this is an optimization-direction measurement, **not speaker similarity**. By step 2, development gained 23 character errors versus initialization, 12 on `dev-003` seed 2027. Every checkpoint was evaluated; updates 3 and 4 were not attempted after the registered second consecutive warning. No checkpoint entered Original114.

Exploratory paired block bootstrap: step-1 custom cosine change interval [−0.000833, +0.001355] included zero; step-2 development CER change was +2.1415 percentage points, interval approximately [+0.2757, +4.8780]. These 10,000-draw intervals preserve seed pairs within texts, involve very small reused panels, and are not multiplicity-adjusted.

All 16 codec derivatives reproduced native hard waveforms. One 164-frame sample took 342.89 seconds versus 0.73–2.08 seconds for most first-batch clips; the cause was unresolved. The second batch returned to 0.64–1.59 seconds per clip. This was a performance observation, not an excuse to reinterpret the quality regression. Lower or nearly unchanged token NLL did not ensure preserved generated pronunciation; this is not clean evidence of ordinary validation-loss overfitting.

## 5. Direct shared-speaker proposals: fitting success did not transfer to the custom screen

Source: [direct speaker candidates](../../../0.6b-direct-speaker-candidates.md); `voice-gradient-geometry-probe/`, `direct-speaker-candidates/`, `direct-speaker-evaluation/`, `direct-speaker-evaluation-v2/`.

A read-only geometry probe compared where the voice gradient agreed with the anchor-likelihood gradient:

| Fitting batch/policy | Speaker-adapter gradient cosine | Attention-adapter gradient cosine |
|---|---:|---:|
| First batch, initial attention-4 | +0.381447 | −0.176906 |
| Second batch, direct-attention step 1 | +0.432349 | −0.196000 |

This motivated a **32,768-parameter rank-16 speaker-input-adapter** proposal with the attention/acoustic/codec/scorer fixed. It did not prove generated voice would improve. Fresh AdamW proposals used learning rate 0.003, zero weight decay, FP32 arithmetic, and the common voice/anchor descent constraint.

Eight predefined scales were prepared: 1, 1/2, 1/4, 1/8, 1/16, 1/32, 1/64, 1/128. Indices 0 and 1 were locally infeasible. Indices 2–7 were feasible; the largest feasible, index 2, reduced anchor NLL 3.649726524 → 3.649717011 with maximum input movement 0.161054% across 36 anchors, below the 0.25% new-movement limit. Prepared feasible weights are not evaluated speech results.

The first evaluation attempt omitted explicit CPU thread/TF32 settings. Its nominal no-update control changed all 16 GPU WAV hashes, including one duration. **That v1 attempt stopped at control failure; it evaluated and selected no candidate.** The separately registered v2 reproduced all 16 GPU controls and one CPU control before selection.

The fitting gate used complete matched fit16 speech, preserving CER and increasing mean cosine, with additional limits of no clip losing more than 0.005 cosine and no clip gaining more than one character error. These per-clip rules were experimental safeguards, not extra user goal thresholds.

| Fit16 variant, CUDA FP32 | CER | WER | Original-reference cosine | Actual outcome |
|---|---:|---:|---:|---|
| Baseline | 4.4324% | 10.1351% | 0.9796891 | Matched control |
| Candidate 2, scale 1/4 | 3.3514% | 7.4324% | 0.9780854 | Failed mean voice and up to three added errors |
| Candidate 3, scale 1/8 | 3.4595% | 7.4324% | 0.9798011 | Failed up to three added errors |
| Candidate 4, scale 1/16 | 2.9189% | 6.0811% | 0.9798626 | First fitting pass; selected |

Candidate 4 terminated all 16 clips, added no character errors, had worst cosine change −0.00001073, and maximum anchor input movement 0.039174%. Candidate 5 was not tried after selection; candidates 6 and 7 were outside the registered maximum of four speech trials.

| Candidate 4, complete CPU Screen26 | CER | WER | Cosine | Comparison with initial policy |
|---|---:|---:|---:|---|
| Development20 | 2.0484% | 4.5977% | 0.9852271 | Same pronunciation; cosine up from 0.9850573 |
| Custom6 | 0.9346% | 4.6875% | 0.9789036 | CER down from 1.6355%; cosine down from 0.9790958 |

The custom cosine requirement failed. All 26 recordings passed independent audit with cosine discrepancy zero; the terminal reason was `joint_screen_failed`. No Original114 confirmation or promotion followed. Heterogeneity mattered: one custom clip gained 0.002792 cosine and corrected three characters, another lost 0.004707 and gained three errors, and `custom-007` seed 2026 remained exactly unchanged at cosine 0.964556. Those examples were diagnostics, never fitting targets.

## 6. Reference-specific profile: local descent, mean follow-up, then DPO

### 6.1 Profile definition and collected baseline

Source: [reference voice profile](../../../0.6b-reference-voice-profile.md); `reference-profile-collection/`, `reference-profile-proposals/`, `reference-profile-evaluation/`, `reference-profile-mean-evaluation/`.

The profile adds **1,024 parameters after the shared speaker-input adapter**, activated only by the registered custom reference WAV hash. Other references bypass it. The base was attention-4/speaker-8, all shared weights stayed frozen, and the scorer remained unadapted. The initial residual was zero; all 26 original CPU WAV hashes reproduced exactly.

Twelve texts were selected before synthesis from the preference-training text pool, excluding every Original114 target and reference transcript. Eight fitting texts and four reserved texts used seeds 41201/41202 with the actual custom reference. Original source-speaker recordings supplied **text provenance only**, not substitute custom-voice targets.

| Zero-profile collection | Recordings | CER | WER | Cosine |
|---|---:|---:|---:|---:|
| Fitting | 16 | 3.4449% | 12.9870% | 0.983043551 |
| Reserved validation | 8 | 3.8889% | 10.4651% | 0.980691284 |

All terminated, with independent cosine discrepancy zero. Only fitting16 supplied codec gradients. Hard waveforms and posterior replays matched exactly, shared weights stayed unchanged, and voice/NLL parameter-gradient cosine was +0.0837793. Codec derivative times were 0.719–2.031 seconds; maximum reserved memory was 3,321,888,768 bytes.

### 6.2 Four actual local proposals

The direction combined normalized voice-loss and token-NLL gradients. The 5% smooth profile bound was mapped back to four predeclared **actual** input movements; it was not four arbitrary raw learning rates.

| Candidate | Actual movement | Fitting token NLL | Fit16 CER | Fit16 WER | Fit16 cosine | Speech selection |
|---|---:|---:|---:|---:|---:|---|
| Baseline | 0% | 3.737711011 | 3.4449% | 12.9870% | 0.983043551 | — |
| 0 | 0.5% | 3.735933649 | 1.4764% | 6.4935% | 0.983870205 | Failed per-clip checks |
| 1 | 0.25% | 3.736577428 | 3.2480% | 11.6883% | 0.983096451 | Failed per-clip checks |
| 2 | 0.125% | 3.737101028 | 1.9685% | 8.4416% | 0.982806474 | Failed mean voice and per-clip checks |
| 3 | 0.0625% | 3.737398420 | 2.8543% | 11.6883% | 0.982843269 | Failed mean voice and per-clip checks |

All four were locally feasible and all 64 candidate fitting recordings terminated and passed audit. Worst individual cosine changes were −0.010572731, −0.006915331, −0.008023262, and −0.007267952; maximum new character errors were respectively 2, 5, 2, and 4. The original run finished `no_fitting_speech_pass`; reserved/screen results did not select a scale.

Candidate 0's exploratory fitting bootstrap CER-change interval was −3.5985 to −0.4434 percentage points, but its cosine-change interval was −0.002206 to +0.003254. A fitting-average pronunciation gain did not establish uniform voice preservation.

### 6.3 Separately registered mean-objective follow-up

After the original process closed, a new experiment selected candidate 0 from the completed audited fitting trials using group means. It did **not retroactively change the failed original gate**. The per-clip failures remained in the record; reserved and Screen26 data did not choose another scale. Selected profile SHA: `7a9a371d489751206f65e064aba71638ac70fbebabb38df9355116014ea397b0`.

| Follow-up partition | CER | WER | Cosine | Result |
|---|---:|---:|---:|---|
| Reserved8 baseline | 3.8889% | 10.4651% | 0.980691284 | Comparator |
| Reserved8 candidate 0 | 3.8889% | 9.3023% | 0.981908970 | Fitting CER gain did not transfer |
| Screen26 development20 | 2.0484% | 4.5977% | 0.985057265 | All 20 WAVs exact fallback |
| Screen26 custom6 | 3.2710% | 7.8125% | 0.980227629 | Both custom thresholds failed |

Reserved mean cosine rose 0.001217686, but the worst clip lost 0.008557141 and another added five character errors. Four-text bootstrap intervals crossed zero for CER and cosine. The custom screen traded higher cosine than the zero-profile 0.979095777 for CER deterioration from 1.6355% to 3.2710%. Terminal `joint_screen_failed`; no promotion or Original114 run. The prepared mean-profile full-confirmation harness remained unlaunched.

### 6.4 Fresh preference collection and support diagnostics

Source: [reference-profile preferences](../../../0.6b-reference-profile-preferences.md); `reference-profile-preferences/`, `reference-profile-preference-training/`, `reference-profile-preference-observability.json`, `profile-dpo-bootstrap.json`.

The next pool used **16 further fresh texts**, excluding Original114 and all 12 earlier profile texts. Four seeds 41301–41304 produced 64 complete recordings from the original zero profile. One known GPU control first reproduced exactly. All outputs were retained in the audit, not filtered to report a favorable pool.

Joint within-text preference pairs required no more character errors and no lower cosine, plus at least one corrected character or a 0.001 cosine gain. The audit found **67 valid pairs across all 16 texts: 27 pronunciation-improving and 40 voice-only pairs**. Equal-text objective mass was 41.5625% pronunciation-improving versus 58.4375% voice-only.

Relative winners were not uniformly adequate absolute targets. Pair-endpoint-weighted winner CER was 1.0543%, cosine 0.982151592. Only eight of 64 unique recordings, covering seven of 16 texts, passed both absolute per-recording thresholds; eight of 38 distinct winners did so. Only 30.8333% of case-balanced preference mass had a jointly passing winner. Five texts had no sampled cosine above 0.984, one had sampled minimum CER 4.396%, and three met individual thresholds in different draws without a joint pass. These are finite-pool support diagnostics, **not quality ceilings** or newly imposed filters.

### 6.5 DPO actual checkpoints and complete results

Only the 1,024 FP32 profile parameters trained, with beta 0.1, fixed zero-profile likelihoods, summed raw 16-codebook-plus-EOS sequence log probability, case-balanced weighting, AdamW learning rate 0.0005, zero weight decay, clip 1, and a 1% extra input-movement cap. No per-token normalization replaced raw DPO sequence likelihood. Registered checkpoints were 0, 4, 8, 16.

Every checkpoint had **50 complete audited recordings**: training16 at seed 41301, reserved8 at seeds 41201/41202, and Screen26. Step 0 reproduced all 50 baseline WAVs and scores. All later development20 WAVs remained exact, all recordings terminated, and original weights remained frozen.

| Step | Group | Recordings | CER | WER | Cosine |
|---:|---|---:|---:|---:|---:|
| 0 | Development | 20 | 2.0484% | 4.5977% | 0.985057265 |
| 0 | Custom screen | 6 | 1.6355% | 4.6875% | 0.979095777 |
| 0 | Reserved | 8 | 3.8889% | 10.4651% | 0.980691284 |
| 0 | Training probe | 16 | 2.2290% | 4.9689% | 0.981831670 |
| 4 | Development | 20 | 2.0484% | 4.5977% | 0.985057265 |
| 4 | Custom screen | 6 | 4.9065% | 14.0625% | 0.979899585 |
| 4 | Reserved | 8 | 6.8519% | 15.1163% | 0.982042432 |
| 4 | Training probe | 16 | 2.6342% | 4.3478% | 0.982676607 |
| 8 | Development | 20 | 2.0484% | 4.5977% | 0.985057265 |
| 8 | Custom screen | 6 | 4.2056% | 14.0625% | 0.981745362 |
| 8 | Reserved | 8 | 4.2593% | 12.7907% | 0.981453955 |
| 8 | Training probe | 16 | 2.2290% | 5.5901% | 0.980851866 |
| 16 | Development | 20 | 2.0484% | 4.5977% | 0.985057265 |
| 16 | Custom screen | 6 | 3.9720% | 14.0625% | 0.981111487 |
| 16 | Reserved | 8 | 8.7037% | 16.2791% | 0.982466087 |
| 16 | Training probe | 16 | 2.6342% | 7.4534% | 0.979805529 |

| Post-update fitting diagnostic | Step 4 | Step 8 | Step 16 |
|---|---:|---:|---:|
| DPO loss | 0.653494 | 0.617935 | 0.576447 |
| Preference accuracy | 94.375% | 94.375% | 95.625% |
| Actual additional input movement | 0.317181% | 0.623534% | 1% cap active |
| Likelihood replay error | 0 | 0 | 0 |

Projection activated at updates 14, 15, and 16. No pair mass had both chosen and rejected likelihoods falling; final log-ratio/length correlation was 0.1642. Those observed diagnostics do not explain away the speech failures. Step 8 improved custom CER/cosine versus step 4, but both remained below the required quality. Step 16 improved custom CER slightly while cosine reversed, reserved CER rose, and training-probe cosine fell again.

The registered stop was two consecutive checkpoints Pareto-dominated in custom CER/cosine by an earlier checkpoint, or a strict joint screen pass, or the update limit. It ended normally at **`maximum_updates_reached`**, not a Pareto stop. No checkpoint passed; the DPO-specific full-confirmation runner was prepared/tested but never launched. Increasing the movement cap or refreshing trajectories was not performed in this run.

The supplemental paired text-block bootstrap preserves both seeds within each screen/reserved text; training probes have one seed. For step 4, custom CER change was +3.2710 percentage points, exploratory 95% interval [−1.5152, +5.7895]; cosine change was +0.000803808, interval [−0.000821620, +0.003433347]. Step-8 custom cosine change versus baseline was +0.002649585, interval approximately [+0.001000, +0.004932], while CER remained 2.5701 percentage points worse. Three custom text blocks, repeated checkpoint selection, and no multiplicity correction limit interpretation. The intervals were reporting diagnostics, not replacement stopping criteria.

## 7. Recurrent full-PAD ablation: first joint small-screen pass, then full-panel failure

Source: [next reference-policy probes](../../../0.6b-next-reference-policy-probes.md); `reference-prefix-generation-pad/`, `reference-prefix-generation-pad-confirmation/`, `reference-prefix-generation-pad-observability.json`, `prefix-pad-goal-audit.json`.

The earlier broad half-prefix router used halfway reference/control matrices and full target/generated matrices, but also carried a halfway text PAD embedding into every generated frame. The new no-training ablation changed **only** `trailing_text_hidden` and `tts_pad_embed` at the native generation boundary to full-model PAD. It retained the old half prefill and original sampler. Exact reference-hash dispatch activated both routing and this intervention only for the supplied reference; development remained centered-speaker step 16.

The disabled control reproduced all six old routed custom WAVs and all 20 original centered-step-16 development WAVs. The intervention preserved **all 26 prefill input/mask hashes and first 16-codebook frames**, with all 20 development WAVs unchanged. Both stages were complete and independently audited, with unchanged weights, no truncation, and finite cosine recomputation discrepancy zero.

| Screen26 stage | Group | CER | WER | Cosine | Decision |
|---|---|---:|---:|---:|---|
| Half-PAD control | Development20 | 1.6760% | 5.7471% | 0.984636086 | Diagnostic exact fallback |
| Half-PAD control | Custom6 | 5.6075% | 15.6250% | 0.985577792 | Pronunciation fails |
| Full-generation-PAD | Development20 | 1.6760% | 5.7471% | 0.984636086 | Passes screen |
| Full-generation-PAD | Custom6 | 2.8037% | 9.3750% | 0.984976113 | Passes screen |

This was an actual inference-conditioning improvement on the custom screen, with **zero optimizer updates**. It was the first joint Screen26 pass in these late experiments. The driver closed with `needs_broader_confirmation`, then a separate process evaluated the unchanged Original114 panel.

| Full confirmation | Recordings | CER | WER | Cosine | Strict requirement |
|---|---:|---:|---:|---:|---|
| Development | 88 | 4.222615% | 10.730594% | 0.984226523 | CER fails |
| Custom | 26 | 3.279632% | 8.270677% | 0.983555874 | CER and cosine fail |

All 114 terminated. All 26 screen WAVs, scores, codec trajectories, prefills and first frames reproduced. Independent finite cosine discrepancy was zero; the separate stdlib final audit recomputed normalization/edit distances and verified hashes, metadata, sources, scope and thresholds. **Implementation verification passed; quality failed. No model was promoted.** Final independent audit SHA: `4b743778ea3bb71afb277867721f0db480a60904580349ca1fb29bf759dc377d`.

The broader failure is not explained by changing the screen between runs. The additional 68 development recordings had CER about 4.819%, versus 1.676% on the original 20; the additional 20 custom recordings had CER about 3.435%, cosine 0.983130, versus 2.804%/0.984976 on the original six. Development routing was inactive, so that failure belongs to the shared centered-step-16 fallback; a custom-only adjustment cannot repair it.

Error concentration diagnosis (`prefix-pad-confirmation-diagnosis.json/.md`) retained every case:

- The top three development text pairs contributed 38.5% of character errors; the top three custom pairs contributed 57.9%. `dev-extra-027` contributed 54 of 239 development errors, including 40 at seed 2026 with shortened ASR output.
- `custom-extra-005` had zero character errors at both seeds but mean cosine about 0.976054. Pronunciation and voice failures are not the same cases.
- Custom seed 2026 had CER about 3.913%/cosine 0.984764; seed 2027 had 2.647%/0.982348. Neither seed independently satisfied both requirements. Reporting only the better pronunciation seed or better voice seed would hide the tradeoff.
- These are ASR/encoder observations, not established phonetic causes, transcription-error adjudications, or reasons to drop difficult examples.

## 8. Attention-4/speaker-8 with the same PAD idea: full panel from the outset

Source: [attention/PAD full-panel findings](../../../0.6b-attention-pad-full-panel-findings.md); `reference-prefix-lora-pad/{baseline,control,candidate}/`, `prefix-lora-pad-observability.json`.

This was a separate **zero-training** comparison using the earlier attention-4/speaker-8 acoustic policy. The router handled the sixteen outer Q/V `LowRankLinear` modules without merging matrices or leaking their residual onto protected reference rows. It retained all full-branch arithmetic when inactive. Registered stages were baseline114, half-PAD control26 on **all 13 custom texts**, and full-PAD candidate114. That control26 is entirely custom and must not be confused with Screen26's 20+6 mixture.

| Stage | Group / recordings | CER | WER | Cosine |
|---|---|---:|---:|---:|
| Unrouted baseline114 | Development88 | 4.399293% | 11.986301% | 0.984458801 |
| Unrouted baseline114 | Custom26 | 3.624856% | 10.150376% | 0.978581518 |
| Half-prefix, half-PAD control | Custom26 | 4.315305% | 11.278195% | 0.982505299 |
| Half-prefix, full-PAD candidate114 | Development88 | 4.399293% | 11.986301% | 0.984458801 |
| Half-prefix, full-PAD candidate114 | Custom26 | 4.085155% | 11.278195% | 0.982749196 |

All three stages passed native audits and independent observer checks. The baseline reproduced its 26 historical screen outputs. All 88 candidate development WAVs exactly matched the new baseline; all 26 custom prefills and first codec frames matched the new half-PAD control. All native conditioning/PAD traces verified, all weights remained fixed, no clip truncated, and finite independent cosine discrepancy was zero.

Full PAD improved both custom CER and cosine slightly versus its own half-PAD control. Compared with the unrouted baseline, the final custom policy gained cosine but worsened CER. Compared with the earlier centered-step-16/full-PAD policy, it had worse development CER and worse custom CER/cosine, though slightly higher development cosine. **Neither full-panel policy passed.**

Registered protocol SHA: `928b7fc9f84f4a620d6f90a95d0789ed071f0b17ada9784558fda2721487e23d`; baseline results SHA: `84509f277ba6a30ca72146ec9bba6886b2d8b71ba24db5ff46a5e3d6ed78f7c4`; candidate results SHA: `01132883192d4cb71b726540e28c81ac847d7f15407d173a91f75bca04a1519b`.

The paired diagnosis of the two baseline/full-policy families found development errors shifted: 22 recordings improved and 29 worsened; short sentences also regressed. This did not support a long-text-only curriculum. The centered-step-16/full-PAD policy was selected as the frozen starting point for the subsequent shared pronunciation adapter because its remaining joint gaps were smaller—not promoted as a goal-achieving model.

## 9. Real-speech data preparation and native features

Source: [reference-masked supervision](../../../0.6b-reference-masked-supervision.md); `reference-masked-supervision-data-v1/`, `reference-masked-supervision-plan-v1.json`, `reference-masked-supervision-features-v1/`.

### Actual eligible population and exclusions

The original supervised cohort contained 28,001 rows/435 speakers. Restricting it to 48 original preference-fitting speaker labels gave 2,800 rows. Original114 target/reference text and source/audio exclusions removed none from that subset; excluding reserved calibration transcript identities reduced it to 2,765. Repeatedly removing any row whose reference ID no longer remained selected produced **2,718 paired rows from 47 speakers**, about 3.21281 target hours.

Each retained pair has a distinct same-label reference/target utterance, distinct bound audio identity, and a reference inside the fitting population. Calibration comprises **32 real cases from 16 disjoint new-adapter-fitting speakers**. Those speakers occurred in historical base training/prior calibration, so “held out” here does not mean unseen to every earlier model. Fitting/calibration clean WAVs cover 2,773 distinct real utterances with hashes checked against source inventories. Original raw MP3 paths from a legacy machine were unavailable; cleaned WAVs are the available audio evidence.

Leakage exclusion uses NFC/casefold/alphanumeric-only text keys, separately from ASR scoring normalization. The original confirmation target/reference transcript keys and available IDs/source/audio identities were excluded. This preserves fitting separation while acknowledging repeated evaluation-panel inspection.

Data protocol SHA: `89b0819de650369fb03556fd190a4b7129e58bfc32ebdfd5886e09addb98e62d`; audit SHA: `284091de8f2a133e09fbb32c920dd4ef5e9ed068dc8fa79df67e134efea38057`.

### Fixed bounded schedule and custom calibration

The plan fixes 32 updates × eight distinct fitting speakers/pairs per update: **256 pair visits, 201 unique scheduled fitting pairs**. Adding the fixed 16-pair fitting probe yields 204 unique fitting pairs. Each visited real pair is replayed in full- and half-reference modes with equal mode/pair weighting. Selection predates speech-quality inspection; no error-heavy Original114 example enters training.

Eight custom calibration texts are selected from distinct target-text keys among the 32 reserved real cases by ascending SHA-256 of `reference-masked-supervision-custom-calibration-v1:20261111:` plus the normalized key, with lexicographically smallest case ID resolving duplicates. They use the actual custom reference at inference and seeds 2026/2027. **There are no real paired target utterances for that custom voice; another speaker's real target codes are never assigned to it.** Plan SHA: `a4cefa5a77ca97696048df57a31f023c5a0441f88a626811e46edd13470c0c34`.

### Fresh native feature extraction, actually completed

The bounded union requires **323 unique real utterances plus the original custom reference = 324 assets**. Historical caches contain int16 codec IDs and FP16 speaker vectors; merely casting cached FP16 vectors to FP32 does not reproduce native FP32 extraction. The feature-only pipeline freshly encoded every required target and reference through the exact native voice-clone path, CPU FP32/SDPA/four threads/TF32 disabled, preserving the original speaker vector before any shared adapter.

Contracts include long token IDs with native target/reference role framing, long `[T,16]` codec IDs, original FP32 `[1024]` speaker vectors, WAV hashes/sample-rate metadata, source/runtime/model/tokenizer identities and immutable per-asset receipts. The tokenizer mirrors native `fix_mistral_regex=True`. Deduplication is by bound utterance identity, not by silently substituting similar audio. All 324 assets completed extraction and final contract/provenance audit; no legacy half-precision cache was reused as authoritative features.

Feature protocol SHA: `394d668d34cf01bf9e777952d2dc0c202cb1702d05c1bc038724c87a142d42d6`; complete feature audit SHA: `4b36bb802beaa26346976e994f5c4b0f62ac08695181de2bfd99033282d7caad`. This audit verifies extraction provenance, exact coverage, tensor shape/dtype/range/finiteness and hashes; it is not a second independent re-extraction of every feature. Native replay and output controls are separate prerequisites.

## 10. Reference-masked pronunciation LoRA: v1 numerical stop and v2 current evidence

### Shared policy and fitting objective

The frozen starting policy is centered-step-16/full-PAD. A **fresh additive rank-4 Q/V LoRA on main decoder layers 10–17** may change target-text and generated/teacher rows only. Original acoustic weights, prior adapters, text projections, embeddings, heads, residual predictor, codec, speaker encoder and halfway matrices stay fixed. Direct adapter residual is excluded from reference/control rows, but their hidden/KV states are not detached: target gradients can flow through frozen operations. This constraint is not a guarantee that the generated voice will remain unchanged.

Deployment chooses half-reference mode only for the exact registered custom-reference SHA; other references use full mode. Fitting real same-speaker pairs in both modes is augmentation, not fabricated custom speech. Half-mode native prefix PAD remains half; recurrent generated PAD and teacher suffix PAD remain full. Teacher replay and native inference must implement this same policy.

For each pair/mode the objective is `mean(main-codebook CE including real EOS) + 0.3 × mean(15 residual-codebook CEs on real target frames)`. Average over the two modes, then eight pairs. There is no residual EOS, DPO term, or generated chosen-output anchor. AdamW learning rate is `2e-5`, weight decay zero, FP32 forward/parameters/moments, no autocast, dropout zero, clip 1, fresh optimizer. Maximum 32 updates; checkpoints 0, 8, 16, 32; seed 20261111. These are bounded engineering settings, not established global optima.

### v1: completed implementation failure before training

Run `reference-masked-supervision-v1`, protocol SHA `d67a683747488e455231098e0aef90abf17e5513c233ede8706cd56d546901fe`, stopped before **any optimizer update**. CUDA memory preflight had passed all three sequence/frame-frontier cases in both modes, with finite nonzero gradients, exact restoration and peak reserved memory **5,825,888,256 bytes (about 5.43 GiB)**. Both historical zero-update WAV controls reproduced exactly.

Native conditioning embeddings matched exactly, but residual teacher/native logits failed the originally registered elementwise `2e-5 + 2e-5 × abs(teacher)` tolerance. The failure was not suppressed or rewritten. A separate read-only diagnosis found maximum residual-logit difference 0.000200272, maximum forward KL `2.61e-10`, and all **1,080 residual argmax predictions identical**, with frozen and new-adapter state unchanged. This supports a cache/batch-shape numerical explanation on those inputs; it does not prove all-input or trained-adapter equivalence. Diagnosis SHA: `96af10a4551829788234ea52e6e91fa0673b7fd6267b774fd27573d07d7cd87d`.

v1 produced no complete80 speech-quality result and trained no adapter. Its teacher diagnostics and memory/parity work are implementation evidence only.

### v2: separately registered numerical contract and controls

v1 remained immutable. v2 registered before new controls: maximum absolute logit error `5e-4`, mean absolute logit error `5e-5`, maximum all-vocabulary log-probability error `5e-4`, maximum chosen-token log-probability error `2e-4`, and maximum forward/reverse KL `1e-8`. A changed winner near a numerical tie is allowed only with each distribution's cross-winner regret ≤`2e-4`; stable argmax disagreements remain forbidden. Counts, frame/head identity, EOS coverage and exact native conditioning are checked separately. These are explicit local engineering tolerances, not thresholds prescribed by PyTorch.

Five fitting-only probes—the original probe plus the fitting sequence/frame-length frontier—were tested in both route modes and at zero/nonzero new adapter state: **20 controls**. Nonzero controls perturb only B using seed 20261112, clipped standard-normal draws in [−3,3] times `1e-4`, A unchanged, with exact post-probe restoration. No calibration pair supplies an optimizer update or a numerical-control trajectory.

All 20 production controls and both historical waveform controls passed. Across residual branches, maximum absolute logit error was 0.000200272, maximum chosen-token log-probability error 0.000062701, and maximum forward KL `4.10e-10`. This is implementation parity within registered bounds, **not a quality pass**.

v2 protocol SHA: `89463810f5f9fc5d75d3eabf16144e30872aa1488b0605418f4f604abee12651`; driver-input SHA: `e5b40f491e7832b1174e8ec270d36a3dfe952c5ab0b98f66be1807984648a495`; native-proof SHA: `d238c7792b5d064c36bbf3d7ac742e8d8ba54e23626f2d0b410406e350dd5e96`. Eleven training tests, four speech-artifact tests and eleven numerical-parity tests preceded execution.

### v2: audited calibration baseline and steps 8/16/32

| Checkpoint / group, calibration80 | Recordings | CER | WER | Cosine | Status |
|---|---:|---:|---:|---:|---|
| Step 0, development | 64 | 3.905519% | 10.182371% | 0.982569867 | Complete audit; both numeric targets fail |
| Step 0, custom | 16 | 1.773836% | 3.623188% | 0.984323889 | Complete audit; pooled targets pass |
| Step 8, development | 64 | 3.858746% | 10.486322% | 0.982577998 | Complete audit; both numeric targets fail |
| Step 8, custom | 16 | 0.997783% | 3.623188% | 0.983990882 | Complete audit; cosine fails |
| Step 16, development | 64 | 4.747428% | 12.613982% | 0.982527697 | Complete audit; both numeric targets fail |
| Step 16, custom | 16 | 2.771619% | 10.144928% | 0.983500279 | Complete audit; cosine fails |
| Step 32, development | 64 | 4.607109% | 12.158055% | 0.982479587 | Complete audit; both numeric targets fail |
| Step 32, custom | 16 | 0.443459% | 2.173913% | 0.984215401 | Complete audit; pooled targets pass |

The step-0 audit verified complete scope, no truncation, native inputs, zero adapter movement, unchanged frozen state, and exact independently recomputed cosines. Receipt SHA: `c6d8e5e7390a10b50eb391513998d727afc392d86bf90d6500bbf4eaf140c72b`. These are the starting policy on a **different calibration panel**, not a new model or a replacement Original114 result.

The first eight optimizer updates completed. Fixed fitting objective changed **3.394105673 → 3.393595599**; fixed calibration objective changed **3.426517192 → 3.425955288**; the post-update gap was **0.032359689**. Both losses decreased slightly. The subsequent complete speech audit verifies lower pooled CER but a custom cosine tradeoff, not freedom from overfit or a new best. Development character edits changed 167/4,276 → 165/4,276, and word edits 67/658 → 69/658. Custom character edits changed 16/902 → 9/902; word edits remained 5/138. Custom has one better, one worse, and fourteen unchanged CER recordings. Seventeen development and four custom waveforms remain byte-identical to baseline. The paired-change observer includes every case/seed and makes no significance claim.

Checkpoint-8 receipt SHA: `39a6abe9e7c634ab0ae09db82fd413e05fda3796ac50603d54fe73809260b115`. All 80 recordings passed native/artifact verification with no truncation. The driver proceeded to checkpoint-16 training. No original114 confirmation or new-best example was triggered by this tradeoff.

Checkpoint 16 also passed its complete audit without truncation, but both groups regressed. Development character/word counts are 203/4,276 and 83/658; custom counts are 25/902 and 14/138. Both baseline and step 8 are better on all four group CER/cosine measures. Seven development and one custom waveform remain identical to baseline. Fixed fitting/calibration objectives decreased further to 3.392970063/3.425355233; that teacher-loss behavior does not override the generated-speech regression. Receipt SHA: `a06384f2d7ec0f967711ad9f4190456f52e1380889596a9ee9eb833c5d477b55`. This was the first consecutive dominated checkpoint, so the registered progression continued to the final checkpoint at 32. No model was promoted.

Checkpoint 32 passed all 80 native/artifact checks, with no truncation, finite recomputed cosines and zero cosine discrepancy. Relative to baseline, development character edits rose 167→197/4,276 and word edits 67→80/658; custom character edits fell 16→4/902 and word edits 5→3/138. Development has 17 better/26 equal/21 worse CER recordings; custom has 2 better/12 equal/2 worse. Twenty-one development and twelve custom ASR outputs are unchanged; only one development WAV is byte-identical. Custom cosine recovers after 16 and passes its threshold, but remains below baseline. Development still fails both thresholds. This is a tradeoff, not a new best or proof of a uniform monotonic overfitting pattern.

| Fixed teacher objectives, diagnostic only | Fitting | Calibration |
|---|---:|---:|
| Step 0 | 3.394105673 | 3.426517192 |
| Step 8 | 3.393595599 | 3.425955288 |
| Step 16 | 3.392970063 | 3.425355233 |
| Step 32 | 3.391651064 | 3.423979275 |

The final maximum effective projection movement is 0.0795483% (mean 0.0674450%). Lower fixed teacher loss does not establish better generated speech. Final audit SHA: `9fd25e34176efe729c5c2ba5b949457b89fb96c14ba3c845eee0953de7bb0baf`; checkpoint SHA: `8110e01146001a297faa82039ee7d1bd684885c2e373284bddf07a49f68e16ad`; teacher-report SHA: `2dfd1d7ff1a34ea10bbbdf4790c5f6c37ab160da1d710bf8b6118b943a29f54b`. The run ended at **`maximum_updates_reached`**, and original process 33720 closed with exit code 0. All 0/8/16/32 audits are complete; no checkpoint qualified for original114 or promotion. The sharing helper reports `no_unshared_audited_improvement`.

The separate `masked-supervision-update-scale-review-v1.md` checked immutable source hashes, small checkpoint 8 tensors, Adam states and stored teacher arithmetic without native inference. At that checkpoint all 32 factor tensors had finite nonzero moments and step counters of eight; both A and B changed. No extra normalization, inactive mask, or clipping suppression was identified. Maximum effective BA/base projection movement was 0.027755%. Approximately 95.6% of the calibration teacher-loss reduction came from half-reference mode; full-reference calibration objective improved by only 0.000049323. At step 8 only 64 distinct fitting pairs had been visited, including three of sixteen fixed fitting probes. These are bounded update diagnostics, not an epoch or a convergence demonstration.

A nonzero positive checkpoint qualifies for confirmation only after complete independently audited calibration80, no truncation, and strict two-group CER/cosine thresholds. Step zero cannot qualify the already failed historical policy for redundant unchanged Original114. Two consecutive checkpoints Pareto-dominated on all four group CER/cosine metrics trigger stopping; otherwise the finite 32-update limit applies. The separate original114 confirmation runner is prepared/reviewed but **unregistered and unlaunched at this snapshot**. A calibration pass alone will not promote the model.

### Completed natural-audio ASR control and next diagnostic

The separate exact-32 real development target control completed after the pilot closed: 7/2,138 character edits (**CER 0.327409%**) and 4/329 word edits (**WER 1.215805%**), 32 recordings from 16 speakers. It used the unchanged pinned BuzzASR manifest/decoding and original target WAV hashes/transcripts. Receipt status is `verified_diagnostic`; original process 69898 closed with exit code 0. Verification SHA: `32b90e0c985842039ccfbdf7fa8743e6f206d0e1bd43aeb17e1a8750939f02ea`. The [supervision note](../../../0.6b-reference-masked-supervision.md#completed-natural-audio-asr-control) records protocol/results hashes and scope.

This is an exact-panel recognizer diagnostic, not paired custom-speaker human evidence. Its errors are not subtracted from generated scores; no goal, evaluation case or fitting label changes.

The fitting-only main/residual/conditional-KL gradient probe subsequently completed with zero optimizer updates, and native process 16468 closed successfully. Independent CPU recomputation verified saved FP32 vectors using FP64 geometry for all 30 fitting pairs × 2 modes, including fixed 16, flagged 15 and unflagged 15 strata. The enriched panel is diagnostic, not a population sample. The auditor did not rerun native derivatives, conditioning, teacher logits or restoration controls; those remain the separately bound native receipts.

| Fixed 16 fitting probes, modes equally averaged | Measured local gradient statistic |
|---|---:|
| Main CE vs residual CE cosine | +0.120889699 |
| Norm of 0.3 × residual CE / norm of main CE | 0.080428437 |
| Combined task vs main CE cosine | +0.996888567 |
| Main teacher-KL vs task cosine | −0.347002710 |
| Residual teacher-KL vs task cosine | −0.322758995 |

There is no aggregate main/residual opposition or weighted residual dominance on these fixed probes. The KL gradients oppose the task gradient locally at checkpoint 32: an infinitesimal negative raw task-gradient step would increase the measured KL to checkpoint 0. This does not predict the actual Adam update or demonstrate that adding KL improves pronunciation or voice. The evidence does not select residual removal, a KL coefficient, data pruning or a next training recipe.

The separate CPU saved-vector cross-mode readout (`masked-gradient-cross-mode-v1.json`) finds fixed 16 full/half aggregate cosines +0.822046243 for main CE, +0.889591924 for residual CE and +0.814485713 for the combined task. This likewise shows no aggregate mode opposition at checkpoint 32; it is not a recommendation to reweight modes or a prediction of Adam/speech outcomes.

The final paired-error review (`masked-supervision-final-error-review-v1.md`) localizes 11 of the 12 net custom character edits removed to custom002 at seed 2027. Five development texts account for 103/197 final edits, versus 67 at baseline; the other 27 texts improve 100→94. Four of those five high-error texts have exact natural-audio ASR. This supports further diagnosis of generated behavior, not a phonetic-cause claim, case pruning or training on inspected evaluation targets.

Native result SHA: `49170ce3720bf3b4d52d9281cf24e3cee04039f5f9b9293220e7f957a6b44bbf`; independent vector audit SHA: `2bf2efa4fbbbff80af20aff5e479a43cd50b42a42bec3b53d1bacd2f122ac03e`, status `independent_cpu_vector_audit_passed`. The external artifacts are `masked-supervision-gradient-probe-v1/results.json` and `masked-gradient-probe-independent-audit-v1.json`. These are implementation/mechanism results, not a new speech-quality trial.

### Completed conditioning comparison: all-half route rejected

The inference-only 2×2 preserved checkpoints 0/32 and all 64 development cases,
references and seeds 2026/2027. Both new half-reference/full-continuation-PAD arms
completed native audits with no truncation and zero cosine recomputation error.
There are 128 new WAVs and 160 reused recordings: 128 full-route development
controls plus 16 unchanged custom recordings per endpoint. Custom reuse is
shared evidence, not new independent output.

| Checkpoint / development route | Character edits / 4,276 | CER | Word edits / 658 | WER | Cosine |
|---|---:|---:|---:|---:|---:|
| 0 / full | 167 | 3.905519% | 67 | 10.182371% | 0.982569867 |
| 0 / half | 248 | 5.799813% | 97 | 14.741641% | 0.981883879 |
| 32 / full | 197 | 4.607109% | 80 | 12.158055% | 0.982479587 |
| 32 / half | 222 | 5.191768% | 95 | 14.437690% | 0.981689340 |

At step 32, full→half gives 22 better/20 equal/22 worse CER recordings; 10 better,
5 equal and 17 worse texts; 5 better and 11 worse speakers. The five largest text
increases add 39 edits while the other 27 remove 14, giving 25 net added errors.
Both routes lose cosine from 0→32. Half-route CER improves 248→222 while full-route
CER worsens 167→197; this is a descriptive interaction, not evidence that half
routing wins or a general causal explanation. Both half arms are inferior to
their matched full controls on all three aggregate measures.

Independent scalar review recomputed all 256 hypotheses/edit counts and finite
speaker-macro averages, checked exact condition metadata and result/audit hashes,
and matched canonical process 23429 exit 0. The full native implementation audit
is separately bound; this review did not rerun embeddings, tensors or synthesis.
The all-half route is rejected: no joint qualification, new best or original114
confirmation. The predeclared checkpoint 32-only confirmation helper therefore
remains unregistered and unlaunched. The fitting-only KV diagnostic also completed without changing student waveforms;
an original-base KV calibration80 screen completed and was rejected for development regression. The registered broader-data pilot is tracked separately.

Protocol SHA: `faaca2f70c9b47dd3a3373c5de0605a8a6d2bd1efe0cf2c0f952cba50c024ee9`;
step 32 audit SHA: `1d0f06317eac78c2b2d380e302542089cdeeb6e41040295dd87611600f0fb262`;
final results SHA: `e07ada42a68dbcca15e7acce2523b10b00ef33bde8e0b0eda9482059ab1cccc6`.
The external `masked-routing-final-findings-v1.md` contains paired seed/speaker
counts and all source hashes. [Current supervision notes](../../../0.6b-reference-masked-supervision.md#completed-conditioning-comparison-failed)
retain the conditional confirmation rules.

### Completed fitting-only KV intervention: exact output identity

The registered probe tested two fitting pairs × full/half reference modes ×
teacher0/self-graft0/student32/graft32, at seed 2026: 16 WAVs, 12 comparison outputs
and four self-graft controls. The teacher0 donor uses reference inputs only;
this is not an original-base donor. All four student32/graft32 pair-mode
conditions had exact WAV hashes, cosine, ASR hypotheses and CER/WER. All teacher
identity controls passed. Structural verification covers all 16 native outputs
and independently recomputed original-reference cosine with zero discrepancy.
Canonical original process 35473 closed with exit 0.

No useful waveform effect was observed in these four fitting conditions. This
`diagnostic_no_waveform_effect` is not an inference about unseen speakers/texts,
not a quality-frontier point, and not proof that reference KV never matters.
The already adapted checkpoint 0 donor does not by itself address the inherited
original-base voice deficit. Tiny fitting CER/WER summaries remain in the raw
report and are not promoted as calibration/generalization measurements.

Results SHA: `91bfa85483a94ebab2c8e5e2fc108bfc6e22e366302e4b4a1a89b5ad53f0a266`;
structural SHA: `9c38f0128cf35d2a9ca92cfaf210040bf39c65cb6d5233e586d4d7021b2429f9`.
Independent saved-artifact review also matched all 16 codec-group/EOS tensors
for the four student/graft and four teacher/self-graft pairs, and independently
recomputed all 16 ASR edit counts. Receipt SHA:
`3ff7b1defe1be2efc9837d0d943d110e85ac65e0e6099b498f565f9e9569856d`
(`masked-kv-final-findings-v1.json`). This peer check did not rerun native models,
ASR or embedding extraction.

The original-base KV calibration80 conditioning screen completed and was rejected.
No new best or original114 confirmation followed.

## 11. New natural-speech consistency diagnosis: a future data audit, not pruning

Sources: `masked-natural-voice-controls-v1.json`; `masked-fitting-voice-outliers-v1.json`, `.md`, and `masked-fitting-voice-outliers-v1-interpretation.md`. The detailed report SHA is `b9006023fcf1d1584d22f5deb378a1544b744f56c5debd864cc4e7a49f4cbe61`.

These read-only diagnostics used already bound cached FP32 native vectors and WAV PCM summaries, **no new model inference** and no dataset/training edits. They reproduced the 204 fitting-pair FP32 cosines, then inspected normalized FP64 within-label similarity matrices.

| Natural real-pair group | Pairs / supplied labels | Speaker-macro cosine | Additional observation |
|---|---|---:|---|
| Selected fitting pairs | 204 / 47 | Approximately 0.984155 | Mean pair cosine 0.983662; minimum 0.903901 |
| Reserved calibration pairs | 32 / 16 | Approximately 0.986877 | Every target ranked its paired reference above the tested other-label representatives |

Fifteen fitting pairs across seven supplied labels scored below at least one other-label representative. The ranking compares 46 alternatives in fitting and 15 in calibration, so the two ranking failure rates are not a controlled label-quality comparison. An internal cosine comparison cannot establish that a recording belongs to a different person.

The fifteen pairs occur in **15/256 planned pair visits (5.86%)**, or 30/512 mode exposures: three visits through update 8, six through update 16, fifteen through update 32. This is an exposure count, not their percentage of gradient magnitude or responsibility for any model outcome.

Using a diagnostic within-label graph connection at cosine 0.97—not an exclusion or identity threshold—four supplied-label groups showed separated sets:

| Supplied label prefix | Set sizes | Within-set mean cosines | Cross-set mean cosine |
|---|---|---|---:|
| `269aa7` | 4 + 3 | 0.971980824 / 0.988524600 | 0.942624978 |
| `ed34daf` | 8 + 2 | 0.975462518 / 0.975480318 | 0.929369051 |
| `685c20` | 6 + 2 | 0.981984355 / 0.985445938 | 0.956365708 |
| `5d2907` | 3 + 6 | 0.990347624 / 0.984695743 | 0.957790516 |

The worst `269aa7` pair joins target `18618091` to reference `18604135`, cosine about 0.903901. The three `18618091/92/93` recordings agree strongly with one another. Removing that one reference from the diagnostic still leaves broader separation, so “one bad reference” is not an adequate explanation. `ed34daf` similarly has two unusual recordings appearing in opposite target/reference roles that agree with each other more strongly than with the other eight. Related patterns also occur under `685c20` and `5d2907`.

All 27 direct low-margin endpoint WAVs were present with complete PCM reads, durations **2.166–6.708 seconds**, zero observed current-file saturation and zero near-clipping at amplitude ≥0.99. **These checks do not rule out prior clipping followed by normalization/resampling, other earlier preprocessing artifacts, or incomplete spoken content.** Two unusual `ed34daf` files were quieter (RMS −35.44/−32.67 dBFS; quiet 20-ms blocks 55.9%/49.4%) than most peers, but another quiet file remained in the main group, so level alone is not the diagnosis. Across all low-margin pairs, quiet proportion was not generally higher than the fitting population.

Possible recording-channel, delivery, processing, or supplied-label inconsistency warrants a focused source/listening audit before a larger follow-up. None is established here. **No rows were removed, labels changed, references replaced, or the current bounded schedule altered.** Natural-pair averages are neither an identity threshold nor a hard upper bound on generated voice cosine. The findings do not establish a trained-model quality gain or failure.

## 12. Observability, examples, and what “better” means

Source: [result-sharing preference](../../../0.6b-result-sharing.md); external `analyze-reference-masked-supervision-v2.py` and `summarize-masked-supervision-improvement.py`.

The reporting layer keeps four types of evidence separate:

1. **Implementation:** source/driver/test hashes, fixed data/reference identities, native conditioning/token/frame/EOS checks, frozen states, complete artifact bindings, finite scores, replay errors and memory.
2. **Fitting:** pre-update versus post-update objective, main/residual/per-codebook NLL, full/half mode losses, fit/calibration gap, gradient clipping, optimizer settings and effective movement. Teacher figures are hash-bound producer diagnostics, not independently rerun losses in the stdlib observer.
3. **Generated speech:** only complete audited panel CER/WER/original-reference cosine, per-speaker/seed changes, termination/duration and paired clips. Partial generation shows progress only.
4. **Decision:** strict actual scope and thresholds, Pareto comparisons, whether the result is calibration-only, screen-only, or full114, and whether it really permits a next stage.

The user requested an example whenever there is an actual better result. For v2, a new calibration best means either a positive eligible checkpoint crosses both groups' quality/termination thresholds when the previous best's **raw measured quality** did not, or it is no worse in all four group CER/cosine values and strictly better in at least one. WER is reported separately. Step zero being ineligible for training qualification is not itself a quality failure: an identical positive checkpoint must not be announced as an improvement just because its step number is positive.

The listening example is always **`masked-custom-calibration-000`, seed 2026**, selected by registered order, not by favorable score. The helper compares its text/audio/individual CER/WER/cosine and all group metrics against the previous audited best, flags tradeoffs, and binds final WAV hashes. It neither sends a message nor marks notification delivered. Root renders verified local audio and may record only examples actually sent. All audio remains outside Git. A favorable clip or provisional calibration improvement is never labeled global Original114 success.

## 13. Literature use and unexecuted alternatives

The experimental literature connections are documented rather than treated as proofs of transfer to this model/language:

| Source discussed in the project | Concrete influence here | What was not reproduced or established |
|---|---|---|
| [DiffRO](https://arxiv.org/html/2507.05911v1) | Differentiable audio reward motivated testing a frozen native-codec gradient path | Its CosyVoice2 English setup/reward-model scale and published WER gains are not Qwen/Mongolian results |
| [RRPO](https://arxiv.org/html/2512.04552v1) | Reward exploitation concern; keep hard speech, ASR and original-reference scoring visible | Internal cosine increase is not a human identity/perceptual guarantee |
| [CSP-FT](https://arxiv.org/html/2501.14273v2) | Recognition-based localization as an alternative to guessing arbitrary “speaker directions” | Its training scale/layer findings do not identify universal Qwen speaker layers |
| [DPO](https://arxiv.org/abs/2305.18290) | Raw sequence preference objective with fixed reference and explicit paired conditions | Metric-derived TTS labels and a 1,024-dimensional profile are a local adaptation, not human-preference performance replication |
| [Beyond Fixed Targets](https://arxiv.org/html/2610.17838) | Reference-specific current-policy feedback; avoid reconstructing the short reference as the target | No LLaSA GRPO/self-distillation reproduction, no transferable ECAPA threshold |
| [VoiceTTA](https://arxiv.org/html/2606.26534) | Lightweight reference-specific generated-speech feedback | No F5 flow-matching loss reproduction; Qwen profile/ST replay is different |
| [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/) and the project's [cross-task](https://arxiv.org/abs/2611.09382)/[robot-policy](https://arxiv.org/abs/2611.09376) readings | Diagnose a particular mechanism, use matched controls, retain failures | No generic weight editing claim or established speaker circuit; no edit from the unsupported subspace hypothesis |
| [PyTorch numerical accuracy guidance](https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html) | Distinguish cache/batch arithmetic differences from semantic replay errors | PyTorch does not prescribe the locally selected v2 probability/logit tolerances |

The following were **proposals or prepared guards, not completed quality experiments** in this late inventory:

- A speaker-subspace subtraction edit after the negative geometric diagnostic: not executed.
- Refreshing four new draws per DPO fitting text from checkpoint 4, followed by a matched four-update comparison: proposed, not run.
- Widening the 1% profile cap or enlarging its parameterization: not done in the completed 16-update pilot.
- Direct-attention updates 3/4, direct-speaker candidate 5 after selection, and later unselected feasible scales: not speech-evaluated.
- Profile mean and DPO Original114 confirmation harnesses: prepared/tested, never launched because their screens failed.
- A larger real-speech run or automatic removal of the newly diagnosed outlier groups: not performed.
- Masked-supervision Original114 confirmation: prepared/tested, not launched at this report snapshot; did not run because no positive checkpoint qualified; the parent process has now closed successfully.

## 14. Compact evidence index

| Family | Primary external evidence | Completed scope / terminal state |
|---|---|---|
| Speaker-subspace geometry | `weight-space-voice-diagnostic/results.json` | Five ranks, training-only diagnostic; zero edits |
| Target-only transfer | `protected-target-weight-transfer-v2/results.json`, `verification.json` | Screen26; quality failed, zero updates |
| Codec gradient feasibility | `native-codec-gradient-probe/results.json` | One fitting trajectory; zero updates |
| Separated policy replay | `policy-codec-replay-probe/results.json` | Exact replay/finite gradients; zero updates |
| Direct attention | `direct-voice-pilot/verification-000000.json`, `verification-000001.json`, `verification-000002.json`; `direct-voice-observability.json` | Three complete Screen26 checkpoints; stopped after two speech-harm warnings |
| Direct speaker | `direct-speaker-evaluation-v2/{gate-02,gate-03,gate-04,screen-verification-04,results}.json` | Three fit16 trials; selected candidate4 Screen26 failed |
| Profile local proposals | `reference-profile-evaluation/`; `reference-profile-observability.json` | Four fit16 trials; no original fitting gate pass |
| Profile mean follow-up | `reference-profile-mean-evaluation/{selection,validation-verification-00,screen-verification-00}.json` | Reserved8 + Screen26; joint screen failed |
| Profile DPO | `reference-profile-preference-training/verification-000000.json`, `verification-000004.json`, `verification-000008.json`, `verification-000016.json` | Four checkpoint50 audits; maximum updates reached without pass |
| Centered16 PAD | `reference-prefix-generation-pad/`; `reference-prefix-generation-pad-confirmation/verification.json`; `prefix-pad-goal-audit.json` | Control26/intervention26 passed audit; provisional screen pass; full114 quality failed |
| Attention4 PAD | `reference-prefix-lora-pad/{baseline,control,candidate}/verification.json`; `prefix-lora-pad-observability.json` | 114/26-custom/114 complete; quality failed |
| Real supervised data/plan | `reference-masked-supervision-data-v1/{protocol,audit}.json`; `reference-masked-supervision-plan-v1.json` | 2,718 eligible pairs; bounded schedule fixed |
| Fresh native features | `reference-masked-supervision-features-v1/` | All 324 features audited |
| Masked supervision v1 | `reference-masked-supervision-v1-status.json`; `masked-native-replay-diagnosis-v1.json` | Failed numerical control before any update |
| Masked supervision v2 | `reference-masked-supervision-v2/{protocol,native-verification,verification-000000,verification-000008,verification-000016,verification-000032,teacher-000032}.json`; `reference-masked-supervision-v2-status.json` | Twenty native controls and all four calibration80 audits passed; terminal update limit, no joint quality pass |
| Data consistency | `masked-natural-voice-controls-v1.json`; `masked-fitting-voice-outliers-v1.json` and interpretation note | Read-only diagnostic; no pruning or model change |

The strongest causal late evidence is the matched recurrent-PAD ablation: it improved the small custom screen while preserving exact prefixes/first frames and inactive development outputs. Its full-panel failure, the failed attention/PAD comparison, and the mismatch between improving DPO loss and generated speech explain the move to a shared real-speech pronunciation adapter with explicit reference routing. The completed supervised pilot improved custom pronunciation at step 32 but did not satisfy the joint tradeoff; its falling teacher loss did not predict development improvement.

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
