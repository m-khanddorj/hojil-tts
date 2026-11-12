# Evidence inventory: preference alignment, conditioning and routing experiments

Updated November 11, 2026. [Summary](../README.md) · [Comprehensive report](../report.md).

**Jump to:** [Panel definitions](#reading-the-evidence) · [Experiments and decisions](#experiment-by-experiment-changes-and-outcomes) · [Research use](#research-sources-and-actual-local-use) · [Exact metric tables](#exact-saved-metric-tables) · [Implementation map](#implementation-entry-points-new-historical-modules-not-edits-to-registered-runs)

This inventory consolidates preference alignment, conditioning and routing experiments from fourteen historical research notes and their saved result JSONs. It does not change a running experiment, select a new model, or imply that a prepared implementation was executed. The current masked-supervision experiment is outside this inventory.

## Reading the evidence

The unchanged requested goal is pooled CER **strictly below 3%** and original-reference speaker-macro cosine **strictly above 0.984**, separately for development and the supplied reference voice. WER is reported but is not the goal threshold. Internal cosine is an encoder measurement, not proof of perceptual identity; values from other papers' encoders are not comparable.

The common **26-recording screen** has ten development targets/speakers × two seeds (20 recordings), plus three custom-reference texts × two seeds (6 recordings). Seeds are 2026 and 2027. Its 1,074 development and 428 custom reference-character denominators make one/few errors consequential. Screen synthesis is CPU FP32 SDPA; ASR runs separately with the pinned Mongolian recognizer. Preference rollouts use CUDA FP32 and different generation draws. Pool, calibration, screen and expanded-confirmation metrics must not be ranked as though they measured the same panel.

The original **114-recording confirmation** has 44 development targets from ten speakers × two seeds (88 recordings), and thirteen custom-reference texts from one speaker × two seeds (26). It was registered before the preference outcomes, but has since been inspected repeatedly: it is expanded development/personalization evidence, **not an untouched test**. Preference-validation speakers were excluded from each new preference fit but appeared in the original supervised training corpus; they are not necessarily unseen by the acoustic model.

Every exact metric table below is transcribed mechanically from existing external JSON summaries, with CER/WER expressed as percentages to six decimal places and cosine to nine. Those display digits are rounding, not new measurement precision. JSON retains original floats. The literature and narrative documents sometimes round to two/four decimals. Missing metrics are explicitly identified; no value is inferred from another panel.

External evidence root (`ROOT` throughout): `C:/Users/Khan/.codex/visualizations/2026/11/09/01a1bf77-0b17-7391-9781-f859f1fbb0b1`. Audio, datasets, weights and generated artifacts remain outside Git. This inventory reads saved results and documented audit conclusions; it does not rerun ASR, synthesis, tensor likelihoods or full artifact audits.

## Experiment-by-experiment changes and outcomes

### 1. Protected text-adapter strength and target-only weight transfer

[Literature/experiment record](../../../0.6b-joint-goal-literature.md). The accepted halfway voice was the zero-strength endpoint of a new reference-protected text adapter; 100% was its step-44 endpoint. Intermediate strengths 25/50/75% applied the strength once to each LoRA B factor, producing 78 new screen recordings. Endpoints reused independently verified evaluations. All failed the joint goal. Increasing strength was not monotonically better for pronunciation or voice. The exact screen tables below include available WER from saved JSON; the prose document itself omitted strength-search WER.

A separate target-only transfer restored the two target-text projection matrices to full epoch-2 values while keeping reference/control processing and the rest of the halfway backbone unchanged. This required no optimizer and tested a different direction from the small DPO updates. Variant **101 is an inference identifier, not an update count**. Development improved, but custom pronunciation regressed severely; no promotion followed. The first preparation/control snapshot `protected-target-weight-transfer` was superseded by the explicitly provenance-hardened `protected-target-weight-transfer-v2` experiment; do not count those preparations as two successful quality experiments. Evidence: `voice-strength-search/`, `protected-target-weight-transfer-v2/`, their protocol/results/verification receipts.

### 2. Small-pool protected-text DPO

[Protocol/results](../../../0.6b-joint-preference-training.md). Collected 128 samples: sixteen original-training speakers, two texts each, four fixed generation seeds. Twelve speakers fitted preferences and four supplied validation. The audited pool produced 74 fitting pairs across 22 usable cases and 25 validation pairs across 8 cases; no truncation. Of fitting pairs, 47 improved pronunciation and 27 were voice-only; 22 of the first 32 visited pairs improved pronunciation. Failure was not explained by exclusively voice-only fitting.

Trainable capacity: 114,688 FP32 protected target-text LoRA parameters. Acoustic backbone, original encoder/codec and reference/control projections stayed frozen. Standard DPO beta 0.1 used raw summed likelihoods of all 16 codebooks plus natural EOS, frozen halfway-policy reference likelihoods, no real-audio auxiliary CE, dropout disabled, four case-balanced pairs/update, 24 updates, LR 5e-5→1e-5, two warmup updates, clip 1 and no weight decay. Single-trajectory gradient replay matched the full objective in tests.

All checkpoints 4/8/16/24 completed and were audited: 104 new screen recordings plus 26 reused baseline records. Voice means stayed above 0.984, but pronunciation worsened. Updates 16 and 24 produced identical 26 WAV hashes despite fitting accuracy rising to 90.68% and validation accuracy 48.96%. The bounded run ended with no model promotion. This is failure of this small pool/capacity/recipe, not proof that DPO generally fails. Evidence: `joint-preference-rollouts/`, `joint-preference-pilot/{eval-*,decision.json}`.

### 3. Per-reference speaker-input reconstruction

[Protocol/results](../../../0.6b-speaker-reference-adaptation.md). Frozen full epoch-2 acoustic LoRA; only a per-reference input speaker vector changed. Four hash-selected original-training speakers, a distinct target text each, two seeds: 8 clips/checkpoint at 0/8/32 steps, 24 total. Normalized FP32 gradient descent bounded displacement to 5%; loss was main CE+0.3 residual CE. Reference-only reconstruction deliberately omitted the same-reference ICL prefix during optimization, whereas evaluation used normal full ICL and a different transcript. This train/inference mismatch is explicit.

All reference reconstruction losses fell (3.094→2.902, 3.457→3.088, 3.691→3.440, 3.368→3.182), but both adapted checkpoints increased CER and failed the expansion gate. No development/custom expansion occurred. The 32 step cosine difference was tiny; the defensible conclusion is absent voice gain and worse pronunciation, not a perceptually established degradation. Evidence: `speaker-reference-calibration/{results.json,verification.json,decision.json,report.html}`.

### 4. Broader speaker-input DPO

[Protocol/results](../../../0.6b-balanced-speaker-preferences.md). Full epoch-2 source policy; 64 original-training speakers, 48 fitting and 16 validation. After one transcript-overlap exclusion: 95 fitting cases+32 validation, four draws per case, 508 samples, no truncations. Audited 352 fit pairs (187 pronunciation-improving) and 116 validation pairs (65 pronunciation-improving). Pool CER/WER/cosine: fitting 6.03%/13.30%/0.978845; validation 4.34%/10.33%/0.980512. These are GPU candidate-pool summaries, not the CPU screen.

New shared rank 16 speaker-input adapter: 32,768 FP32 parameters, zero output initially, smooth 5% total input bound. Full acoustic weights frozen. DPO beta 0.1, four speaker-balanced pairs/update, FP32 AdamW LR 0.003→0.0003 over 64 planned updates, four warmup, clip 1, WD 0. All 26 baseline WAVs and native no-op controls validated the loader.

Update 8 improved group CER and cosine relative to full-model control, but development CER and custom cosine still failed. Worst paired clip cosine fell 0.005345, a registered warning. At 24, preference-validation loss still improved yet development CER worsened and worst decline reached 0.005584, the second speech warning. Stopped 24; **64 was not trained**. Fit/validation losses at 24 were 0.562056/0.612143, accuracies 85.97%/84.69%; this is speech regression despite improving preference validation, not the declared validation-loss overfit trigger. Evidence: `speaker-preference-pilot/`, `speaker-preference-fit-registration.json`; collection in `balanced-full-lora-rollouts/`.

Read-only error analysis compared 130 matched screen records across five policies. Development errors grew 39→51 from 8→24; substitutions 14→29, deletions 11→12, insertions 14→10 under one deterministic minimum-edit alignment. Space-excluded diagnostics also worsened 3.91%→4.89%, but did not replace official CER. ASR alignments are not phonetic labels. Evidence: `joint-error-analysis/report.html` and JSON.

### 5. Refreshed current-policy speaker DPO

[Protocol/results](../../../0.6b-refreshed-speaker-preferences.md). Start from original speaker-update 8; same 127 conditions/64 speakers, new batch seeds 17301+case_index, four draws/case: 508 new trajectories, zero truncation. Audited 357 fit pairs/94 usable cases/48 speakers (199 pronunciation-improving),120 validation pairs/32 cases/16 speakers (61 pronunciation-improving). Pool fitting 6.42%/14.80%/0.979898; validation 4.83%/12.23%/0.981006. Different draws prevent a paired comparison with the prior pool.

Fresh optimizer and frozen reference at speaker-adapter update 8, 16 additional updates, four speaker-balanced pairs, LR 0.001→0.0001, two warmup, seed 9017; same rank 16/5% bound. Refreshing trajectories, reference/optimizer reset and LR change were bundled; causality is not isolated. Stage 0 reproduced all 26 original speaker-adapter update 8 WAVs. Additional 4/8/16 all completed, no joint pass. The old cross-group stop rule allowed 8 despite substantial custom regression because development cosine improved and worst clip loss stayed below 0.005. The running rule was not rewritten retroactively. Later designs tightened this per-group weakness. Evidence: `refreshed-speaker-rollouts/`, `refreshed-speaker-pilot/`, `refreshed-speaker-fit-registration.json`.

A one-prompt native likelihood diagnostic checked 8 main and 105 residual predictions: exact prefix, main / residual max differences 1.38e-5/1.34e-4, all argmaxes equal, selected log probability difference 0.000153 over 112 tokens. This was numerical evidence on one short continuation, not quality or universal parity. `speaker-adapter-geometry.json` found 98.26%/98.47% common correction energy in original update-8 fitting / validation; nearly parallel input updates do not establish generated-voice collapse.

### 6. Centered speaker-dependent residual DPO

[Protocol/results](../../../0.6b-centered-speaker-preferences.md). Keep original speaker-adapter update 8 and acoustic model frozen; add a zero-initialized 1024×16 matrix (16,384 FP32 parameters) on centered/whitened coordinates from 95 fitting references / 48 equally weighted speakers. Sixteen directions explain 79.64% of fitting-speaker variation; validation/custom refs did not fit the basis. Apply the total original+new correction through the original 5% bound.

Reuse refreshed 508 trajectories/477 pairs. Sixteen speaker-balanced pairs/update, 24 updates, checkpoints 0/8/16/24, LR 0.001→0.0001, two warmup, beta 0.1, seed 11017. Changed centering, freezing, batch coverage and schedule together. Baseline 26 WAVs exactly matched speaker-adapter update 8. At 8, development gained only one character 39→38 and WER increased; custom errors 8→5. At 16/24 development passed both screen thresholds, but custom CER and cosine deteriorated twice, triggering stop 24. No joint pass. Validation accuracies near chance (53.59% at 16; 51.77% at 24) and falling validation loss did not satisfy the specific loss-rise overfit trigger. Evidence: `centered-speaker-pilot/`, `centered-speaker-fit-registration.json`, `centered-speaker-basis.pt`.

The new correction was speaker-dependent in the measured input space: common-energy fraction 0.0113% for fitting and 5.07% for validation at 8. This diagnoses parameterization, not speech identity. `preference-confirmation-separation.json` found zero normalized-text, speaker-ID and reference-WAV-hash overlap between 127 preference conditions and 57 confirmation conditions; it does not audit original SFT/pretraining exposure.

### 7. Three frozen-weight reference-routing ablations

All start from centered update 16, use the same 26-recording screen, no optimizer, no per-case selection, unchanged original speaker adapter and full generated-frame weights. Native no-op controls reproduced the same historical WAV. All candidates completed with natural EOS and independent matrix/state/trajectory/ASR/cosine audits.

- [Broad reference/control routing](../../../0.6b-reference-prefix-routing.md): halfway 196 main-decoder matrices plus 2 text projections on reference/control prefill rows; full target-text rows and cached frames. Both cosine means passed but CER failed. Attention couples branches, so this was not a claim of isolated voice circuitry. Evidence: `reference-prefix-half-routing/`.
- [Codec/BOS-only routing](../../../0.6b-reference-codec-routing.md): 196 main matrices only on reference codec/BOS rows; text projections, control, speaker and text rows remain full. An extra independent full-prefill cache proved all earlier hidden states exact on every case. Both CERs and custom cosine failed. Evidence: `reference-codec-half-routing/`.
- [Speaker+codec routing](../../../0.6b-reference-condition-routing.md): add speaker position 6 to codec-only mask; positions 0–5 exact, text hidden states can change. Custom CER worsened sharply; rejected. Evidence: `reference-condition-half-routing/`.

No arbitrary position sweep or successful 114-recording confirmation follows these three failures.

### 8. Middle-attention DPO

[Protocol/results](../../../0.6b-middle-attention-preferences.md). Return to frozen original speaker-adapter update 8/full epoch 2 and zero new rank-4 Q/V LoRA in main layers 10–17: 16 projections, 163,840 FP32 trainable parameters. Unlike later masked supervision, these adapters act throughout reference, target and generated rows. Reuse 357 fit/120 validation joint pairs from 508 refreshed samples. Beta 0.1, 16 speaker-balanced pairs/update, 32 updates, checkpoints 0/4/8/16/32, LR 2e-5→2e-6, two warmup, seed 12117, WD 0, clip 1, effective BA matrix norm cap 1%.

All 480 paired trajectory likelihoods initially equaled the frozen reference cache; all 26 baseline WAVs matched speaker-adapter update 8. Update 4 passed pronunciation in both groups but missed custom voice; 8 passed custom CER but missed development CER/custom voice. Updates 16/32 triggered two speech-harm warnings, stopping 32. Fit/validation losses continued improving to 0.675364/0.688956 and accuracies 92.41%/71.30%. Maximum effective displacement 0.04770% was small, yet speech regressed. Small weight movement is not a voice guarantee. Evidence: `middle-attention-pilot/`; `middle-attention-paired-diagnostics.json` notes 14/26 WAVs unchanged 4→8 and one development clip accounts 12 of 16 extra character errors.

### 9. Absolute-quality-filtered DPO

[Protocol/results](../../../0.6b-quality-preference-proposal.md). Strict chosen CER <3% and cosine >0.984 retained 137/357 fitting pairs, 41 cases/29 speakers versus 94 cases/48 speakers. Full 120 pair/16 speaker validation remained unfiltered; 39 validation pairs passing the same filter was a read-only coverage diagnostic, not its fitting/eval scope. Zero CER plus the voice threshold would leave 119 pairs/26 speakers; not used. Same middle-attention baseline/capacity/settings/checkpoints, changed subset and its speaker-balanced visit schedule. No evaluation text entered gradients.

All 0/4/8/16/32 screens audited; none qualified. Stopped 32 on speech regression. Effective fitting accuracy 97.79%, full validation 65.73%, losses 0.644927/0.687899; lower chosen NLL/greater chosen likelihood did not prevent voice degradation. Maximum matrix movement 0.06900%. Diversity loss and worse development 4 result do not support adopting filtering as established improvement. Evidence: `quality-filtered-pilot/`, `quality-filtered-fit-registration.json`, `quality-filtered-comparison.{json,html}`.

Exploratory 10,000-draw paired block bootstrap kept both seeds together (10 development speaker blocks; only 3 custom text blocks). Filtered-minus-unfiltered 4 development CER+1.676 pp, marginal 95% interval+0.179..+3.517; custom CER−0.234 pp interval−2.273..+4.717, custom cosine−0.000155 interval−0.001095..+0.000422. These are inspected-screen diagnostics, not population guarantees or replacements for strict thresholds. A custom zero-CER clip had cosine 0.964556 (seed 2026) versus 0.986501 (seed 2027), duration 4.48/4.64 s: duration alone does not explain voice drift, and the bad seed was retained.

### 10. CE-anchor gradient diagnostic and calibrated DPO+CE

[Protocol/results](../../../0.6b-calibrated-quality-preferences.md). The proposed weight 0.1 mean chosen-token NLL arm was **not trained**. Training-only no-update gradient measurement over 29 fitting speakers: DPO norm 1.764817, unit CE norm 0.089572, cosine 0.270828. Weight 0.1 contributed 0.508% of DPO norm and combined-direction cosine 0.999988, so a 25% relative gradient target gave 4.925715, rounded to 5 before fitting. First 16 pair probe gave 0.369% at weight 0.1. This is local engineering calibration, not an optimal universal coefficient.

Actually trained objective: `DPO+5×chosen mean token NLL`, same 137 fitting pairs, full 120 validation, speaker-adapter update 8 baseline, rank-4 Q/V/matrices/settings/checkpoints as filtered DPO. Initial 480 likelihoods/26 WAVs exact. At 4, 23/26 WAVs matched filtered 4 including all 6 custom; development gained 4 errors. At 8/16 speech warnings accumulated, stop 16; **32 was not run**. Effective-fitting / validation DPO losses 0.656860/0.687764 and chosen NLL fell while speech worsened. Actual update 4 BA differed 24.97% of filtered update norm, cosine 0.969210, showing a meaningful parameter change despite no speech gain. Evidence: `quality-calibrated-pilot/`, `quality-anchor-gradient-diagnostic.json`, `quality-anchor-update-diagnostic.json`; optional 0.1 arm has no CER/WER/cos result.

### 11. Residual-temperature calibration

[Protocol/results](../../../0.6b-residual-sampling-calibration.md). Freeze unfiltered middle-attention update 4+speaker-adapter update 8. Change only residual sampling temperature 0.9→0.6; both keep residual top-k 50/top-p 1/sampling, main temperature 0.9/top-k 50/top-p 1/maximum 320. Residual embeddings feed the next main step, so this is not acoustics-only. Four hash-selected preference-validation speakers, two cases each, seeds 2026/2027: 16/arm, 32 total. No speaker / text/reference hash overlap with confirmation; original SFT exposure remains.

Both arms audited, no truncation. Group averages improved, but worst clip cosine loss 0.00794077 exceeded 0.005 limit; two clips failed. Seed 2026 CER worsened 5.54→6.88% while 2027 improved 4.97→3.06%. The expansion gate failed; **no 26-recording screen or 114 expansion**. Preparation initially rejected matching explicit `subtalker_dosample=True`; guard fixed before successful registration, before generation, not a failed quality trial. Evidence: `residual-temperature-calibration/{protocol.json,results.json,verification.json,report.html}`.

### 12. Confirmation implementations versus actual confirmation

[Inventory](../../../0.6b-joint-confirmation.md). Prepared support for speaker, precision, refreshed, centered, broad prefix, codec, condition, attention, quality and calibrated candidates does **not** mean each ran 114. These screen candidates failed entry. Reference-profile and profile-DPO also had prepared separate confirmation harnesses; no qualifying native confirmation result is claimed for those in this family.

The later **custom-reference full-generation PAD intervention** did pass 26-recording screen (development CER 1.675978%, cosine 0.984636086; custom CER 2.803738%, cosine 0.984976113). Its separate original 114-recording confirmation actually completed and failed. All 26-recording screen WAVs/scores/code trajectories reproduced, all 114 terminated, frozen weights unchanged, finite independent cosine error 0. The unchanged/inactive 88 development branch revealed the centered update 16 fallback's wider pronunciation gap; custom-only routing cannot repair that branch. Exact 114 table below. Evidence: `reference-prefix-generation-pad-confirmation/`; independent `prefix-pad-goal-audit.json` says `goal_criteria_met:false`. No promotion. The pre-existing generic confirmation document contains earlier chronological “not launched yet” passages; its later completed PAD section supersedes those only for PAD, not every candidate type.

### ASR calibration on real human recordings

The pinned recognizer measured 0.318% CER across 44 real development clips and 0.186% on the ten-clip screening subset, as documented in `0.6b-joint-goal-literature.md` and external `real-development-asr-calibration/`. WER and cosine are not quoted in that source passage and are not supplied here. These observations do not establish a mathematical synthesis floor or justify changing thresholds.

## Research sources and actual local use

The following summarizes the already recorded literature review, not a fresh claim to reproduce these papers or independently revalidate every publication result. Links point to the primary sources cited there; architectures/languages/scorers differ.

| Source | Relevant idea | What was actually tested locally / limitation |
|---|---|---|
| [Koel-TTS](https://arxiv.org/html/2502.05236v2) | Joint CER/speaker Pareto preference selection | Generated-versus-generated joint filters and bounded DPO; not its full reward-aware recipe, codec or scale. |
| [Preference Alignment Improves LM-Based TTS](https://arxiv.org/html/2409.12403v1) | Generated preferred/rejected pairs; balanced quality objectives | Local generated pairs with immutable all-sample reporting; no assumption real audio is automatically preferred. |
| [INTP](https://aclanthology.org/2025.acl-long.598/) | Error-category-targeted intelligibility preferences | Motivation for diagnostics; its corpus/multilingual recipe not reproduced. |
| [Beyond Fixed Targets](https://arxiv.org/html/2610.17838v1) | Current-policy trajectories, GRPO/self-distillation | Refreshed staged DPO tested; no GRPO or on-policy self-distillation reproduction. Residual KL prototype tested numerically only, not trained. |
| [Qwen3-TTS report](https://arxiv.org/html/2601.15621v1) and [implementation](https://github.com/QwenLM/Qwen3-TTS/blob/main/qwen_tts/core/models/modeling_qwen3_tts.py) | Main semantic/residual codebook architecture | All 16 raw likelihoods+EOS, selective Q/V and residual temperature/routing comparisons; semantic/acoustic split does not guarantee identity isolation. |
| [PEFT-TTS](https://www.isca-archive.org/interspeech_2025/kwon25_interspeech.pdf) | Pronunciation/identity trade-off and prompt / text / attention adapters | Protected text then bounded attention adaptation; F5 flow/Korean adapter recipe not copied. Rank increase alone not evidence of solution. |
| [TKTO](https://aclanthology.org/2026.acl-short.59/) | Token-level preferences via contrastive policies | Deferred; no local TKTO training or phonetic-label corpus claimed. |
| [Weight-Space Programming report](https://rikiyat.github.io/blog/weight-space-programming/), [cross-task](https://arxiv.org/abs/2611.09382), [robot-policy](https://arxiv.org/abs/2611.09376) | Mechanism-specific edits with matched controls and negative results | Training-only subspace diagnostic on 28,001 clips/435 equally weighted speakers: top 16 explained 72.01% variation but only 1.08% text-update overlap vs 1.52% random mean across 64 controls. Hypothesis unsupported; no subtraction edit. Local routing/transfer/PAD were separate controlled inference mechanisms, not WSP paper reproductions. |
| [ASRRL-TTS](https://arxiv.org/html/2407.05421v1) | Optimize speaker conditioning with speech rewards | Reference-reconstruction calibration failed; later shared speaker DPO differed from PPO. Neither failure disproves ASRRL's generated speech reward method. |
| [VoiceTTA](https://arxiv.org/html/2606.26534v1) | Small learned prefixes at inference | Motivation only; no local F5 prefix/flow-likelihood method implemented. |
| [Qwen accent steering](https://arxiv.org/html/2603.05977v1) | Layer-sensitive generated token interventions | Informed bounded middle-layer choice; no matched native/accented Mongolian activation direction learned. Paper's Resemblyzer metric differs. |
| [UtterTune](https://arxiv.org/html/2508.09767v2) | Attention LoRA with explicit pronunciation annotations | Motivation for actual pronunciation supervision, not a local Japanese/phoneme-token reproduction. |
| [ReStyle-TTS](https://arxiv.org/html/2601.03632v1) | Separate conditioning guidance and orthogonal updates | Not reproduced: current Qwen lacks assumed trained unconditional branches/distinct style adapters. |
| [MPO](https://arxiv.org/html/2509.00685v1) | Absolute chosen quality and CE regularization | Filtered DPO and calibrated weight 5 anchor actually tested, both failed the joint screen. Weight 0.1 remained a proposal. |
| [Does Expert Supervision Improve TTS?](https://arxiv.org/html/2610.25292v1) | Chosen likelihood not reliable quality proxy | Fixed 480 trajectory probes and actual speech measured; no real/expert bridge objective claimed. |
| [Magpie preference guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/tts/magpietts-po.html) | Varied generated candidates/absolute CER filtering | Local quality filter tested; no transplanted Magpie online GRPO or claim that an anchor is universally needed. |

## Exact saved metric tables

The following tables are generated from saved JSON summaries, rather than rounded prose. Every two-group row in the screen table uses the common 26 panel. The same zero-update baseline can appear in several runs because complete waveform parity was required; these are separate recorded controls, not independent evidence of improvement. A row's successful integrity audit does not mean its quality goal passed.

### Protected-adapter strength (same 26-recording screen)

| Strength | Development CER % | WER % | Cosine | Custom CER % | WER % | Cosine |
|---|---:|---:|---:|---:|---:|---:|
| 0% | 9.310987 | 23.563218 | 0.986513475 | 7.242991 | 23.437500 | 0.984960198 |
| 25% | 9.217877 | 24.712644 | 0.986805737 | 8.177570 | 26.562500 | 0.984920104 |
| 50% | 7.914339 | 19.540230 | 0.987309915 | 8.177570 | 26.562500 | 0.985059530 |
| 75% | 9.683426 | 24.712644 | 0.987212324 | 9.579439 | 28.125000 | 0.984864563 |
| 100% | 7.541899 | 18.965517 | 0.987029678 | 6.074766 | 18.750000 | 0.983740389 |

Evidence: `ROOT/voice-strength-search/results.json` for 25/50/75%; `ROOT/reference-protected-pilot/eval-000000/results.json` and `eval-000044/results.json` for reused 0/100% endpoints.

### Common 26-recording screen: all saved fitted checkpoints

| Run / checkpoint | Development CER % | WER % | Cosine | Custom CER % | WER % | Cosine |
|---|---:|---:|---:|---:|---:|---:|
| joint-preference-pilot / 000000 | 9.310987 | 23.563218 | 0.986513475 | 7.242991 | 23.437500 | 0.984960198 |
| joint-preference-pilot / 000004 | 9.590317 | 24.712644 | 0.986487085 | 7.242991 | 23.437500 | 0.984960198 |
| joint-preference-pilot / 000008 | 10.893855 | 25.862069 | 0.986521396 | 7.242991 | 23.437500 | 0.985224863 |
| joint-preference-pilot / 000016 | 10.893855 | 25.862069 | 0.986523429 | 8.177570 | 26.562500 | 0.985185583 |
| joint-preference-pilot / 000024 | 10.893855 | 25.862069 | 0.986523429 | 8.177570 | 26.562500 | 0.985185583 |
| speaker-preference-pilot / 000000 | 4.283054 | 12.643678 | 0.984564760 | 3.271028 | 10.937500 | 0.975647797 |
| speaker-preference-pilot / 000008 | 3.631285 | 8.620690 | 0.984888551 | 1.869159 | 6.250000 | 0.979494135 |
| speaker-preference-pilot / 000024 | 4.748603 | 12.643678 | 0.985096195 | 1.869159 | 7.812500 | 0.980299274 |
| refreshed-speaker-pilot / 000000 | 3.631285 | 8.620690 | 0.984888551 | 1.869159 | 6.250000 | 0.979494135 |
| refreshed-speaker-pilot / 000004 | 5.400372 | 13.793103 | 0.985295412 | 2.102804 | 4.687500 | 0.979138851 |
| refreshed-speaker-pilot / 000008 | 7.635009 | 17.241379 | 0.985090950 | 7.710280 | 18.750000 | 0.978850146 |
| refreshed-speaker-pilot / 000016 | 3.165736 | 9.195402 | 0.985069680 | 3.738318 | 9.375000 | 0.981217593 |
| centered-speaker-pilot / 000000 | 3.631285 | 8.620690 | 0.984888551 | 1.869159 | 6.250000 | 0.979494135 |
| centered-speaker-pilot / 000008 | 3.538175 | 9.770115 | 0.985470495 | 1.168224 | 1.562500 | 0.980313410 |
| centered-speaker-pilot / 000016 | 1.675978 | 5.747126 | 0.984636086 | 3.738318 | 9.375000 | 0.977098246 |
| centered-speaker-pilot / 000024 | 1.024209 | 3.448276 | 0.984933525 | 4.205607 | 10.937500 | 0.975869189 |
| middle-attention-pilot / 000000 | 3.631285 | 8.620690 | 0.984888551 | 1.869159 | 6.250000 | 0.979494135 |
| middle-attention-pilot / 000004 | 2.048417 | 4.597701 | 0.985057265 | 1.635514 | 4.687500 | 0.979095777 |
| middle-attention-pilot / 000008 | 3.538175 | 8.620690 | 0.985181090 | 0.934579 | 3.125000 | 0.979519933 |
| middle-attention-pilot / 000016 | 5.214153 | 11.494253 | 0.985060337 | 3.271028 | 9.375000 | 0.979460220 |
| middle-attention-pilot / 000032 | 4.189944 | 10.344828 | 0.984637341 | 3.271028 | 9.375000 | 0.979164014 |
| quality-filtered-pilot / 000000 | 3.631285 | 8.620690 | 0.984888551 | 1.869159 | 6.250000 | 0.979494135 |
| quality-filtered-pilot / 000004 | 3.724395 | 8.045977 | 0.985655445 | 1.401869 | 6.250000 | 0.978940795 |
| quality-filtered-pilot / 000008 | 3.724395 | 8.045977 | 0.985864696 | 0.934579 | 4.687500 | 0.979057531 |
| quality-filtered-pilot / 000016 | 6.052142 | 13.793103 | 0.985048652 | 1.401869 | 6.250000 | 0.976852943 |
| quality-filtered-pilot / 000032 | 5.679702 | 12.068966 | 0.985088491 | 3.738318 | 12.500000 | 0.977800051 |
| quality-calibrated-pilot / 000000 | 3.631285 | 8.620690 | 0.984888551 | 1.869159 | 6.250000 | 0.979494135 |
| quality-calibrated-pilot / 000004 | 4.096834 | 8.620690 | 0.985617495 | 1.401869 | 6.250000 | 0.978940795 |
| quality-calibrated-pilot / 000008 | 3.724395 | 7.471264 | 0.985615444 | 1.635514 | 6.250000 | 0.978588094 |
| quality-calibrated-pilot / 000016 | 5.586592 | 12.068966 | 0.985406554 | 5.140187 | 14.062500 | 0.975899378 |
| protected-target-weight-transfer-v 2 / 000000 | 9.310987 | 23.563218 | 0.986513475 | 7.242991 | 23.437500 | 0.984960198 |
| protected-target-weight-transfer-v 2 / 000101 | 7.541899 | 18.965517 | 0.986651519 | 13.551402 | 39.062500 | 0.984899938 |
| reference-prefix-half-routing | 4.934823 | 14.367816 | 0.984177110 | 5.607477 | 15.625000 | 0.985577792 |
| reference-codec-half-routing | 3.631285 | 10.344828 | 0.984572732 | 3.037383 | 6.250000 | 0.981488377 |
| reference-condition-half-routing | 3.072626 | 10.344828 | 0.984678105 | 7.710280 | 20.312500 | 0.982240309 |

Evidence for each fitted row: `ROOT/<run>/eval-<checkpoint>/results.json`; routing rows: `ROOT/<run>/results.json`. Independent checkpoint audits are stored alongside/in the owning run as documented above. Row 101 is the target-transfer inference variant, not 101 training updates.

### Separate reference-reconstruction calibration (8 recordings per checkpoint)

| Input gradient steps | CER % | WER % | Original-reference cosine |
|---|---:|---:|---:|
| 0 | 3.021978 | 3.448276 | 0.983795375 |
| 8 | 4.120879 | 8.620690 | 0.983140402 |
| 32 | 4.120879 | 13.793103 | 0.983786754 |

Evidence: `ROOT/speaker-reference-calibration/results.json`. Four speakers, two seeds; this is not the development/custom screen.

### Separate residual-temperature calibration (16 recordings per arm)

| Arm | CER % | WER % | Original-reference cosine |
|---|---:|---:|---:|
| control | 5.258126 | 15.243902 | 0.983575914 |
| candidate | 4.971319 | 12.804878 | 0.984245777 |

Evidence: `ROOT/residual-temperature-calibration/results.json`; control 0.9, candidate 0.6. This four-speaker calibration is neither the 26-recording screen nor the 114-recording confirmation.

### Actually completed original 114 PAD confirmation

| Group | Recordings | CER % | WER % | Original-reference cosine |
|---|---:|---:|---:|---:|
| development | 88 | 4.222615 | 10.730594 | 0.984226523 |
| custom | 26 | 3.279632 | 8.270677 | 0.983555874 |

Evidence: `ROOT/reference-prefix-generation-pad-confirmation/speech/results.json`, its native `verification.json`, and `ROOT/prefix-pad-goal-audit.json`. Strict goal failed in both groups on CER, and custom also failed cosine.

### Coverage and non-result exclusions

- Reference-protected residual-KL prototype: numerical tests only; no trained checkpoint or generated quality result in these source documents.
- Weight-space speaker-subspace subtraction: no edit, because the alignment diagnostic did not support it.
- Optional quality-anchor coefficient 0.1: gradient diagnostic only; no speech evaluation. The actually fitted coefficient was 5.
- Speaker-preference update 64 and calibrated-quality update 32: stopped before those checkpoints; no result to invent.
- Prepared generic/profile/profile-DPO confirmation variants: implementation and entry-gate tests are not native 114-recording results. Only the separate PAD confirmation above is a completed 114 result in this inventory.
- Superseded target-transfer preparation and residual-temperature registration rejection: provenance/guard events, not additional successful model-quality trials.
- Prose descriptions saying “running” or “not yet” are historical status entries; later complete audited sections supersede them only for the named experiment.
- Current masked-supervision v 2 checkpoint 8 is outside this inventory; do not merge a partial/live result into these completed tables.

## Implementation entry points (new historical modules, not edits to registered runs)

| Experiment | Main implementation |
|---|---|
| Strength / target transfer | [voice_strength_search.py](../../../../training/voice_strength_search.py), [target_weight_transfer.py](../../../../training/target_weight_transfer.py) |
| Small-pool DPO | [preference_rollouts.py](../../../../training/preference_rollouts.py), [preference_pilot.py](../../../../training/preference_pilot.py) |
| Reference reconstruction | [speaker_reference_adaptation.py](../../../../training/speaker_reference_adaptation.py) |
| Broad speaker DPO | [balanced_preference_rollouts.py](../../../../training/balanced_preference_rollouts.py), [speaker_preference_adapter.py](../../../../training/speaker_preference_adapter.py), [speaker_preference_training.py](../../../../training/speaker_preference_training.py) |
| Refreshed speaker DPO | [refreshed_speaker_rollouts.py](../../../../training/refreshed_speaker_rollouts.py), [refreshed_speaker_training.py](../../../../training/refreshed_speaker_training.py) |
| Centered residual | [centered_speaker_adapter.py](../../../../training/centered_speaker_adapter.py), [centered_speaker_training.py](../../../../training/centered_speaker_training.py) |
| Broad prefix routing | [reference_prefix_routing.py](../../../../training/reference_prefix_routing.py), [reference_prefix_comparison.py](../../../../training/reference_prefix_comparison.py) |
| Codec routing | [reference_codec_routing.py](../../../../training/reference_codec_routing.py), [reference_codec_comparison.py](../../../../training/reference_codec_comparison.py) |
| Speaker+codec routing | [reference_condition_routing.py](../../../../training/reference_condition_routing.py), [reference_condition_comparison.py](../../../../training/reference_condition_comparison.py) |
| Middle attention | [middle_attention_adapter.py](../../../../training/middle_attention_adapter.py), [middle_attention_training.py](../../../../training/middle_attention_training.py) |
| Filtered / anchored preferences | [quality_preference_objective.py](../../../../training/quality_preference_objective.py), [quality_preference_training.py](../../../../training/quality_preference_training.py), [calibrated_quality_training.py](../../../../training/calibrated_quality_training.py) |
| Residual sampling | [residual_sampling_calibration.py](../../../../training/residual_sampling_calibration.py) |
| Original confirmation | [joint_confirmation.py](../../../../training/joint_confirmation.py), [confirm_joint_candidate.py](../../../../training/confirm_joint_candidate.py), [reference_prefix_generation_pad_confirmation.py](../../../../training/reference_prefix_generation_pad_confirmation.py) |
