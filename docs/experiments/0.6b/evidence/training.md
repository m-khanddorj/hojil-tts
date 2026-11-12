# Early 0.6B training and voice-preservation evidence inventory

[Summary page](../README.md) · [Comprehensive report](../report.md)

**Jump to:** [Training audit](#1-recovered-training-and-optimization-audit) · [Literature comparisons](#2-literature-guided-short-continuations) · [Epochs and stopping](#4-continuation-through-epoch-6-and-the-stopping-decision) · [Reserved test](#5-reserved-test-chronology-and-interpretation) · [Reference voice](#6-user-reference-voice-natural-controls-and-halfway-feedback) · [Targeted training](#8-two-early-pronunciation-training-pilots-on-the-halfway-backbone) · [Exact metric tables](#quantitative-appendix-extracted-from-raw-reports)

This factual source inventory supports the comprehensive experiment report. It covers the recovered historical training, FP32 continuation and literature comparisons, six fresh epochs, reserved-test evaluation, the user's reference-voice investigation, the two early targeted training pilots, and merge precision. Later preference, routing, and masked-supervision experiments have separate inventories. No model job, synthesis, ASR inference, checkpoint edit, or selection change was performed to assemble this document.

The quantitative appendix below is extracted from existing JSON reports rather than transcribed from rounded documentation. CER/WER percentages are formatted for readability; error-count fractions preserve the exact pooled rates where denominators are available. Cosines are dimensionless internal-encoder diagnostics, **not percentages or human similarity ratings**. Existing verification receipts are evidence of checks performed at the time; this inventory does not claim to have rerun every historical tensor/audio audit.

## Summary for the final report

| Phase | What actually happened | Result and decision |
| --- | --- | --- |
| Recovered historical full-cohort run | Rank-16 LoRA, 12,009,472 trainable parameters; 28,001 clips; one BF16-state epoch | Much better development pronunciation than the base; no evidence that this was optimal. It was not full-parameter fine-tuning. |
| Historical BF16 continuation | 176 more updates using the historical adapter and optimizer | Development CER/WER worsened; not selected. Many low-precision adapter elements stayed bit-identical. |
| FP32 continuation v4 | FP32 trainable parameters and Adam state; frozen BF16 base/forward; 176 updates, selected update 88 | Better single-seed development CER/WER; became the incumbent. Precision, batching, selection and extra training changed together. |
| Literature-guided optimizer comparison | Ratio-1 control and LoRA+ B/A=4 short continuations | Both lost to the incumbent on first-seed WER. No second-seed confirmation was triggered, and LoRA+ was not adopted. |
| Fresh FP32 epoch 1 | Fresh adapters and optimizer, full 28,001-clip pass | Lower teacher-forced losses, but worse pooled development CER/WER than the incumbent; incumbent retained. |
| Fresh FP32 epoch 2 | Exact continuation of epoch 1, lower decaying LR | Improved pooled development CER/WER; selected provisionally. |
| Fresh FP32 epochs 3–6 | Constant 3e-6 LR, exact state continuity, per-epoch two-seed speech checks | Epoch 3 was the development winner. Training stopped after epoch 6 for three consecutive speech regressions, **not** the codec-loss divergence guard. |
| Initial reserved test | Frozen epoch 3 versus previous-best epoch 2, 30 targets × two seeds | Epoch 2 was better on pooled test CER/WER. Historical development selection was not silently rewritten. |
| Expanded six-epoch test | Added epochs 1,4,5,6 after the initial test had been inspected | Epoch 6 had the lowest observed test CER/WER. This ranking is exploratory and did not change the checkpoint pointer. |
| Reference-voice investigation | Exact supplied reference; original encoder equality; controlled base/adapted comparison | User heard the base as closer. The internal cosine direction agreed for this example; no formal listening score was collected. |
| Halfway interpolation | FP32 interpolation of base and exported epoch-2 endpoint | Recovered much of the voice similarity at a pronunciation cost. User: “voice is closer but pronunciation is worse.” |
| Three small inference ablations | Lower main temperature; stronger text projection; full main decoder with original residual predictor | All failed the joint pronunciation/voice screening criterion. All outputs retained. |
| Global text-only LoRA pilot | 114,688 new parameters on frozen halfway backbone; updates 0/22/44 evaluated | Stopped at 44 for voice screening; initial halfway baseline retained. |
| Reference-protected text LoRA pilot | Same scope and training order; route new adapter only through target text | Better pooled pronunciation at 44, but custom voice and a seedwise WER regressed; initial baseline retained. |
| Exact FP32 half-merge | Inference-only test of base + half original FP32 LoRA update | Both group CERs worsened versus accepted halfway blend. Rejected; removing merge rounding was not sufficient. |

## Evidence namespaces and comparability

- **R**, the current external artifact root: `C:/Users/Khan/.codex/visualizations/2026/11/09/01a1bf77-0b17-7391-9781-f859f1fbb0b1`.
- **H**, the recovered historical root: `C:/Users/Khan/.codex/visualizations/2026/11/09/01a1beb0-7b7b-7f73-993e-71ad9a8eefd3`.
- **DEV44**: 44 unseen-speaker development targets, ten speakers. Seed 2026 alone gives 44 recordings, 2,830 normalized reference characters and 438 words. Both 2026/2027 give 88 recordings, 5,660 characters and 876 words. These targets were repeatedly used for selection.
- **LOSS135**: all 44 unseen development clips plus one fixed familiar validation clip from each of 91 eligible speakers; the selection objective averages the familiar and unseen cohort losses equally. Later observability adds 91 fixed matched training clips without adding them to selection. Fixed probes disable dropout and use the same non-streaming ICL contract.
- **TEST30**: 30 targets from ten reserved unseen test speakers, each with a distinct same-speaker reference from the test split; 2026/2027 give 60 recordings, 3,262 normalized characters and 512 words per model. The separate 1,764-row familiar-speaker test cohort was not included in this benchmark.
- **CUSTOM3**: the three previously requested texts, one supplied reference voice. One seed gives three recordings/214 normalized characters/32 words; two seeds give six recordings/428 characters/64 words. These were qualitative/custom diagnostics, separate from DEV44 and TEST30 scores. The texts originated in the example/test presentation, so they must not be described as fresh independent unseen evaluation after their use in these diagnostics.
- **SCREEN26**: ten fixed development prompts, one per development speaker, plus the three custom prompts, each at 2026/2027. This gives 20 development recordings and six custom recordings per checkpoint. Development denominators are 1,074 characters/174 words; custom denominators are 428/64. It is a small screening panel, **not the full DEV44 two-seed panel** and not the later original-114 panel.

Development and test evaluation used Auto language, non-streaming conditioning and fixed seeds reset per case. The preserved test protocol explicitly records main temperature 0.9, top-p 1, top-k 50 and max-new-tokens 320. ASR is `BuzzASR/mongolian`, revision `185f1bf1f562ad4b1bdbfc71811fb165a9235416`, deterministic decoding with beams 1, no-repeat-ngram-size 3, repetition penalty 1.2 and `do_sample=false`. Rates here use punctuation-normalized scoring, not the historical strict-punctuation scores. Later custom/inference trials use their separately pinned CPU FP32 generation protocols; identical panel names alone do not establish identical model construction or numerical backend.

## 1. Recovered training and optimization audit

Source narrative: [training audit](../../../0.6b-training-analysis.md). Primary structured evidence: `R/audit.json`; historical runs `H/qwen06-cv27-full` and `H/qwen06-cv27-continuation`.

“Full” in `qwen06-cv27-full` meant all retained training data. The original 0.6B run froze base parameters and trained rank-16, alpha-16, dropout-0.05 LoRA in 233 linear layers: 196 main-decoder matrices, two text-projection matrices, and 35 residual-predictor matrices. There are 466 A/B tensors and 12,009,472 trainable parameters. Merging these adapters for inference does not make the training full fine-tuning. The older README's full-talker experiment was for 1.7B. A configured 0.6B full-talker alternative (905,788,672 trainable parameters according to the audit) remained unrun; the user ultimately prioritized LoRA. No Runpod GPU was started for this audit.

The historical data were 28,001 Common Voice 27 clips, 32.6970997917 hours, 435 speakers. The top ten speakers supplied 48.32684547% of clips; the median speaker had five. Training used 1,751 updates, nominally 16 clips/update, with a one-clip final update; AdamW weight decay 0.01, clip norm 1, 50-update warmup, peak 5e-5 and cosine decay to 1e-5; about 80% distinct same-speaker ICL and about 50% non-streaming conditioning. Base weights, adapters, gradients and Adam moments were BF16. Recorded environment: torch 2.8.0+cu128, qwen-tts 0.1.1, transformers 4.57.3, peft 0.21.2, local RTX 5070 Laptop 8 GB. Last timer was 2,251.719 seconds, excluding preparation/evaluation.

The historical 176-update continuation used 2,816 clips and LR 1e-5→3e-6, preserving the old optimizer. Its DEV44 WER rose by 1.5981735 percentage points versus the original epoch; the speaker-bootstrap 95% interval was [-4.1450777,+8.2294264] points. This is a rejected observed result, not proof that all continuation hurts.

The audit found 93.55176366% of A elements and 49.76800522% of B elements bit-identical after those updates. That is consistent with low-precision updates being rounded away, but is not a causal proof of a particular speech error. Original matched teacher losses were 3.3347681351 training, 3.3505473157 familiar validation and 3.4104775172 unseen validation; a small gap does not establish either adequate fitting or underfitting.

Implemented corrections were FP32 adapter/gradient/Adam state; fixed development selection with a 0.002 improvement threshold and starting-model eligibility; separate selected versus latest resume state; supervised-position-only vocabulary projection; full-sequence and target-frame memory budgets; persistent smaller OOM budgets; no retry after a possibly partially applied optimizer step; merging the tiny tail into the preceding update (1,750 updates, last group 17 clips); source/data/base/feature fingerprints; target/reference leakage checks; and preserved RNG/optimizer state. None changes the objective into a speaker-identity loss.

### Interrupted or empty setup attempts

At inventory time, `R/fp32-continuation` and `R/fp32-continuation-v2` were empty directories. They are setup attempts with no retained completed result in those directories; their names do not establish completed training or a measured quality comparison. `R/fp32-continuation-v3` contains initialization, a step-zero selected adapter and metrics through update 40, but no completed result or speech comparison. The historical audit says it was interrupted when priorities changed; this inventory does not infer its final optimizer step from a sparse log or load its checkpoint. `fp32-continuation-v4` is the separately retained completed restart.

### Completed FP32 continuation v4

V4 completed 176 updates and selected update 88 using the then-registered 0.002 threshold. DEV44 seed-2026 CER/WER improved versus the historical epoch, but the WER change interval [-5.9523810,+1.1695906] percentage points included no improvement. It became the provisional incumbent, not a listener-validated universal winner. Peak reserved memory was 6.06 GiB versus 6.50 GiB in the historical short continuation; recorded loop timers were 363.36 versus 395.67 seconds, with differing validation/setup boundaries. These are individual laptop observations, not controlled repeated speed benchmarks.

V4's `resume.pt` is step 176 and is **not** the selected update-88 adapter. The selected FP32 adapter SHA is `a549dea51dba99a5cf59f81f8378fe103213b2a5ae7b015659b63b7d7b375e01`; its exported inference model SHA is `8fd169396451ff649c1f3a5bc900258d8f3fadd0967c5cbb8b32826abef96d40`. Export still followed the historical BF16 merge policy for comparability. The experiment combines precision, implementation and extra-training changes; it does not isolate FP32 as a causal factor.

## 2. Literature-guided short continuations

Source narrative: [LoRA literature experiments](../../../0.6b-lora-literature.md). Runs: `R/literature-control` and `R/literature-loraplus-r4`; configuration: `configs/qwen06-lora-continuation.json` and `configs/qwen06-loraplus.json`.

The review motivated an explicit LoRA+ B/A learning-rate ratio-4 comparison against ratio 1, from the same historical epoch-one adapter **and matching Adam state**. B retained the 1e-5→3e-6 maximum/end schedule while A received one-quarter in ratio 4. Both used rank 16/dropout 0.05, identical data order, up to 176 updates, evaluations every 44 updates and patience 2. The absolute minimum validation-loss checkpoint was separated from the patience threshold, and its matching optimizer/RNG was saved.

Both started at selection loss 3.4244882302 with identical initial/first-step loss. The control ran 176 updates, selected 132, best loss 3.420481 (documentation precision). LoRA+ stopped at 88, selected 44, best loss 3.422729. Both reserved at most 6,499,074,048 GPU bytes in logged training. Neither improved seed-2026 DEV44 WER over v4, so the prespecified second-seed challenger confirmation **did not run**. The control WER interval versus incumbent was [-2.5996534,+6.2962963] percentage points; LoRA+'s documented interval was approximately [-2.15,+8.31]. Keep the negative result without generalizing to fresh LoRA+, other ratios, or other TTS architectures.

The spectrum diagnostic (`R/incumbent-adapter-spectrum.json`) measured the learned update, not pretrained weights: median ranks for 90%/99% update energy were main decoder 7/15, text projection 3.5/10.5, residual predictor 10/16. It motivated future capacity tests but did not validate pruning or larger ranks.

The existing literature document records LoRA+, rsLoRA, DoRA, PiSSA, EVA, LoRA-FA, the Qwen3-TTS report, model-selection overfitting literature, low-resource VoxCPM2 adaptation and AnLR-LoRA. In this early round **only the ratio-1/ratio-4 comparison was executed**. Rank sweeps, rsLoRA/DoRA/PiSSA/EVA/LoRA-FA/AnLR changes and data/speaker-sampling interventions were proposed, not measured successes. This inventory preserves those historical distinctions and is not a new independent verification of every paper claim.

## 3. Fresh FP32 epochs 1 and 2

Source narratives: [fresh FP32 epoch](../../../0.6b-fresh-fp32.md), [second epoch](../../../0.6b-second-epoch.md). Runs: `R/fresh-fp32-epoch1`, `R/fresh-fp32-epoch2`.

Fresh epoch 1 started from the original pinned base with new LoRA and Adam state. “FP32” means trainable parameters, gradients and moments; frozen base storage and training autocast remained BF16. Rank/alpha/dropout, conditioning mixture and first-codebook CE + 0.3 residual-codebook CE were retained. It completed 28,001 clips/1,750 updates, final/best update 1,750, with peak reserved memory 6.12 GiB, no OOM retries and a loop timer of 3,622 seconds before final evaluation. Lower matched training and both validation losses did not translate into a two-seed speech win. V4 remained incumbent after the fresh model's pooled WER worsened by 0.4566210 points (95% interval [-4.2039356,+6.7524116]).

All outputs stayed in scores. Fresh epoch 1 improved WER on 32 recordings, tied 28, worsened 28; WER≥50% cases rose from three to six. Seed-2026 case 020 alone added 12 word and 57 character errors, more than the net +4 words/+50 characters; duration rose 4.40→8.64 seconds. This was a generation-robustness flag, not a human phonetic diagnosis.

Observability added fixed dropout-disabled matched train/familiar/unseen probes, component and token-weighted/equal-speaker losses, gap curves, component/global gradient norms, clipping frequency, adapter movement/unchanged fractions, learning rates, coverage, timing, memory and OOM counts, CSV/JSON exports and refreshed charts. It fixed a duplicate patience increment and kept training probes out of selection. Direct comparison of dropout-on minibatch loss with dropout-off validation was explicitly discouraged.

Epoch 2 restored epoch 1's exact selected adapter, Adam and RNG, continued the next deterministic shuffle epoch, and used LR 1e-5→3e-6 without a new warmup. It completed another 1,750 updates/28,001 clips: cumulative 3,500 updates/56,002 exposures. Final/best selection loss was 3.36690, down from 3.38686; peak reserved memory 6.13 GiB; no OOM retries; about 65.6 minutes before final evaluation.

Epoch 2 became the provisional development winner: versus v4, 23 fewer word and 48 fewer character errors, pooled WER -2.6255708 points (95% interval [-5.5276382,+1.3377926]). It beat fresh epoch 1 on both seeds but was slightly worse than v4 on seed 2026 and much better on 2027. WER improved/tied/worsened on 35/33/20 recordings versus v4; WER≥50% cases stayed at three. No independent-test or naturalness claim followed from this selection.

## 4. Continuation through epoch 6 and the stopping decision

Source narrative: [until-overfit experiment](../../../0.6b-until-overfit.md). Begin at `fresh-fp32-epoch2/best-checkpoint-001750.pt`; keep Adam/RNG and LR 3e-6 without warmup restart.

The initial operational codec-loss guard required either validation cohort to be ≥0.005 above its previous minimum **and** its matched training probe ≥0.002 below the training value at that minimum, for five consecutive checks. Checks occurred every 100 updates and at boundaries; anchors/counters persisted; duplicate cumulative checks did not count. The configured ten-additional-epoch work block was a work limit, not an overfit finding.

After the user's request for CER/WER each epoch, the continuous run was paused at cumulative update 7,200. Exact epoch-3/4 snapshots at 5,250/7,000 were preserved. The epochwise driver resumed from the exact epoch-4 state, replaying the already saved 200 partial-epoch-5 updates rather than skipping data. Subsequent complete endpoints were scored before advancing. The separate continuous-run artifacts remain historical; partial and accepted-trajectory work must not be double-counted as distinct complete epochs.

Epoch 3 became the development winner. Versus epoch 2 it removed 20 word and 86 character errors, WER≥50% recordings fell three→zero, and the descriptive speaker-bootstrap WER interval was [-4.3668122,-0.5714286] percentage points. Epoch 4 regressed on both seeds; WER delta versus 3 was +2.2831050 points, interval approximately [+0.11,+4.38]. Epoch 5 partially recovered but did not beat 3 pooled; delta +0.7990868 points, interval approximately [-1.77,+3.11].

Before seeing epoch-6 results, an additional speech rule was registered: stop after three consecutive endpoints worse in **both** pooled CER and WER than the selected endpoint while matched training loss is ≥0.002 lower. Epochs 4/5/6 met this rule. Stop reason was `speech_regression_patience`; the codec-loss guard **never triggered**, both counters remained zero, and validation losses continued to improve. Epoch 6 WER delta versus 3 was +1.8264840 points, descriptive interval approximately [-1.06,+4.99]. Do not rewrite this as proven classical validation-loss overfitting.

The best combined validation loss was 3.34933 at cumulative update 10,500, exactly the epoch-6 adapter already speech-evaluated; tensor equality justified reusing its audio. It did not beat epoch 3. Accepted trajectory: six complete epochs, 10,500 updates, 168,006 clip exposures. Epoch 7 was stopped during initialization; retained files include run metadata/probe/cache but no initial checkpoint or training update. The partial earlier epoch-5 work remains separately preserved. No seventh completed epoch or score exists in this evidence.

### Fixed-probe losses (five-decimal display)

| Endpoint | Matched training | Familiar validation | Unseen validation | Interpretation |
| --- | ---: | ---: | ---: | --- |
| V4 incumbent, measured on fresh-run probes | 3.34997 | 3.41274 | 3.43130 | Same probe identities as fresh run |
| Fresh epoch 1 | 3.30862 | 3.37518 | 3.39855 | Better loss but rejected speech comparison |
| Fresh epoch 2 | 3.28816 | 3.35227 | 3.38152 | All improved; provisional speech selection |
| Fresh epoch 3 | 3.27862 | 3.34946 | 3.37782 | Speech-stop training anchor |
| Fresh epoch 4 | 3.27108 | 3.34274 | 3.37379 | First speech-regression endpoint |
| Fresh epoch 5 | 3.26484 | 3.33861 | 3.36898 | Both codec guard counters zero |
| Fresh epoch 6 | 3.25807 | 3.33190 | 3.36676 | Gaps .07383/.10869; codec guard zero |

Epoch-3/4 validation entries come directly from their retained `best.json` metrics; the remaining displayed probe values follow the historical documentation. These are rounded display values, not exact-equality claims.

## 5. Reserved-test chronology and interpretation

Source narrative: [held-out test](../../../0.6b-heldout-test.md). Initial comparison `R/heldout-epoch3-vs-epoch2`; expansion `R/heldout-all-epochs`.

The initial epoch-3 versus epoch-2 comparison froze both checkpoint/model/adapter hashes before opening TEST30. Split/cohort/manifest, reference identity, text/PCM/speaker leakage and original reserved-speaker assignment were checked before synthesis. Every target used a distinct same-speaker test reference. Epoch 3 was slightly better at seed 2026 but materially worse at 2027; pooled totals added 13 word/35 character errors. WER delta was +2.5390625 points, 95% speaker-bootstrap interval [0,+5.0847458]. It worsened pooled WER for eight of ten speakers; recording-level improved/tied/worse counts were 13/28/19; WER≥50% recordings increased one→four. These are sample-specific ASR results, not population certainty or naturalness ratings.

Listening examples 001/004/007 were chosen before scoring (first prompt from the first three test speakers), not for favorable errors. The earlier preview during synthesis was explicitly development case 001. All-case blind pages were generated, but **no formal human ratings were collected**. The original 120 recordings were all retained and verified.

The user then requested remaining epoch evaluations. Epochs 1/4/5/6 added 240 recordings, with the original 120 reused, for 360 across six frozen endpoints. Epoch 6 had the lowest observed pooled rates; its WER difference from epoch 2 was -1.953125 points, interval [-8.1081081,+2.0771513]. Because the test was already inspected and six endpoints were ranked, this was explicitly exploratory. It did not justify retuning on that test, claiming untouched confirmation, or changing the historical development-selected epoch 3. It also prevents the earlier development regression from being described as universal generalization deterioration.

## 6. User reference voice, natural controls and halfway feedback

Source narrative: [reference investigation](../../../0.6b-voice-reference-analysis.md). Supplied audio: `C:/Users/Khan/.codex/visualizations/2026/11/10/01a1c49d-d4b6-78b2-89ab-baeed81a18ec/hojil-reference/reference.wav`; duration 3.8 seconds; SHA256 `860372aef1859fa8295d68a16a20fa0b0f9bad2e19b30048da0ee1039b32827d`; reference transcript: `Энэ өдрийг тэрээр тэсэн ядан хүлээжээ.`

Full conditioning used both audio codec tokens and the speaker embedding; a preset speaker was not substituted. All 76 speaker-encoder tensors were identical between base and exported epoch 2; their original-reference embeddings matched exactly. Training had optimized codec cross-entropy, not an explicit speaker-similarity objective, and original checkpoint selection had not measured voice similarity. This is a plausible route for a pronunciation/voice tradeoff, not a proof of a unique cause.

The first controlled diagnostic used the same first text, seed 2026, CPU FP32 and reference: base/full, base/embedding-only, epoch2/full, epoch2/embedding-only. It reused the original epoch2/full waveform. A single text/seed cannot establish general cloning quality; embedding-only mode was not adopted.

Natural controls were observational: ten natural same-speaker target/reference pairs from the already examined test speakers had mean cosine 0.9865264297 (range .9752212167–.9936868548); 90 cross-speaker pairs had mean .9386402104 (range .8826342821–.9668728113); custom reference versus ten other natural speaker references averaged .9391655922. These indicate speaker information in the encoder, but sharing speakers creates dependencies and they do not calibrate a perceptual threshold. A pitch-tracking attempt had inadequate confident-frame coverage, including zero confident frames for epoch2/full at the fixed threshold; it cannot support a pitch-change explanation.

Post-hoc voice measurements reused and verified all 528 DEV44 recordings (six epochs × two seeds) without regeneration. Epoch 1 had the highest mean/macro internal cosine while epoch 3 had the best development pronunciation. Differences were small and nonmonotonic; this did not make epoch 1 a listener-validated voice winner or change selection.

The user explicitly judged the original base closer than epoch 2 while ignoring pronunciation. This was **unblinded example feedback**, not a MOS, preference percentage, or multi-speaker listening study. A single halfway coefficient was then registered: `W = .5*W_base + .5*W_epoch2_merged`, constructed/stored in FP32. The epoch-2 endpoint had already been BF16-exported, so this is not identical to applying half the original FP32 LoRA delta. Nine recordings covered the three variants × three CUSTOM3 texts at seed 2026; endpoints were reused when available. All 233 changed tensors were in the talker; the speaker encoder stayed original.

The halfway variant recovered much of the measured voice similarity but increased pronunciation errors relative to full epoch 2. The user's follow-up was “voice is closer but pronunciation is worse.” This established the user-preferred halfway starting point for subsequent trials, not a broad automatic deployment claim. No other early comparison collected formal human ratings.

## 7. Small inference ablations before targeted training

All three trials used the same six CUSTOM3 draws (2026/2027), full conditioning, pinned ASR and internal original-reference cosine, retaining every output. They shared baseline draws; do not add reused baselines as new synthesis. Across the round there were 24 unique recordings, 21 newly generated relative to the earlier three halfway seed-2026 examples.

1. **Main temperature .9→.6**, fixed halfway weights and residual sampling (`R/half-blend-pronunciation`): CER/WER worsened on both seeds; 15→20 word errors/64. Mean cosine was similar. Rejected as a pronunciation change.
2. **Full epoch-2 text projection only**, other matrices halfway (`R/half-blend-text-mapping`): pooled CER/WER improved, 15→11 word errors, but mean cosine fell, seed-2026 WER worsened, and case007/2027 lost .0086777210 cosine. Rejected as an inconsistent voice/pronunciation tradeoff.
3. **Full epoch-2 main decoder + text projection, original residual predictor**, 198 adapted matrices versus 35 restored (`R/main-only-adaptation`): 10/64 word errors, but mean cosine fell to .9736502270; seed-2026 WER worsened. Rejected. Restoring the residual predictor did not eliminate voice drift under this full main-decoder condition; it does not prove a clean content/identity decomposition.

## 8. Two early pronunciation training pilots on the halfway backbone

Sources: [targeted text training](../../../0.6b-targeted-voice-training.md), [reference-protected training](../../../0.6b-reference-protected-training.md). Configurations: `configs/qwen06-text-voice-pilot.json` and `configs/qwen06-reference-protected.json`.

Both froze the complete halfway FP32 backbone, embeddings, encoder and codec; only four new FP32 A/B tensors on two text-projection matrices trained: rank/alpha 16, dropout .05, 114,688 parameters. Fresh AdamW used WD .01, clip 1, 16 clips/update, seed 2026 + epoch 2 deterministic order, peak 5e-5/end 1e-5, eight-update warmup. Training forward/backward used BF16 autocast and gradient checkpointing; synthesis used CPU FP32. Maximum 88 updates/1,408 clips was planned, with 0/22/44/66/88 checks. Actual runs stopped at 44/704 clips; **66 and 88 were not evaluated endpoints**. Every training sample used same-speaker reference ICL and non-streaming layout.

The global text pilot changed the shared projection for target text, reference text, roles and TTS special/PAD inputs. It required initial custom 001/2026 waveform parity with the accepted halfway sample and frozen-parameter hashes before/after updates. The protected variant routed the fresh adapter only onto target-transcript positions inside the native ICL construction. It preserved direct reference/control projections, kept full-shape computation to preserve numerical paths, and rejected unsupported streaming/non-ICL modes. The protected adapter must remain unmerged; merging would defeat that conditional route. Direct input protection does not freeze downstream attention or perceived voice.

SCREEN26 ran at every actual checkpoint. The registered joint screen required lower pooled development WER; no pooled CER or seedwise WER increase in either group; no group macro-cosine reduction (protected implementation tolerance 1e-7); and no single recording losing more than .005 cosine. Two consecutive voice-screen failures stopped the pilot. These are engineering screening constraints, not calibrated perceptual noninferiority thresholds.

Both pilots rejected steps 22 and 44 and selected step 0 locally with `retain_halfway_baseline`, stop `voice_guard`, no global pointer/deployment change. Each retained 78 recordings and verified provenance/frozen state/FP32 moments. The protected pilot regenerated all 26 baseline waveforms, ASR transcripts and cosine values exactly. Direct reference/control projection changes stayed exactly zero after training; target projection movement was nonzero.

Global text step 44 improved development CER/WER but custom CER rose, custom seed 2026 WER worsened and development cosine fell, including a worst paired drop .0165171623. Protected step 44 improved pooled development and custom pronunciation more, but custom cosine fell and case004/2026 dropped .0088900328; custom WER went 18.75→28.125% for 2026 and28.125→9.375% for 2027. It failed the joint rule despite favorable pooled errors. No new-adapter listening judgments were collected. The protected step 44 WER interval versus baseline was approximately [-12.21,+5.23] percentage points; repeated small-panel selection was not corrected away.

Neither stop was classical loss-divergence overfit. The protected matched training/familiar/unseen losses decreased from 3.76352/4.11231/4.17405 to 3.75679/4.10866/4.16915; the global text pilot also had lower corresponding losses at 44 than 0. Voice guards, not loss evidence, ended these runs.

## 9. Merge precision: measurement and separate inference test

Source: [merge precision](../../../0.6b-merge-precision.md). `R/merge-precision-diagnostic.json` verified the original base, selected epoch-2 FP32 adapter and exported model. Historical export cast the talker/adapters to BF16 before merging; reloading that export as FP32 cannot restore discarded values.

Across 233 adapted matrices/525,336,576 elements, the intended update norm was 5.1231796937 and difference norm .6413716404: relative **update** error .1251901512, direction cosine .9922420447 and .2014688675 of nonzero update elements rounded to zero. This is 12.52% of the small update norm, **not 12.52% of total model parameters, learning, or speech quality**. Installed PEFT CPU BF16 merge reproduced every exported element exactly (zero mismatches), tying the result to export policy.

`R/exact-fp32-half-merge` tested the original FP32 base plus half the original FP32 adapter delta, against accepted halfway endpoint interpolation. No optimizer ran. Before synthesis, reconstructed legacy custom 001/2026 exactly matched accepted audio. All26 SCREEN26 outputs completed without truncation. Independent checks verified every model tensor, all 233 adapted matrices, untouched unrelated parameters, WAV hashes, recomputed original-reference cosines and stored ASR edit counts (not a second ASR inference).

An initial auditor using two CPU threads failed to reproduce the model hash. Reconstructing matrix products at registered four threads reproduced it exactly. The audit was corrected to the registered backend; candidate weights/audio were not changed. Generation-only voice means passed the later absolute .984 threshold in both groups, but complete ASR showed worse CER in both groups. Custom WER improved while development WER worsened. The candidate was rejected; no optimizer, checkpoint pointer or accepted audio changed. Eliminating rounding alone did not solve generated quality.

## Reporting constraints and missing evidence

- “Best” is always conditional: incumbent at that time, best DEV44, best exploratory TEST30 point estimate, user-preferred example, or locally retained step 0. These are not interchangeable.
- No early experiment here establishes CER<3% and cosine>.984 jointly for both groups on the later complete original 114 requirement. Those requirements and later panels must be reported in the later inventory.
- Voice numbers were absent from original pronunciation selection and added post hoc; no new voice measurements were invented for historical continuations/LoRA+ or the test expansion.
- A blind listening page is an artifact, not collected ratings. User feedback is quoted only as qualitative/unblinded feedback.
- Setup/interruption, numerical correction, rejected quality trial and planned method are different outcomes. No rank/full-fine-tune/DoRA/etc result should be inferred from a configuration or literature mention.
- Source hashes below identify the JSON files read now. Checkpoint/model hashes embedded in historical protocols are reported as recorded provenance; this inventory avoids hashing multi-gigabyte models or loading native models alongside the live experiment.

## Quantitative appendix extracted from raw reports

The following tables and file SHA256 inventory are populated directly from the retained JSON evidence. Ratios use normalized error counts; percentages are rounded to eight decimals. The source JSON retains full floating-point values and detailed case/seed diagnostics.

### A. DEV44 pronunciation

Single-seed denominators: 2,830 characters/438 words; two-seed: 5,660/876. Fractions below are recovered from these denominators and checked against stored pooled rates; this is not a fresh ASR/edit-count audit.

| Model/checkpoint | Seeds | Character edits / chars | CER % | Word edits / words | WER % |
| --- | --- | --- | --- | --- | --- |
| Original base | 2026 | 692/2830 | 24.45229682 | 247/438 | 56.39269406 |
| Historical BF16 epoch 1 | 2026 | 205/2830 | 7.24381625 | 68/438 | 15.52511416 |
| BF16 continuation | 2026 | 235/2830 | 8.30388693 | 75/438 | 17.12328767 |
| FP32 v4 selected 88 | 2026 | 160/2830 | 5.65371025 | 58/438 | 13.24200913 |
| Literature ratio 1 selected 132 | 2026 | 162/2830 | 5.72438163 | 63/438 | 14.38356164 |
| LoRA+ ratio 4 selected 44 | 2026 | 168/2830 | 5.93639576 | 69/438 | 15.75342466 |
| FP32 v4 incumbent | 2026+2027 | 319/5660 | 5.63604240 | 122/876 | 13.92694064 |
| Fresh epoch 1 | 2026+2027 | 369/5660 | 6.51943463 | 126/876 | 14.38356164 |
| Fresh epoch 1 | 2026 | 220/2830 | 7.77385159 | 67/438 | 15.29680365 |
| Fresh epoch 1 | 2027 | 149/2830 | 5.26501767 | 59/438 | 13.47031963 |
| Fresh epoch 2 | 2026+2027 | 271/5660 | 4.78798587 | 99/876 | 11.30136986 |
| Fresh epoch 2 | 2026 | 187/2830 | 6.60777385 | 61/438 | 13.92694064 |
| Fresh epoch 2 | 2027 | 84/2830 | 2.96819788 | 38/438 | 8.67579909 |
| Fresh epoch 3 | 2026+2027 | 185/5660 | 3.26855124 | 79/876 | 9.01826484 |
| Fresh epoch 3 | 2026 | 92/2830 | 3.25088339 | 40/438 | 9.13242009 |
| Fresh epoch 3 | 2027 | 93/2830 | 3.28621908 | 39/438 | 8.90410959 |
| Fresh epoch 4 | 2026+2027 | 229/5660 | 4.04593640 | 99/876 | 11.30136986 |
| Fresh epoch 4 | 2026 | 141/2830 | 4.98233216 | 55/438 | 12.55707763 |
| Fresh epoch 4 | 2027 | 88/2830 | 3.10954064 | 44/438 | 10.04566210 |
| Fresh epoch 5 | 2026+2027 | 210/5660 | 3.71024735 | 86/876 | 9.81735160 |
| Fresh epoch 5 | 2026 | 133/2830 | 4.69964664 | 53/438 | 12.10045662 |
| Fresh epoch 5 | 2027 | 77/2830 | 2.72084806 | 33/438 | 7.53424658 |
| Fresh epoch 6 | 2026+2027 | 220/5660 | 3.88692580 | 95/876 | 10.84474886 |
| Fresh epoch 6 | 2026 | 115/2830 | 4.06360424 | 51/438 | 11.64383562 |
| Fresh epoch 6 | 2027 | 105/2830 | 3.71024735 | 44/438 | 10.04566210 |

### B. TEST30: initial comparison and exploratory expansion

Epochs 2 and 3 formed the initial reserved-test comparison. The six-epoch extension reused that already inspected test: 60 recordings/model, 360 total, 240 newly generated. Single-seed denominators are 1,631 characters/256 words; pooled 3,262/512. Epoch 6 had the lowest observed WER but did not change the development-selected pointer.

| Model/checkpoint | Seeds | Character edits / chars | CER % | Word edits / words | WER % |
| --- | --- | --- | --- | --- | --- |
| Epoch 1 | 2026+2027 | 188/3262 | 5.76333538 | 69/512 | 13.47656250 |
| Epoch 1 | 2026 | 112/1631 | 6.86695279 | 42/256 | 16.40625000 |
| Epoch 1 | 2027 | 76/1631 | 4.65971796 | 27/256 | 10.54687500 |
| Epoch 2 | 2026+2027 | 146/3262 | 4.47578173 | 56/512 | 10.93750000 |
| Epoch 2 | 2026 | 78/1631 | 4.78234212 | 32/256 | 12.50000000 |
| Epoch 2 | 2027 | 68/1631 | 4.16922134 | 24/256 | 9.37500000 |
| Epoch 3 | 2026+2027 | 181/3262 | 5.54874310 | 69/512 | 13.47656250 |
| Epoch 3 | 2026 | 77/1631 | 4.72103004 | 30/256 | 11.71875000 |
| Epoch 3 | 2027 | 104/1631 | 6.37645616 | 39/256 | 15.23437500 |
| Epoch 4 | 2026+2027 | 124/3262 | 3.80134887 | 49/512 | 9.57031250 |
| Epoch 4 | 2026 | 63/1631 | 3.86266094 | 27/256 | 10.54687500 |
| Epoch 4 | 2027 | 61/1631 | 3.74003679 | 22/256 | 8.59375000 |
| Epoch 5 | 2026+2027 | 174/3262 | 5.33415083 | 66/512 | 12.89062500 |
| Epoch 5 | 2026 | 98/1631 | 6.00858369 | 37/256 | 14.45312500 |
| Epoch 5 | 2027 | 76/1631 | 4.65971796 | 29/256 | 11.32812500 |
| Epoch 6 | 2026+2027 | 119/3262 | 3.64806867 | 46/512 | 8.98437500 |
| Epoch 6 | 2026 | 73/1631 | 4.47578173 | 27/256 | 10.54687500 |
| Epoch 6 | 2027 | 46/1631 | 2.82035561 | 19/256 | 7.42187500 |

Speaker-cluster bootstrap WER differences versus epoch 2, in percentage points. These are exploratory intervals, not multiple-comparison-corrected checkpoint selection.

| Epoch | WER difference | 95% interval |
| --- | --- | --- |
| 1 | 2.53906250 | [-2.26130653, 6.67870036] |
| 2 | 0.00000000 | [0.00000000, 0.00000000] |
| 3 | 2.53906250 | [0.00000000, 5.08474576] |
| 4 | -1.36718750 | [-8.10810811, 3.59375000] |
| 5 | 1.95312500 | [-3.60169492, 6.50000000] |
| 6 | -1.95312500 | [-8.10810811, 2.07715134] |

### C. Post hoc DEV44 voice observability

Existing 88 waveforms/epoch; original-reference encoder cosine. Per-seed values are recording means, not speaker macros. No regeneration or selection change.

| Epoch | Recording mean | Speaker macro | 2026 mean | 2027 mean |
| --- | --- | --- | --- | --- |
| 1 | 0.9810090159828012 | 0.9816280113822884 | 0.9801052497191862 | 0.9819127822464163 |
| 2 | 0.9797399098222906 | 0.9805219641990132 | 0.9793534820730035 | 0.9801263375715776 |
| 3 | 0.9792260310866616 | 0.9801313706735769 | 0.9786575070836327 | 0.9797945550896905 |
| 4 | 0.9790010892532088 | 0.9800573871864213 | 0.978363187475638 | 0.9796389910307798 |
| 5 | 0.9800315621224317 | 0.9809105548262597 | 0.9799382998184725 | 0.9801248244263909 |
| 6 | 0.9790977720509876 | 0.9803107900751961 | 0.9785183261741291 | 0.979677217927846 |

### D. Custom-reference diagnosis and halfway comparison

Four-way diagnostic: one text, seed 2026. CER/WER were not reported for this four-way result. Only epoch-2 full-reference audio was reused.

| Variant | Original-reference cosine | Seconds | Reused audio |
| --- | --- | --- | --- |
| base_full_reference | 0.9861574769020081 | 4.32 | False |
| base_speaker_embedding_only | 0.9799924492835999 | 4.8 | False |
| epoch2_full_reference | 0.9729294180870056 | 4.72 | True |
| epoch2_speaker_embedding_only | 0.9763362407684326 | 3.68 | False |

Three-text halfway comparison, seed 2026 only: 214 characters/32 words per variant.

| Variant | CER % | WER % | Mean cosine |
| --- | --- | --- | --- |
| base | 12.14953271 | 50.00000000 | 0.9855965971946716 |
| half_blend | 5.60747664 | 18.75000000 | 0.983937939008077 |
| epoch2 | 2.80373832 | 6.25000000 | 0.9682466387748718 |

### E. Inference-only CUSTOM3 alternatives

Two seeds, six recordings/variant, pooled 428 characters/64 words. Each seed has 214/32. Baseline is shared across comparisons, not three independent baseline samples.

| Variant | Seeds | CER % | WER % | Mean cosine | Word edits / words |
| --- | --- | --- | --- | --- | --- |
| current | 2026+2027 | 7.24299065 | 23.43750000 | 0.9849601984024048 | 15/64 |
| current | 2026 | 5.60747664 | 18.75000000 | 0.983937939008077 | 6/32 |
| current | 2027 | 8.87850467 | 28.12500000 | 0.9859824577967325 | 9/32 |
| lower_temperature | 2026+2027 | 10.98130841 | 31.25000000 | 0.9852717419465383 | 20/64 |
| lower_temperature | 2026 | 9.81308411 | 25.00000000 | 0.985028088092804 | 8/32 |
| lower_temperature | 2027 | 12.14953271 | 37.50000000 | 0.9855153958002726 | 12/32 |
| text_mapping | 2026+2027 | 5.14018692 | 17.18750000 | 0.9842076698939005 | 11/64 |
| text_mapping | 2026 | 6.07476636 | 21.87500000 | 0.9851200183232626 | 7/32 |
| text_mapping | 2027 | 4.20560748 | 12.50000000 | 0.9832953214645386 | 4/32 |
| main_only | 2026+2027 | 5.37383178 | 15.62500000 | 0.9736502269903818 | 10/64 |
| main_only | 2026 | 7.94392523 | 21.87500000 | 0.9717419544855753 | 7/32 |
| main_only | 2027 | 2.80373832 | 9.37500000 | 0.9755584994951884 | 3/32 |

### F. Every targeted/protected training checkpoint

SCREEN26: development 20 recordings/10 speakers, custom 6 recordings/one speaker. Pooled denominators are development 1,074 characters/174 words and custom 428/64. Both independently verified step-0 baselines are retained. Cosine is the pooled speaker macro.

| Pilot | Step | Group | CER % | WER % | Speaker-macro cosine | 2026 CER / WER % | 2027 CER / WER % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| targeted-voice-pilot | 0 | custom | 7.24299065 | 23.43750000 | 0.9849601984024048 | 5.60747664 / 18.75000000 | 8.87850467 / 28.12500000 |
| targeted-voice-pilot | 0 | development | 9.31098696 | 23.56321839 | 0.9865134745836258 | 10.05586592 / 26.43678161 | 8.56610801 / 20.68965517 |
| targeted-voice-pilot | 22 | custom | 8.17757009 | 25.00000000 | 0.9851191540559133 | 8.41121495 / 25.00000000 | 7.94392523 / 25.00000000 |
| targeted-voice-pilot | 22 | development | 8.75232775 | 24.71264368 | 0.9860816210508346 | 10.24208566 / 33.33333333 | 7.26256983 / 16.09195402 |
| targeted-voice-pilot | 44 | custom | 8.41121495 | 23.43750000 | 0.9852085411548615 | 8.87850467 / 25.00000000 | 7.94392523 / 21.87500000 |
| targeted-voice-pilot | 44 | development | 8.19366853 | 20.68965517 | 0.9860676348209381 | 7.82122905 / 20.68965517 | 8.56610801 / 20.68965517 |
| reference-protected-pilot | 0 | custom | 7.24299065 | 23.43750000 | 0.9849601984024048 | 5.60747664 / 18.75000000 | 8.87850467 / 28.12500000 |
| reference-protected-pilot | 0 | development | 9.31098696 | 23.56321839 | 0.9865134745836258 | 10.05586592 / 26.43678161 | 8.56610801 / 20.68965517 |
| reference-protected-pilot | 22 | custom | 8.17757009 | 26.56250000 | 0.9848418434460958 | 7.47663551 / 25.00000000 | 8.87850467 / 28.12500000 |
| reference-protected-pilot | 22 | development | 8.93854749 | 24.13793103 | 0.9867320537567139 | 8.19366853 / 26.43678161 | 9.68342644 / 21.83908046 |
| reference-protected-pilot | 44 | custom | 6.07476636 | 18.75000000 | 0.9837403893470764 | 7.94392523 / 28.12500000 | 4.20560748 / 9.37500000 |
| reference-protected-pilot | 44 | development | 7.54189944 | 18.96551724 | 0.9870296776294708 | 9.12476723 / 24.13793103 | 5.95903166 / 13.79310345 |

### G. Merge-precision measurement and inference result

Error is relative to the intended small adapter update, not total model quality. Rounded-to-zero fraction applies to nonzero intended update elements.

| Group | Matrices | Elements | Relative update error | Delta cosine | Rounded-to-zero fraction | BF16 reproduction mismatches |
| --- | --- | --- | --- | --- | --- | --- |
| all | 233 | 525336576 | 0.12519015119152116 | 0.9922420447412126 | 0.20146886745612777 | 0 |
| main_decoder | 196 | 440401920 | 0.12508350560463305 | 0.9922545126847778 | 0.20965617043631418 | 0 |
| text_projection | 2 | 6291456 | 0.11010282082706789 | 0.9939882240363915 | 0.1667760213216146 | 0 |
| residual_predictor | 35 | 78643200 | 0.12979046976818648 | 0.9916713277079736 | 0.15839539845784506 | 0 |

Exact FP32 half-merge candidate on SCREEN26; baseline is the shared step-0 row in F.

| Group | CER % | WER % | Speaker-macro cosine | 2026 CER / WER % | 2027 CER / WER % |
| --- | --- | --- | --- | --- | --- |
| custom | 7.47663551 | 18.75000000 | 0.9847533106803894 | 7.00934579 / 18.75000000 | 7.94392523 / 18.75000000 |
| development | 10.70763501 | 29.31034483 | 0.9870005458593368 | 12.29050279 / 33.33333333 | 9.12476723 / 25.28735632 |

### H. Recorded checkpoint/export identities

Copied from the original six-epoch test protocol, not recomputed from model files during this documentation audit. Adapter and exported-model identities are distinct.

| Epoch | Selection kind | Adapter SHA256 | Exported model SHA256 |
| --- | --- | --- | --- |
| 1 | completed_epoch_endpoint | `c81b251ed1d04bb0bf24b07f7842fa340c85cdeaec32f95ef2489cacaeeea205` | `5cbed03811ed445f1c8ac0bded31034a9cc332e2f97a0a9b87d2083f04af29e3` |
| 2 | completed_epoch_endpoint | `39e51daf27c74ebda37126e6a0554ac5d5d112ae9204d4f7fbbadf4cb9227bb5` | `84025c2d5b62f1c02cf186bf66ee6b624fe74f204248c03013f1b33eb684ae0a` |
| 3 | exact_epoch_endpoint | `17f0f91508a5b96adc38177b374be299b7824051aa95f8e3169ae39b2522af55` | `3181fba3d03b467652d7c4e10515489ab10d908e1361d763b76534c6dd9cf4e2` |
| 4 | exact_epoch_endpoint | `929e92df0773eb2f5a3943bf8c4088118e9000e330b37c3162bb65c9a4856d57` | `280d78d4c26911ce47466c4ae168119bb17b46186a78d0b4eaf6ecf881e11df9` |
| 5 | exact_epoch_endpoint | `547a98a81ec6a3b4188911c7370f7b91e18dcef601af7fea88ef3d745597bf96` | `0e73b574110222e0edf4734a637739ad0d691b10f1888fa495507d2f3df3feb6` |
| 6 | exact_epoch_endpoint | `c26e88a64abf2792be750440b6c651dc6f05d98a55fab3e89556e2e1e8106e1f` | `403f540d553d1303d22f51fedd26a4f0e64c6a2f7b56e135439d1093a73305f5` |

### I. Raw evidence-file SHA256 index

All linked artifacts remain outside Git. Hashes below were recomputed from the small retained report/protocol files; this is not a new native model/audio audit. Links are machine-local. Legacy documentation links above provide portable narrative context.

| Evidence file under R | SHA256 |
| --- | --- |
| [audit.json](C:/Users/Khan/.codex/visualizations/2026/11/09/01a1bf77-0b17-7391-9781-f859f1fbb0b1/audit.json) | `15a19bc93ed716eb0bcbc1f4d0ac03232ff2491b0fd59b7cee77fddd87ba5091` |
| [fp32-continuation-v3/run.json](C:/Users/Khan/.codex/visualizations/2026/11/09/01a1bf77-0b17-7391-9781-f859f1fbb0b1/fp32-continuation-v3/run.json) | `2d7ceae7d03d008c1fe1f5806ad064ecce41b62bc9fe50a1344aa2e63a6b91c7` |
| [fp32-continuation-v3/best.json](C:/Users/Khan/.codex/visualizations/2026/11/09/01a1bf77-0b17-7391-9781-f859f1fbb0b1/fp32-continuation-v3/best.json) | `0677073789dd535f65621fe675d9c37925d5cdfc003bd63902a3e85aa86c3b8d` |
| [fp32-continuation-v3/metrics-000040.json](C:/Users/Khan/.codex/visualizations/2026/11/09/01a1bf77-0b17-7391-9781-f859f1fbb0b1/fp32-continuation-v3/metrics-000040.json) | `b19484ba506a6fb922cd2cac100a9a4e9e0ba686d3ca3dd39e9fe1fdffe29554` |
| [fp32-continuation-v4/result.json](C:/Users/Khan/.codex/visualizations/2026/11/09/01a1bf77-0b17-7391-9781-f859f1fbb0b1/fp32-continuation-v4/result.json) | `b178405b01ec944cab974fbed4b67504dc9ea12fbb88ac325b8d16eda063ecdf` |
| [fp32-continuation-v4/best.json](C:/Users/Khan/.codex/visualizations/2026/11/09/01a1bf77-0b17-7391-9781-f859f1fbb0b1/fp32-continuation-v4/best.json) | `22cf875981e1206b1be33e465fdb9e9cceeacdbe5dc577e404ae12ea47a1db38` |
| [fp32-continuation-v4/precision-verification.json](C:/Users/Khan/.codex/visualizations/2026/11/09/01a1bf77-0b17-7391-9781-f859f1fbb0b1/fp32-continuation-v4/precision-verification.json) | `04554da688790f856c7ee2ef9bdbc13a0ee6c23621f2a5db109f86c57ff123e2` |
| [fp32-continuation-v4/speech-validation/comparison.json](C:/Users/Khan/.codex/visualizations/2026/11/09/01a1bf77-0b17-7391-9781-f859f1fbb0b1/fp32-continuation-v4/speech-validation/comparison.json) | `bde4fb4a7c084da0b98af3fda43890479ccfb53e0e1cca9a56bde7d41c8a7d75` |
| [incumbent-adapter-spectrum.json](C:/Users/Khan/.codex/visualizations/2026/11/09/01a1bf77-0b17-7391-9781-f859f1fbb0b1/incumbent-adapter-spectrum.json) | `c5f5d80006029b161b2fca980b9620d66e2984c785ef2dcc09514ceb440d0366` |
| [fresh-paired-comparison.json](C:/Users/Khan/.codex/visualizations/2026/11/09/01a1bf77-0b17-7391-9781-f859f1fbb0b1/fresh-paired-comparison.json) | `453739c15727222cb3a17886e1a2e3a4da959af9dbeca77bc4c5d26c2117b6c8` |
| [fresh-round-result.json](C:/Users/Khan/.codex/visualizations/2026/11/09/01a1bf77-0b17-7391-9781-f859f1fbb0b1/fresh-round-result.json) | `bd05a065c8c031ffc882a523cfe139509e7915d5a23717531274db5a51973fd8` |
| [incumbent-matched-probe.json](C:/Users/Khan/.codex/visualizations/2026/11/09/01a1bf77-0b17-7391-9781-f859f1fbb0b1/incumbent-matched-probe.json) | `f4cc70600e440f4ea4c7788ed759bff5d4b39d6f78dd4c9fa20d220412d526a0` |
| [fresh-fp32-epoch1/verification.json](C:/Users/Khan/.codex/visualizations/2026/11/09/01a1bf77-0b17-7391-9781-f859f1fbb0b1/fresh-fp32-epoch1/verification.json) | `75ba51ed7d14755a63ebbcd59b7f688a18b7c956f148b9dd78ea2daa87abf480` |
| [epoch2-vs-incumbent.json](C:/Users/Khan/.codex/visualizations/2026/11/09/01a1bf77-0b17-7391-9781-f859f1fbb0b1/epoch2-vs-incumbent.json) | `c48d993e39fa0a7092ee8c1d632d9b38c6fddfb654f6995ce146278347794ccc` |
| [epoch2-vs-epoch1.json](C:/Users/Khan/.codex/visualizations/2026/11/09/01a1bf77-0b17-7391-9781-f859f1fbb0b1/epoch2-vs-epoch1.json) | `876882fdc490f4815f4dd9f1f2f17e2e75dc3e7f98e4b1c464a846860b09ea6f` |
| [epoch2-round-result.json](C:/Users/Khan/.codex/visualizations/2026/11/09/01a1bf77-0b17-7391-9781-f859f1fbb0b1/epoch2-round-result.json) | `d0eb8682ce1441942ad85cf799fdb21f2048d4aa0424ca61e31eced5d61acab1` |
| [fresh-fp32-epoch2/verification.json](C:/Users/Khan/.codex/visualizations/2026/11/09/01a1bf77-0b17-7391-9781-f859f1fbb0b1/fresh-fp32-epoch2/verification.json) | `7d6208c293629aedbb69cbb2b0378f83a84732c547e5a5069b76d96e32c15335` |
| [fresh-fp32-epoch2/continuation-verification.json](C:/Users/Khan/.codex/visualizations/2026/11/09/01a1bf77-0b17-7391-9781-f859f1fbb0b1/fresh-fp32-epoch2/continuation-verification.json) | `ad0d77806ff932af44819ecac8517cbb0561477ea1e6cda816f1cd61aa7b15f1` |
| [epochwise-results.json](C:/Users/Khan/.codex/visualizations/2026/11/09/01a1bf77-0b17-7391-9781-f859f1fbb0b1/epochwise-results.json) | `042b49e065eae4741a1ea00f0631618519d4b28d988640d9b0a4b2f1048d5a24` |
| [epochwise-final.json](C:/Users/Khan/.codex/visualizations/2026/11/09/01a1bf77-0b17-7391-9781-f859f1fbb0b1/epochwise-final.json) | `95a94defa6d560c2173bbb01af4e550a7bd195c9c57ad3dda10fb1be7db0924f` |
| [speech-stop-status.json](C:/Users/Khan/.codex/visualizations/2026/11/09/01a1bf77-0b17-7391-9781-f859f1fbb0b1/speech-stop-status.json) | `a506c7241c870e24ee174b7db25bc92761603e86260bb06cb22c9007ef0142b2` |
| [epochwise-final-best-validation-comparison.json](C:/Users/Khan/.codex/visualizations/2026/11/09/01a1bf77-0b17-7391-9781-f859f1fbb0b1/epochwise-final-best-validation-comparison.json) | `e668b38b3390e72df53c81453878a9c5c59293aae7346a7da17cc9789084240c` |
| [development-voice-observability/results.json](C:/Users/Khan/.codex/visualizations/2026/11/09/01a1bf77-0b17-7391-9781-f859f1fbb0b1/development-voice-observability/results.json) | `681333a1774ec6ea3bcdddb7f9ad4d445b6e060e52a5998dc797b62a38a7508c` |
| [merge-precision-diagnostic.json](C:/Users/Khan/.codex/visualizations/2026/11/09/01a1bf77-0b17-7391-9781-f859f1fbb0b1/merge-precision-diagnostic.json) | `9100ced404d0c619effa68b3f75eecca48ad77d987e05fe547f726a286581b35` |
| [literature-control/run.json](C:/Users/Khan/.codex/visualizations/2026/11/09/01a1bf77-0b17-7391-9781-f859f1fbb0b1/literature-control/run.json) | `357be350e4fa7b34071c1af4303fedf02236fc0fae9fe1dce9fc0358800d0e4a` |
| [literature-control/best.json](C:/Users/Khan/.codex/visualizations/2026/11/09/01a1bf77-0b17-7391-9781-f859f1fbb0b1/literature-control/best.json) | `e36aa908450db3f60c9d95cc03a381c22594a29bdb7d50b9a0109cf661ad8b4a` |
| [literature-control/verification.json](C:/Users/Khan/.codex/visualizations/2026/11/09/01a1bf77-0b17-7391-9781-f859f1fbb0b1/literature-control/verification.json) | `64b798fd088a233fd6b8dd77700e15d7afbb1d9178c03ff63420753299d27908` |
| [literature-control/speech-validation/comparison.json](C:/Users/Khan/.codex/visualizations/2026/11/09/01a1bf77-0b17-7391-9781-f859f1fbb0b1/literature-control/speech-validation/comparison.json) | `ad6258ca771f4833b1bb3f7484aa087441592a3d19d60da2a3a8a33229a1c755` |
| [literature-loraplus-r4/run.json](C:/Users/Khan/.codex/visualizations/2026/11/09/01a1bf77-0b17-7391-9781-f859f1fbb0b1/literature-loraplus-r4/run.json) | `caf4624efa0601d5b21831691bb6c9558cbbf1730b6fe952870429aa89604ca3` |
| [literature-loraplus-r4/best.json](C:/Users/Khan/.codex/visualizations/2026/11/09/01a1bf77-0b17-7391-9781-f859f1fbb0b1/literature-loraplus-r4/best.json) | `9d897fa815e5493e8ecad90fdaee6125738e8b91faf4182ba1f15c53feb1bf90` |
| [literature-loraplus-r4/verification.json](C:/Users/Khan/.codex/visualizations/2026/11/09/01a1bf77-0b17-7391-9781-f859f1fbb0b1/literature-loraplus-r4/verification.json) | `8500840e24d560bfe50c884d5e71b215d056ab321e77ad123ac065536b925885` |
| [literature-loraplus-r4/speech-validation/comparison.json](C:/Users/Khan/.codex/visualizations/2026/11/09/01a1bf77-0b17-7391-9781-f859f1fbb0b1/literature-loraplus-r4/speech-validation/comparison.json) | `eb1efdea24d78e84965fec1e5ac090d2191239f421d33949e4bcf24a26edb813` |
| [heldout-epoch3-vs-epoch2/protocol.json](C:/Users/Khan/.codex/visualizations/2026/11/09/01a1bf77-0b17-7391-9781-f859f1fbb0b1/heldout-epoch3-vs-epoch2/protocol.json) | `b9a93b1567e600606fc003b8fe45e9b5f2e8b6b302dbda1127869a8acf0b77d3` |
| [heldout-epoch3-vs-epoch2/comparison.json](C:/Users/Khan/.codex/visualizations/2026/11/09/01a1bf77-0b17-7391-9781-f859f1fbb0b1/heldout-epoch3-vs-epoch2/comparison.json) | `063286ba260384b4ba51f70b8dae81859f053067e0cc65515e1db0397b9a0a2b` |
| [heldout-epoch3-vs-epoch2/verification.json](C:/Users/Khan/.codex/visualizations/2026/11/09/01a1bf77-0b17-7391-9781-f859f1fbb0b1/heldout-epoch3-vs-epoch2/verification.json) | `cd4faf8b489c20c4e6f9b7def00fec3e379816ad867b71103e4f098b8250d14d` |
| [heldout-all-epochs/protocol.json](C:/Users/Khan/.codex/visualizations/2026/11/09/01a1bf77-0b17-7391-9781-f859f1fbb0b1/heldout-all-epochs/protocol.json) | `346baacad839f65a4987035ef53fb0922a27d063fa64113aa9b7d8bb46ba1d94` |
| [heldout-all-epochs/results.json](C:/Users/Khan/.codex/visualizations/2026/11/09/01a1bf77-0b17-7391-9781-f859f1fbb0b1/heldout-all-epochs/results.json) | `4e9a84ef313face37a2c113afd7321b59a3a4420e8b3569f1e97c088d4c5e692` |
| [heldout-all-epochs/verification.json](C:/Users/Khan/.codex/visualizations/2026/11/09/01a1bf77-0b17-7391-9781-f859f1fbb0b1/heldout-all-epochs/verification.json) | `8cdc972fa9a4194f241924fd23e5a2dc954078a2e2afc0b7da2a110cdda7d6db` |
| [reference-voice-diagnostic/result.json](C:/Users/Khan/.codex/visualizations/2026/11/09/01a1bf77-0b17-7391-9781-f859f1fbb0b1/reference-voice-diagnostic/result.json) | `684ccfcd2841124defecb3d6e34d7772ea0c63fe73eb43d20cb895a208bbd95e` |
| [reference-voice-diagnostic/calibration.json](C:/Users/Khan/.codex/visualizations/2026/11/09/01a1bf77-0b17-7391-9781-f859f1fbb0b1/reference-voice-diagnostic/calibration.json) | `6b6f4429e4dbf15673a375bba134212d2ce9de8a8c68150bb8ac0c561dd93094` |
| [reference-voice-diagnostic/pitch.json](C:/Users/Khan/.codex/visualizations/2026/11/09/01a1bf77-0b17-7391-9781-f859f1fbb0b1/reference-voice-diagnostic/pitch.json) | `201c837c899f348c87f7c58ad98bbbb0d61e937d397311a956664272254686f6` |
| [reference-half-blend/protocol.json](C:/Users/Khan/.codex/visualizations/2026/11/09/01a1bf77-0b17-7391-9781-f859f1fbb0b1/reference-half-blend/protocol.json) | `51e1b39993c4a2318849e947d9e6e85f63fc7f938afca7da539cb2ee295f7ba7` |
| [reference-half-blend/results.json](C:/Users/Khan/.codex/visualizations/2026/11/09/01a1bf77-0b17-7391-9781-f859f1fbb0b1/reference-half-blend/results.json) | `ec806f74a747b992dfd399914bd24fa0a492e44b777f78285555af1d28e4a513` |
| [reference-half-blend/verification.json](C:/Users/Khan/.codex/visualizations/2026/11/09/01a1bf77-0b17-7391-9781-f859f1fbb0b1/reference-half-blend/verification.json) | `b79071d0f553aa4245987d0a5e701e2f8fcb0d9a1f315719dcf9a500c1d342a5` |
| [half-blend-pronunciation/protocol.json](C:/Users/Khan/.codex/visualizations/2026/11/09/01a1bf77-0b17-7391-9781-f859f1fbb0b1/half-blend-pronunciation/protocol.json) | `28362c64b0cb6fb60f4b312204092d9f720bd6df0b6e603e8982c5bd57381c7d` |
| [half-blend-pronunciation/results.json](C:/Users/Khan/.codex/visualizations/2026/11/09/01a1bf77-0b17-7391-9781-f859f1fbb0b1/half-blend-pronunciation/results.json) | `f01c6f139e9488e856e743eb25d231b523c28ec0f659786d179733b108cedda2` |
| [half-blend-pronunciation/status.json](C:/Users/Khan/.codex/visualizations/2026/11/09/01a1bf77-0b17-7391-9781-f859f1fbb0b1/half-blend-pronunciation/status.json) | `ec645a950b1790ad14899458c39d15a40122483470948bc1d299cbefc8a7ed82` |
| [half-blend-text-mapping/protocol.json](C:/Users/Khan/.codex/visualizations/2026/11/09/01a1bf77-0b17-7391-9781-f859f1fbb0b1/half-blend-text-mapping/protocol.json) | `70cf3dd5968df2093d5e7688389252d2e60771eb0aa2e05a53dd08538985debd` |
| [half-blend-text-mapping/results.json](C:/Users/Khan/.codex/visualizations/2026/11/09/01a1bf77-0b17-7391-9781-f859f1fbb0b1/half-blend-text-mapping/results.json) | `86b3317ebc29eca8a05f7be2dcf08d404f933c6d17969279ecad1a6d8e013459` |
| [half-blend-text-mapping/status.json](C:/Users/Khan/.codex/visualizations/2026/11/09/01a1bf77-0b17-7391-9781-f859f1fbb0b1/half-blend-text-mapping/status.json) | `ec645a950b1790ad14899458c39d15a40122483470948bc1d299cbefc8a7ed82` |
| [main-only-adaptation/protocol.json](C:/Users/Khan/.codex/visualizations/2026/11/09/01a1bf77-0b17-7391-9781-f859f1fbb0b1/main-only-adaptation/protocol.json) | `6951947440301e5820ba8efe3dc61384606bdba3df91cdf731042b19603f2e56` |
| [main-only-adaptation/results.json](C:/Users/Khan/.codex/visualizations/2026/11/09/01a1bf77-0b17-7391-9781-f859f1fbb0b1/main-only-adaptation/results.json) | `35a249972e1d7d9043b83716c7bee20433b801451ec2bc2f75f5ddd13da84e12` |
| [main-only-adaptation/status.json](C:/Users/Khan/.codex/visualizations/2026/11/09/01a1bf77-0b17-7391-9781-f859f1fbb0b1/main-only-adaptation/status.json) | `ec645a950b1790ad14899458c39d15a40122483470948bc1d299cbefc8a7ed82` |
| [targeted-voice-pilot/protocol.json](C:/Users/Khan/.codex/visualizations/2026/11/09/01a1bf77-0b17-7391-9781-f859f1fbb0b1/targeted-voice-pilot/protocol.json) | `c669be114158df5a99bdfc700289f34d224664c657e09235e5a2153f61c8d396` |
| [targeted-voice-pilot/selection.json](C:/Users/Khan/.codex/visualizations/2026/11/09/01a1bf77-0b17-7391-9781-f859f1fbb0b1/targeted-voice-pilot/selection.json) | `a3836362645c595b58f0e76ac0abb1cd69fcbf2cca68dd05cf357a8beaa27733` |
| [targeted-voice-pilot/status.json](C:/Users/Khan/.codex/visualizations/2026/11/09/01a1bf77-0b17-7391-9781-f859f1fbb0b1/targeted-voice-pilot/status.json) | `083886cdc13d8f7e202dd98a9d6ff96f6fa5e3a52edeef943e2d1b2cd61a0c30` |
| [targeted-voice-pilot/verification.json](C:/Users/Khan/.codex/visualizations/2026/11/09/01a1bf77-0b17-7391-9781-f859f1fbb0b1/targeted-voice-pilot/verification.json) | `346982e99d2c0a1b5be9e5f6a94817d3a43dd98c6601f81e1a1580c0df3c617e` |
| [targeted-voice-pilot/eval-000000/results.json](C:/Users/Khan/.codex/visualizations/2026/11/09/01a1bf77-0b17-7391-9781-f859f1fbb0b1/targeted-voice-pilot/eval-000000/results.json) | `ac263ac2b9f588d0fb451e89871a681bb74ca02824090299b44f095267702722` |
| [targeted-voice-pilot/eval-000022/results.json](C:/Users/Khan/.codex/visualizations/2026/11/09/01a1bf77-0b17-7391-9781-f859f1fbb0b1/targeted-voice-pilot/eval-000022/results.json) | `5d771d2f4fe169dc93239c6833773552b62a0b528ebbe858464ca819debccd4b` |
| [targeted-voice-pilot/eval-000044/results.json](C:/Users/Khan/.codex/visualizations/2026/11/09/01a1bf77-0b17-7391-9781-f859f1fbb0b1/targeted-voice-pilot/eval-000044/results.json) | `1a29132d6b3314ed053868dfb7deba3b5ea54975e6c1bb72f7d65bfda279ffb3` |
| [reference-protected-pilot/protocol.json](C:/Users/Khan/.codex/visualizations/2026/11/09/01a1bf77-0b17-7391-9781-f859f1fbb0b1/reference-protected-pilot/protocol.json) | `bdaba55fc5b11e8c90c485cded831b5d362f6a92c2ee35cf68c5cd474df65c15` |
| [reference-protected-pilot/selection.json](C:/Users/Khan/.codex/visualizations/2026/11/09/01a1bf77-0b17-7391-9781-f859f1fbb0b1/reference-protected-pilot/selection.json) | `a3836362645c595b58f0e76ac0abb1cd69fcbf2cca68dd05cf357a8beaa27733` |
| [reference-protected-pilot/status.json](C:/Users/Khan/.codex/visualizations/2026/11/09/01a1bf77-0b17-7391-9781-f859f1fbb0b1/reference-protected-pilot/status.json) | `c1a94b8236519a7ead01991dd530faad3bbadc1087fa6bb9f06e773f530c8944` |
| [reference-protected-pilot/verification.json](C:/Users/Khan/.codex/visualizations/2026/11/09/01a1bf77-0b17-7391-9781-f859f1fbb0b1/reference-protected-pilot/verification.json) | `e2052312d81bb371d56d2dcfe3d762e6e3275257e6b3d4a64b3b932cbae2f6c3` |
| [reference-protected-pilot/eval-000000/results.json](C:/Users/Khan/.codex/visualizations/2026/11/09/01a1bf77-0b17-7391-9781-f859f1fbb0b1/reference-protected-pilot/eval-000000/results.json) | `647d3786a727a456ca13e6f93b48f70bdcfcf784c832d8f5fa657060a19d7f1c` |
| [reference-protected-pilot/eval-000022/results.json](C:/Users/Khan/.codex/visualizations/2026/11/09/01a1bf77-0b17-7391-9781-f859f1fbb0b1/reference-protected-pilot/eval-000022/results.json) | `3634d308644a5f27e9c0705c65080748f24815361294593df416179adb949ce7` |
| [reference-protected-pilot/eval-000044/results.json](C:/Users/Khan/.codex/visualizations/2026/11/09/01a1bf77-0b17-7391-9781-f859f1fbb0b1/reference-protected-pilot/eval-000044/results.json) | `ecda94e18f55c0a736c4fd0ca43b108320e84cccd2c972f194b5e047c8243463` |
| [exact-fp32-half-merge/protocol.json](C:/Users/Khan/.codex/visualizations/2026/11/09/01a1bf77-0b17-7391-9781-f859f1fbb0b1/exact-fp32-half-merge/protocol.json) | `ce8f6ca66488988d11197a65fa6dbdb2fdbd1eefb67a75effa617c982214ffae` |
| [exact-fp32-half-merge/results.json](C:/Users/Khan/.codex/visualizations/2026/11/09/01a1bf77-0b17-7391-9781-f859f1fbb0b1/exact-fp32-half-merge/results.json) | `61e2e06f9f60e7d92745b495edbb071fafb30c0f9936c51acef5608d64ec5468` |
| [exact-fp32-half-merge/verification.json](C:/Users/Khan/.codex/visualizations/2026/11/09/01a1bf77-0b17-7391-9781-f859f1fbb0b1/exact-fp32-half-merge/verification.json) | `8b092eef2397b37cfdcb1ead6aa6ec18eb28a6aa6a8427ca0f2d86b10784f996` |
| [exact-fp32-half-merge/control.json](C:/Users/Khan/.codex/visualizations/2026/11/09/01a1bf77-0b17-7391-9781-f859f1fbb0b1/exact-fp32-half-merge/control.json) | `bba43f576b23ff3c8011dfd20c2fdd15ff6033c5f65a1a1370e1dcdae4a0fc8a` |

| [fresh-fp32-epoch3/best.json](C:/Users/Khan/.codex/visualizations/2026/11/09/01a1bf77-0b17-7391-9781-f859f1fbb0b1/fresh-fp32-epoch3/best.json) | `473f1c0ec484262a66be7aebb0fd81571de69da892c3d6c03b1676218e348897` |
| [fresh-fp32-epoch4/best.json](C:/Users/Khan/.codex/visualizations/2026/11/09/01a1bf77-0b17-7391-9781-f859f1fbb0b1/fresh-fp32-epoch4/best.json) | `345499206fd8755378b3e748f23ff55d5957c7846897248f690e15996a5e1f8b` |
