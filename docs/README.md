# Documentation

For the active 0.6B optimization work, start with the
[experiment summary](experiments/0.6b/README.md). It links the
[comprehensive report](experiments/0.6b/report.md) and evidence appendices with
individual experiments, results, failures, and decisions. The joint pronunciation
and reference-voice goal remains in progress.

Start with these two documents:

1. [Open-source data review](1-open-source-data-review.md) catalogs the available sources, overlapping distributions, and license evidence. It preserves the original research snapshot dated October 21, 2026.
2. [Data preparation](2-data-preparation.md) explains the cleaning strategy, the actual source inputs, review-driven corrections, and the final corpus results.

For implementation and reproduction:

- [Pipeline reference](data-pipeline.md): acquisition, configuration, CLI commands, splits, audits, and export contracts.
- [Cleaning research and decisions](data-cleaning-research.md): evidence behind the enhancement and trimming choices.
- [Initial pilot](data-pilot-2026-10-22.md): historical small-sample experiments before the full preparation and later corrections.
- [Preparation code](../data/README.md), [example configuration](../configs/example.json), and [verification dependencies](../requirements-verification.txt).

For model evaluation and local inference:

- [0.6B training audit and optimization](0.6b-training-analysis.md): actual LoRA recipe, precision findings, validation selection, and reproduction commands.
- [LoRA literature and experiments](0.6b-lora-literature.md): research review, controlled LoRA+ comparison, and promotion criteria.
- [Fresh FP32 epoch and observability](0.6b-fresh-fp32.md): matched train/validation probes, optimization diagnostics, and generated-speech comparison.
- [Second FP32 epoch](0.6b-second-epoch.md): optimizer-preserving continuation and comparison against the first epoch and incumbent.
- [Continuation until overfitting](0.6b-until-overfit.md): persistent divergence guard, checkpoint selection and extended training protocol.
- [Reserved test comparison](0.6b-heldout-test.md): frozen epoch-3 versus epoch-2 evaluation on unseen test speakers, with paired listening examples.
- [Reference voice investigation](0.6b-voice-reference-analysis.md): conditioning checks, base-versus-LoRA audio, and speaker-similarity diagnostics across all six epochs.
- [Targeted voice-preserving training](0.6b-targeted-voice-training.md): frozen halfway backbone, text-only LoRA pilot, and pronunciation/voice checks at every checkpoint.
- [Reference-protected pronunciation training](0.6b-reference-protected-training.md): target-text-only adapter routing with unchanged reference/control projections and joint checkpoint screening.
- [Joint CER/voice goal and literature review](0.6b-joint-goal-literature.md): evidence for preference alignment, Weight-Space Programming assessment, evaluator calibration, and the explicit two-group target.
- [Joint pronunciation/voice preference training](0.6b-joint-preference-training.md): generated-trajectory DPO, training/validation separation, checkpoint observability and the preregistered broader confirmation panel.
- [Reference-only speaker-input adaptation](0.6b-speaker-reference-adaptation.md): bounded per-reference conditioning updates on the frozen full LoRA model, first calibrated on training speakers.
- [Broader generated-speech speaker preferences](0.6b-balanced-speaker-preferences.md): 64-speaker collection, batched trajectory integrity, and a bounded low-rank speaker-conditioning adapter.
- [Historical merge precision diagnostic](0.6b-merge-precision.md): measured BF16 export rounding and the completed, rejected FP32 merge comparison.
- [Expanded joint-goal confirmation](0.6b-joint-confirmation.md): execution and independent audit of the pre-registered 114-recording panel, gated on a verified screening pass.
- [Refreshed speaker preferences](0.6b-refreshed-speaker-preferences.md): new update-8 trajectories, a shorter fitting stage, and unchanged joint speech evaluation.
- [Centered speaker preferences](0.6b-centered-speaker-preferences.md): a fitting-only speaker basis and bounded residual, prepared after diagnosing the shared input shift.
- [Reference-prefix weight routing](0.6b-reference-prefix-routing.md): a controlled hypothesis that preserves halfway matrices on reference/control prompt rows while retaining full target/generation mappings.
- [Reference-audio-only routing](0.6b-reference-codec-routing.md): an acoustic-prefix ablation with exact full-weight text/control hidden-state checks on every generation.
- [Speaker plus reference-audio routing](0.6b-reference-condition-routing.md): a controlled addition of halfway processing at the native speaker-conditioning position.
- [Middle-attention joint preferences](0.6b-middle-attention-preferences.md): a bounded Q/V LoRA fit with the earlier speaker adapter frozen and independent audits at every checkpoint.
- [Absolute-quality preference comparison](0.6b-quality-preference-proposal.md): the completed filtered-DPO run, its speech-regression stop, full validation and fixed-probe diagnostics.
- [Calibrated supervised-loss comparison](0.6b-calibrated-quality-preferences.md): a training-only gradient diagnostic sets a meaningful supervised-loss scale while preserving the full evaluation scope.
- [Residual-code sampling calibration](0.6b-residual-sampling-calibration.md): a frozen-checkpoint comparison of residual temperatures, with separate main sampling and a paired calibration gate before expanded evaluation.
- [Differentiable codec feasibility](0.6b-differentiable-codec-feasibility.md): primary-literature review and a read-only native forward/gradient check before considering direct speaker-loss training.
- [Codec-to-policy gradient replay](0.6b-codec-policy-replay.md): verified memory-separated gradients through the real frozen codec into the 0.6B LoRA, with fixed-trajectory limitations made explicit.
- [Fresh-rollout direct voice pilot](0.6b-direct-voice-pilot.md): bounded projected updates with fixed pronunciation anchors, fresh training audio and full speech checks after every update.
- [Direct speaker candidates](0.6b-direct-speaker-candidates.md): measured gradient alignment, frozen attention and candidate selection using actual matched training speech before the full evaluation screen.
- [Reference-specific voice profile](0.6b-reference-voice-profile.md): isolated per-reference conditioning, disjoint fitting/validation texts, and generated-speech feedback while preserving the two-group goal.
- [Reference-specific joint preferences](0.6b-reference-profile-preferences.md): fresh multi-draw training texts with explicit joint pronunciation/voice comparisons and unchanged evaluation scope.
- [Full-panel attention/PAD findings](0.6b-attention-pad-full-panel-findings.md): complete comparison against the earlier reference-preserving policy.
- [Real-speech masked supervision](0.6b-reference-masked-supervision.md): current data, implementation validation, checkpoint training, and calibration results.
- [Result-sharing rules](0.6b-result-sharing.md): verified improvements, fixed listening examples, and provisional-versus-confirmed claims.

- [Recovery and backup scope](recovery-2026-11-04.md): verified final checkpoint, model, dataset and metadata copies outside Runpod; exclusions and restoration steps.

- [Latest CER/WER measurement](latest-evaluation-2026-11-04.md): current reference/configuration, exact scoring method, parity checks, and selected-retry results.

- [Listening findings](listening-findings-2026-11-04.md): human ratings, reference selection, chunking results, seed retries, and limits of the evidence.
- [CPU playground](playground.md): local generation, version comparison, feedback export, and launch instructions.

Recordings, model weights, local listening pages, and generated reports remain
outside Git. The documents above summarize their outcomes and provenance.
