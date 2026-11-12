# CI portability inventory

[Summary page](../README.md) · [Comprehensive report](../report.md) · [Publication checklist](#publication-checklist-and-limits) · [Local validation](#local-implementation-validation)

This document began as a static source audit and now records the separately authorized CI implementation and wheel packaging fix. Existing experiment sources/tests and registered logs remain unchanged. The new runner and lightweight data lane were exercised locally; the later packaging check built wheels in an external snapshot and tested them in isolated subprocesses. No dependency installation, training test suite, remote CI run, model load or model job was performed for these changes. Clean Ubuntu CI is **not claimed to pass**.

## Findings and minimal structure

Before this change, the data workflow discovered the entire test directory without installing the training dependencies. Several test files import Torch or Qwen unconditionally, so installing only the package/data extras cannot collect the present suite successfully on a clean runner. Installing the training stack alone also does not make unrestricted discovery portable: two tests require the author's retained experiment tree.

The implementation preserves the Python 3.11/3.12 × core/audio data matrix, explicitly selects its eight data test modules plus the new runner tests, and adds one Python 3.11 CPU training job with the pinned training stack, the remaining test modules, and **exactly two named artifact-dependent methods excluded**. Their other synthetic tests remain selected. Keep CUDA resume tests and retained-experiment verification as separate, explicitly requested integration coverage. Do not download model weights or copy datasets/audio into Git to make ordinary CI pass.

The initial inspected tree contained **83 test modules and 341 statically declared `test_*` methods**. Adding the seven runner tests made the CI-separation snapshot 84 modules/348 declared methods. Eight data modules contained 65 methods; the other 75 modules contained 276. These are AST source counts, not executed/pass counts. Two methods require retained artifacts. Seven further methods have explicit CUDA availability gates and will skip in a CPU-only training job. Any other skip/error needs inspection rather than being counted as a successful numerical check. Subsequent gradient-diagnostic preparation adds another module with nine portable tests; the 274-method collection receipt below predates that addition and is not a current full-suite execution claim.

## Configuration evidence before the CI separation

| Source | Pre-change behavior | Implication |
| --- | --- | --- |
| [.github/workflows/data-preparation.yml](../../../../.github/workflows/data-preparation.yml) | `ubuntu-latest`; Python 3.11/3.12; profiles core/audio; `pip install -e . build`; audio installs FFmpeg and `.[audio,parquet,neural]`; both run unrestricted `unittest discover -s tests -v`; builds a wheel | Data dependencies do not cover all discovered tests. The source audit establishes the import mismatch; it does not assert a particular remote workflow run has failed. |
| [pyproject.toml](../../../../pyproject.toml) | Base dependencies empty; extras audio/parquet/neural/asr/youtube; Python ≥3.11; packages explicitly `data`, `training` | There is no training extra, so `pip install -e .` does not install Torch/Qwen/PEFT. |
| [requirements-training.txt](../../../../requirements-training.txt) | Torch/torchaudio 2.8.0, qwen-tts 0.1.1, transformers 4.57.3, peft 0.21.2; bounded NumPy, soundfile, safetensors | This is the actual training requirements file. **`training/requirements.txt` does not exist.** It is a partly pinned requirements list, not a complete transitive lock. Its CUDA note does not preclude installing the matching CPU Torch wheels for tiny tests. |
| [requirements-verification.txt](../../../../requirements-verification.txt) | Exact local data/audio environment versions, including NumPy, soundfile, PyArrow and ONNX Runtime | Separate from the training requirements. Do not assume this local snapshot is an already tested Linux CI lock. |
| [.gitignore](../../../../.gitignore) | Ignores common audio, dataset archives, Parquet, `.pt`, `.safetensors`, caches and build products | Keep generated artifacts outside Git as required by the repository instructions; ignore patterns are not permission to commit other artifact formats. |

The original build checked wheel creation without isolated wheel functionality. A local external-snapshot build succeeded, but its wheel contained 96 training Python files and no `scripts` entries. Importing `training.evaluate` from that unpacked wheel under `python -I -S` failed with `ModuleNotFoundError: No module named 'scripts'`. Both `training.evaluate` and `training.reference_masked_supervision_evaluation` import `scripts.score_latest_audio`; checkout/editable tests had masked the omission. The separately reviewed fix adds `scripts` to the explicit package list and adds the isolated check described below. Artifact-bound research runners still require their checkout tests, retained protocols and external model/data assets; this is not a claim that every training API runs from a wheel alone.

## Portable data coverage

| Module | Declared methods | Dependencies and fixture behavior |
| --- | ---: | --- |
| `test_bulk` | 5 | Temporary metadata/audio fixtures; mocks; imports `test_pipeline` helper. |
| `test_pipeline` | 23 | Standard-library fixture WAV/tar/JSON creation and mocked tokenizer; no pretrained model required. |
| `test_repartition` | 3 | Temporary WAV/metadata fixtures. |
| `test_preprocessing` | 12 | Pure preprocessing checks plus real FFmpeg denoising checks guarded by executable availability. |
| `test_source_audit` | 8 | Synthetic metadata/WAV fixtures; NumPy-dependent WAV audit is guarded. |
| `test_playground` | 5 | Local HTTP server and temporary fixtures; NumPy/soundfile-dependent test is guarded. No hosted service is required. |
| `test_youtube` | 7 | Temporary fixtures and mocked downloader/service calls, not a live YouTube download. |
| `test_integration` | 2 | **Portable integration**, despite its name: creates local audio/Parquet fixtures, guarded on FFmpeg + metadata reader + PyArrow. No retained experiment files. |

The core profile is allowed to skip its declared optional audio tests; the audio profile should exercise those branches. Installing the neural extra does not by itself prove a real neural model was downloaded or tested. Existing tests use fixtures/mocks where appropriate.

Implemented data workflow commands, run from the checkout root after the existing profile-specific install steps:

```bash
python -m data --help
hojil-data --help
python scripts/run_ci_tests.py --lane data
python -m build --wheel
python scripts/check_wheel.py dist
```

The runner adds the checkout and test directory to the import path, supporting existing bare sibling imports. It selects the data modules before importing tests, avoiding unrelated training dependency imports. The CLI help checks, all eight data modules, optional audio install and wheel build are preserved in the workflow.

## Portable training and tiny-model coverage

The remaining tests cover scheduling/leakage guards, text scoring and selection, adapter algebra, gradients, deterministic resume, native prefix/codec alignment, numerical parity, provenance and synthetic artifact tampering. Many functions are pure standard-library code, but their test modules can transitively import Torch. For a minimal reliable division, use the complete pinned training environment for this whole group rather than infer dependencies from filenames.

Examples of pure or guarded lightweight tests are `test_training_continuation`, `test_training_overfit`, `test_training_direct_speaker_gate`, `test_reference_masked_supervision_plan`, `test_reference_masked_supervision_evaluation`, `test_training_evaluation` and `test_training_test_protocol`. `test_training`, `test_training_optimization` and `test_training_observability` deliberately catch missing Torch and skip their numerical portions. That pattern is **not universal**.

Concrete unconditional import examples:

- `tests/test_training_targeted_voice.py:3` and `tests/test_reference_masked_supervision_parity.py:3` import Torch directly.
- `tests/test_training_differentiable_codec_probe.py:3–5` imports Torch and Qwen codec classes directly, before any skip guard.
- `tests/test_training_reference_profile_collection.py:3` imports a production module whose lines 4–5 import NumPy and Torch. A test using only text-selection fixtures can still require the training environment to collect.
- Several routing tests import Torch before their `skipUnless(AVAILABLE)` class decorator. The decorator cannot rescue a failed module import.

The shared [tiny model helper](../../../../tests/test_training_sequence.py) constructs a random Qwen model directly from a small configuration: 32 hidden dimensions, two decoder layers, small vocabularies. It does not use `from_pretrained`. The codec probe likewise constructs a tiny codec from configuration. Tests of actual model-class logic are not full native 0.6B jobs, but do need the compatible installed Qwen/Transformers implementation. Feature-preparation tests mock `from_pretrained`; they do not extract the real 324 assets.

Synthetic files in temporary directories are **not** external artifact requirements. This includes the masked-supervision speech/confirmation tests and the reference-profile/prefix confirmation tests: they write mock protocols, waveform bytes and hash receipts, then test rejection/coverage behavior. Do not exclude these valuable checks just because their names contain “confirmation”, “artifact” or “speech”.

Implemented CPU training workflow setup on Ubuntu/Python 3.11 (installation not run locally):

```bash
python -m pip install --upgrade pip
python -m pip install torch==2.8.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e . -r requirements-training.txt
python -m pip check
```

The CPU wheel installation is a proposed clean-runner recipe, not a tested installation result. Verify that subsequent dependency resolution preserves the selected CPU Torch build. Then run with `CUDA_VISIBLE_DEVICES` empty, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `NUMBA_DISABLE_JIT=1`, and a writable runner temporary/cache directory. Keep offline variables on the test step, after dependency installation. No Hugging Face token or model download is needed for the intended portable tests.

The implementation is [scripts/run_ci_tests.py](../../../../scripts/run_ci_tests.py), with [focused stdlib tests](../../../../tests/test_ci_runner.py). It selects modules before import, supports both historical sibling import styles, rejects missing declared data modules, retains only test classes defined by each requested module, logs imported helper exclusions, rejects genuine duplicate test IDs, fails loader errors, and prints the exact two artifact exclusions. Both exclusions must actually be found. Training skips are permitted only for the seven exact CUDA method IDs with the existing reason `Qwen and CUDA required`; other skips fail the lane. Ordinary test/import/file-not-found errors remain failures.

```bash
python scripts/run_ci_tests.py --lane training
```

[training-cpu.yml](../../../../.github/workflows/training-cpu.yml) verifies `torch.version.cuda is None` after installation and keeps model access offline during tests. It does not run a GPU or artifact integration lane. No training tests were executed during this implementation task.

This is preferable to excluding both complete trainer files: their eight v1 and ten v2 remaining methods test normalization, gradients, forbidden calibration inputs, optimizer/RNG restoration, tampering, CPU/GPU admission rules and native-proof completeness using tiny/synthetic fixtures. Import failures must fail the job, not be silently filtered. Retain the skip list in logs and require all intended numerical tests to execute when their dependencies are installed.

## Nonportable and CUDA-gated cases

### Exactly two retained-experiment dependencies

Both `TrainingTests.test_pinned_real_metadata_plan_scope_and_tamper_fail_closed` methods are unconditional. They call `fit.plan_data(ROOT / 'reference-masked-supervision-plan-v1.json')`, with `ROOT` set at line 21 to the author-specific Windows visualization directory. V1 calls it around line 222; v2 around line 264. There is no missing-artifact skip.

The dependency is larger than one JSON: `plan_data` checks the plan's fixed hash, referenced source hashes, paired-data protocol/audit and every manifest before rebuilding the 204-pair/32-calibration/323-feature schedule. V2 additionally calls `prior_failure_evidence(ROOT)` and verifies the failed-v1 protocol, original logs/driver inputs, checkpoint zero, diagnosis script/result and all bound hashes. It must not be made portable by fabricating those historical receipts or copying a substitute JSON into the hard-coded path.

The same trainer tests pass `ROOT` to `save_teacher` in a negative CPU-admission test. That call is under `assertRaisesRegex` and rejects the CPU backend before writing; it is not another positive external artifact test. Keep this distinction when refactoring later.

A future artifact integration lane should be manual/trusted and use the complete external evidence bundle at its registered paths, preserving hashes and run identity. With current immutable sources, that means the original Windows environment; changing paths or source hashes is a separately versioned portability change. These tests mostly inspect/hash retained evidence, not synthesize new audio, but the v2 chain can read checkpoint files. Do not schedule them alongside a live native model run or claim that merely having a CUDA runner supplies their required artifacts.

### Seven tiny CUDA resume methods

| Module/class | CUDA-gated methods |
| --- | --- |
| `test_training_centered_speaker_training.CenteredTrainingTests` | `test_optimizer_resume_and_frozen_initial_policy` |
| `test_training_middle_attention_training.MiddleTrainingTests` | `test_cuda_resume_preserves_frozen_speaker_and_acoustic_parameters` |
| `test_training_refreshed_speaker_training.RefreshedTrainingTests` | `test_optimizer_resume_matches_uninterrupted_tiny_qwen_training` |
| `test_training_speaker_preference_training.SpeakerPreferenceTrainingTests` | `test_optimizer_resume_matches_uninterrupted_tiny_qwen_training` |
| `test_training_calibrated_quality_training.CalibratedQualityTrainingTests` | `test_calibrated_anchor_resume_and_full_validation` (guard in helper `run_resume_case`) |
| `test_training_quality_preference_training.QualityTrainingTests` | `test_filtered_dpo_resume_and_full_validation`, `test_anchored_dpo_resume_and_full_validation` (guard in helper) |

They build tiny models and temporary pools, but intentionally exercise CUDA training/resume; a CPU pass cannot establish those behaviors. A separate optional GPU job can run them with a compatible CUDA Torch environment. It should not be a required hosted-Ubuntu job unless a dedicated GPU runner is provisioned. Do not replace their skip guards with a claim that CPU checks covered GPU semantics.

## Publication checklist and limits

1. Review the new CI-only publication changes separately from immutable experiment execution. Packaging changes remain deferred to a separate review; no package source or pyproject was edited here.
2. Test the proposed clean Python 3.11 CPU installation and core/audio data matrix; report exact collected/executed/skipped counts and dependency versions. Only the limited local validation below occurred here; the clean installation/matrix remains unrun.
3. Keep the two explicit artifact exclusions visible. If new external dependencies appear, fail/review rather than broadening an implicit skip rule.
4. Preserve current historical test/source bytes and receipts. New synthetic fixtures or relocated integration tests should be versioned without retroactively altering a registered experiment.
5. Retain the isolated wheel smoke step after construction; editable checkout tests cannot establish that the shared scorer and data assets are packaged.
6. Before any eventual remote write, verify GitHub identity is `m-khanddorj`; no remote write was made during this audit. Keep audio, datasets, weights and generated artifacts outside Git.

## Local implementation validation

Before editing, the live v2 driver-input, training-protocol and speech-protocol registries were checked: none of the five edited/new target paths was registered, and all 25 registered repository-file hashes matched. The same hashes were checked again after implementation. Receipt: `R/ci-publication-source-boundary-v1.json`; `R` is the external root defined in [training evidence](training.md). No runtime-bound source/test/log was modified.

- **Latest focused runner tests: 7 passed**, exit 0, in 0.014 seconds, using the existing local Python 3.12 environment and external writable TEMP. Log: `R/ci-runner-stdlib-tests-v3.log`. The new regression test covers a helper TestCase imported into a second module, keeps its defining-module execution once, logs the exclusion, and still rejects real duplicate IDs. Previous logs remain: v2 had six passing tests; v1 failed because the sandbox default temporary directory denied access.
- **Data lane: 71 tests, 69 passed, 2 errors, no skips**, exit 1, in 15.757 seconds. Log: `R/ci-data-lane-tests-v1.log`. The errors were `test_audio_path_traversal_denied` and `test_post_requires_token_and_local_origin`, both at localhost socket connection with Windows `WinError 10013`. They remain enabled in the workflow. This is not a passing data-suite claim or an Ubuntu result.
- **Localhost recheck: all five playground tests passed**, exit 0, in 2.607 seconds after running outside the socket-restricted sandbox. Log: `R/ci-playground-localhost-recheck-v1.log`. This includes both previously errored methods; no test code changed. The original failure log remains intact. Results across these two runs cover every selected data test, but do not establish a clean Ubuntu matrix or CPU training-suite pass.
- **Training metadata collection succeeded: 274 unique selected methods from 276 declared training methods**, exactly the two registered artifact-method exclusions, and all seven expected CUDA methods retained. An independent AST-derived set matched the collected set exactly. Three imported codec-helper tests in `test_training_direct_voice_objective` were logged and excluded there, while retained under their defining `test_training_differentiable_codec_probe` module. Logs/receipt: `R/ci-training-collection-v2.log` and `.json`. CUDA was hidden, Torch threads/inter-op threads were one; no suite, test method or setup fixture ran. A guard allowed only parameter-free import-time activation objects, blocked other model construction, parameter allocation and tensor loading. The first collection diagnostic (`ci-training-collection-v1.log`) stopped because an overly broad guard also rejected Transformers activation singletons; it is retained, and no native model ran.
- The current data lane selects 72 methods (65 existing plus seven runner tests); the earlier complete data-lane attempt selected 71 before the new regression test was added. The latest seven-test runner pass and parent localhost recheck supplement that earlier run; there was no new combined 72-test run.
- During the initial CI-separation work, no dependencies were installed and no training/tiny-Qwen/native suite, GPU work, wheel build, remote workflow, commit or remote write occurred. The new Python files were syntax-parsed; document links were checked. The subsequent isolated wheel build is documented below. A clean hosted runner still needs to validate dependency resolution and all training tests.

## Isolated wheel verification

The packaging fix changes only the package list, adds [scripts/check_wheel.py](../../../../scripts/check_wheel.py), and appends a wheel smoke step to the existing data workflow. Previous package/workflow edits are preserved. Before editing, a transitive metadata check of the active `fd8a504ab2567e4cecefc6a91e3eb91961329e31db1a501a1f9887a8b882ddec` training registration inspected 327 JSON documents and 4,725 absolute paths. None of the four edited/new packaging/document paths appeared in those active bindings. The experiment's pinned runtime sources, tests, drivers and logs were not changed.

The checker accepts a wheel or a directory containing exactly one wheel. It verifies required members, rejects duplicate or escaping archive paths, and unpacks into a temporary directory outside the checkout. A subprocess runs with `python -I -S`, an external working directory, and only that unpacked wheel added to its import path. No installation, network access, model load or optional training dependency is needed. The check verifies:

- Package-local imports of `data.cli`, `training.evaluate`, `training.reference_masked_supervision_evaluation`, and `scripts.score_latest_audio`.
- Unicode normalization and edit-count behavior of the packaged scorer.
- All three console entry-point definitions and `hojil-data --help` through its packaged entry point.
- Both packaged HTML resources, `playground.html` and `youtube.html`.
- That Torch, Qwen, Transformers, PEFT and NumPy were not imported.

Local validation used the existing setuptools 78.1.0 backend offline in an external source snapshot. The rebuilt wheel passed all checks. The same checker rejected the original wheel specifically because `scripts/score_latest_audio.py` was missing, demonstrating regression coverage for the actual defect. Nothing was installed into the active environment. The wheel, build logs and positive/negative receipts remain outside Git at `C:/Users/Khan/.codex/visualizations/2026/11/09/01a1bf77-0b17-7391-9781-f859f1fbb0b1/packaging-wheel-fix-verification-v1/`; its `verification.json` records exact source hashes and subprocess results. The binding-scope receipt is the adjacent `packaging-active-binding-scope-v1.json`.

These checks establish wheel contents and dependency-free import/CLI/resource behavior. They do not establish a clean Linux dependency installation, all optional CLI functionality, native training portability, or a successful hosted CI run.

The classification below is static and conservative: “CPU training” means eligible for the proposed full training-stack CPU lane, not that every module needs every dependency or has already passed on Ubuntu. Numerical tolerances, optional-library installation and source namespace behavior still need clean-runner verification. Inspection covered the imports, all declared test methods and their setup/helpers, filesystem/model-loader call sites, missing-dependency decorators and relevant delegated production read/load paths. No further unconditional retained-experiment dependency was identified in that static audit. This is not a proof against every possible transitive side effect: any additional file-not-found/import error remains a failing test, and unexpected skips also fail the proposed training job rather than expanding the exclusion list automatically.

## Complete module inventory

| Test module | Declared methods | CI classification |
| --- | ---: | --- |
| [test_ci_runner.py](../../../../tests/test_ci_runner.py) | 7 | New stdlib infrastructure; data lane |
| [test_bulk.py](../../../../tests/test_bulk.py) | 5 | Data |
| [test_centered_speaker_adapter.py](../../../../tests/test_centered_speaker_adapter.py) | 4 | CPU training |
| [test_integration.py](../../../../tests/test_integration.py) | 2 | Data |
| [test_pipeline.py](../../../../tests/test_pipeline.py) | 23 | Data |
| [test_playground.py](../../../../tests/test_playground.py) | 5 | Data |
| [test_preprocessing.py](../../../../tests/test_preprocessing.py) | 12 | Data |
| [test_reference_codec_routing.py](../../../../tests/test_reference_codec_routing.py) | 3 | CPU training |
| [test_reference_condition_routing.py](../../../../tests/test_reference_condition_routing.py) | 3 | CPU training |
| [test_reference_masked_supervision.py](../../../../tests/test_reference_masked_supervision.py) | 8 | CPU training |
| [test_reference_masked_supervision_confirmation.py](../../../../tests/test_reference_masked_supervision_confirmation.py) | 6 | CPU training |
| [test_reference_masked_supervision_evaluation.py](../../../../tests/test_reference_masked_supervision_evaluation.py) | 4 | CPU training |
| [test_reference_masked_supervision_features.py](../../../../tests/test_reference_masked_supervision_features.py) | 5 | CPU training |
| [test_reference_masked_supervision_parity.py](../../../../tests/test_reference_masked_supervision_parity.py) | 11 | CPU training |
| [test_reference_masked_supervision_plan.py](../../../../tests/test_reference_masked_supervision_plan.py) | 4 | CPU training |
| [test_reference_masked_supervision_speech.py](../../../../tests/test_reference_masked_supervision_speech.py) | 4 | CPU training |
| [test_reference_masked_supervision_speech_v2.py](../../../../tests/test_reference_masked_supervision_speech_v2.py) | 4 | CPU training |
| [test_reference_masked_supervision_training.py](../../../../tests/test_reference_masked_supervision_training.py) | 9 | Mixed CPU + retained-artifact method |
| [test_reference_masked_supervision_training_v2.py](../../../../tests/test_reference_masked_supervision_training_v2.py) | 11 | Mixed CPU + retained-artifact method |
| [test_reference_prefix_generation_pad.py](../../../../tests/test_reference_prefix_generation_pad.py) | 6 | CPU training |
| [test_reference_prefix_lora_routing.py](../../../../tests/test_reference_prefix_lora_routing.py) | 6 | CPU training |
| [test_reference_prefix_routing.py](../../../../tests/test_reference_prefix_routing.py) | 3 | CPU training |
| [test_repartition.py](../../../../tests/test_repartition.py) | 3 | Data |
| [test_source_audit.py](../../../../tests/test_source_audit.py) | 8 | Data |
| [test_training.py](../../../../tests/test_training.py) | 7 | CPU training |
| [test_training_acoustic_distillation.py](../../../../tests/test_training_acoustic_distillation.py) | 3 | CPU training |
| [test_training_balanced_preference_rollouts.py](../../../../tests/test_training_balanced_preference_rollouts.py) | 3 | CPU training |
| [test_training_calibrated_confirmation.py](../../../../tests/test_training_calibrated_confirmation.py) | 1 | CPU training |
| [test_training_calibrated_quality_objective.py](../../../../tests/test_training_calibrated_quality_objective.py) | 1 | CPU training |
| [test_training_calibrated_quality_training.py](../../../../tests/test_training_calibrated_quality_training.py) | 1 | CPU lane; CUDA method(s) skip |
| [test_training_centered_speaker_training.py](../../../../tests/test_training_centered_speaker_training.py) | 2 | CPU lane; CUDA method(s) skip |
| [test_training_codec_policy_replay.py](../../../../tests/test_training_codec_policy_replay.py) | 3 | CPU training |
| [test_training_continuation.py](../../../../tests/test_training_continuation.py) | 4 | CPU training |
| [test_training_differentiable_codec_probe.py](../../../../tests/test_training_differentiable_codec_probe.py) | 3 | CPU training |
| [test_training_direct_speaker_gate.py](../../../../tests/test_training_direct_speaker_gate.py) | 3 | CPU training |
| [test_training_direct_speaker_proposal.py](../../../../tests/test_training_direct_speaker_proposal.py) | 2 | CPU training |
| [test_training_direct_voice_objective.py](../../../../tests/test_training_direct_voice_objective.py) | 2 | CPU training |
| [test_training_direct_voice_training.py](../../../../tests/test_training_direct_voice_training.py) | 2 | CPU training |
| [test_training_evaluation.py](../../../../tests/test_training_evaluation.py) | 5 | CPU training |
| [test_training_fp32_merge_comparison.py](../../../../tests/test_training_fp32_merge_comparison.py) | 3 | CPU training |
| [test_training_joint_confirmation_execution.py](../../../../tests/test_training_joint_confirmation_execution.py) | 8 | CPU training |
| [test_training_middle_attention_adapter.py](../../../../tests/test_training_middle_attention_adapter.py) | 3 | CPU training |
| [test_training_middle_attention_training.py](../../../../tests/test_training_middle_attention_training.py) | 1 | CPU lane; CUDA method(s) skip |
| [test_training_observability.py](../../../../tests/test_training_observability.py) | 6 | CPU training |
| [test_training_optimization.py](../../../../tests/test_training_optimization.py) | 4 | CPU training |
| [test_training_overfit.py](../../../../tests/test_training_overfit.py) | 6 | CPU training |
| [test_training_preference_objective.py](../../../../tests/test_training_preference_objective.py) | 3 | CPU training |
| [test_training_preference_pilot.py](../../../../tests/test_training_preference_pilot.py) | 2 | CPU training |
| [test_training_preference_rollouts.py](../../../../tests/test_training_preference_rollouts.py) | 3 | CPU training |
| [test_training_profile_descent.py](../../../../tests/test_training_profile_descent.py) | 2 | CPU training |
| [test_training_quality_preference_objective.py](../../../../tests/test_training_quality_preference_objective.py) | 3 | CPU training |
| [test_training_quality_preference_training.py](../../../../tests/test_training_quality_preference_training.py) | 2 | CPU lane; CUDA method(s) skip |
| [test_training_reference_prefix_generation_pad_confirmation.py](../../../../tests/test_training_reference_prefix_generation_pad_confirmation.py) | 4 | CPU training |
| [test_training_reference_prefix_generation_pad_evaluation.py](../../../../tests/test_training_reference_prefix_generation_pad_evaluation.py) | 6 | CPU training |
| [test_training_reference_prefix_lora_pad_evaluation.py](../../../../tests/test_training_reference_prefix_lora_pad_evaluation.py) | 5 | CPU training |
| [test_training_reference_profile_collection.py](../../../../tests/test_training_reference_profile_collection.py) | 2 | CPU training |
| [test_training_reference_profile_confirmation.py](../../../../tests/test_training_reference_profile_confirmation.py) | 3 | CPU training |
| [test_training_reference_profile_evaluation.py](../../../../tests/test_training_reference_profile_evaluation.py) | 1 | CPU training |
| [test_training_reference_profile_gate.py](../../../../tests/test_training_reference_profile_gate.py) | 2 | CPU training |
| [test_training_reference_profile_mean_evaluation.py](../../../../tests/test_training_reference_profile_mean_evaluation.py) | 2 | CPU training |
| [test_training_reference_profile_mean_selection.py](../../../../tests/test_training_reference_profile_mean_selection.py) | 2 | CPU training |
| [test_training_reference_profile_optimization.py](../../../../tests/test_training_reference_profile_optimization.py) | 2 | CPU training |
| [test_training_reference_profile_preference_collection.py](../../../../tests/test_training_reference_profile_preference_collection.py) | 2 | CPU training |
| [test_training_reference_profile_preference_confirmation.py](../../../../tests/test_training_reference_profile_preference_confirmation.py) | 4 | CPU training |
| [test_training_reference_profile_preference_evaluation.py](../../../../tests/test_training_reference_profile_preference_evaluation.py) | 4 | CPU training |
| [test_training_reference_profile_preference_objective.py](../../../../tests/test_training_reference_profile_preference_objective.py) | 3 | CPU training |
| [test_training_reference_profile_preference_training.py](../../../../tests/test_training_reference_profile_preference_training.py) | 3 | CPU training |
| [test_training_reference_protected.py](../../../../tests/test_training_reference_protected.py) | 5 | CPU training |
| [test_training_reference_voice_profile.py](../../../../tests/test_training_reference_voice_profile.py) | 3 | CPU training |
| [test_training_refreshed_speaker_rollouts.py](../../../../tests/test_training_refreshed_speaker_rollouts.py) | 2 | CPU training |
| [test_training_refreshed_speaker_training.py](../../../../tests/test_training_refreshed_speaker_training.py) | 1 | CPU lane; CUDA method(s) skip |
| [test_training_residual_sampling_calibration.py](../../../../tests/test_training_residual_sampling_calibration.py) | 3 | CPU training |
| [test_training_sequence.py](../../../../tests/test_training_sequence.py) | 4 | CPU training |
| [test_training_speaker_diagnostics.py](../../../../tests/test_training_speaker_diagnostics.py) | 3 | CPU training |
| [test_training_speaker_preference_adapter.py](../../../../tests/test_training_speaker_preference_adapter.py) | 3 | CPU training |
| [test_training_speaker_preference_training.py](../../../../tests/test_training_speaker_preference_training.py) | 4 | CPU lane; CUDA method(s) skip |
| [test_training_speaker_reference_adaptation.py](../../../../tests/test_training_speaker_reference_adaptation.py) | 2 | CPU training |
| [test_training_target_weight_transfer.py](../../../../tests/test_training_target_weight_transfer.py) | 2 | CPU training |
| [test_training_targeted_voice.py](../../../../tests/test_training_targeted_voice.py) | 6 | CPU training |
| [test_training_test_protocol.py](../../../../tests/test_training_test_protocol.py) | 5 | CPU training |
| [test_training_voice_gradient_geometry.py](../../../../tests/test_training_voice_gradient_geometry.py) | 2 | CPU training |
| [test_training_voice_preferences.py](../../../../tests/test_training_voice_preferences.py) | 4 | CPU training |
| [test_training_voice_strength_search.py](../../../../tests/test_training_voice_strength_search.py) | 3 | CPU training |
| [test_youtube.py](../../../../tests/test_youtube.py) | 7 | Data |

## Initial audit configuration snapshot identities

These hashes identify the initial pre-separation configuration bytes inspected, including the previous data workflow; they are not hashes of the updated workflow and not test-pass receipts. No external weights were read.

| Source | SHA256 |
| --- | --- |
| [.github/workflows/data-preparation.yml](../../../../.github/workflows/data-preparation.yml) | `d7ffb23e17953970ee9b2ca68fef8604dd94fcdd802ed4b903684b5a6ccf09aa` |
| [pyproject.toml](../../../../pyproject.toml) | `240f684d875190c1266eaabf89e8cddc870d05a7ca4ab552681e363c280324f8` |
| [requirements-training.txt](../../../../requirements-training.txt) | `bbfd80b941c76860d01a9332beb37e62cdfc771ccab93190ef07552a89c3c62d` |
| [requirements-verification.txt](../../../../requirements-verification.txt) | `93ec6a6fcd1f5153612ab134a097d9397261f43630efe68772ca8b280068f044` |

