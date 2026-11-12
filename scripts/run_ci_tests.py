"""Select portable data or CPU training tests without importing excluded modules."""
import argparse
from contextlib import contextmanager
from pathlib import Path
import sys
import types
import unittest


DATA_MODULES = frozenset({
    'test_bulk', 'test_pipeline', 'test_repartition', 'test_preprocessing',
    'test_source_audit', 'test_playground', 'test_youtube', 'test_integration',
})
INFRA_MODULES = frozenset({'test_ci_runner'})
ARTIFACT_METHODS = frozenset(
    f'{module}.TrainingTests.test_pinned_real_metadata_plan_scope_and_tamper_fail_closed'
    for module in ('test_reference_masked_supervision_training',
                   'test_reference_masked_supervision_training_v2')
)
CUDA_SKIPS = frozenset({
    'test_training_centered_speaker_training.CenteredTrainingTests.test_optimizer_resume_and_frozen_initial_policy',
    'test_training_middle_attention_training.MiddleTrainingTests.test_cuda_resume_preserves_frozen_speaker_and_acoustic_parameters',
    'test_training_refreshed_speaker_training.RefreshedTrainingTests.test_optimizer_resume_matches_uninterrupted_tiny_qwen_training',
    'test_training_speaker_preference_training.SpeakerPreferenceTrainingTests.test_optimizer_resume_matches_uninterrupted_tiny_qwen_training',
    'test_training_calibrated_quality_training.CalibratedQualityTrainingTests.test_calibrated_anchor_resume_and_full_validation',
    'test_training_quality_preference_training.QualityTrainingTests.test_filtered_dpo_resume_and_full_validation',
    'test_training_quality_preference_training.QualityTrainingTests.test_anchored_dpo_resume_and_full_validation',
})


def canonical_id(case):
    return case.id().removeprefix('tests.')


def module_names(available, lane):
    if lane not in ('data', 'training'):
        raise ValueError(f'Unknown CI lane: {lane}')
    available = set(available)
    missing = (DATA_MODULES | INFRA_MODULES) - available
    if missing:
        raise ValueError(f'Required CI modules missing: {sorted(missing)}')
    selected = ((DATA_MODULES | INFRA_MODULES) if lane == 'data'
                else available - DATA_MODULES - INFRA_MODULES)
    if not selected:
        raise ValueError(f'Empty {lane} suite')
    return sorted(selected)


def cases(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from cases(item)
        else:
            yield item


def select_cases(suite, lane, stream):
    selected, excluded, seen = [], set(), set()
    for case in cases(suite):
        name = canonical_id(case)
        if name in seen:
            raise ValueError(f'Duplicate collected test: {name}')
        seen.add(name)
        if lane == 'training' and name in ARTIFACT_METHODS:
            excluded.add(name)
        else:
            selected.append(case)
    if lane == 'training' and excluded != ARTIFACT_METHODS:
        raise ValueError(f'Artifact exclusion coverage changed: {sorted(excluded)}')
    if not selected:
        raise ValueError(f'Empty {lane} suite')
    print(f'CI lane: {lane}; selected tests: {len(selected)}', file=stream)
    print(f'Explicit artifact exclusions ({len(excluded)}):', file=stream)
    for name in sorted(excluded):
        print(f'  {name}', file=stream)
    return unittest.TestSuite(selected)


def build_suite(root, lane, stream, loader=None):
    loader = loader or unittest.TestLoader()
    names = module_names((p.stem for p in (root / 'tests').glob('test*.py')), lane)
    loaded = [(name, loader.loadTestsFromName(name)) for name in names]
    if loader.errors:
        raise ValueError('Test collection failed:\n' + '\n'.join(loader.errors))
    owned = []
    for name, suite in loaded:
        for case in cases(suite):
            owner = case.__class__.__module__.removeprefix('tests.')
            if owner != name:
                print(f'Imported helper exclusion: {canonical_id(case)} '
                      f'(defined in {owner}, imported by {name})', file=stream)
            else:
                owned.append(case)
    return select_cases(unittest.TestSuite(owned), lane, stream)


@contextmanager
def test_imports(root):
    """Support existing bare sibling and tests.* imports from this checkout."""
    old_path = sys.path[:]
    old_package = sys.modules.get('tests')
    package = types.ModuleType('tests')
    package.__path__ = [str(root / 'tests')]
    sys.modules['tests'] = package
    sys.path[:0] = [str(root), str(root / 'tests')]
    try:
        yield
    finally:
        sys.path[:] = old_path
        if old_package is None:
            sys.modules.pop('tests', None)
        else:
            sys.modules['tests'] = old_package


def successful(result, lane, stream):
    unexpected = []
    for case, reason in result.skipped:
        name = canonical_id(case)
        expected = lane == 'training' and name in CUDA_SKIPS and reason == 'Qwen and CUDA required'
        print(f'{"Expected CUDA" if expected else "Optional data" if lane == "data" else "UNEXPECTED"}'
              f' skip: {name}: {reason}', file=stream)
        if lane == 'training' and not expected:
            unexpected.append(name)
    print(f'Tests run: {result.testsRun}; skipped: {len(result.skipped)}', file=stream)
    return result.wasSuccessful() and not unexpected


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lane', choices=('data', 'training'), required=True)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    try:
        with test_imports(root):
            suite = build_suite(root, args.lane, sys.stderr)
            result = unittest.TextTestRunner(verbosity=2).run(suite)
            return 0 if successful(result, args.lane, sys.stderr) else 1
    except (ValueError, ImportError) as error:
        print(f'CI collection/configuration error: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
