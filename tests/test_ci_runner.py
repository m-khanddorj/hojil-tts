"""Standard-library tests of CI selection; never collect the native suite."""
import io
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from scripts import run_ci_tests as runner


def case(name):
    value = unittest.FunctionTestCase(lambda: None)
    value.id = lambda: name
    return value


class CIRunnerTests(unittest.TestCase):
    def test_data_allowlist_and_training_default_include_new_modules(self):
        modules = runner.DATA_MODULES | runner.INFRA_MODULES | {'test_new_training'}
        self.assertEqual(set(runner.module_names(modules, 'data')), runner.DATA_MODULES | runner.INFRA_MODULES)
        self.assertEqual(runner.module_names(modules, 'training'), ['test_new_training'])
        with self.assertRaisesRegex(ValueError, 'missing'):
            runner.module_names(modules - {'test_pipeline'}, 'data')
        with self.assertRaisesRegex(ValueError, 'Unknown'):
            runner.module_names(modules, 'typo')

    def test_exact_artifact_exclusions_keep_synthetic_methods_visible(self):
        retained = 'test_reference_masked_supervision_training.TrainingTests.test_optimizer'
        inputs = [case('tests.' + name) for name in runner.ARTIFACT_METHODS] + [case(retained)]
        stream = io.StringIO()
        suite = runner.select_cases(unittest.TestSuite([unittest.TestSuite(inputs)]), 'training', stream)
        self.assertEqual([runner.canonical_id(v) for v in runner.cases(suite)], [retained])
        for name in runner.ARTIFACT_METHODS:
            self.assertIn(name, stream.getvalue())

    def test_missing_exclusion_duplicate_and_empty_fail_closed(self):
        with self.assertRaisesRegex(ValueError, 'coverage changed'):
            runner.select_cases(unittest.TestSuite([case('ordinary')]), 'training', io.StringIO())
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            runner.select_cases(unittest.TestSuite([case('same'), case('tests.same')]), 'data', io.StringIO())
        with self.assertRaisesRegex(ValueError, 'Empty'):
            runner.select_cases(unittest.TestSuite(), 'data', io.StringIO())

    def test_loader_errors_abort_before_test_execution(self):
        class Loader:
            errors = []
            def loadTestsFromName(self, name):
                self.errors = ['missing dependency or artifact during import']
                return unittest.TestSuite([case(name)])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root / 'tests').mkdir()
            for name in runner.DATA_MODULES | runner.INFRA_MODULES:
                (root / 'tests' / (name + '.py')).touch()
            with self.assertRaisesRegex(ValueError, 'collection failed'):
                runner.build_suite(root, 'data', io.StringIO(), Loader())

    def test_imported_classes_run_only_under_their_defining_module(self):
        own = type('OwnTests', (unittest.TestCase,), {
            '__module__': 'test_owner', 'test_one': lambda self: None})
        helper = type('HelperTests', (unittest.TestCase,), {
            '__module__': 'tests.test_helper', 'test_helper': lambda self: None})
        class Loader:
            errors = []
            def loadTestsFromName(self, name):
                return unittest.TestSuite([own('test_one'), helper('test_helper')]
                                          if name == 'test_owner' else [helper('test_helper')])
        stream = io.StringIO()
        with patch.object(runner, 'module_names', return_value=['test_owner', 'test_helper']):
            suite = runner.build_suite(Path('/synthetic-checkout'), 'data', stream, Loader())
        self.assertEqual([runner.canonical_id(v) for v in runner.cases(suite)],
                         ['test_owner.OwnTests.test_one', 'test_helper.HelperTests.test_helper'])
        self.assertIn('Imported helper exclusion:', stream.getvalue())
        self.assertIn('imported by test_owner', stream.getvalue())
        class DuplicateLoader(Loader):
            def loadTestsFromName(self, name):
                return unittest.TestSuite([own('test_one'), own('test_one')])
        with patch.object(runner, 'module_names', return_value=['test_owner']):
            with self.assertRaisesRegex(ValueError, 'Duplicate collected'):
                runner.build_suite(Path('/synthetic-checkout'), 'data', io.StringIO(), DuplicateLoader())

    def test_unknown_skips_and_changed_cuda_reasons_are_failures(self):
        expected = case(next(iter(runner.CUDA_SKIPS)))
        def result(skipped, passed=True):
            return types.SimpleNamespace(skipped=skipped, testsRun=1, wasSuccessful=lambda: passed)
        self.assertTrue(runner.successful(result([(expected, 'Qwen and CUDA required')]), 'training', io.StringIO()))
        self.assertFalse(runner.successful(result([(expected, 'Missing local artifact')]), 'training', io.StringIO()))
        self.assertFalse(runner.successful(result([(case('new.test'), 'missing fixture')]), 'training', io.StringIO()))
        self.assertFalse(runner.successful(result([], False), 'training', io.StringIO()))
        self.assertTrue(runner.successful(result([(case('audio.test'), 'optional audio')]), 'data', io.StringIO()))

    def test_checkout_namespace_and_path_restore_on_exception(self):
        path = sys.path[:]; package = sys.modules.get('tests')
        with self.assertRaisesRegex(RuntimeError, 'fixture'):
            with runner.test_imports(Path('/synthetic-checkout')):
                self.assertEqual(sys.path[0], str(Path('/synthetic-checkout')))
                self.assertEqual(sys.modules['tests'].__path__, [str(Path('/synthetic-checkout/tests'))])
                raise RuntimeError('fixture')
        self.assertEqual(sys.path, path)
        self.assertIs(sys.modules.get('tests'), package)


if __name__ == '__main__':
    unittest.main()
