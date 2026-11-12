"""Eight metadata/artifact tests; no native model or speech inference.

Inherited tests exercise the intentionally unchanged original panel/scalar and
artifact helpers. The new parent gate and shared-zero phase history are direct.
"""
from pathlib import Path
import copy
import unittest
from tests import test_reference_masked_supervision_confirmation as fixtures
from training import reference_masked_supervision_broad_confirmation as confirmation


class BroadConfirmationTests(fixtures.ConfirmationTests):
    def test_qualified_positive_checkpoint_and_authoritative_process_closure_required(self):
        history=[self.calibration(0),self.calibration(8)]
        bindings=dict(driver_sha256='d',terminal_status_sha256='s',inputs_sha256='i',verification_sha256='v',
            protocol_sha256='p',checkpoint_sha256='c')
        status=dict(status='complete',phase='complete',step=8,progression=confirmation.progression(history),
            summary=history[-1]['summary'],original114_confirmation_required=True,goal_achieved=False,
            global_model_selection_changed=False,inputs_sha256='i',verification_sha256='v')
        closure=dict(status='closed',exit_code=0,original_session_id=41410,step=8,**bindings)
        self.assertEqual(confirmation.validate_parent(status,closure,history,8,bindings),confirmation.progression(history))
        for changes in ({'status':'running'},{'exit_code':1},{'exit_code':False},{'original_session_id':33720},
                {'step':16},{'checkpoint_sha256':'changed'},{'inputs_sha256':'changed'}):
            with self.assertRaisesRegex(ValueError,'process-closed'):
                confirmation.validate_parent(status,dict(closure,**changes),history,8,bindings)
        with self.assertRaisesRegex(ValueError,'Positive registered'):confirmation.validate_parent(status,closure,history,0,bindings)
        for group,metric,value in (('custom','pooled_cer',.03),('development','speaker_macro_cosine',.984)):
            bad=copy.deepcopy(history);bad[-1]['summary'][group][metric]=value
            with self.assertRaisesRegex(ValueError,'qualification'):confirmation.validate_parent(status,closure,bad,8,bindings)
        with self.assertRaisesRegex(ValueError,'Terminal qualified'):
            confirmation.validate_parent(dict(status,status='running'),closure,history,8,bindings)

    def test_shared_zero_phase_has_no_fabricated_generation(self):
        for step in (8,16,32):
            phases=confirmation.phases_for(step)
            self.assertIn('reuse-baseline',phases)
            self.assertNotIn('generate-000000',phases);self.assertNotIn('score-000000',phases);self.assertNotIn('audit-000000',phases)
            expected_positive=sum(s<=step for s in (8,16,32))
            self.assertEqual(len(phases),6+4*expected_positive)
            self.assertIn(f'audit-{step:06d}',phases)
        with self.assertRaises(ValueError):confirmation.phases_for(0)

    def test_same_pure_measurements_and_test_log_gate(self):
        self.assertIs(confirmation.measurement,confirmation.original.measurement)
        self.assertIs(confirmation.bound_generation,confirmation.original.bound_generation)
        self.assertIs(confirmation.fresh_reference,confirmation.original.fresh_reference)
        self.assertIs(confirmation.complete_audit,confirmation.original.complete_audit)
        path=Path(self.root)/confirmation.TEST_LOG
        for text in ('Ran 7 tests\n\nOK\n','Ran 8 tests\n\nOK (skipped=1)\n','Ran 8 tests\nFAILED\n'):
            path.write_text(text)
            with self.assertRaises(ValueError):confirmation.test_evidence(self.root)
        path.write_text('Ran 8 tests in 0.1s\n\nOK\n')
        self.assertEqual(confirmation.test_evidence(self.root),(path,confirmation.digest(path)))


if __name__=='__main__':unittest.main()
