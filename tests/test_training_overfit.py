import copy
import json
import unittest

from training.overfit import update_guard


class OverfitGuardTests(unittest.TestCase):
    config = dict(patience=5, validation_delta=.005, training_delta=.002)

    def advance(self, state, step, train=3, familiar=3.1, unseen=3.2):
        return update_guard(state, step=step, train=train, familiar=familiar,
                            unseen=unseen, config=self.config)

    def test_plateau_and_joint_deterioration_do_not_claim_overfit(self):
        for train in (3, 3.1):
            state = self.advance(None, 0)
            for step in range(1, 20):
                state = self.advance(state, step, train=train, familiar=3.2, unseen=3.3)
            self.assertFalse(state['stopped'])

    def test_requires_five_consecutive_checks_in_same_cohort(self):
        state = self.advance(None, 0)
        for step in range(1, 5):
            state = self.advance(state, step, train=2.99, unseen=3.21)
            self.assertFalse(state['stopped'])
        state = self.advance(state, 5, train=2.99, unseen=3.21)
        self.assertEqual(state['triggered_groups'], ['unseen'])

    def test_recovery_resets_counter_and_new_minimum_reanchors(self):
        state = self.advance(None, 0)
        state = self.advance(state, 1, train=2.99, familiar=3.11)
        state = self.advance(state, 2, train=2.99, familiar=3.101)
        self.assertEqual(state['groups']['familiar']['bad_checks'], 0)
        state = self.advance(state, 3, train=2.98, familiar=3.09)
        self.assertEqual(state['groups']['familiar']['train'], 2.98)
        state = self.advance(state, 4, train=2.979, familiar=3.11)
        self.assertEqual(state['groups']['familiar']['bad_checks'], 0)

    def test_serialization_and_duplicate_boundary_preserve_state(self):
        state = self.advance(None, 3500)
        state = self.advance(state, 3600, train=2.99, familiar=3.11)
        before = copy.deepcopy(state)
        restored = json.loads(json.dumps(state))
        self.assertEqual(self.advance(restored, 3600), before)
        self.advance(state, 3700, train=2.99, familiar=3.11)
        self.assertEqual(state, before)

    def test_nonfinite_and_out_of_order_probes_fail_closed(self):
        state = self.advance(None, 100)
        with self.assertRaises(ValueError):
            self.advance(state, 99)
        for value in (float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                self.advance(state, 200, train=value)

    def test_stopped_state_cannot_silently_restart(self):
        state = self.advance(None, 0)
        for step in range(1, 6):
            state = self.advance(state, step, train=2.99, familiar=3.11)
        self.assertEqual(self.advance(state, 6, familiar=3), state)


if __name__ == '__main__':
    unittest.main()
