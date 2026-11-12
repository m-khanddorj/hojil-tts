import json
from pathlib import Path
import tempfile
import unittest

from training.observability import (adapter_movement, fit_signals, gradient_metrics,
                                   matched_probe, summarize_losses, window_summary)

try:
    import torch
except ImportError:
    torch = None


class ObservationTests(unittest.TestCase):
    def test_token_weighted_loss_and_equal_speaker_mean_are_distinct(self):
        records = [dict(id='a', group='familiar', speaker_id='a', first_sum=20,
                        first_tokens=10, residual_sum=40, residual_tokens=10),
                   dict(id='b', group='familiar', speaker_id='b', first_sum=4,
                        first_tokens=1, residual_sum=6, residual_tokens=1),
                   dict(id='c', group='unseen', speaker_id='c', first_sum=3,
                        first_tokens=1, residual_sum=5, residual_tokens=1)]
        result = summarize_losses(records)
        self.assertAlmostEqual(result['familiar']['combined'], 24/11+.3*46/11)
        self.assertAlmostEqual(result['familiar']['speaker_macro_loss'], (3.2+5.8)/2)
        self.assertAlmostEqual(result['selection_loss'], (result['familiar']['combined']+4.5)/2)

    def test_probe_matches_speaker_and_length_without_using_losses(self):
        train = [dict(id='a',speaker_id='s'),dict(id='b',speaker_id='s')]
        val = [dict(id='v',speaker_id='s',validation_group='familiar')]
        features = {k:dict(codes=[0]*n) for k,n in [('a',5),('b',10),('v',9)]}
        probe = matched_probe(train, val, features, 42)
        self.assertEqual(probe[0]['id'], 'b')
        self.assertEqual(probe[0]['validation_group'], 'train_matched')

    def test_window_aggregates_tokens_and_clipping_not_mean_loss(self):
        events = [dict(first_sum=20,first_tokens=10,residual_sum=40,residual_tokens=10,
                       gradient_norm=.5,clips=1,codec_frames=10,step_seconds=1),
                  dict(first_sum=4,first_tokens=1,residual_sum=6,residual_tokens=1,
                       gradient_norm=2,clips=2,codec_frames=1,step_seconds=2)]
        result = window_summary(events)
        self.assertAlmostEqual(result['window_loss'], 24/11+.3*46/11)
        self.assertEqual(result['clipping_fraction'], .5)
        self.assertEqual(result['clips_per_second'], 1)

    def test_diagnostics_are_trends_not_absolute_loss_thresholds(self):
        probes = [dict(train=3-i*.01,familiar=3+i*.01,unseen=3+i*.01) for i in range(3)]
        self.assertIn('Possible overfitting', fit_signals(probes)[0])
        probes = [dict(train=5,familiar=5,unseen=5)]*3
        self.assertIn('does not prove underfitting', fit_signals(probes)[0])

    def test_dashboard_ignores_per_case_files_and_exports_probes(self):
        from training.monitor import render
        group = dict(combined=3,first=2,residual=10/3,clips=1,speakers=1,speaker_macro_loss=3)
        event = dict(step=0,metrics=dict(familiar=group,unseen=group),
                     train_probe=dict(all=group),familiar_generalization_gap=0,
                     unseen_generalization_gap=0,best_step=0,bad_checks=0,epoch_fraction=0)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'validation-000000.json').write_text(json.dumps(event))
            (root/'validation-cases-000000.json').write_text('[]')
            report = render(root)
            self.assertEqual(report['latest_probe']['train'], 3)
            self.assertTrue((root/'fit-probes.csv').is_file())
            self.assertIn('Fixed train/validation probes', (root/'observability.html').read_text())


@unittest.skipIf(torch is None, 'optional training dependencies not installed')
class NumericalObservationTests(unittest.TestCase):
    def test_gradient_metrics_and_movement_do_not_change_parameters_or_rng(self):
        p = torch.nn.Parameter(torch.tensor([3.,4.]))
        p.grad = torch.tensor([3.,4.])
        rng = torch.get_rng_state().clone()
        values = gradient_metrics([('code_predictor.layer.lora_B.weight',p)])
        self.assertEqual(values['residual_predictor'], 5)
        torch.testing.assert_close(p.grad, torch.tensor([3.,4.]))
        torch.testing.assert_close(torch.get_rng_state(),rng)
        movement = adapter_movement({'x':torch.tensor([0.,0.])}, {'x':torch.tensor([0.,1.])})
        self.assertEqual(movement['talker_decoder']['delta_l2'],1)
        self.assertIsNone(movement['talker_decoder']['relative_delta_l2'])
        self.assertEqual(movement['talker_decoder']['unchanged_fraction'],.5)


if __name__ == '__main__':
    unittest.main()
