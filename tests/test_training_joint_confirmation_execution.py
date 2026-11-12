import copy
import unittest
from training.confirm_joint_candidate import validate_panel, validate_records
from scripts.score_latest_audio import score


class ConfirmationExecutionTests(unittest.TestCase):
    def fixture(self):
        cases = [dict(case=f'dev-{i}', group='development', speaker_id=str(i % 10),
            text='a'*100, reference_sha256=f'ref-{i % 10}') for i in range(44)]
        cases += [dict(case=f'custom-{i}', group='custom', speaker_id='custom',
            text='a'*100, reference_sha256='custom-ref') for i in range(13)]
        panel = dict(cases=cases, seeds=[2026, 2027], development_recordings=88,
            custom_recordings=26, criteria=dict(pooled_cer_strictly_below=.03,
                speaker_macro_cosine_strictly_above=.984, groups=['development', 'custom'],
                require_complete_panel=True))
        records = [dict(c, seed=s, terminated=True, cosine=.985,
            score=score(c['text'], 'a'*99+'b', True)) for c in cases for s in panel['seeds']]
        return panel, records

    def test_complete_panel_and_strict_joint_thresholds(self):
        p, r = self.fixture()
        result = validate_records(p, r)
        self.assertTrue(result['confirmation_pass'])
        self.assertEqual(result['summary']['development']['recordings'], 88)
        self.assertEqual(result['summary']['custom']['recordings'], 26)
        for group in ('development', 'custom'):
            changed = copy.deepcopy(r)
            for row in changed:
                if row['group'] == group:
                    row['score'] = score(row['text'], 'a'*97+'bbb', True)
            self.assertFalse(validate_records(p, changed)['confirmation_pass'])
            changed = copy.deepcopy(r)
            for row in changed:
                if row['group'] == group:
                    row['cosine'] = .984
            self.assertFalse(validate_records(p, changed)['confirmation_pass'])

    def test_favorable_subset_duplicates_or_reference_changes_are_rejected(self):
        p, r = self.fixture()
        for changed in (r[:-1], r[:-1]+[r[0]], [v for v in r if v['seed'] == 2027]):
            with self.assertRaisesRegex(ValueError, 'Incomplete'):
                validate_records(p, changed)
        r[0]['reference_sha256'] = 'other-reference'
        with self.assertRaisesRegex(ValueError, 'case changed'):
            validate_records(p, r)

    def test_truncated_audio_cannot_pass(self):
        p, r = self.fixture()
        r[-1]['terminated'] = False
        result = validate_records(p, r)
        self.assertEqual(result['truncated'], 1)
        self.assertFalse(result['confirmation_pass'])

    def test_weaker_scope_or_thresholds_cannot_replace_registration(self):
        p, _ = self.fixture()
        for key, value in (('seeds', [2027]), ('development_recordings', 20), ('custom_recordings', 6)):
            changed = copy.deepcopy(p)
            changed[key] = value
            with self.assertRaisesRegex(ValueError, 'scope'):
                validate_panel(changed)
        p['criteria']['pooled_cer_strictly_below'] = .04
        with self.assertRaisesRegex(ValueError, 'thresholds'):
            validate_panel(p)

    def test_centered_candidate_keeps_frozen_initial_adapter_and_acoustic_identity(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        import torch
        from training.speaker_preference_adapter import SpeakerAdapter
        from training.centered_speaker_adapter import CenteredSpeakerAdapter,fit_basis
        from training import centered_speaker_training as trainer
        from training.confirm_joint_candidate import load_candidate
        torch.manual_seed(71)
        initial=SpeakerAdapter(8,2)
        with torch.no_grad():initial.up.normal_(0,.01)
        coordinates=fit_basis({str(i):[torch.randn(8)] for i in range(5)},2)
        model=torch.nn.Module()
        model.backbone=torch.nn.Linear(8,8).requires_grad_(False)
        model.speaker_preference_adapter=CenteredSpeakerAdapter(initial,coordinates)
        expected=trainer.frozen_digest(model)
        p=dict(kind='centered',candidate='unused',step=8,
               evidence=dict(identity=dict(frozen_backbone_sha256=expected)))
        saved=copy.deepcopy(model.speaker_preference_adapter.state_dict())
        saved['residual'].normal_(0,.01)
        with patch.object(trainer,'load',return_value=SimpleNamespace(model=model)), \
             patch.object(trainer,'read_checkpoint',return_value=dict(adapter=saved)):
            self.assertIs(load_candidate(p,{}).model,model)
        self.assertEqual(trainer.frozen_digest(model),expected)
        saved['initial.up'][0,0] += 1
        with patch.object(trainer,'load',return_value=SimpleNamespace(model=model)), \
             patch.object(trainer,'read_checkpoint',return_value=dict(adapter=saved)):
            with self.assertRaisesRegex(ValueError,'coordinates changed'):
                load_candidate(p,{})

    def test_attention_candidate_keeps_original_speaker_and_acoustic_identity(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        import torch
        from test_training_sequence import tiny
        from training.speaker_preference_adapter import install
        from training import middle_attention_adapter as api,middle_attention_training as trainer
        from training.confirm_joint_candidate import load_candidate
        torch.set_num_threads(1)
        model=tiny().requires_grad_(False);install(model,2);api.install(model,[0,1],2)
        identity=dict(frozen_backbone_sha256=api.frozen_digest(model),frozen_speaker_sha256=api.speaker_digest(model))
        state=dict(adapter=api.snapshot(model),frozen_speaker_sha256=identity['frozen_speaker_sha256'])
        p=dict(kind='attention',candidate='unused',step=4,evidence=dict(identity=identity))
        with patch.object(trainer,'load',return_value=SimpleNamespace(model=model)), \
             patch.object(trainer,'read_checkpoint',return_value=state):
            self.assertIs(load_candidate(p,{}).model,model)
            with torch.no_grad():model.speaker_preference_adapter.up[0,0].add_(1)
            with self.assertRaisesRegex(ValueError,'frozen speaker'):load_candidate(p,{})

    def test_quality_candidate_keeps_original_speaker_and_acoustic_identity(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        import torch
        from test_training_sequence import tiny
        from training.speaker_preference_adapter import install
        from training import middle_attention_adapter as api,quality_preference_training as trainer
        from training.confirm_joint_candidate import load_candidate
        torch.set_num_threads(1)
        model=tiny().requires_grad_(False);install(model,2);api.install(model,[0,1],2)
        identity=dict(frozen_backbone_sha256=api.frozen_digest(model),frozen_speaker_sha256=api.speaker_digest(model))
        state=dict(adapter=api.snapshot(model),frozen_speaker_sha256=identity['frozen_speaker_sha256'])
        p=dict(kind='quality',candidate='unused',step=4,evidence=dict(identity=identity))
        with patch.object(trainer,'load',return_value=SimpleNamespace(model=model)), \
             patch.object(trainer,'read_checkpoint',return_value=state):
            self.assertIs(load_candidate(p,{}).model,model)
            with torch.no_grad():model.speaker_preference_adapter.up[0,0].add_(1)
            with self.assertRaisesRegex(ValueError,'frozen speaker'):load_candidate(p,{})

    def test_routing_candidates_reject_changed_alternative_matrices(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        import torch
        from training import reference_prefix_comparison,reference_codec_comparison,reference_condition_comparison
        from training.centered_speaker_training import frozen_digest
        from training.confirm_joint_candidate import load_candidate
        model=torch.nn.Linear(4,4).requires_grad_(False)
        identity=dict(frozen_backbone_sha256=frozen_digest(model),speaker_adapter_sha256='speaker',
                      alternative_weights_sha256='registered',matrices=198,main_matrices=196)
        for kind,comparison in (('prefix',reference_prefix_comparison),('codec',reference_codec_comparison),
                                ('condition',reference_condition_comparison)):
            identity['matrices']=198 if kind=='prefix' else 196
            p=dict(kind=kind,evidence=dict(identity=identity))
            with patch.object(comparison,'load',return_value=SimpleNamespace(model=model)), \
                 patch.object(comparison,'model_identity',return_value=dict(identity)):
                self.assertIs(load_candidate(p,{}).model,model)
            changed=dict(identity,alternative_weights_sha256='changed')
            with patch.object(comparison,'load',return_value=SimpleNamespace(model=model)), \
                 patch.object(comparison,'model_identity',return_value=changed):
                with self.assertRaisesRegex(ValueError,'reference-routing model differs'):
                    load_candidate(p,{})


if __name__ == '__main__':
    unittest.main()
