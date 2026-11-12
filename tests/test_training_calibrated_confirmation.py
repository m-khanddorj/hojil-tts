import unittest

class CalibratedConfirmationTests(unittest.TestCase):
    def test_calibrated_candidate_keeps_original_speaker_and_acoustic_identity(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        import torch
        from test_training_sequence import tiny
        from training.speaker_preference_adapter import install
        from training import middle_attention_adapter as api,calibrated_quality_training as trainer
        from training.confirm_calibrated_candidate import load_candidate
        torch.set_num_threads(1)
        model=tiny().requires_grad_(False);install(model,2);api.install(model,[0,1],2)
        identity=dict(frozen_backbone_sha256=api.frozen_digest(model),frozen_speaker_sha256=api.speaker_digest(model))
        state=dict(adapter=api.snapshot(model),frozen_speaker_sha256=identity['frozen_speaker_sha256'])
        p=dict(kind='calibrated',candidate='unused',step=4,evidence=dict(identity=identity))
        with patch.object(trainer,'load',return_value=SimpleNamespace(model=model)), \
             patch.object(trainer,'read_checkpoint',return_value=state):
            self.assertIs(load_candidate(p,{}).model,model)
            with torch.no_grad():model.speaker_preference_adapter.up[0,0].add_(1)
            with self.assertRaisesRegex(ValueError,'frozen speaker'):load_candidate(p,{})


if __name__=='__main__':unittest.main()
