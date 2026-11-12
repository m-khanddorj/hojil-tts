import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from training.engine import write,digest
from training.reference_profile_mean_evaluation import fitting_evidence,permitted


class MeanEvaluationTests(unittest.TestCase):
    def test_requires_terminal_original_and_no_original_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);prior=root/'prior';prior.mkdir()
            with patch('training.reference_profile_mean_evaluation.original.verify',return_value=({}, {}, {})):
                write(root/'reference-profile-evaluation-status.json',dict(status='running'))
                with self.assertRaises(ValueError):fitting_evidence(prior)
                write(root/'reference-profile-evaluation-status.json',dict(status='complete',reason='no_fitting_speech_pass'))
                write(prior/'selection.json',dict(index=1))
                with self.assertRaises(ValueError):fitting_evidence(prior)

    def test_cannot_search_fitting_or_change_selected_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);p=dict(mean_decision=dict(selected=0));write(out/'protocol.json',p)
            write(out/'selection.json',dict(index=0,decision=p['mean_decision'],protocol_sha256=digest(out/'protocol.json')))
            permitted(out,p,0,'validation');permitted(out,p,0,'screen')
            for kind in ('fit','control'):
                with self.assertRaises(ValueError):permitted(out,p,0,kind)
            with self.assertRaises(ValueError):permitted(out,p,1,'screen')
            write(out/'protocol.json',dict(mean_decision=dict(selected=1)))
            with self.assertRaises(ValueError):permitted(out,p,0,'screen')


if __name__=='__main__':unittest.main()
