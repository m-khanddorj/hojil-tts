import tempfile
import unittest
from pathlib import Path
from training.engine import write,digest
from training.reference_profile_evaluation import permitted


class EvaluationSelectionTests(unittest.TestCase):
    def test_no_skipping_or_search_after_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);p=dict(trials=[dict(index=0),dict(index=2)])
            write(out/'control-verification.json',dict(verified=True))
            with self.assertRaises(FileNotFoundError):permitted(out,p,2,'fit')
            write(out/'gate-00.json',dict(passed=False));permitted(out,p,2,'fit')
            write(out/'gate-02.json',dict(passed=True))
            write(out/'selection.json',dict(index=2,gate_sha256=digest(out/'gate-02.json')))
            with self.assertRaises(ValueError):permitted(out,p,2,'fit')
            permitted(out,p,2,'validation');permitted(out,p,2,'screen')
            with self.assertRaises(ValueError):permitted(out,p,0,'validation')
            write(out/'gate-02.json',dict(passed=False))
            with self.assertRaises(ValueError):permitted(out,p,2,'screen')


if __name__=='__main__':unittest.main()
