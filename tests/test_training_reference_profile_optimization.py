import unittest
from training.reference_profile_optimization import fitting_records


class ReferenceProfileOptimizationTests(unittest.TestCase):
    def setUp(self):
        self.p=dict(cases=[dict(case=str(i),group='profile-fit' if i<8 else 'profile-validation') for i in range(12)],
                    settings=dict(seeds=[1,2]))
        self.report=dict(records=[dict(c,seed=s,terminated=True) for s in (1,2) for c in self.p['cases']])

    def test_reserved_validation_records_are_never_fitting_inputs(self):
        selected=fitting_records(self.p,self.report)
        self.assertEqual(len(selected),16)
        self.assertTrue(all(r['group']=='profile-fit' for r in selected))
        self.assertFalse({r['case'] for r in selected}&{'8','9','10','11'})

    def test_incomplete_duplicate_or_truncated_fitting_records_fail(self):
        records=self.report['records']
        for invalid in (records[1:],records[1:]+[records[1]],
                        [dict(records[0],terminated=False)]+records[1:]):
            with self.assertRaises(ValueError):fitting_records(self.p,dict(records=invalid))


if __name__=='__main__':unittest.main()
