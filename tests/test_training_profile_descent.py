import unittest
import torch
from training.profile_descent import common_direction,raw_for_displacement


class ProfileDescentTests(unittest.TestCase):
    def test_compatible_and_conflicting_gradients_both_decrease(self):
        voice=torch.tensor([1.,0.])
        for anchor in (torch.tensor([1.,1.]),torch.tensor([-1.,1.]),torch.zeros(2)):
            d,metrics=common_direction(voice,anchor)
            self.assertLess(float(d.dot(voice)),0)
            if anchor.norm()>0:self.assertLess(float(d.dot(anchor)),0)
            self.assertAlmostEqual(float(d.norm()),1,places=6)
        with self.assertRaises(ValueError):common_direction(voice,-voice)

    def test_requested_fraction_matches_bounded_profile_forward(self):
        d=torch.tensor([3.,4.]);radius=.05
        for fraction in (.005,.0025,.00125,.000625):
            raw=raw_for_displacement(d,fraction,radius)
            actual=radius*raw/(1+raw.square().sum()).sqrt()
            self.assertAlmostEqual(float(actual.norm()),fraction,places=8)
        with self.assertRaises(ValueError):raw_for_displacement(d,.05,radius)


if __name__=='__main__':unittest.main()
