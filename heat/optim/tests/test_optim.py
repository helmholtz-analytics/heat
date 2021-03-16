import heat as ht

from heat.core.tests.test_suites.basic_test import TestCase


class TestOptim(TestCase):
    def test_optim_getattr(self):
        with self.assertRaises(AttributeError):
            ht.optim.asdf()


class TestLRScheduler(TestCase):
    def test_lr_scheduler_callthrough(self):
        import torch.optim.lr_scheduler as lrs

        htlrs = ht.optim.lr_scheduler

        self.assertTrue(htlrs.LambdaLR == lrs.LambdaLR)
        self.assertTrue(htlrs.MultiplicativeLR == lrs.MultiplicativeLR)
        self.assertTrue(htlrs.StepLR == lrs.StepLR)
        self.assertTrue(htlrs.MultiStepLR == lrs.MultiStepLR)
        self.assertTrue(htlrs.ExponentialLR == lrs.ExponentialLR)
        self.assertTrue(htlrs.CosineAnnealingLR == lrs.CosineAnnealingLR)
        self.assertTrue(htlrs.ReduceLROnPlateau == lrs.ReduceLROnPlateau)
        self.assertTrue(htlrs.CyclicLR == lrs.CyclicLR)
        self.assertTrue(htlrs.CosineAnnealingWarmRestarts == lrs.CosineAnnealingWarmRestarts)
        self.assertTrue(htlrs.OneCycleLR == lrs.OneCycleLR)
