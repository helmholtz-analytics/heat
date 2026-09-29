import numpy as np
import heat as ht
from heat import array_api as xp
from heat.testing.basic_test import TestCase


class TestConstants(TestCase):
    def test_constants(self):
        self.assertTrue(float("inf") == ht.Inf)
        self.assertTrue(ht.inf == np.inf)
        self.assertTrue(np.isnan(ht.nan))
        self.assertTrue(3 < ht.inf)
        self.assertTrue(np.isinf(ht.inf))
        self.assertTrue(ht.pi == np.pi)
        self.assertTrue(ht.e == np.e)
        self.assertTrue(ht.newaxis is None)

    def test_array_api_constants(self):
        self.assertTrue(float("inf") == xp.inf)
        self.assertTrue(xp.inf == np.inf)
        self.assertTrue(np.isnan(xp.nan))
        self.assertTrue(3 < xp.inf)
        self.assertTrue(np.isinf(xp.inf))
        self.assertTrue(xp.pi == np.pi)
        self.assertTrue(xp.e == np.e)
        self.assertTrue(xp.newaxis is None)
