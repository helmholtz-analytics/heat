import unittest
import numpy as np
import os

import heat as ht

ht.use_device(os.environ.get("DEVICE"))


class TestConstants(unittest.TestCase):
    def test_constants(self):
        self.assertTrue(float("inf") == ht.Inf)
        self.assertTrue(ht.inf == np.inf)
        self.assertTrue(np.isnan(ht.nan))
        self.assertTrue(3 < ht.inf)
        self.assertTrue(np.isinf(ht.inf))
        self.assertTrue(ht.pi == np.pi)
        self.assertTrue(ht.e == np.e)
