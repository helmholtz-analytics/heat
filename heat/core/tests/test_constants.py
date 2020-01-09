import unittest
import numpy as np
import os

import heat as ht

if os.environ.get("DEVICE") == "gpu" and ht.torch.cuda.is_available():
    ht.use_device("gpu")
    ht.torch.cuda.set_device(ht.torch.device(ht.get_device().torch_device))
else:
    ht.use_device("cpu")
device = ht.get_device().torch_device
ht_device = None
if os.environ.get("DEVICE") == "lgpu" and ht.torch.cuda.is_available():
    device = ht.gpu.torch_device
    ht_device = ht.gpu
    ht.torch.cuda.set_device(device)


class TestConstants(unittest.TestCase):
    def test_constants(self):
        self.assertTrue(float("inf") == ht.Inf)
        self.assertTrue(ht.inf == np.inf)
        self.assertTrue(np.isnan(ht.nan))
        self.assertTrue(3 < ht.inf)
        self.assertTrue(np.isinf(ht.inf))
        self.assertTrue(ht.pi == np.pi)
        self.assertTrue(ht.e == np.e)
