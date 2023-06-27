import unittest
import heat as ht
import numpy as np
import sklearn.preprocessing
from mpi4py import MPI
import os

from ...core.tests.test_suites.basic_test import TestCase

envar = os.getenv("HEAT_TEST_USE_DEVICE", "cpu")


class TestStandardScaler(TestCase):
    def test_standard_scaler(self):
        if envar not in ["cpu"]:
            dts = [ht.float32]
        else:
            dts = [ht.float32, ht.float64]
        for dt in dts:
            if dt == ht.float32:
                atol = 1e-7
            if dt == ht.float64:
                atol = 1e-14

            X_ht = ht.random.rand(MPI.COMM_WORLD.Get_size() * 10, 3, split=0, dtype=dt)
            X_np = X_ht.numpy()
            Z_ht = ht.random.randn(MPI.COMM_WORLD.Get_size() * 5, 3, split=0, dtype=dt)
            Z_np = Z_ht.numpy()

            # test fit
            scaler_ht = ht.preprocessing.StandardScaler()
            scaler_sk = sklearn.preprocessing.StandardScaler()

            scaler_ht.fit(X_ht)
            scaler_sk.fit(X_np)

            self.assertTrue(np.allclose(scaler_ht.mean_.numpy(), scaler_sk.mean_, atol=atol))
            self.assertTrue(np.allclose(scaler_ht.var_.numpy(), scaler_sk.var_, atol=atol))

            # test transform
            Z_ht_trafo = scaler_ht.transform(Z_ht)
            Z_np_trafo = scaler_sk.transform(Z_np)
            self.assertTrue(np.allclose(Z_ht_trafo.numpy(), Z_np_trafo, atol=atol))

            # test inverse_transform
            Z_ht_invtrafo = scaler_ht.inverse_transform(Z_ht_trafo)
            Z_np_invtrafo = scaler_sk.inverse_transform(Z_np_trafo)

            self.assertTrue(np.allclose(Z_ht_invtrafo.numpy(), Z_np_invtrafo, atol=atol))
            self.assertTrue(np.allclose(Z_ht_invtrafo, Z_ht, atol=atol))

            # test fit_transform
            Z_ht_trafo = scaler_ht.fit_transform(Z_ht)
            Z_np_trafo = scaler_sk.fit_transform(Z_np)
            self.assertTrue(np.allclose(scaler_ht.mean_.numpy(), scaler_sk.mean_, atol=atol))
            self.assertTrue(np.allclose(scaler_ht.var_.numpy(), scaler_sk.var_, atol=atol))
            self.assertTrue(np.allclose(Z_ht_trafo.numpy(), Z_np_trafo, atol=atol))
