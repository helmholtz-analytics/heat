import unittest
import heat as ht
import numpy as np
from mpi4py import MPI
import os

from ...core.tests.test_suites.basic_test import TestCase

"""
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
"""
atol_fit = 1e-6
atol_inv = 1e-4


class TestStandardScaler(TestCase):
    def test_StandardScaler(self):
        for copy in [True, False]:
            X = ht.random.rand(MPI.COMM_WORLD.Get_size() * 10, 3, split=0)
            scaler = ht.preprocessing.StandardScaler(copy=copy)
            scaler.fit(X)
            Y = scaler.transform(X)
            self.assertTrue(ht.allclose(Y.mean(axis=0), ht.zeros(Y.shape[1]), atol=atol_fit))
            self.assertTrue(ht.allclose(Y.var(axis=0), ht.ones(Y.shape[1]), atol=atol_fit))
            Y = scaler.inverse_transform(Y)
            self.assertTrue(ht.allclose(Y, X, atol=atol_inv))


class TestMinMaxScaler(TestCase):
    def test_MinMaxScaler(self):
        for copy in [True, False]:
            X = ht.random.randn(MPI.COMM_WORLD.Get_size() * 10, 3, split=0)
            scaler = ht.preprocessing.MinMaxScaler(copy=copy)
            scaler.fit(X)
            Y = scaler.transform(X)
            self.assertTrue(ht.allclose(Y.min(axis=0), ht.zeros(Y.shape[1]), atol=atol_fit))
            self.assertTrue(ht.allclose(Y.max(axis=0), ht.ones(Y.shape[1]), atol=atol_fit))
            Y = scaler.inverse_transform(Y)
            self.assertTrue(ht.allclose(Y, X, atol=atol_inv))
            with self.assertRaises(ValueError):
                scaler = ht.preprocessing.MinMaxScaler(feature_range=(0.5, 0.5), copy=copy)


class TestNormalizer(TestCase):
    def test_Normalizer(self):
        for copy in [True, False]:
            norms = ["l2", "l1", "max"]
            ords = [2, 1, ht.inf]
            for k in range(3):
                X = ht.random.randn(MPI.COMM_WORLD.Get_size() * 10, 3, split=0)
                scaler = ht.preprocessing.Normalizer(norm=norms[k], copy=copy)
                scaler.fit(X)
                X = scaler.transform(X)
                self.assertTrue(
                    ht.allclose(ht.norm(X, axis=1, ord=ords[k]), ht.ones(X.shape[0]), atol=atol_fit)
                )
            with self.assertRaises(NotImplementedError):
                scaler = ht.preprocessing.Normalizer(norm="l3", copy=copy)


class TestMaxAbsScaler(TestCase):
    def test_MaxAbsScaler(self):
        for copy in [True, False]:
            X = ht.random.randn(MPI.COMM_WORLD.Get_size() * 10, 3, split=0)
            scaler = ht.preprocessing.MaxAbsScaler(copy=copy)
            scaler.fit(X)
            Y = scaler.transform(X)
            self.assertTrue(
                ht.allclose(ht.max(ht.abs(Y), axis=0), ht.ones(Y.shape[1]), atol=atol_fit)
            )
            Y = scaler.inverse_transform(Y)
            self.assertTrue(ht.allclose(Y, X, atol=atol_inv))


class TestRobustScaler(TestCase):
    def test_RobustScaler(self):
        for copy in [True, False]:
            for with_centering in [True, False]:
                for with_scaling in [True, False]:
                    X = ht.random.randn(MPI.COMM_WORLD.Get_size() * 10, 3, split=0)
                    scaler = ht.preprocessing.RobustScaler(
                        quantile_range=(24.0, 76.0),
                        copy=copy,
                        with_centering=with_centering,
                        with_scaling=with_scaling,
                    )
                    scaler.fit(X)
                    Y = scaler.transform(X)
                    if with_centering:
                        self.assertTrue(
                            ht.allclose(ht.median(Y, axis=0), ht.zeros(Y.shape[1]), atol=atol_fit)
                        )
                    if with_scaling:
                        self.assertTrue(
                            ht.allclose(
                                ht.percentile(Y, 76.0, axis=0) - ht.percentile(Y, 24.0, axis=0),
                                ht.ones(1),
                                atol=atol_fit,
                            )
                        )
                    Y = scaler.inverse_transform(Y)
                    self.assertTrue(ht.allclose(X, Y, atol=atol_inv))
                    with self.assertRaises(ValueError):
                        scaler = ht.preprocessing.RobustScaler(
                            quantile_range=(-0.1, 3.0),
                            copy=copy,
                            with_centering=with_centering,
                            with_scaling=with_scaling,
                        )
                        scaler = ht.preprocessing.RobustScaler(
                            quantile_range=(3.0, 101.0),
                            copy=copy,
                            with_centering=with_centering,
                            with_scaling=with_scaling,
                        )
                        scaler = ht.preprocessing.RobustScaler(
                            quantile_range=(75.0, 23.0),
                            copy=copy,
                            with_centering=with_centering,
                            with_scaling=with_scaling,
                        )
