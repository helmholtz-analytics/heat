import unittest
import heat as ht
import numpy as np
from mpi4py import MPI
import os

from ...core.tests.test_suites.basic_test import TestCase

atol_fit = 1e-5
atol_inv = 1e-4


# generates a test data set with varying mean and variation per feature; variances of the two last features are zero, mean of the last feature is also zero, whereas mean of second last feature is nonzero.
def _generate_test_data_set(n_data_points, n_features, split, dtype=ht.float32):
    mu = ht.arange(0, n_features)
    mu[-1] = 1e-12
    sigma = ht.arange(n_features - 1, -1, -1)
    sigma[-1] = 1e-9
    sigma[-2] = 1e-9
    return ht.random.normal(mu, sigma, shape=(n_data_points, n_features), split=split, dtype=dtype)


class TestStandardScaler(TestCase):
    def test_StandardScaler(self):
        for split in [0, 1]:
            # first option: with copy = True
            copy = True
            X = _generate_test_data_set(
                MPI.COMM_WORLD.Get_size() * 10,
                MPI.COMM_WORLD.Get_size() * 4,
                split,
                dtype=ht.float32,
            )
            scaler = ht.preprocessing.StandardScaler(copy=copy)
            scaler.fit(X)
            Y = scaler.transform(X)
            self.assertTrue(ht.allclose(Y.mean(axis=0), ht.zeros(Y.shape[1]), atol=atol_fit))
            # last two features have variance 0 and are not scaled therefore
            self.assertTrue(
                ht.allclose(Y.var(axis=0)[:-2], ht.ones(Y.shape[1])[:-2], atol=atol_fit)
            )
            self.assertTrue(ht.allclose(Y.var(axis=0)[-2:], ht.zeros(2), atol=atol_fit))
            Y = scaler.inverse_transform(Y)
            self.assertTrue(ht.allclose(Y, X, atol=atol_inv))
            Z = ht.zeros(
                (MPI.COMM_WORLD.Get_size() * 10, MPI.COMM_WORLD.Get_size() * 2), split=split
            )
            with self.assertRaises(ValueError):
                scaler.transform(Z)

            # second option: copy = False
            copy = False
            X = _generate_test_data_set(
                MPI.COMM_WORLD.Get_size() * 10,
                MPI.COMM_WORLD.Get_size() * 4,
                split,
                dtype=ht.float32,
            )
            X_cpy = X.copy()  # for later checks only...
            scaler = ht.preprocessing.StandardScaler(copy=copy)
            scaler.fit(X)
            scaler.transform(X)
            self.assertTrue(ht.allclose(X.mean(axis=0), ht.zeros(X.shape[1]), atol=atol_fit))
            # last two features have variance 0 and are not scaled therefore
            self.assertTrue(
                ht.allclose(X.var(axis=0)[:-2], ht.ones(X.shape[1])[:-2], atol=atol_fit)
            )
            self.assertTrue(ht.allclose(X.var(axis=0)[-2:], ht.zeros(2), atol=atol_fit))
            scaler.inverse_transform(X)
            self.assertTrue(ht.allclose(X, X_cpy, atol=atol_inv))

        scaler = ht.preprocessing.StandardScaler()
        with self.assertRaises(TypeError):
            scaler.fit(["abc"])
        with self.assertRaises(ValueError):
            scaler.fit(ht.zeros((10, 10, 10), dtype=ht.float32))
        with self.assertRaises(TypeError):
            scaler.fit(ht.zeros(10, 10, dtype=ht.int32))


class TestMinMaxScaler(TestCase):
    def test_MinMaxScaler(self):
        for split in [0, 1]:
            # first case: use standard operations
            copy = True
            X = _generate_test_data_set(
                MPI.COMM_WORLD.Get_size() * 10,
                MPI.COMM_WORLD.Get_size() * 4,
                split,
                dtype=ht.float32,
            )
            scaler = ht.preprocessing.MinMaxScaler(copy=copy)
            scaler.fit(X)
            Y = scaler.transform(X)
            self.assertTrue(
                ht.allclose(Y.min(axis=0)[:-2], ht.zeros(Y.shape[1])[:-2], atol=atol_fit)
            )
            self.assertTrue(
                ht.allclose(Y.max(axis=0)[:-2], ht.ones(Y.shape[1])[:-2], atol=atol_fit)
            )
            self.assertTrue(ht.allclose(Y.min(axis=0)[-2:], ht.zeros(2), atol=atol_fit))
            self.assertTrue(ht.allclose(Y.max(axis=0)[-2:], ht.zeros(2), atol=atol_fit))
            Y = scaler.inverse_transform(Y)
            self.assertTrue(ht.allclose(Y, X, atol=atol_inv))
            with self.assertRaises(ValueError):
                scaler = ht.preprocessing.MinMaxScaler(feature_range=(0.5, 0.5), copy=copy)
            Z = ht.zeros(
                (MPI.COMM_WORLD.Get_size() * 10, MPI.COMM_WORLD.Get_size() * 2), split=split
            )
            with self.assertRaises(ValueError):
                scaler.transform(Z)
            # second case: use in-place operations
            copy = False
            X = _generate_test_data_set(
                MPI.COMM_WORLD.Get_size() * 10,
                MPI.COMM_WORLD.Get_size() * 4,
                split,
                dtype=ht.float32,
            )
            X_cpy = X.copy()  # for comparison only
            scaler = ht.preprocessing.MinMaxScaler(copy=copy)
            scaler.fit(X)
            scaler.transform(X)
            self.assertTrue(
                ht.allclose(X.min(axis=0)[:-2], ht.zeros(X.shape[1])[:-2], atol=atol_fit)
            )
            self.assertTrue(
                ht.allclose(X.max(axis=0)[:-2], ht.ones(X.shape[1])[:-2], atol=atol_fit)
            )
            self.assertTrue(ht.allclose(X.min(axis=0)[-2:], ht.zeros(2), atol=atol_fit))
            self.assertTrue(ht.allclose(X.max(axis=0)[-2:], ht.zeros(2), atol=atol_fit))
            scaler.inverse_transform(X)
            self.assertTrue(ht.allclose(X, X_cpy, atol=atol_inv))

        scaler = ht.preprocessing.MinMaxScaler()
        with self.assertRaises(TypeError):
            scaler.fit(["abc"])
        with self.assertRaises(ValueError):
            scaler.fit(ht.zeros((10, 10, 10), dtype=ht.float32))
        with self.assertRaises(TypeError):
            scaler.fit(ht.zeros(10, 10, dtype=ht.int32))


class TestNormalizer(TestCase):
    def test_Normalizer(self):
        for split in [0, 1]:
            for copy in [True, False]:
                norms = ["l2", "l1", "max"]
                ords = [2, 1, ht.inf]
                for k in range(3):
                    X = ht.random.randn(
                        MPI.COMM_WORLD.Get_size() * 10, MPI.COMM_WORLD.Get_size() * 4, split=split
                    )
                    X = ht.vstack([X, 1e-16 * ht.random.rand(10, MPI.COMM_WORLD.Get_size() * 4)])
                    scaler = ht.preprocessing.Normalizer(norm=norms[k], copy=copy)
                    scaler.fit(X)
                    if copy:
                        X = scaler.transform(X)
                    else:
                        scaler.transform(X)
                    self.assertTrue(
                        ht.allclose(
                            ht.norm(X, axis=1, ord=ords[k])[:-10],
                            ht.ones(X.shape[0])[:-10],
                            atol=atol_fit,
                        )
                    )
                    self.assertTrue(
                        ht.allclose(
                            ht.norm(X, axis=1, ord=ords[k])[-10:],
                            ht.zeros(X.shape[0])[-10:],
                            atol=atol_fit,
                        )
                    )
                with self.assertRaises(NotImplementedError):
                    scaler = ht.preprocessing.Normalizer(norm="l3", copy=copy)
        scaler = ht.preprocessing.Normalizer()
        with self.assertRaises(TypeError):
            scaler.transform(["abc"])
        with self.assertRaises(ValueError):
            scaler.transform(ht.zeros((10, 10, 10), dtype=ht.float32))
        with self.assertRaises(TypeError):
            scaler.transform(ht.zeros(10, 10, dtype=ht.int32))


class TestMaxAbsScaler(TestCase):
    def test_MaxAbsScaler(self):
        for split in [0, 1]:
            # first case: use normal operations
            copy = True
            X = _generate_test_data_set(
                MPI.COMM_WORLD.Get_size() * 10,
                MPI.COMM_WORLD.Get_size() * 4,
                split,
                dtype=ht.float32,
            )
            scaler = ht.preprocessing.MaxAbsScaler(copy=copy)
            scaler.fit(X)
            Y = scaler.transform(X)
            self.assertTrue(
                ht.allclose(ht.max(ht.abs(Y), axis=0)[:-1], ht.ones(Y.shape[1])[:-1], atol=atol_fit)
            )
            self.assertTrue(ht.allclose(ht.max(ht.abs(Y), axis=0)[-1], ht.zeros(1), atol=atol_fit))
            Y = scaler.inverse_transform(Y)
            self.assertTrue(ht.allclose(Y, X, atol=atol_inv))
            Z = ht.zeros(
                (MPI.COMM_WORLD.Get_size() * 10, MPI.COMM_WORLD.Get_size() * 2), split=split
            )
            with self.assertRaises(ValueError):
                scaler.transform(Z)
            # second case: use in-place operations
            copy = False
            X = _generate_test_data_set(
                MPI.COMM_WORLD.Get_size() * 10,
                MPI.COMM_WORLD.Get_size() * 4,
                split,
                dtype=ht.float32,
            )
            X_cpy = X.copy()  # for later comparison only
            scaler = ht.preprocessing.MaxAbsScaler(copy=copy)
            scaler.fit(X)
            scaler.transform(X)
            self.assertTrue(
                ht.allclose(ht.max(ht.abs(X), axis=0)[:-1], ht.ones(X.shape[1])[:-1], atol=atol_fit)
            )
            self.assertTrue(ht.allclose(ht.max(ht.abs(X), axis=0)[-1], ht.zeros(1), atol=atol_fit))
            scaler.inverse_transform(X)
            self.assertTrue(ht.allclose(X, X_cpy, atol=atol_inv))

        scaler = ht.preprocessing.MaxAbsScaler()
        with self.assertRaises(TypeError):
            scaler.fit(["abc"])
        with self.assertRaises(ValueError):
            scaler.fit(ht.zeros((10, 10, 10), dtype=ht.float32))
        with self.assertRaises(TypeError):
            scaler.fit(ht.zeros(10, 10, dtype=ht.int32))


class TestRobustScaler(TestCase):
    def test_RobustScaler(self):
        for split in [0, 1]:
            for with_centering in [False, True]:
                for with_scaling in [False, True]:
                    if not with_centering and not with_scaling:
                        with self.assertRaises(ValueError):
                            scaler = ht.preprocessing.RobustScaler(
                                quantile_range=(24.0, 76.0),
                                with_centering=with_centering,
                                with_scaling=with_scaling,
                            )
                    else:
                        # first case: use standard operations
                        copy = True
                        X = _generate_test_data_set(
                            MPI.COMM_WORLD.Get_size() * 10,
                            MPI.COMM_WORLD.Get_size() * 4,
                            split,
                            dtype=ht.float32,
                        )
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
                                ht.allclose(
                                    ht.median(Y, axis=0), ht.zeros(Y.shape[1]), atol=atol_fit
                                )
                            )
                        if with_scaling:
                            self.assertTrue(
                                ht.allclose(
                                    ht.percentile(Y, 76.0, axis=0)[:-2]
                                    - ht.percentile(Y, 24.0, axis=0)[:-2],
                                    ht.ones(Y.shape[1])[:-2],
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
                        Z = ht.zeros(
                            (MPI.COMM_WORLD.Get_size() * 10, MPI.COMM_WORLD.Get_size() * 2),
                            split=split,
                        )
                        with self.assertRaises(ValueError):
                            scaler.transform(Z)
                        # second case: use in-place operations
                        copy = False
                        if not with_scaling:  # THIS IS A PROBLEM!
                            X = _generate_test_data_set(
                                MPI.COMM_WORLD.Get_size() * 10,
                                MPI.COMM_WORLD.Get_size() * 4,
                                split,
                                dtype=ht.float32,
                            )
                            X_cpy = X.copy()  # for comparison only
                            scaler = ht.preprocessing.RobustScaler(
                                quantile_range=(24.0, 76.0),
                                copy=copy,
                                with_centering=with_centering,
                                with_scaling=with_scaling,
                            )
                            scaler.fit(X)
                            print("before trafo", X)
                            scaler.transform(X)
                            if with_centering:
                                self.assertTrue(
                                    ht.allclose(
                                        ht.median(X, axis=0), ht.zeros(X.shape[1]), atol=atol_fit
                                    )
                                )
                            if with_scaling:
                                self.assertTrue(
                                    ht.allclose(
                                        ht.percentile(X, 76.0, axis=0)[:-2]
                                        - ht.percentile(X, 24.0, axis=0)[:-2],
                                        ht.ones(X.shape[1])[:-2],
                                        atol=atol_fit,
                                    )
                                )
                            scaler.inverse_transform(X)
                            self.assertTrue(ht.allclose(X, X_cpy, atol=atol_inv))
        scaler = ht.preprocessing.RobustScaler()
        with self.assertRaises(TypeError):
            scaler.fit(["abc"])
        with self.assertRaises(ValueError):
            scaler.fit(ht.zeros((10, 10, 10), dtype=ht.float32))
        with self.assertRaises(TypeError):
            scaler.fit(ht.zeros(10, 10, dtype=ht.int32))
