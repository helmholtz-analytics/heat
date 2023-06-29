import unittest
import heat as ht
import numpy as np
from mpi4py import MPI
import os

from ...core.tests.test_suites.basic_test import TestCase

atol_fit = 1e-6
atol_inv = 1e-4


class TestStandardScaler(TestCase):
    def test_StandardScaler(self):
        for split in [0, 1]:
            for copy in [True, False]:
                X = ht.random.rand(
                    MPI.COMM_WORLD.Get_size() * 10, MPI.COMM_WORLD.Get_size() * 3, split=split
                )
                scaler = ht.preprocessing.StandardScaler(copy=copy)
                scaler.fit(X)
                Y = scaler.transform(X)
                self.assertTrue(ht.allclose(Y.mean(axis=0), ht.zeros(Y.shape[1]), atol=atol_fit))
                self.assertTrue(ht.allclose(Y.var(axis=0), ht.ones(Y.shape[1]), atol=atol_fit))
                Y = scaler.inverse_transform(Y)
                self.assertTrue(ht.allclose(Y, X, atol=atol_inv))
                Z = ht.zeros(
                    (MPI.COMM_WORLD.Get_size() * 10, MPI.COMM_WORLD.Get_size() * 2), split=split
                )
                with self.assertRaises(ValueError):
                    scaler.transform(Z)
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
            for copy in [True, False]:
                X = ht.random.randn(
                    MPI.COMM_WORLD.Get_size() * 10, MPI.COMM_WORLD.Get_size() * 3, split=split
                )
                scaler = ht.preprocessing.MinMaxScaler(copy=copy)
                scaler.fit(X)
                Y = scaler.transform(X)
                self.assertTrue(ht.allclose(Y.min(axis=0), ht.zeros(Y.shape[1]), atol=atol_fit))
                self.assertTrue(ht.allclose(Y.max(axis=0), ht.ones(Y.shape[1]), atol=atol_fit))
                Y = scaler.inverse_transform(Y)
                self.assertTrue(ht.allclose(Y, X, atol=atol_inv))
                with self.assertRaises(ValueError):
                    scaler = ht.preprocessing.MinMaxScaler(feature_range=(0.5, 0.5), copy=copy)
                Z = ht.zeros(
                    (MPI.COMM_WORLD.Get_size() * 10, MPI.COMM_WORLD.Get_size() * 2), split=split
                )
                with self.assertRaises(ValueError):
                    scaler.transform(Z)
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
                        MPI.COMM_WORLD.Get_size() * 10, MPI.COMM_WORLD.Get_size() * 3, split=split
                    )
                    scaler = ht.preprocessing.Normalizer(norm=norms[k], copy=copy)
                    scaler.fit(X)
                    X = scaler.transform(X)
                    self.assertTrue(
                        ht.allclose(
                            ht.norm(X, axis=1, ord=ords[k]), ht.ones(X.shape[0]), atol=atol_fit
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
            for copy in [True, False]:
                X = ht.random.randn(
                    MPI.COMM_WORLD.Get_size() * 10, MPI.COMM_WORLD.Get_size() * 3, split=split
                )
                scaler = ht.preprocessing.MaxAbsScaler(copy=copy)
                scaler.fit(X)
                Y = scaler.transform(X)
                self.assertTrue(
                    ht.allclose(ht.max(ht.abs(Y), axis=0), ht.ones(Y.shape[1]), atol=atol_fit)
                )
                Y = scaler.inverse_transform(Y)
                self.assertTrue(ht.allclose(Y, X, atol=atol_inv))
                Z = ht.zeros(
                    (MPI.COMM_WORLD.Get_size() * 10, MPI.COMM_WORLD.Get_size() * 2), split=split
                )
                with self.assertRaises(ValueError):
                    scaler.transform(Z)
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
            for copy in [True, False]:
                for with_centering in [True, False]:
                    for with_scaling in [True, False]:
                        if not with_centering and not with_scaling:
                            with self.assertRaises(ValueError):
                                scaler = ht.preprocessing.RobustScaler(
                                    quantile_range=(24.0, 76.0),
                                    copy=copy,
                                    with_centering=with_centering,
                                    with_scaling=with_scaling,
                                )
                        else:
                            X = ht.random.randn(
                                MPI.COMM_WORLD.Get_size() * 10,
                                MPI.COMM_WORLD.Get_size() * 3,
                                split=split,
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
                                        ht.percentile(Y, 76.0, axis=0)
                                        - ht.percentile(Y, 24.0, axis=0),
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
                            Z = ht.zeros(
                                (MPI.COMM_WORLD.Get_size() * 10, MPI.COMM_WORLD.Get_size() * 2),
                                split=split,
                            )
                            with self.assertRaises(ValueError):
                                scaler.transform(Z)
        scaler = ht.preprocessing.RobustScaler()
        with self.assertRaises(TypeError):
            scaler.fit(["abc"])
        with self.assertRaises(ValueError):
            scaler.fit(ht.zeros((10, 10, 10), dtype=ht.float32))
        with self.assertRaises(TypeError):
            scaler.fit(ht.zeros(10, 10, dtype=ht.int32))
