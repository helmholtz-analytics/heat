import os
import unittest
import torch
import numpy as np

import heat as ht
from heat.core.tests.test_suites.basic_test import BasicTest

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


class TestGaussianNB(BasicTest):
    def test_fit_iris(self):
        # load sklearn train/test sets and resulting probabilities
        X_train = ht.load(
            "heat/datasets/data/iris_X_train.csv", sep=";", dtype=ht.float32, device=ht_device
        )
        X_test = ht.load(
            "heat/datasets/data/iris_X_test.csv", sep=";", dtype=ht.float32, device=ht_device
        )
        y_train = ht.load(
            "heat/datasets/data/iris_y_train.csv", sep=";", dtype=ht.int32, device=ht_device
        ).squeeze()
        y_test = ht.load(
            "heat/datasets/data/iris_y_test.csv", sep=";", dtype=ht.int32, device=ht_device
        ).squeeze()
        y_pred_proba_sklearn = ht.load(
            "heat/datasets/data/iris_y_pred_proba.csv", sep=";", dtype=ht.float32, device=ht_device
        )

        # test ht.GaussianNB
        from heat.naive_bayes import GaussianNB

        gnb_heat = GaussianNB()

        # test GaussianNB locally
        local_fit = gnb_heat.fit(X_train, y_train)
        y_pred_local = local_fit.predict(X_test)
        y_pred_proba_local = local_fit.predict_proba(X_test)
        self.assertIsInstance(y_pred_local, ht.DNDarray)
        self.assertEqual((y_pred_local != y_test).sum(), ht.array(4))
        self.assertTrue(ht.isclose(y_pred_proba_sklearn, y_pred_proba_local, atol=1e-1).all())

        # test GaussianNB when sample_weight is not None, sample_weight not distributed
        sample_weight = ht.ones((y_train.gshape[0]), dtype=ht.float32, split=None)
        local_fit_weight = gnb_heat.fit(X_train, y_train, sample_weight=sample_weight)
        y_pred_local_weight = local_fit_weight.predict(X_test)
        y_pred_proba_local_weight = local_fit_weight.predict_proba(X_test)
        self.assertIsInstance(y_pred_local_weight, ht.DNDarray)
        self.assert_array_equal(y_pred_local_weight, y_pred_local.numpy())
        self.assertTrue(ht.isclose(y_pred_proba_sklearn, y_pred_proba_local_weight).all())

        # test GaussianNB, data and labels distributed along split axis 0
        size = ht.MPI_WORLD.size
        if size in range(7):
            X_train_split = ht.resplit(X_train, axis=0)
            X_test_split = ht.resplit(X_test, axis=0)
            y_train_split = ht.resplit(y_train, axis=0)
            y_test_split = ht.resplit(y_test, axis=0)
            y_pred_split = gnb_heat.fit(X_train_split, y_train_split).predict(X_test_split)
            self.assert_array_equal(y_pred_split, y_pred_local.numpy())
            self.assertEqual((y_pred_split != y_test_split).sum(), ht.array(4))
            sample_weight_split = ht.ones(y_train_split.gshape[0], dtype=ht.float32, split=0)
            y_pred_split_weight = gnb_heat.fit(
                X_train_split, y_train_split, sample_weight=sample_weight_split
            ).predict(X_test_split)
            self.assertIsInstance(y_pred_split_weight, ht.DNDarray)
            self.assert_array_equal(y_pred_split_weight, y_pred_split.numpy())

        # test GaussianNB internal functions

        # test exceptions
        X_torch = torch.ones(75, 4)
        y_np = np.zeros(75)
        y_2D = ht.ones((75, 1), split=None, device=ht_device)
        weights_torch = torch.zeros(75)
        X_3D = ht.ones((75, 4, 4), split=None, device=ht_device)
        y_wrong_size = ht.zeros(76, device=ht_device)
        X_train_split = ht.resplit(X_train, axis=0)
        y_train_split = ht.resplit(y_train, axis=0)
        weights_2D_split = y_2D = ht.ones((75, 1), split=0, device=ht_device)
        weights_wrong_size = ht.ones(76, device=ht_device)

        with self.assertRaises(ValueError):
            gnb_heat.fit(X_torch, y_train)
        with self.assertRaises(ValueError):
            gnb_heat.fit(X_train, y_np)
        with self.assertRaises(ValueError):
            gnb_heat.fit(X_train, y_2D)
        with self.assertRaises(ValueError):
            gnb_heat.fit(X_train, y_train, sample_weight=weights_torch)
        with self.assertRaises(ValueError):
            gnb_heat.fit(X_3D, y_train)
        with self.assertRaises(ValueError):
            gnb_heat.fit(X_train, y_wrong_size)
        with self.assertRaises(ValueError):
            gnb_heat.fit(X_train_split, y_train_split, sample_weight=weights_2D_split)
        with self.assertRaises(ValueError):
            gnb_heat.fit(X_train, y_train, sample_weight=weights_wrong_size)
