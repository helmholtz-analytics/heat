import os
import numpy as np
import torch

import heat as ht
from heat.core.tests.test_suites.basic_test import TestCase


class TestGaussianNB(TestCase):
    def test_classifier(self):
        gnb = ht.naive_bayes.GaussianNB()
        self.assertTrue(ht.is_estimator(gnb))
        self.assertTrue(ht.is_classifier(gnb))

    def test_get_and_set_params(self):
        gnb = ht.naive_bayes.GaussianNB()
        params = gnb.get_params()

        self.assertEqual(params, {"priors": None, "var_smoothing": 1e-9})

        params["var_smoothing"] = 1e-10
        gnb.set_params(**params)
        self.assertEqual(1e-10, gnb.var_smoothing)

    def test_fit_iris(self):
        # load sklearn train/test sets and resulting probabilities
        X_train = ht.load("heat/utils/data/datasets/iris_X_train.csv", sep=";", dtype=ht.float64)
        X_test = ht.load("heat/utils/data/datasets/iris_X_test.csv", sep=";", dtype=ht.float64)
        y_train = ht.load(
            "heat/utils/data/datasets/iris_y_train.csv", sep=";", dtype=ht.int64
        ).squeeze()
        y_test = ht.load(
            "heat/utils/data/datasets/iris_y_test.csv", sep=";", dtype=ht.int64
        ).squeeze()
        y_pred_proba_sklearn = ht.load(
            "heat/utils/data/datasets/iris_y_pred_proba.csv", sep=";", dtype=ht.float64
        )

        # test ht.GaussianNB
        from heat.naive_bayes import GaussianNB

        gnb_heat = GaussianNB()
        self.assertEqual(gnb_heat.priors, None)
        with self.assertRaises(AttributeError):
            gnb_heat.classes_
        with self.assertRaises(AttributeError):
            gnb_heat.class_prior_
        with self.assertRaises(AttributeError):
            gnb_heat.epsilon_

        # test GaussianNB locally, no weights
        local_fit = gnb_heat.fit(X_train, y_train)
        self.assert_array_equal(gnb_heat.classes_, np.array([0, 1, 2]))
        local_fit_no_classes = gnb_heat.partial_fit(X_train, y_train, classes=None)
        y_pred_local = local_fit_no_classes.predict(X_test)
        y_pred_proba_local = local_fit.predict_proba(X_test)
        sklearn_class_prior = np.array([0.38666667, 0.26666667, 0.34666667])
        sklearn_epsilon = np.array([3.6399040000000003e-09])
        sklearn_theta = ht.array(
            [
                [4.97586207, 3.35862069, 1.44827586, 0.23448276],
                [5.935, 2.71, 4.185, 1.3],
                [6.77692308, 3.09230769, 5.73461538, 2.10769231],
            ],
            dtype=X_train.dtype,
        )
        sklearn_sigma = ht.array(
            [
                [0.10321047, 0.13208086, 0.01629013, 0.00846612],
                [0.256275, 0.0829, 0.255275, 0.046],
                [0.38869823, 0.10147929, 0.31303255, 0.04763314],
            ],
            dtype=X_train.dtype,
        )
        self.assertIsInstance(y_pred_local, ht.DNDarray)
        self.assertEqual((y_pred_local != y_test).sum(), ht.array(4))
        self.assert_array_equal(gnb_heat.class_prior_, sklearn_class_prior)
        self.assert_array_equal(gnb_heat.epsilon_, sklearn_epsilon)
        self.assertTrue(ht.isclose(gnb_heat.theta_, sklearn_theta).all())
        self.assertTrue(ht.isclose(gnb_heat.sigma_, sklearn_sigma, atol=1e-1).all())
        self.assertTrue(ht.isclose(y_pred_proba_sklearn, y_pred_proba_local, atol=1e-1).all())

        # test GaussianNB when sample_weight is not None, sample_weight not distributed
        sample_weight = ht.ones((y_train.gshape[0]), dtype=ht.float32, split=None)
        local_fit_weight = gnb_heat.fit(X_train, y_train, sample_weight=sample_weight)
        y_pred_local_weight = local_fit_weight.predict(X_test)
        y_pred_proba_local_weight = local_fit_weight.predict_proba(X_test)
        self.assertIsInstance(y_pred_local_weight, ht.DNDarray)
        self.assert_array_equal(gnb_heat.class_prior_, sklearn_class_prior)
        self.assert_array_equal(gnb_heat.epsilon_, sklearn_epsilon)
        self.assertTrue(ht.isclose(gnb_heat.theta_, sklearn_theta).all())
        self.assertTrue(ht.isclose(gnb_heat.sigma_, sklearn_sigma, atol=1e-1).all())
        self.assert_array_equal(y_pred_local_weight, y_pred_local.numpy())
        self.assertTrue(ht.isclose(y_pred_proba_sklearn, y_pred_proba_local_weight).all())

        # test GaussianNB, data and labels distributed along split axis 0
        X_train_split = ht.resplit(X_train, axis=0)
        X_test_split = ht.resplit(X_test, axis=0)
        y_train_split = ht.resplit(y_train, axis=0)
        y_test_split = ht.resplit(y_test, axis=0)
        y_pred_split = gnb_heat.fit(X_train_split, y_train_split).predict(X_test_split)
        self.assert_array_equal(gnb_heat.class_prior_, sklearn_class_prior)
        self.assert_array_equal(gnb_heat.epsilon_, sklearn_epsilon)
        self.assertTrue(ht.isclose(gnb_heat.theta_, sklearn_theta).all())
        self.assertTrue(ht.isclose(gnb_heat.sigma_, sklearn_sigma, atol=1e-1).all())
        self.assert_array_equal(y_pred_split, y_pred_local.numpy())
        self.assertEqual((y_pred_split != y_test_split).sum(), ht.array(4))
        sample_weight_split = ht.ones(y_train_split.gshape[0], dtype=ht.float32, split=0)
        y_pred_split_weight = gnb_heat.fit(
            X_train_split, y_train_split, sample_weight=sample_weight_split
        ).predict(X_test_split)
        self.assertIsInstance(y_pred_split_weight, ht.DNDarray)
        self.assert_array_equal(y_pred_split_weight, y_pred_split.numpy())

        # test exceptions
        X_torch = torch.ones(75, 4)
        y_np = np.zeros(75)
        y_2D = ht.ones((75, 1), split=None)
        weights_torch = torch.zeros(75)
        X_3D = ht.ones((75, 4, 4), split=None)
        X_wrong_size = ht.ones((75, 5), split=None)
        y_wrong_size = ht.zeros(76)
        X_train_split = ht.resplit(X_train, axis=0)
        y_train_split = ht.resplit(y_train, axis=0)
        weights_2D_split = y_2D = ht.ones((75, 1), split=0)
        weights_wrong_size = ht.ones(76)
        priors_wrong_shape = ht.random.randn(4)
        priors_wrong_sum = ht.random.randn(3, dtype=ht.float32)
        priors_wrong_sign = ht.array([-0.3, 0.7, 0.6])
        wrong_classes = ht.array([3, 4, 5])

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
            gnb_heat.fit(X_train, y_train)
            gnb_heat.predict(X_torch)
        with self.assertRaises(ValueError):
            gnb_heat.fit(X_train, y_train)
            gnb_heat.partial_fit(X_wrong_size, y_train)
        with self.assertRaises(ValueError):
            gnb_heat.fit(X_train, y_train)
            gnb_heat.partial_fit(X_train, y_train, classes=wrong_classes)
        with self.assertRaises(ValueError):
            gnb_heat.classes_ = None
            gnb_heat.partial_fit(X_train, y_train, classes=None)
        with self.assertRaises(ValueError):
            gnb_heat.fit(X_train_split, y_train_split, sample_weight=weights_2D_split)
        with self.assertRaises(ValueError):
            gnb_heat.fit(X_train, y_train, sample_weight=weights_wrong_size)
        with self.assertRaises(ValueError):
            gnb_heat.priors = priors_wrong_shape
            gnb_heat.fit(X_train, y_train)
        with self.assertRaises(ValueError):
            gnb_heat.priors = priors_wrong_sum
            gnb_heat.fit(X_train, y_train)
        with self.assertRaises(ValueError):
            gnb_heat.priors = priors_wrong_sign
            gnb_heat.fit(X_train, y_train)

    def test_exception(self):
        with self.assertRaises(ValueError):
            ht.naive_bayes.GaussianNB().set_params(foo="bar")
