import os
import unittest
import numpy as np
import torch
import heat as ht

from heat.core.tests.test_suites.basic_test import BasicTest


class TestLasso(BasicTest):
    @classmethod
    def setUpClass(cls):
        super(TestLasso, cls).setUpClass()

    def test_regressor(self):
        lasso = ht.regression.Lasso()
        self.assertTrue(ht.is_estimator(lasso))
        self.assertTrue(ht.is_regressor(lasso))

    def test_get_and_set_params(self):
        lasso = ht.regression.Lasso()
        params = lasso.get_params()

        self.assertEqual(params, {"lam": 0.1, "max_iter": 100, "tol": 1e-6})

        params["max_iter"] = 200
        lasso.set_params(**params)
        self.assertEqual(200, lasso.max_iter)

    def test_exceptions(self):
        with self.assertRaises(ValueError):
            ht.regression.Lasso().set_params(foo="bar")

    if ht.io.supports_hdf5():

        def test_lasso(self):
            # ToDo: add additional tests
            # get some test data
            X = ht.load_hdf5(
                os.path.join(os.getcwd(), "heat/datasets/data/diabetes.h5"),
                dataset="x",
                device=self.ht_device,
                split=0,
            )
            y = ht.load_hdf5(
                os.path.join(os.getcwd(), "heat/datasets/data/diabetes.h5"),
                dataset="y",
                device=self.ht_device,
                split=0,
            )

            # normalize dataset
            X = X / ht.sqrt((ht.mean(X ** 2, axis=0)))
            m, n = X.shape
            # HeAT lasso instance
            estimator = ht.regression.lasso.Lasso(max_iter=100, tol=None)
            # check whether the results are correct
            self.assertEqual(estimator.lam, 0.1)
            self.assertTrue(estimator.theta is None)
            self.assertTrue(estimator.n_iter is None)
            self.assertEqual(estimator.max_iter, 100)
            self.assertEqual(estimator.coef_, None)
            self.assertEqual(estimator.intercept_, None)

            estimator.fit(X, y)

            # check whether the results are correct
            self.assertEqual(estimator.lam, 0.1)
            self.assertIsInstance(estimator.theta, ht.DNDarray)
            self.assertEqual(estimator.n_iter, 100)
            self.assertEqual(estimator.max_iter, 100)
            self.assertEqual(estimator.coef_.shape, (n - 1, 1))
            self.assertEqual(estimator.intercept_.shape, (1,))

            yest = estimator.predict(X)

            # check whether the results are correct
            self.assertIsInstance(yest, ht.DNDarray)
            self.assertEqual(yest.shape, (m, 1))

            with self.assertRaises(ValueError):
                estimator.fit(X, ht.zeros((3, 3, 3)))
            with self.assertRaises(ValueError):
                estimator.fit(ht.zeros((3, 3, 3)), ht.zeros((3, 3)))
