import os
import unittest
import numpy as np
import torch
import heat as ht


class TestLasso(unittest.TestCase):
    def test_lasso(self):
        #ToDo: add additional tests
        # get some test data
        X = ht.load_hdf5(os.path.join(os.getcwd(), 'heat/datasets/data/diabetes.h5'), dataset='x')
        y = ht.load_hdf5(os.path.join(os.getcwd(), 'heat/datasets/data/diabetes.h5'), dataset='y')

        # normalize dataset
        X = X / ht.sqrt((ht.mean(X**2, axis = 0))) 
        m, n = X.shape
        # HeAT lasso instance 
        estimator = ht.ml.regression.lasso.HeatLasso(max_iter=100, tol=None)
        # check whether the results are correct
        self.assertEqual(estimator.lam, .1)
        self.assertTrue(estimator.theta is None)
        self.assertTrue(estimator.n_iter is None)
        self.assertEqual(estimator.max_iter, 100)
        self.assertEqual(estimator.coef_, None)
        self.assertEqual(estimator.intercept_, None)

        estimator.fit(X,y)

        # check whether the results are correct
        self.assertEqual(estimator.lam, .1)
        self.assertIsInstance(estimator.theta, ht.DNDarray)
        self.assertTrue(estimator.n_iter is 100)
        self.assertEqual(estimator.max_iter, 100)
        self.assertEqual(estimator.coef_.shape, (n-1, 1))
        self.assertEqual(estimator.intercept_.shape, (1,))     

        yest = estimator.predict(X)

        # check whether the results are correct
        self.assertIsInstance(yest, ht.DNDarray)
        self.assertEqual(yest.shape, (m,))


        X = ht.load_hdf5(os.path.join(os.getcwd(), 'heat/datasets/data/diabetes.h5'), dataset='x')
        y = ht.load_hdf5(os.path.join(os.getcwd(), 'heat/datasets/data/diabetes.h5'), dataset='y')

        # Now the same stuff again in PyTorch
        X = torch.tensor(X._DNDarray__array)
        y = torch.tensor(y._DNDarray__array)

        # normalize dataset
        X = X / torch.sqrt((torch.mean(X**2, 0))) 
        m, n = X.shape

        estimator = ht.ml.regression.lasso.PytorchLasso(max_iter=100, tol=None)
        # check whether the results are correct
        self.assertEqual(estimator.lam, .1)
        self.assertTrue(estimator.theta is None)
        self.assertTrue(estimator.n_iter is None)
        self.assertEqual(estimator.max_iter, 100)
        self.assertEqual(estimator.coef_, None)
        self.assertEqual(estimator.intercept_, None)

        estimator.fit(X,y)

        # check whether the results are correct
        self.assertEqual(estimator.lam, .1)
        self.assertIsInstance(estimator.theta, torch.Tensor)
        self.assertTrue(estimator.n_iter is 100)
        self.assertEqual(estimator.max_iter, 100)
        self.assertEqual(estimator.coef_.shape, (n-1, 1))
        self.assertEqual(estimator.intercept_.shape, (1,))     

        yest = estimator.predict(X)

        # check whether the results are correct
        self.assertIsInstance(yest, torch.Tensor)
        self.assertEqual(yest.shape, (m,))

        X = ht.load_hdf5(os.path.join(os.getcwd(), 'heat/datasets/data/diabetes.h5'), dataset='x')
        y = ht.load_hdf5(os.path.join(os.getcwd(), 'heat/datasets/data/diabetes.h5'), dataset='y')

        # Now the same stuff again in PyTorch
        X = X._DNDarray__array.numpy()
        y = y._DNDarray__array.numpy()

        # normalize dataset
        X = X / np.sqrt((np.mean(X**2, axis=0, keepdims=True))) 
        m, n = X.shape

        estimator = ht.ml.regression.lasso.NumpyLasso(max_iter=100, tol=None)
        # check whether the results are correct
        self.assertEqual(estimator.lam, .1)
        self.assertTrue(estimator.theta is None)
        self.assertTrue(estimator.n_iter is None)
        self.assertEqual(estimator.max_iter, 100)
        self.assertEqual(estimator.coef_, None)
        self.assertEqual(estimator.intercept_, None)

        estimator.fit(X,y)

        # check whether the results are correct
        self.assertEqual(estimator.lam, .1)
        self.assertIsInstance(estimator.theta, np.ndarray)
        self.assertTrue(estimator.n_iter is 100)
        self.assertEqual(estimator.max_iter, 100)
        self.assertEqual(estimator.coef_.shape, (n-1, 1))
        self.assertEqual(estimator.intercept_.shape, (1,))     

        yest = estimator.predict(X)

        # check whether the results are correct
        self.assertIsInstance(yest, np.ndarray)
        self.assertEqual(yest.shape, (m,))
    
