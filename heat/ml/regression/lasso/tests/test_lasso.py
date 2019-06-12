import os
import unittest

import heat as ht


class TestLasso(unittest.TestCase):
    def test_lasso(self):
        #ToDo: add additional tests
        # get some test data
        X = ht.load_hdf5(os.path.join(os.getcwd(), 'heat/datasets/data/diabetes.h5'), dataset='x', split=0)
        y = ht.load_hdf5(os.path.join(os.getcwd(), 'heat/datasets/data/diabetes.h5'), dataset='y', split=0)

        # normalize dataset
        X = X / ht.sqrt((ht.mean(X**2, axis = 0))) 
        _, n = X.shape
        # HeAT lasso instance 
        estimator = ht.ml.regression.lasso.HeatLasso(max_iter=100)


        estimator.fit(X,y)

        # check whether the results are correct
        self.assertIsInstance(estimator.theta, ht.DNDarray)
        self.assertEqual(estimator.theta.shape, (n, 1))
