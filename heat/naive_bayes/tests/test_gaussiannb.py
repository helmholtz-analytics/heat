import os
import unittest

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
        # benchmark result with scikit-learn
        from sklearn.datasets import load_iris
        from sklearn.model_selection import train_test_split
        from sklearn.naive_bayes import GaussianNB

        X, y = load_iris(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
        gnb = GaussianNB()
        y_pred = gnb.fit(X_train, y_train).predict(X_test)

        # test ht.GaussianNB
        from heat.core.naive_bayes import GaussianNB

        gnb_heat = GaussianNB()

        # test ht.GaussianNB locally
        X_train_local = ht.array(X_train)
        X_test_local = ht.array(X_test)
        y_train_local = ht.array(y_train)
        y_test_local = ht.array(y_test)
        y_pred_local = gnb_heat.fit(X_train_local, y_train_local).predict(X_test_local)
        self.assertIsInstance(y_pred_local, ht.DNDarray)
        self.assert_array_equal(y_pred_local, y_pred)
        self.assertEqual((y_pred_local != y_test_local).sum(), ht.array(4))

        # #test ht.GaussianNB, data and labels distributed along split axis 0
        X_train_split = ht.array(X_train, split=0)
        X_test_split = ht.array(X_test, split=0)
        y_train_split = ht.array(y_train, split=0)
        y_test_split = ht.array(y_test, split=0)
        y_pred_split = gnb_heat.fit(X_train_split, y_train_split).predict(X_test_split)
        self.assert_array_equal(y_pred_split, y_pred)
        self.assertEqual((y_pred_split != y_test_split).sum(), ht.array(4))
