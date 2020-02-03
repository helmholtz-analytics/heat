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
        #benchmark result with scikit-learn
        from sklearn.datasets import load_iris
        from sklearn.model_selection import train_test_split
        from sklearn.naive_bayes import GaussianNB
        X, y = load_iris(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
        gnb = GaussianNB()
        y_pred = gnb.fit(X_train, y_train).predict(X_test)

        #test ht.GaussianNB locally
        from heat.core.naive_bayes import GaussianNB
        X_train = ht.array(X_train)   
        X_test = ht.array(X_test) 
        y_train = ht.array(y_train)
        y_test = ht.array(y_test)
        gnb_heat = GaussianNB()
        y_pred_heat_local = gnb_heat.fit(X_train, y_train).predict(X_test)
        self.assertIsInstance(y_pred_heat_local, ht.DNDarray)
        self.assert_array_equal(y_pred_heat_local, y_pred)

        #test ht.GaussianNB, both data and labels distributed along same split dimension
        from heat.core.naive_bayes import GaussianNB
        X_train = ht.array(X_train, split=0)   
        X_test = ht.array(X_test, split=0) 
        y_train = ht.array(y_train, split=0)
        y_test = ht.array(y_test, split=0)
        gnb_heat = GaussianNB()
        y_pred_heat_split = gnb_heat.fit(X_train, y_train).predict(X_test)
        self.assert_array_equal(y_pred_heat_split, y_pred)
