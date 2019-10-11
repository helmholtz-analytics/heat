import numpy as np
import torch
import sys

sys.path.append("../../")

import heat as ht
from matplotlib import pyplot as plt
from sklearn import datasets
import heat.ml.regression.lasso as lasso
import plotfkt

# read scikit diabetes data set
diabetes = datasets.load_diabetes()

# load diabetes dataset from hdf5 file
X = ht.load_hdf5("../../heat/datasets/data/diabetes.h5", dataset="x", split=0)
y = ht.load_hdf5("../../heat/datasets/data/diabetes.h5", dataset="y", split=0)

# normalize dataset #DoTO this goes into the lasso fit routine soon as issue #106 is solved
X = X / ht.sqrt((ht.mean(X ** 2, axis=0)))

# HeAT lasso instance
estimator = lasso.HeatLasso(max_iter=100)

# List  lasso model parameters
theta_list = list()

# Range of lambda values
lamda = np.logspace(0, 4, 10) / 10

# compute the lasso path
for l in lamda:
    estimator.lam = l
    estimator.fit(X, y)
    theta_list.append(estimator.theta.numpy().flatten())

# Stack estimated model parameters into one numpy array
theta_lasso = np.stack(theta_list).T

# Stack into numpy array
theta_lasso = np.stack(theta_list).T[1:, :]


# plot lasso paths
plt.subplot(3, 1, 1)
plotfkt.plot_lasso_path(
    lamda, theta_lasso, diabetes.feature_names, title="Lasso Paths - HeAT implementation"
)

if X.is_distributed():
    distributed = X.comm.rank
else:
    distributed = False

# Now the same stuff in numpy
X = diabetes.data.astype("float32")
y = diabetes.target.astype("float32")

m, _ = X.shape
X = np.concatenate((np.ones((m, 1)).astype("float32"), X), axis=1)

# normalize dataset
X = X / np.sqrt((np.mean(X ** 2, axis=0)))

# Numpy lasso instance
estimator = lasso.NumpyLasso(max_iter=100)

# List  lasso model parameters
theta_list = list()

# Range of lambda values
lamda = np.logspace(0, 4, 10) / 10

# compute the lasso path
for l in lamda:
    estimator.lam = l
    estimator.fit(X, y)
    theta_list.append(estimator.theta.flatten())

# Stack estimated model parameters into one numpy array
theta_lasso = np.stack(theta_list).T

# Stack into numpy array
theta_lasso = np.stack(theta_list).T[1:, :]

# plot lasso paths
plt.subplot(3, 1, 2)
plotfkt.plot_lasso_path(
    lamda, theta_lasso, diabetes.feature_names, title="Lasso Paths - Numpy implementation"
)

# Now the same stuff again in PyTorch
X = torch.tensor(X)
y = torch.tensor(y)

# HeAT lasso instance
estimator = lasso.PytorchLasso(max_iter=100)

# List lasso model parameters
theta_list = list()

# Range of lambda values
lamda = np.logspace(0, 4, 10) / 10

# compute the lasso path
for l in lamda:
    estimator.lam = l
    estimator.fit(X, y)
    theta_list.append(estimator.theta.numpy().flatten())

# Stack estimated model parameters into one numpy array
theta_lasso = np.stack(theta_list).T

# Stack into numpy array
theta_lasso = np.stack(theta_list).T[1:, :]

# plot lasso paths
plt.subplot(3, 1, 3)
plotfkt.plot_lasso_path(
    lamda, theta_lasso, diabetes.feature_names, title="Lasso Paths - PyTorch implementation"
)

# plot only with first rank
if distributed is False:
    plt.show()
elif distributed == 0:
    plt.show()
