import numpy as np
import torch
import sys

sys.path.append("../../")

import heat as ht
from matplotlib import pyplot as plt
from sklearn import datasets
import heat.regression.lasso as lasso
import plotfkt

# read scikit diabetes data set
diabetes = datasets.load_diabetes()

# load diabetes dataset from hdf5 file
X = ht.load_hdf5("../../heat/datasets/data/diabetes.h5", dataset="x", split=0)
y = ht.load_hdf5("../../heat/datasets/data/diabetes.h5", dataset="y", split=0)

# normalize dataset #DoTO this goes into the lasso fit routine soon as issue #106 is solved
X = X / ht.sqrt((ht.mean(X ** 2, axis=0)))

# HeAT lasso instance
estimator = lasso.Lasso(max_iter=100)

# List  lasso model parameters
theta_list = list()

# Range of lambda values
lamda = np.logspace(0, 4, 10) / 10

# compute the lasso path
for la in lamda:
    estimator.lam = la
    estimator.fit(X, y)
    theta_list.append(estimator.theta.numpy().flatten())

# Stack estimated model parameters into one numpy array
theta_lasso = np.stack(theta_list).T

# Stack into numpy array
theta_lasso = np.stack(theta_list).T[1:, :]

# plot lasso paths
plotfkt.plot_lasso_path(
    lamda, theta_lasso, diabetes.feature_names, title="Lasso Paths - HeAT implementation"
)

if X.is_distributed():
    distributed = X.comm.rank
else:
    distributed = False

# plot only with first rank
if distributed is False:
    plt.show()
elif distributed == 0:
    plt.show()
