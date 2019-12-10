import numpy as np
import heat as ht
import heat.regression.lasso as lasso


# load diabetes dataset from hdf5 file
X = ht.load_hdf5("./heat/datasets/data/diabetes.h5", dataset="x", split=0)
y = ht.load_hdf5("./heat/datasets/data/diabetes.h5", dataset="y", split=0)

# normalize dataset
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
