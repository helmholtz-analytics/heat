# Cluster Analysis
# ================
#
# This tutorial is an interactive version of our static [clustering tutorial on ReadTheDocs](https://heat.readthedocs.io/en/stable/tutorial_clustering.html).
#
# We will demonstrate memory-distributed analysis with k-means and k-medians from the ``heat.cluster`` module. As usual, we will run the analysis on a small dataset for demonstration. We need to have an `ipcluster` running to distribute the computation.
#
# We will use matplotlib for visualization of data and results.


import heat as ht

# The Iris Dataset
# ------------------------------
# The _iris_ dataset is a well known example for clustering analysis. It contains 4 measured features for samples from
# three different types of iris flowers. A subset of 150 samples is included in formats h5, csv and netcdf in the [Heat repository under 'heat/heat/datasets'](https://github.com/helmholtz-analytics/heat/tree/main/heat/datasets), and can be loaded in a distributed manner with Heat's parallel dataloader.
#
# **NOTE: you might have to change the path to the dataset in the following cell.**

iris = ht.load("~/heat/tutorials/hpc/02_loading_preprocessing/iris.csv", sep=";", split=0)


# Feel free to try out the other [loading options](https://heat.readthedocs.io/en/stable/autoapi/heat/core/io/index.html#heat.core.io.load) as well.
#
# Fitting the dataset with `kmeans`:

k = 3
kmeans = ht.cluster.KMeans(n_clusters=k, init="kmeans++")
kmeans.fit(iris)

# Let's see what the results are. In theory, there are 50 samples of each of the 3 iris types: setosa, versicolor and virginica. We will plot the results in a 3D scatter plot, coloring the samples according to the assigned cluster.

labels = kmeans.predict(iris).squeeze()

# Select points assigned to clusters c1, c2 and c3
c1 = iris[ht.where(labels == 0), :]
c2 = iris[ht.where(labels == 1), :]
c3 = iris[ht.where(labels == 2), :]
# After slicing, the arrays are not distributed equally among the processes anymore; we need to balance
# TODO is balancing really necessary?
c1.balance_()
c2.balance_()
c3.balance_()

print(
    f"Number of points assigned to c1: {c1.shape[0]} \n"
    f"Number of points assigned to c2: {c2.shape[0]} \n"
    f"Number of points assigned to c3: {c3.shape[0]}"
)


# compare Heat results with sklearn
from sklearn.cluster import KMeans
import sklearn.datasets

k = 3
iris_sk = sklearn.datasets.load_iris().data
kmeans_sk = KMeans(n_clusters=k, init="k-means++").fit(iris_sk)
labels_sk = kmeans_sk.predict(iris_sk)

c1_sk = iris_sk[labels_sk == 0, :]
c2_sk = iris_sk[labels_sk == 1, :]
c3_sk = iris_sk[labels_sk == 2, :]
print(
    f"Number of points assigned to c1: {c1_sk.shape[0]} \n"
    f"Number of points assigned to c2: {c2_sk.shape[0]} \n"
    f"Number of points assigned to c3: {c3_sk.shape[0]}"
)
