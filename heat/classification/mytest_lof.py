"""Tests during the implementation of the Local Outlier Factor (LOF) algorithm"""

import heat as ht
import torch
from heat.spatial import distance
from localoutlierfactor import LocalOutlierFactor
from heat.core import types

# from heat.classification import localoutlierfactor
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerPathCollection

ht.use_device("gpu")

"""
list=ht.array([0,10,20,30])
idx=ht.array([[0,1,2],[1,2,3],[3,2,1],[2,0,3]])

list_test = list[idx]
print(f"\n\n list={list} \n\n idx={idx} \n\n lrd_neighbors={list_test} \n\n ") """


""" list = ht.array([0, 10, 20, 30])
idx = ht.array([[0, 1, 2], [1, 2, 3], [3, 2, 1], [2, 0, 3]])

list_test=ht.zeros((list.shape[0], idx.shape[1]),split=None)
for i in range(list.shape[0]):
    list_test[i,:] = list[idx[i,:]]

# Ergebnis ausgeben
print(f"\n\n list={list} \n\n idx={idx} \n\n lrd_neighbors={list_test} \n\n ") """


# Generate random data with outliers
np.random.seed(42)

X_inliers = 0.3 * np.random.randn(100, 2)
X_inliers = np.r_[X_inliers + 2, X_inliers - 2]
X_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))
# X_inliers = np.array([[1,1.2],[0,1.7],[1.3,0],[2.4,1],[1.8,2.5],[2.1,2.9],[2.3,0],[0,2.2],[3.6,1.7],[3.4,2.5],[3.2,0],[0,3.8],[1.1,3.2],[2.5,3.7],[1.5,1.1],[2.5,1.3],[3.5,1.4],[1.5,2.6],[1.5,3.3]])
# X_outliers = np.array([[10,0],[0,10],[10,10],[-10,-10]])
X = np.r_[X_inliers, X_outliers]

n_outliers = len(X_outliers)
ground_truth = np.ones(len(X), dtype=int)
ground_truth[-n_outliers:] = -1

# Convert data to HeAT tensor
ht_X = ht.array(X, split=0)

# Compute the LOF scores
lof = LocalOutlierFactor(n_neighbors=10, threshold=2)
lof.fit(ht_X)

# Get LOF scores and anomaly labels
lof_scores = lof.lof_scores.numpy()
# print(f"lof_scores={lof_scores}")
anomaly = lof.anomaly.numpy()

# Plot data points with LOF scores
plt.figure(figsize=(10, 6))
scatter = plt.scatter(
    X[:, 0], X[:, 1], c=lof_scores, cmap="coolwarm", edgecolors="k", s=60, alpha=0.8
)

# Highlight outliers with a larger marker
outlier_indices = np.where(anomaly == 1)[0]
plt.scatter(
    X[outlier_indices, 0],
    X[outlier_indices, 1],
    facecolors="none",
    edgecolors="black",
    s=120,
    linewidths=2,
    label="Outliers",
)

# Add colorbar to indicate LOF score intensity
plt.colorbar(scatter, label="LOF Score")

# Labels and title
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Local Outlier Factor (LOF) - Anomaly Detection")
plt.legend()
if ht.MPI_WORLD.rank == 0:
    plt.show()
