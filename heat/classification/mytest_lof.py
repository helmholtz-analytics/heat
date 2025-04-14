"""Tests during the implementation of the Local Outlier Factor (LOF) algorithm"""

import heat as ht
import torch
from heat.spatial import distance
from localoutlierfactor import LocalOutlierFactor
from heat.core import types
from mpi4py import MPI

# from heat.classification import localoutlierfactor
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerPathCollection

print("Start")
ht.use_device("gpu")

# X=ht.array([[1,2],[2,3],[3,4],[100,200],[2,2],[2,6],[0,1],[3,6],[7,8],[3,2],[1,1]],split=0)
# lof = LocalOutlierFactor(n_neighbors=3, fully_distributed=False)
# lof.fit(X)

# # Get LOF scores and anomaly labels
# lof_scores = lof.lof_scores.numpy()
# print(f"lof_scores={lof_scores}")


# Generate random data with outliers
""" np.random.seed(42)

X_inliers = ht.random.randn(100, 2, split=0)
X_inliers = ht.concatenate((X_inliers + 2, X_inliers - 2), axis=0)
X_outliers = ht.array(
    [[10, 10], [4, 7], [8, 3], [-2, 6], [5, -9], [-1, -10], [7, -2], [-6, 4], [-5, -8]], split=0
)
X = ht.concatenate((X_inliers, X_outliers), axis=0)


X = X.numpy()

n_outliers = len(X_outliers)

# Convert data to Heat tensor
ht_X = ht.array(X, split=0)


# Compute the LOF scores
lof = LocalOutlierFactor(n_neighbors=10, threshold=3)
# lof = LocalOutlierFactor(n_neighbors=10, binary_decision="top_n", top_n=n_outliers)
lof.fit(ht_X)

# Get LOF scores and anomaly labels
lof_scores = lof.lof_scores.numpy()
# print(f"lof_scores={lof_scores}")
anomaly = lof.anomaly.numpy()

if anomaly[X_outliers.shape[0] :].all() == 1:
    print("\n\n The anomaly matrix is correct\n\n ")
# print(f"anomaly={anomaly}")

# Plot data points with LOF scores
plt.rc("text", usetex=True)
plt.rc("font", family="serif")
plt.figure(figsize=(10, 6))
scatter = plt.scatter(
    X[:, 0], X[:, 1], c=lof_scores, cmap="coolwarm", edgecolors="k", s=60, alpha=0.8
)

# Highlight outliers with a larger marker
outlier_indices = np.where(anomaly == 1)[0]
# print(f"outlier_indices={outlier_indices}")

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
if ht.MPI_WORLD.rank==0:
    plt.show() """

""" idx=ht.array([2,0,1],split=0)
dist=ht.array([[1,1],[2,2],[3,3]],split=0)
dist=dist[idx]
print(f"dist={dist},\n dist[2]={dist[2].larray}") """


# size = comm.Get_size()
# _, displ, _ = comm.counts_displs_shape(idx.shape, idx.split)
# mapped_idx = ht.zeros_like(idx)
# for rank in range(size):
#     lower_bound = displ[rank]
#     if rank == size - 1:  # size-1 is the last rank
#         upper_bound = idx.shape[0]
#     else:
#         upper_bound = displ[rank + 1]
#     mask = (idx >= lower_bound) & (idx < upper_bound)
#     print(f"rank={rank}, mask.larray.shape={mask.larray.shape}")
#     mapped_idx[mask] = rank
# return mapped_idx


# idx = ht.array([2, 0, 1], split=0)
# mask = ht.array([True, False, True], split=0)
# mapped_idx = ht.zeros_like(idx)

# mapped_idx[mask] = 42
# print(idx, mask, mapped_idx)


# vec = ht.array([0,10,20,30,40,50], split=0)
# mat = ht.array([[1, 2], [2, 3], [3, 4]], split=0)

# test=vec[mat]
# print(f"test={test}")


vec = ht.array([0, 10, 20, 30, 40, 50], split=0)
mat = ht.array([[1, 2], [2, 3], [3, 4]], split=0)

test = ht.zeros_like(mat)
for i in range(mat.shape[0]):
    test[i] = vec[mat[i]]
print(f"test={test}")
