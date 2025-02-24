"""Tests during the implementation of the Local Outlier Factor (LOF) algorithm"""

import heat as ht
import torch
from heat.spatial import distance
from localoutlierfactor import LocalOutlierFactor

# from heat.classification import localoutlierfactor
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerPathCollection

# ht.use_device("gpu")

""" # Generate random data with outliers
np.random.seed(42)

X_inliers = 0.3 * np.random.randn(100, 2)
X_inliers = np.r_[X_inliers + 2, X_inliers - 2]
X_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))
X = np.r_[X_inliers, X_outliers]

n_outliers = len(X_outliers)
ground_truth = np.ones(len(X), dtype=int)
ground_truth[-n_outliers:] = -1

# Convert data to HeAT tensor
ht_X = ht.array(X, split=0)

# Compute the LOF scores
lof = LocalOutlierFactor(n_neighbors=20, metric="euclidian", binary_decision="threshold", threshold=1.5, top_n=10)
lof.fit(ht_X)

# Get LOF scores and anomaly labels
lof_scores = lof.lof_scores.numpy()
anomaly = lof.anomaly.numpy()

# Plot data points with LOF scores
plt.figure(figsize=(10, 6))
scatter = plt.scatter(X[:, 0], X[:, 1], c=lof_scores, cmap="coolwarm", edgecolors="k", s=60, alpha=0.8)

# Highlight outliers with a larger marker
outlier_indices = np.where(anomaly == 1)[0]
plt.scatter(X[outlier_indices, 0], X[outlier_indices, 1], facecolors='none', edgecolors='black', s=120, linewidths=2, label="Outliers")

# Add colorbar to indicate LOF score intensity
plt.colorbar(scatter, label="LOF Score")

# Labels and title
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Local Outlier Factor (LOF) - Anomaly Detection")
plt.legend()
plt.show() """


# print(f"process: {ht.MPI_WORLD.rank} \n k_dist.larray={k_dist.larray}, \n  rd.larray={rd.larray}\n")

cdist_small = ht.array([[0, 1, 3], [0, 3, 4], [0, 1, 2]], split=0)

indices = ht.array([[0, 1, 2], [1, 2, 0], [2, 0, 1]], split=0)

k_dist = cdist_small[:, -1]
idx_k_dist = indices[:, -1]
dist = cdist_small

# reachability_dist = ht.maximum(k_dist[:, None], cdist_small[idx_k_dist,:])

idx = ht.array([2, 0, 1], split=0)


""" dist = cdist_small.resplit_(None)
dist = dist[idx_k_dist]
dist = dist.resplit_(0)

# dist = cdist_small[idx_k_dist]




# reachability_dist = ht.zeros((3,3),split=0)

# for i in range(3):
#     for j in range(3):
#         reachability_dist[i,j] = ht.maximum(k_dist[i], cdist_small[idx_k_dist[i],j])


reachability_dist = ht.maximum(
            k_dist[:,None], dist[idx_k_dist,:]
        )

expected_result=indices = ht.array(
    [[3,3,3],
     [4,4,4],
     [2,3,4]],split=0) """


def _map_idx_to_proc(idx, comm):
    size = comm.Get_size()
    _, displ, _ = comm.counts_displs_shape(idx.shape, idx.split)
    mapped_idx = ht.zeros_like(idx)
    for rank in range(size):
        lower_bound = displ[rank]
        if rank == size - 1:  # size-1 is the last rank
            upper_bound = idx.shape[0]
        else:
            upper_bound = displ[rank + 1]
        mask = (idx >= lower_bound) & (idx < upper_bound)
        mapped_idx[mask] = rank
    print(f"process {ht.MPI_WORLD.rank}: \n displ={displ}")
    return mapped_idx


# Berechnung der reachability_dist als reine Array-Operation

comm = dist.comm
rank = comm.Get_rank()
_, displ, _ = comm.counts_displs_shape(dist.shape, dist.split)

# print(f"process {ht.MPI_WORLD.rank}: \n displ={displ}")
# TODO: add a type promotion to float32 or float64
# promoted_type = types.promote_types(dist.dtype)
# promoted_type = types.promote_types(promoted_type, types.float32)
# X = X.astype(promoted_type)
# Y = Y.astype(promoted_type)
# if promoted_type == types.float32:
#     torch_type = torch.float32
#     # mpi_type = MPI.FLOAT
# elif promoted_type == types.float64:
#     torch_type = torch.float64
#     # mpi_type = MPI.DOUBLE
# else:
#     raise NotImplementedError(f"Datatype {X.dtype} currently not supported as input")
# map the indices in idx_k_dist to the corresponding MPI process that is responsible for
# the corresponding sample in dist

mapped_idx = _map_idx_to_proc(idx_k_dist, comm)

# reachability_dist = ht.zeros_like(dist)

reachability_dist = ht.zeros_like(dist).larray

local_k_dist = k_dist.larray
local_dist = dist.larray

for i in range(int(idx_k_dist.lshape[0])):
    # evaluate reachability distance for the current process
    if mapped_idx[i] == rank:
        # print(f"\n\n\n process: {ht.MPI_WORLD.rank} \n dist={dist}\n\n\n ")
        # print(f"\n\n\n process: {ht.MPI_WORLD.rank} \n dist[{i},:]={dist[int(idx_k_dist[i]),:]}\n\n\n ")
        reachability_dist[i, :] = torch.maximum(
            local_k_dist[i, None], local_dist[int(idx_k_dist[i]) - displ[rank], :]
        )
        # reachability_dist = ht.maximum(k_dist[:, None], cdist_small[idx_k_dist,:])
    else:
        receiver = rank
        sender = int(mapped_idx[i])
        # print(f"\n\n\n process: {ht.MPI_WORLD.rank} \n idx_k_dist[i]={idx_k_dist[i]},\n sender={sender}\n\n\n ")
        # select the distances to communicate between the processes according to the mapped inidces
        dist_comm = dist[int(idx_k_dist[i]) - displ[sender], :]
        # set a buffer to store the part of Y that is sent to the next process
        buffer = torch.zeros(
            (dist_comm.lshape_map[sender, 0]),
            # dtype=torch_type,
            device=dist_comm.device.torch_device,
        )
        dist_comm = dist_comm.larray
        comm.Isend(dist_comm, dest=receiver, tag=i)
        # receive the part of Y to the next process
        comm.Irecv(buffer, source=sender, tag=i)

        # TODO:
        # check whether
        # - torch.tensors and ht.DNDarrays were used consistently
        # - larrays were used consistently
        # - the correct result is computed
        reachability_dist[i] = torch.maximum(local_k_dist[i, None], buffer[:])
    reachability_dist = ht.array(reachability_dist, is_split=0)


# Erwartetes Ergebnis für den Vergleich
expected_result = ht.array([[3, 3, 3], [4, 4, 4], [2, 3, 4]], split=0)


print(f"process: {ht.MPI_WORLD.rank} \n reachability_dist={reachability_dist},\n ")

if not ht.allclose(reachability_dist, expected_result):
    print(f"process: {ht.MPI_WORLD.rank} \n -----------------Fail!---------------,\n ")
else:
    print(f"process: {ht.MPI_WORLD.rank} \n ###################Success!#############,\n ")
