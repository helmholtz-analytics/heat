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

# ht.use_device("gpu")

# X=ht.array([[0, 1, 3], [0, 3, 4], [0, 1, 2]], split=0)
# data=X[1].larray

# comm=X.comm
# rank = comm.Get_rank()
# sender=0
# receiver=1

# buf= torch.zeros(
#     data.shape,
#     dtype=X.dtype.torch_type(),
#     device=X.device.torch_device,
#     )


# if rank== sender:
#      comm.Send(data, dest=receiver, tag=1)
# if rank== receiver:
#      comm.Recv(buf, source=sender, tag=1)
# print(f"--------------\n process: {ht.MPI_WORLD.rank} \n buf={buf}\n-------------- ")

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


# distance.cdist_small(indices,indices,n_smallest=1)
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
    return mapped_idx


# Berechnung der reachability_dist als reine Array-Operation

comm = dist.comm
rank = comm.Get_rank()
size = comm.Get_size()
_, displ, _ = comm.counts_displs_shape(dist.shape, dist.split)

# print(f"process {ht.MPI_WORLD.rank}: \n displ={displ}")
# # TODO: add a type promotion to float32 or float64
# #promoted_type = types.promote_types(dist.dtype)
# promoted_type = types.promote_types(dist.dtype, types.float32)
# dist= dist.astype(promoted_type)
# if promoted_type == types.float32:
#     torch_type = torch.float32
#     # mpi_type = MPI.FLOAT
# elif promoted_type == types.float64:
#     torch_type = torch.float64
#     # mpi_type = MPI.DOUBLE
# else:
#     raise NotImplementedError(f"Datatype {dist.dtype} currently not supported as input")

# map the indices in idx_k_dist to the corresponding MPI process that is responsible for
# the corresponding sample in dist

mapped_idx = _map_idx_to_proc(idx_k_dist, comm)
mapped_idx_ = mapped_idx.larray


# reachability_dist = ht.zeros_like(dist)

reachability_dist = ht.zeros_like(dist)
reachability_dist = reachability_dist.larray

k_dist_ = k_dist.larray
dist_ = dist.larray
idx_k_dist_ = idx_k_dist.larray
global_idx_k_dist_ = idx_k_dist.resplit_(None)
# k_dist=k_dist.resplit_(None)

ones = ht.ones(int(idx_k_dist.shape[0]), split=0)
proc_id = ones * rank
proc_id_global = proc_id.resplit_(None)
k_dist_global = k_dist.resplit_(None)
idx_k_dist_global = idx_k_dist.resplit_(None)
mapped_idx_global = mapped_idx.resplit_(None)

# buffer to store one row of the distance matrix that is sent to the next process
buffer = torch.zeros(
    (1, dist_.shape[1]),
    dtype=dist.dtype.torch_type(),
    device=dist.device.torch_device,
)

for i in range(int(mapped_idx_global.shape[0])):
    receiver = proc_id_global[i].item()
    sender = mapped_idx_global[i].item()
    print(
        f"------------------ \n process: {ht.MPI_WORLD.rank} \n int(idx_k_dist_global[i])={int(idx_k_dist_global[i])}\n "
    )
    # dist_row = dist_[int(idx_k_dist_global[i]), :]
    # check if current process needs to send the corresponding row of its distance matrix
    if sender != receiver:
        # send
        if rank == sender:
            if rank == size - 1:
                upper_bound = mapped_idx_global.shape[0]
            else:
                upper_bound = displ[rank + 1]

            # only send if the sender is not the same as the current process
            if not displ[rank] <= i < upper_bound:
                # select the row of the distance matrix to communicate between the processes
                print(
                    f"------------------ \n process: {ht.MPI_WORLD.rank} \n displ[sender]={sender}\n "
                )
                dist_row = dist_[int(idx_k_dist_global[i]) - displ[sender], :]
                sent_to_buffer = dist_row
                # send the row to the next process
                print(
                    f"------------------ \n process: {ht.MPI_WORLD.rank} \n sending with tag {i}\n "
                )
                comm.Send(sent_to_buffer, dest=receiver, tag=i)
            # else:
            # reachability_dist=torch.maximum(k_dist_global[i, None], dist_row)
        # receive
        if rank == receiver:
            print(
                f"------------------ \n process: {ht.MPI_WORLD.rank} \n receiving with tag {i}\n "
            )
            comm.Recv(buffer, source=sender, tag=i)
            dist_row = buffer

            print(f"\n\n\n process: {ht.MPI_WORLD.rank} \n buffer={buffer}\n\n\n ")
            # reachability_dist[i]=torch.maximum(dist_row, dist_row)
            # TODO: check if the buffer is overwritten
    # no communication required
    elif sender == receiver:
        # no only take the row of the distance matrix that is already available
        if rank == sender:
            print(
                f"------------------ \n process: {ht.MPI_WORLD.rank} \n calculating w/o communication \n "
            )
            dist_row = dist_[int(idx_k_dist_global[i]), :]

            k_dist_compare = k_dist_global[i - displ[rank], None]
            k_dist_compare = k_dist_compare.larray
            reachability_dist[i] = torch.maximum(k_dist_compare, dist_row)
        else:
            pass
    # TODO: reachability_dist should be a local torch tensor and cannot have i in range(int(mapped_idx_global.shape[0]))
    # entries. How do overcome this issue?

# print(f"\n\n\n process: {ht.MPI_WORLD.rank} \n buffer={buffer}\n\n\n ")


# for i in range(int(idx_k_dist_.shape[0])):
#     for receiver in range(size):

#         sender = int(mapped_idx[i])
#         # select the distances to communicate between the processes according to the mapped inidces

#         if rank == sender:
#             # define    part of distance matrix to communicate
#             dist_comm_ = dist_[int(idx_k_dist_[i]) - displ[sender], :]
#             comm.Isend(dist_comm_, dest=receiver, tag=i)

#             # calculate part of dist that should enter the reachability distance (here: no communication required)
#             if sender == receiver:
#                 dist_row = dist_comm_

#         if rank == receiver:
#                 # receiving only required if sender is not the same as receiver
#                 if sender != receiver:
#                     # setup for receiving
#                     buffer = torch.zeros(
#                         (1,dist_.shape[1]),
#                         dtype=dist.dtype.torch_type(),
#                         device=dist.device.torch_device,
#                     )

#                     # receive part of dist that should enter the reachability distance
#                     comm.Irecv(buffer, source=sender, tag=i)
#                     dist_row = buffer


#         # no communication required
#         if sender == receiver:
#             dist_row=dist_[int(idx_k_dist_[i]) - displ[receiver], :]

#         # communication required
#         else:


#             # Sender schickt die Daten
#             dist_comm = dist[int(idx_k_dist_[i]) - displ[sender], :]
#             dist_comm_=dist_comm.larray
#             comm.Isend(dist_comm_, dest=receiver, tag=receiver)
#             # Starte Empfang zuerst
#             comm.Irecv(buffer, source=sender, tag=receiver)
#             dist_row = buffer

#     # Berechnung der reachability_dist
#     reachability_dist[i] = torch.maximum(k_dist_[i, None], dist_row)


# for receiver in range(size):  # Über alle Prozesse iterieren
#     for i in range(int(idx_k_dist_.shape[0])):
#         sender = int(mapped_idx[i])  # Wo soll die Zeile hin?

#         tag = i  # Einfacher Tag für jeden Datenaustausch

#         if sender == rank:  # Dieser Prozess ist Sender
#             if receiver != sender:  # Nicht an sich selbst senden
#                 dist_comm = dist_[int(idx_k_dist_[i]) - displ[sender], :]
#                 req_send = comm.Isend(dist_comm, dest=receiver, tag=tag)

#         elif receiver == rank:  # Dieser Prozess ist Empfänger
#             buffer = torch.zeros(
#                 (dist_.shape[1],),
#                 dtype=dist.dtype.torch_type(),
#                 device=dist.device.torch_device,
#             )
#             req_recv = comm.Irecv(buffer, source=sender, tag=tag)
#             req_recv.Wait()
#             dist_row = buffer
#         else:
#             continue  # Falls dieser Prozess nicht beteiligt ist

#         if rank == receiver:  # Berechnung nur beim Empfänger
#             reachability_dist[i] = torch.maximum(k_dist_[i, None], dist_row)

#     if sender == rank:
#         req_send.Wait()  # Sicherstellen, dass alle Sends abgeschlossen sind

print(f"\n\n\n process: {ht.MPI_WORLD.rank} \n buffer={buffer}\n\n\n ")

# dist_comm = dist[int(idx_k_dist_[i]) - displ[sender], :]
# #dist_comm=dist
# dist_comm_=dist_comm.larray
# buffer = torch.zeros(
#     dist_comm_.shape,
#     dtype=dist.dtype.torch_type(),
#     device=dist.device.torch_device,
#     )
# comm.Isend(dist_comm_, dest=receiver, tag=i)

# # evaluate reachability distance for the current process
# if sender==receiver:
#     # reachability_dist[i] = torch.maximum(
#     #     k_dist_[i, None], dist_[int(idx_k_dist_[i]) - displ[rank], :]
#     # )
#     dist_row=dist_[int(idx_k_dist_[i]) - displ[rank], :]
# else:
#     dist_row=buffer[:]
# comm.Irecv(buffer, source=sender, tag=i)
# reachability_dist[i] = torch.maximum(k_dist_[i, None], dist_row)

reachability_dist = ht.array(reachability_dist, is_split=0)

# Erwartetes Ergebnis für den Vergleich
expected_result = ht.array([[3, 3, 3], [4, 4, 4], [2, 3, 4]], split=0)
# reachability_dist = ht.zeros(expected_result.shape, split=0)

print(f"process: {ht.MPI_WORLD.rank} \n reachability_dist={reachability_dist},\n ")

if not ht.allclose(reachability_dist, expected_result):
    print(f"process: {ht.MPI_WORLD.rank} \n -----------------Fail!---------------,\n ")
else:
    print(f"process: {ht.MPI_WORLD.rank} \n ###################Success!#############,\n ")
