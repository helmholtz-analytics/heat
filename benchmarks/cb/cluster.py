import heat as ht
import torch
from perun import monitor
from sizes import GSIZE_vTS_S, GSIZE_vTS_L

"""
For clustering we assume very tall skinny data
Benchmarks in this file:
- K-Means (with kmeans++ initialization)
- K-Medians (with kmedians++ initialization)
- K-Medoids (with kmedoids++ initialization)
- BatchParallelKMeans (with k-means++ initialization)
"""

N_CLUSTERS_TO_FIND = 4


@monitor()
def kmeans(data):
    kmeans = ht.cluster.KMeans(n_clusters=N_CLUSTERS_TO_FIND, init="kmeans++")
    kmeans.fit(data)


@monitor()
def kmedians(data):
    kmeans = ht.cluster.KMedians(n_clusters=N_CLUSTERS_TO_FIND, init="kmedians++")
    kmeans.fit(data)


@monitor()
def kmedoids(data):
    kmeans = ht.cluster.KMedoids(n_clusters=N_CLUSTERS_TO_FIND, init="kmedoids++")
    kmeans.fit(data)


@monitor()
def batchparallel_kmeans(data):
    bpkmeans = ht.cluster.BatchParallelKMeans(n_clusters=N_CLUSTERS_TO_FIND, init="k-means++")
    bpkmeans.fit(data)


def run_cluster_benchmarks():
    # N_CLUSTERS_TO_FIND many spherical clusters, "centers" are uniformly distributed in a hypercube [-5,5]^d
    # each cluster is normally distributed with std=1
    data = ht.utils.data.spherical.create_clusters(
        GSIZE_vTS_L,
        GSIZE_vTS_S,
        N_CLUSTERS_TO_FIND,
        10 * (torch.rand.rand(N_CLUSTERS_TO_FIND, GSIZE_vTS_S) - 1),
        1,
    )

    kmeans(data)
    kmedians(data)
    kmedoids(data)
    batchparallel_kmeans(data)
