import heat as ht
import time
import numpy as np


n_samples = 100
n_cluster_max = 3
dataset_path = "/projects/HPDAGrundlagensoftware-Heat/Testdata/JPL_SBDB/sbdb_asteroids.h5"
load_fraction = 0.1
device = "cpu"

# load jpl asteroids dataset and scale the data
data = ht.load(
    dataset_path,
    load_fraction=load_fraction,
    dataset="data",
    split=0,
    dtype=ht.float32,
    device=device,
)
preproc = ht.preprocessing.StandardScaler(copy=False)
data = preproc.fit_transform(data)

n_procs = ht.MPI_WORLD.size

for n_clusters in range(2, n_cluster_max + 1):
    times_bp = []
    times_bph = []
    values_bp = []
    values_bph = []

    for s in range(n_samples):
        ht.MPI_WORLD.Barrier()
        t0 = time.perf_counter()
        clusterer_bp = ht.cluster.BatchParallelKMeans(n_clusters=n_clusters, init="k-means++")
        clusterer_bp.fit_predict(data)
        ht.MPI_WORLD.Barrier()
        t1 = time.perf_counter()
        times_bp.append(t1 - t0)
        values_bp.append(clusterer_bp.functional_value_)
        del clusterer_bp

        ht.MPI_WORLD.Barrier()
        t0 = time.perf_counter()
        clusterer_bph = ht.cluster.BatchParallelKMeans(
            n_clusters=n_clusters, init="k-means++", n_procs_to_merge=2
        )
        clusterer_bph.fit_predict(data)
        ht.MPI_WORLD.Barrier()
        t1 = time.perf_counter()
        times_bph.append(t1 - t0)
        values_bph.append(clusterer_bph.functional_value_)
        del clusterer_bph

    if ht.MPI_WORLD.rank == 0:
        np.savetxt("times_bp_%d_%d.txt" % (n_clusters, n_procs), np.asarray(times_bp))
        np.savetxt("values_bp_%d_%d.txt" % (n_clusters, n_procs), np.asarray(values_bp))
        np.savetxt("times_bph_%d_%d.txt" % (n_clusters, n_procs), np.asarray(times_bph))
        np.savetxt("values_bph_%d_%d.txt" % (n_clusters, n_procs), np.asarray(values_bph))
