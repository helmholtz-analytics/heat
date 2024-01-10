import heat as ht
import time
import numpy as np


n_samples = 10
n_cluster_max = 3
dataset_path = "/projects/HPDAGrundlagensoftware-Heat/Testdata/JPL_SBDB/sbdb_asteroids.h5"
load_fraction = 0.1
device = "cpu"

# load jpl asteroids dataset and scale the data 
data = ht.load(dataset_path, load_fraction=load_fraction, dataset='data', split=0, dtype=ht.float32, device=device)
preproc = ht.preprocessing.StandardScaler(copy=False)
data = preproc.fit_transform(data)

n_procs = ht.MPI_WORLD.size

for n_clusters in range(2, n_cluster_max + 1):

    times_kmpp = []
    times_kmbp = []
    values_kmpp = []
    values_kmbp = []

    for s in range(n_samples):
        ht.MPI_WORLD.Barrier()
        t0 = time.perf_counter()
        clusterer_kmpp = ht.cluster.KMeans(n_clusters=n_clusters, init="kmeans++")
        clusterer_kmpp.fit_predict(data)
        ht.MPI_WORLD.Barrier()
        t1 = time.perf_counter()
        times_kmpp.append(t1 - t0)
        values_kmpp.append(clusterer_kmpp.functional_value_)
        del clusterer_kmpp

        ht.MPI_WORLD.Barrier()
        t0 = time.perf_counter()
        clusterer_kmbp = ht.cluster.KMeans(n_clusters=n_clusters, init="batchparallel")
        clusterer_kmbp.fit_predict(data)
        ht.MPI_WORLD.Barrier()
        t1 = time.perf_counter()
        times_kmbp.append(t1 - t0)
        values_kmbp.append(clusterer_kmbp.functional_value_)
        del clusterer_kmbp

    if ht.MPI_WORLD.rank == 0:
        np.savetxt('times_kmbp_%d_%d.txt' % (n_clusters, n_procs), np.asarray(times_kmbp))
        np.savetxt('values_kmbp_%d_%d.txt' % (n_clusters, n_procs), np.asarray(values_kmbp))
        np.savetxt('times_kmpp_%d_%d.txt' % (n_clusters, n_procs), np.asarray(times_kmpp))
        np.savetxt('values_kmpp_%d_%d.txt' % (n_clusters, n_procs), np.asarray(values_kmpp))
