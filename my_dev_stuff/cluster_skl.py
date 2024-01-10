import h5py
import sklearn.cluster as skl
import time
import numpy as np

n_samples = 10
n_cluster_max = 3
dataset_path = "/projects/HPDAGrundlagensoftware-Heat/Testdata/JPL_SBDB/sbdb_asteroids.h5"

file = h5py.File(dataset_path, "r")
data = file["data"]

# standard scaler in scikit-learn
data = (data - data.mean(axis=0)) / data.std(axis=0)

functional_value = lambda x, centers, labels: ((x - centers[labels]) ** 2).sum()

for n_clusters in range(2, n_cluster_max + 1):
    times_sklkm = []
    times_sklbkm = []
    values_sklkm = []
    values_sklbkm = []

    for s in range(n_samples):
        t0 = time.perf_counter()
        clusterer_sklkm = skl.KMeans(n_clusters=n_clusters, init="k-means++")
        clusterer_sklkm.fit_predict(data)
        t1 = time.perf_counter()
        times_sklkm.append(t1 - t0)
        values_sklkm.append(
            functional_value(data, clusterer_sklkm.cluster_centers_, clusterer_sklkm.labels_)
        )
        del clusterer_sklkm

        t0 = time.perf_counter()
        clusterer_sklbkm = skl.MiniBatchKMeans(n_clusters=n_clusters, init="k-means++")
        clusterer_sklbkm.fit_predict(data)
        t1 = time.perf_counter()
        times_sklbkm.append(t1 - t0)
        values_sklbkm.append(
            functional_value(data, clusterer_sklbkm.cluster_centers_, clusterer_sklbkm.labels_)
        )
        del clusterer_sklbkm

    np.savetxt("times_sklkm_%d.txt" % n_clusters, np.asarray(times_sklkm))
    np.savetxt("values_sklkm_%d.txt" % n_clusters, np.asarray(values_sklkm))
    np.savetxt("times_sklbkm_%d.txt" % n_clusters, np.asarray(times_sklbkm))
    np.savetxt("values_sklbkm_%d.txt" % n_clusters, np.asarray(values_sklbkm))
