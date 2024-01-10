import heat as ht
import matplotlib.pyplot as plt
import torch
import time

# load jpl asteroids dataset
data = ht.load(
    "/projects/HPDAGrundlagensoftware-Heat/Testdata/JPL_SBDB/sbdb_asteroids.h5",
    load_fraction=0.01,
    dataset="data",
    split=0,
    dtype=ht.float32,
)
preproc = ht.preprocessing.StandardScaler(copy=False)
data = preproc.fit_transform(data)

# clustering
# clusterer = ht.cluster.BatchParallelKMeans(n_clusters=10,init="k-means++")
t0 = time.perf_counter()
clusterer = ht.cluster.KMeans(n_clusters=10, init="random")
labels = clusterer.fit_predict(data)
t1 = time.perf_counter()

if data.comm.rank == 0:
    print("Time for Kmeans (random init): ", t1 - t0)
    print("Functional (random init): ", clusterer.functional_value_)

t0 = time.perf_counter()
clusterer = ht.cluster.KMeans(n_clusters=10, init="kmeans++")
labels = clusterer.fit_predict(data)
t1 = time.perf_counter()

if data.comm.rank == 0:
    print("Time for Kmeans (++ init): ", t1 - t0)
    print("Functional (++ init): ", clusterer.functional_value_)

t0 = time.perf_counter()
clusterer = ht.cluster.KMeans(n_clusters=10, init="batchparallel")
labels = clusterer.fit_predict(data)
t1 = time.perf_counter()

if data.comm.rank == 0:
    print("Time for Kmeans (batchparallel init): ", t1 - t0)
    print("Functional (batchparallel init): ", clusterer.functional_value_)


# clustering
t0 = time.perf_counter()
clusterer = ht.cluster.BatchParallelKMeans(n_clusters=10, init="k-means++")
labels = clusterer.fit_predict(data)
t1 = time.perf_counter()

if data.comm.rank == 0:
    print("Time for BatchParallelKmeans: ", t1 - t0)
    print("Functional BatchParallelKmeans: ", clusterer.functional_value_)

# # hSVD
# u,s,v,info = ht.linalg.hsvd_rank(data.T, 9, compute_sv=True)

# coords = u.T @ data.T
# coords = coords.numpy()
# print(labels)

# plt.figure()
# plt.scatter(coords[0,:], coords[1,:], c=c, s=0.5)
# plt.savefig("viz_01.png")

# plt.figure()
# plt.scatter(coords[0,:], coords[2,:], c=c, s=0.5)
# plt.savefig("viz_02.png")

# plt.figure()
# plt.scatter(coords[0,:], coords[3,:], c=c, s=0.5)
# plt.savefig("viz_03.png")

# plt.figure()
# plt.scatter(coords[1,:], coords[2,:], c=c, s=0.5)
# plt.savefig("viz_12.png")

# plt.figure()
# plt.scatter(coords[2,:], coords[3,:], c=c, s=0.5)
# plt.savefig("viz_23.png")
