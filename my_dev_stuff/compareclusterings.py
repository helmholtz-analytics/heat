import heat as ht
import matplotlib.pyplot as plt
import torch
import time

# load jpl asteroids dataset
data = ht.load(
    "/projects/HPDAGrundlagensoftware-Heat/Testdata/JPL_SBDB/sbdb_asteroids.h5",
    load_fraction=0.1,
    dataset="data",
    split=0,
    dtype=ht.float32,
    device="cpu",
)
preproc = ht.preprocessing.StandardScaler(copy=False)
data = preproc.fit_transform(data)

todos = ["bpkm2", "bpkm"]
n_samples = 3
clusternumbers = list(range(2, 5))

for todo in todos:
    mins = []
    maxs = []
    avgs = []
    times = []
    for k in clusternumbers:
        vals = []
        t = 0
        if ht.MPI_WORLD.rank == 0:
            print(todo + f"Clustering with {k} clusters")
        for s in range(n_samples):
            if ht.MPI_WORLD.rank == 0:
                print(f"\t Sample {s}")
            if todo is "bpkm2":
                clusterer = ht.cluster.BatchParallelKMeans(
                    n_clusters=k, init="k-means++", n_procs_to_merge=2
                )
                titlestr = "BatchParallelKMeans (merge 2)" if ht.MPI_WORLD.size > 1 else "KMeans"
            elif todo is "bpkm":
                clusterer = ht.cluster.BatchParallelKMeans(
                    n_clusters=k, init="k-means++", n_procs_to_merge=None
                )
                titlestr = (
                    "BatchParallelKMeans (all-at-once)" if ht.MPI_WORLD.size > 1 else "KMeans"
                )
            else:
                raise NotImplementedError("Not implemented yet")
            ht.MPI_WORLD.barrier()
            t0 = time.perf_counter()
            clusterer.fit_predict(data)
            ht.MPI_WORLD.barrier()
            t1 = time.perf_counter()
            t += t1 - t0
            vals.append(clusterer.functional_value_)
        times.append(t / n_samples)
        mins.append(min(vals))
        maxs.append(max(vals))
        avgs.append(sum(vals) / len(vals))

    filestr = todo if ht.MPI_WORLD.size > 1 else "km"
    if data.comm.rank == 0:
        plt.figure()
        plt.title(titlestr)
        plt.plot(clusternumbers, mins, label="min")
        plt.plot(clusternumbers, maxs, label="max")
        plt.plot(clusternumbers, avgs, label="avg")
        plt.xlabel("n_clusters")
        plt.ylabel("K-clustering functional value")
        plt.legend()
        plt.savefig(filestr + "val.png")

        plt.figure()
        plt.title(titlestring)
        plt.plot(clusternumbers, times, label="time")
        plt.xlabel("n_clusters")
        plt.ylabel("time")
        plt.legend()
        plt.savefig(filestr + "_time.png")
