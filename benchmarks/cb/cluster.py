import heat as ht
from perun import monitor
from sklearn.metrics import silhouette_score as sk_silhouette_score


@monitor()
def kmeans(data):
    kmeans = ht.cluster.KMeans(n_clusters=4, init="kmeans++")
    kmeans.fit(data)


@monitor()
def kmedians(data):
    kmeans = ht.cluster.KMedians(n_clusters=4, init="kmedians++")
    kmeans.fit(data)


@monitor()
def kmedoids(data):
    kmeans = ht.cluster.KMedoids(n_clusters=4, init="kmedoids++")
    kmeans.fit(data)

@monitor()
def spectralclustering(data):
    spectral = ht.cluster.Spectral(
        n_clusters=4, gamma=1.0, metric="rbf", laplacian="fully_connected", eigen_solver="randomized", reigh_rank=10
    )
    spectral.fit(data)

@monitor()
def heat_silhouette(data, labels):
    score = ht.cluster.silhouette_score(data, labels)
    return score

@monitor()
def sklearn_silhouette(data_numpy, labels_numpy):
    score = None
    #if ht.communication.MPI_WORLD.rank == 0:
    #with parallel_backend('threading', n_jobs=4):
    score = sk_silhouette_score(data_numpy, labels_numpy)
    #ht.communication.MPI_WORLD.Barrier()
    return score


def run_cluster_benchmarks():
    n = 5000
    seed = 1
    data = ht.utils.data.spherical.create_spherical_dataset(
        num_samples_cluster=n, radius=1.0, offset=4.0, dtype=ht.float32, random_state=seed
    )

    kmeans(data)
    kmedians(data)
    kmedoids(data)
    spectralclustering(data)

def run_metrics_benchmarks():
    n_array = [1000, 2500, 5000, 7500, 10000]

    for n in n_array:
        seed = 1
        data = ht.utils.data.spherical.create_spherical_dataset(
            num_samples_cluster=n, radius=1.0, offset=4.0, dtype=ht.float32, random_state=seed
        )
        print(data.split)
        print(data.is_distributed())

        km = ht.cluster.KMeans(n_clusters=4)
        km.fit(data)
        labels = km.labels_

        data_np = data.numpy()
        labels_np = labels.numpy().ravel()

        heat_silhouette(data, labels)
        sklearn_silhouette(data_np, labels_np)

def run_heat_silhoette():
    #n_array = [10000, 20000, 30000, 40000, 50000]
    n_array = [5000]

    for n in n_array:
        seed = 1
        data = ht.utils.data.spherical.create_spherical_dataset(
            num_samples_cluster=n, radius=1.0, offset=4.0, dtype=ht.float32, random_state=seed
        )

        km = ht.cluster.KMeans(n_clusters=4)
        km.fit(data)
        labels = km.labels_

        heat_silhouette(data, labels)
        print("done with heat ", n)

def run_sklearn_silhouette():
    n_array = [5000]
    #n_array = [10000, 20000, 30000, 40000, 50000]
    for n in n_array:
        seed = 1
        data = ht.utils.data.spherical.create_spherical_dataset(
            num_samples_cluster=n, radius=1.0, offset=4.0, dtype=ht.float32, random_state=seed
        )

        km = ht.cluster.KMeans(n_clusters=4)
        km.fit(data)
        labels = km.labels_

        data_np = data.numpy()
        labels_np = labels.numpy().ravel()

        sklearn_silhouette(data_np, labels_np)
        print("done with sklearn ", n)

if __name__ == "__main__":
    #run_metrics_benchmarks()
    run_sklearn_silhouette()
