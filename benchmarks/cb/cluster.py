import heat as ht
from perun import monitor


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
