from perun.decorator import monitor

import heat as ht


@monitor()
def kmeans_cpu(data):
    kmeans = ht.cluster.KMeans(n_clusters=4, init="kmeans++")
    kmeans.fit(data)


@monitor()
def kmedians_cpu(data):
    kmeans = ht.cluster.KMedians(n_clusters=4, init="kmedians++")
    kmeans.fit(data)


@monitor()
def kmedoids_cpu(data):
    kmeans = ht.cluster.KMedoids(n_clusters=4, init="kmedoids++")
    kmeans.fit(data)


n = 5000
seed = 1
data = ht.utils.data.spherical.create_spherical_dataset(
    num_samples_cluster=n, radius=1.0, offset=4.0, dtype=ht.float32, random_state=seed
)
kmeans_cpu(data)
kmedians_cpu(data)
kmedoids_cpu(data)
