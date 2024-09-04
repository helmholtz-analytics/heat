"""Create a sperical dataset."""

import heat as ht
import torch


def create_spherical_dataset(
    num_samples_cluster, radius=1.0, offset=4.0, dtype=ht.float32, random_state=1
):
    """
    Creates k=4 sperical clusters in 3D space along the space-diagonal

    Parameters
    ----------
    num_samples_cluster: int
        Number of samples per cluster. Each process will create n // MPI_WORLD.size elements for each cluster
    radius: float
        Radius of the sphere
    offset: float
        Shift of the clusters along the axes. The 4 clusters will be positioned centered around c1=(offset, offset,offset),
        c2=(2*offset,2*offset,2*offset), c3=(-offset, -offset, -offset) and c4=(2*offset, -2*offset, -2*offset)
    dtype: ht.datatype
        Dataset dtype
    random_state: int
        seed of the torch random number generator
    """
    # contains num_samples

    p = ht.MPI_WORLD.size
    # create k sperical clusters with each n elements per cluster. Each process creates k * n/p elements
    num_ele = num_samples_cluster // p
    ht.random.seed(random_state)
    # radius between 0 and 1
    r = ht.random.rand(num_ele, split=0) * radius
    # theta between 0 and pi
    theta = ht.random.rand(num_ele, split=0) * 3.1415
    # phi between 0 and 2pi
    phi = ht.random.rand(num_ele, split=0) * 2 * 3.1415
    # Cartesian coordinates
    x = r * ht.sin(theta) * ht.cos(phi)
    x.astype(dtype, copy=False)
    y = r * ht.sin(theta) * ht.sin(phi)
    y.astype(dtype, copy=False)
    z = r * ht.cos(theta)
    z.astype(dtype, copy=False)

    cluster1 = ht.stack((x + offset, y + offset, z + offset), axis=1)
    cluster2 = ht.stack((x + 2 * offset, y + 2 * offset, z + 2 * offset), axis=1)
    cluster3 = ht.stack((x - offset, y - offset, z - offset), axis=1)
    cluster4 = ht.stack((x - 2 * offset, y - 2 * offset, z - 2 * offset), axis=1)

    data = ht.concatenate((cluster1, cluster2, cluster3, cluster4), axis=0)
    # Note: enhance when shuffel is available
    return data


def create_clusters(
    n_samples, n_features, n_clusters, cluster_mean, cluster_std, cluster_weight=None, device=None
):
    """
    Creates a DNDarray of shape (n_samples, n_features), split=0, and dtype=ht.float32, that is balanced (i.e. roughly same size of samples on each process).
    The data set consists of n_clusters clusters, each of which is sampled from a multivariate normal distribution with mean cluster_mean[k,:] and covariance matrix cluster_std[k,:,:].
    The clusters are of the same size (quantitatively) and distributed evenly over the processes, unless cluster_weight is specified.

    Parameters
    ------------
    n_samples: int
        Number of overall samples
    n_features: int
        Number of features
    n_clusters: int
        Number of clusters
    cluster_mean: torch.Tensor of shape (n_clusters, n_features)
        featurewise mean (center) of each cluster; of course not the true mean, but rather the mean according to which the elements of the cluster are sampled.
    cluster_std: torch.Tensor of shape (n_clusters, n_features, n_features), or (n_clusters,)
        featurewise standard deviation of each cluster from the mean value; of course not the true std, but rather the std according to which the elements of the cluster are sampled.
        If shape is (n_clusters,), std is assumed to be the same in each direction for each cluster
    cluster_weight: torch.Tensor of shape (n_clusters,), optional
        On each process, cluster_weight is assumed to be a torch.Tensor whose entries add up to 1. The i-th entry of cluster_weight on process p specified which amount of the samples on process p
        is sampled according to the distribution of cluster i. Thus, this parameter allows to distribute the n_cluster clusters unevenly over the processes.
        If None, each cluster is distributed evenly over all processes.
    device: Optional[str] = None,
        The device on which the data is stored. If None, the default device is used.
    """
    device = ht.devices.sanitize_device(device)

    if cluster_weight is None:
        cluster_weight = torch.ones(n_clusters) / n_clusters
    else:
        if not isinstance(cluster_weight, torch.Tensor):
            raise TypeError(
                "cluster_weight must be None or a torch.Tensor, but is {}".format(
                    type(cluster_weight)
                )
            )
        elif not cluster_weight.shape == (n_clusters,):
            raise ValueError(
                "If a torch.Tensor, cluster_weight must be of shape (n_clusters,), but is {}".format(
                    cluster_weight.shape
                )
            )
        elif not torch.allclose(torch.sum(cluster_weight), torch.tensor(1.0)):
            raise ValueError(
                "If a torch.Tensor, cluster_weight must add up to 1, but adds up to {}".format(
                    torch.sum(cluster_weight)
                )
            )
    if not isinstance(cluster_mean, torch.Tensor):
        raise TypeError("cluster_mean must be a torch.Tensor, but is {}".format(type(cluster_mean)))
    elif not cluster_mean.shape == (n_clusters, n_features):
        raise ValueError(
            "cluster_mean must be of shape (n_clusters, n_features), but is {}".format(
                cluster_mean.shape
            )
        )
    if not isinstance(cluster_std, torch.Tensor):
        raise TypeError("cluster_std must be a torch.Tensor, but is {}".format(type(cluster_std)))
    elif not cluster_std.shape == (
        n_clusters,
        n_features,
        n_features,
    ) and not cluster_std.shape == (n_clusters,):
        raise ValueError(
            "cluster_std must be of shape (n_clusters, n_features, n_features) or (n_clusters,), but is {}".format(
                cluster_std.shape
            )
        )
    if cluster_std.shape == (n_clusters,):
        cluster_std = torch.stack(
            [torch.eye(n_features) * cluster_std[k] for k in range(n_clusters)], dim=0
        )

    global_shape = (n_samples, n_features)
    local_shape = ht.MPI_WORLD.chunk(global_shape, 0)[1]
    local_size_of_clusters = [int(local_shape[0] * cluster_weight[k]) for k in range(n_clusters)]
    if sum(local_size_of_clusters) != local_shape[0]:
        local_size_of_clusters[0] += local_shape[0] - sum(local_size_of_clusters)
    distributions = [
        torch.distributions.multivariate_normal.MultivariateNormal(
            cluster_mean[k, :], cluster_std[k]
        )
        for k in range(n_clusters)
    ]
    local_data = [
        distributions[k].sample((local_size_of_clusters[k],)).to(device.torch_device)
        for k in range(n_clusters)
    ]
    local_data = torch.cat(local_data, dim=0)
    rand_perm = torch.randperm(local_shape[0])
    local_data = local_data[rand_perm, :]
    data = ht.DNDarray(
        local_data,
        global_shape,
        dtype=ht.float32,
        split=0,
        device=device,
        comm=ht.MPI_WORLD,
        balanced=True,
    )
    return data
