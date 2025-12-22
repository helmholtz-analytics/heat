Module heat.utils.data.spherical
================================
Create a sperical dataset.

Functions
---------

`create_clusters(n_samples, n_features, n_clusters, cluster_mean, cluster_std, cluster_weight=None, device=None)`
:   Creates a DNDarray of shape (n_samples, n_features), split=0, and dtype=ht.float32, that is balanced (i.e. roughly same size of samples on each process).
    The data set consists of n_clusters clusters, each of which is sampled from a multivariate normal distribution with mean cluster_mean[k,:] and covariance matrix cluster_std[k,:,:].
    The clusters are of the same size (quantitatively) and distributed evenly over the processes, unless cluster_weight is specified.

    Parameters
    ----------
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

`create_spherical_dataset(num_samples_cluster, radius=1.0, offset=4.0, dtype=heat.core.types.float32, random_state=1)`
:   Creates k=4 sperical clusters in 3D space along the space-diagonal

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
