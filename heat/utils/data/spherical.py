"""Create a sperical dataset."""
import heat as ht


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
