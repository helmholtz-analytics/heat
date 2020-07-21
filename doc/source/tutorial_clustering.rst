Cluster Analysis
================
**Author**: `Charlotte Debus <https://github.com/Cdebus>`_

This tutorial will demonstrate analysis with k-means and k-medians from the ``cluster`` module.
We will use matplotlib for visualization of data and results :

    import heat as ht
    import matplotlib.pyplot as plt


Spherical Clouds of Datapoints
------------------------------
For a simple demonstration of the clustering process and the differences between the algorithms, we will create an
artificial dataset, consisting of two circularly shaped clusters positioned at :math:`(x_1=2,y_1=2)` and :math:`
(x_2=-2,y_2=-2)` in 2D space.
For each cluster we will sample 100 arbitrary points from a circle with radius of r = 1.0 by drawing random numbers
for the spherical coordinates :math:`( r\in [0,1], \phi \in [0,2\pi])`, translating these to cartesian coordinates
and shifting them by +2 for cluster c1 and -2 for cluster c2. The resulting concatenated dataset ``data`` has shape
(100, 2)  and is distributed among the ``p`` processes along axis 0 (sample axis) ::

    p = ht.MPI_WORLD.size
    # Each process creates 100 / p datapoints per cluster
    num_ele = 100 // p
    ht.random.seed(1)

    # Create default sperical point cloud
    # Sample radius between 0 and 1, and phi between 0 and 2pi
    r = ht.random.rand(num_ele, split=0) * radius
    phi = ht.random.rand(num_ele, split=0) * 2 * 3.1415

    # Transform spherical coordinates to cartesian coordinates
    x = r * ht.cos(phi)
    y = r * ht.sin(phi)


    # Stack the sampled points and shift them to locations (2,2) and (-2, -2)
    cluster1 = ht.stack((x + 2, y + 2), axis=1)
    cluster2 = ht.stack((x - 2, y - 2), axis=1)

    data = ht.concatenate((cluster1, cluster2), axis=0)

Let's plot the data for illustration. In order to do so with matplotlib, we need to unsplit the data (gather it from
all processes) and transform it into a numpy array. Plotting can only be done on rank 0 ::

    data_np = ht.resplit(data, axis=None).numpy()
    if ht.MPI_WORLD.rank == 0:
        plt.plot(data_np[:,0], data_np[:,1], 'bo')

Now we perform the clustering analysis with kmeans. We chose 'kmeans++' as an intelligent way of sampling the
initial centroids ::

    kmeans = ht.cluster.KMeans(n_clusters=2, init="kmeans++")
    labels = kmeans.fit_predict(data)
    labels.resplit_(axis=None)
    centroids = kmeans.cluster_centers_

    # Create two sub-arrays for the assigned cluster points
    c1 = ht.where(labels == 0, data, 0)
    c1 = c1[ht.abs(ht.sum(c1, axis=1))=!0, :]
    c2 = ht.where(labels == 1, data, 0)
    c2 = c2[ht.abs(ht.sum(c2, axis=1))=!0, :]

    print("Number of points assigned to c1: {} \n
           Number of points assigned to c2: {}".format(c1.shape[0], c2.shape[0]))
Let's plot the assigned clusters and the respective centroids: ::

    c1_np = c1.numpy()
    c2_np = c2.numpy()
    plt.plot(c1_np[:,0], c1_np[:,1], 'x', color='#f0781e')
    plt.plot(c2_np[:,0], c2_np[:,1], 'x', color='#5a696e')
    plt.plot(centroids[0,0],centroids[0,1], '^', markersize=3, color='#f0781e' )
    plt.plot(centroids[1,0],centroids[1,1], '^', markersize=3, color='#5a696e')
We can also cluster the data with kmedians. The respective advanced initial centroid sampling is called 'kmedians++' ::

    kmedians = ht.cluster.KMedians(n_clusters=2, init="kmedians++")
    labels = kmedians.fit_predict(data)
    labels.resplit_(axis=None)
    centroids = kmedians.cluster_centers_

    # Create two sub-arrays for the assigned cluster points
    c1 = ht.where(labels == 0, data, 0)
    c1 = c1[ht.abs(ht.sum(c1, axis=1))=!0, :]
    c2 = ht.where(labels == 1, data, 0)
    c2 = c2[ht.abs(ht.sum(c2, axis=1))=!0, :]

    print("Number of points assigned to c1: {} \n
           Number of points assigned to c2: {}".format(c1.shape[0], c2.shape[0]))
Plotting the assigned clusters and the respective centroids: ::

    c1_np = c1.numpy()
    c2_np = c2.numpy()
    plt.plot(c1_np[:,0], c1_np[:,1], 'x', color='#f0781e')
    plt.plot(c2_np[:,0], c2_np[:,1], 'x', color='#5a696e')
    plt.plot(centroids[0,0],centroids[0,1], '^', markersize=3, color='#f0781e' )
    plt.plot(centroids[1,0],centroids[1,1], '^', markersize=3, color='#5a696e')
The Iris Dataset
------------------------------

The _iris_ dataset is a well known example for clustering analysis. It contains 4 measured features for samples from
three different types of iris flowers. A subset of 150 samples is included in formats h5, csv and netcdf in heat,
located under 'heat/heat/datasets/data/iris.h5', and can be loaded in a distributed manner with heat's parallel
dataloader ::

    iris = ht.load("heat/datasets/data/iris.csv", sep=";", split=0)
Fitting the dataset with kmeans: ::

    k = 3
    kmeans = ht.cluster.KMeans(n_clusters=k, init="kmeans++")
    kmeans.fit(iris)

Let's see what the results are. In theory, there are 50 samples of each of the 3 iris types ::

    labels = kmeans.predict(iris)
    labels.resplit_(axis=None)

    c1 = ht.where(labels == 0, data, 0)
    c1 = c1[ht.abs(ht.sum(c1, axis=1))=!0, :]
    c2 = ht.where(labels == 1, data, 0)
    c2 = c2[ht.abs(ht.sum(c2, axis=1))=!0, :]
    c3 = ht.where(labels == 2, data, 0)
    c3 = c3[ht.abs(ht.sum(c3, axis=1))=!0, :]

    print("Number of points assigned to c1: {} \n
           Number of points assigned to c2: {} \n
           Number of points assigned to c3: {} ".format(c1.shape[0], c2.shape[0], c3.shape[0]))
