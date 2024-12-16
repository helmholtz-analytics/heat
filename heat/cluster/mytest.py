"""
Some tests to check the funtionality of the k-means clustering algortihm
"""

import heat as ht
import numpy as np
import torch
import time

ht.use_device("gpu")
# Convert data into DNDarrays
# The shape of this data is (3,5), i.e.,
# 3 data points, each consisting of 5 features
x = [[1, 2, 3, 4, 5], [10, 20, 30, 40, 50], [0, 2, 3, 4, 4]]
unit = ht.ones((3, 5), split=None)
unitvector = ht.ones((1, 5), split=None)
v = [[20, 30, 40, 5, 6], [11, 22, 33, 44, 55], [102, 204, 303, 406, 507], [30, 44, 53, 66, 77]]
y = ht.array(x)
w = ht.array(v)
# Split the data along different axes
y0 = ht.array(x, split=0)
y1 = ht.array(x, split=1)
# Convert data, labels, and centers from heat tensors to numpy arrays
# larray
y_as_np = y0.resplit_(None).larray.cpu().numpy()
# output the shape
y_shape0 = y0.shape
# print the number of features in each data point
n_features = y0.shape[1]
# calculate Euclidean distance between each
# row-vector in y and w
# !!! Important !!!
# ---> the arguments of cdist must be 2D tensors, i.e., ht.array([[1,2,3]]) instead of ht.array([1,2,3])
dist = ht.spatial.distance.cdist(y, w)
# pick the minimum value of a tensor along the axis=1
min_dist = dist.min(axis=0)
# define a tensor with the same dimension as y and fill it with zeros
centroids = ht.zeros((y.shape[0], y.shape[1]))
# replace the 0th row vector of "centroids" by a randomly chosen row vector of y
sample = ht.random.randint(0, y.shape[0] - 1).item()
centroids[0, :] = y[sample]
# Useful for degubbing: keep track auf matrix shapes and the process (i.e., the gpu) the data is assigned to
print(f"centroids.shape{centroids.shape}, process= {ht.MPI_WORLD.rank}\n")
# stack two vectors together
# a=ht.array([1,2,3,4])
# b=ht.array([10,20,30,40])
# a=ht.array(2)
# b=ht.array(3)
# stacked_ab=ht.stack((a,b),axis=0)
# add dimensions
a_vector = ht.array([1, 2, 3, 4])
new_x = ht.expand_dims(a_vector, axis=0)  # output: [[1,2,3,4]]
# stack two vectors together and flatten, so that the outcome is similar to the command "append"
a = ht.array([[1, 2, 3, 4], [1, 5, 3, 4], [1, 2, 3, 42]])
# b=ht.array([[10,20,30,40],[10,20,30,40],[1,2,3,4]])
# stacked_ab=ht.stack((a,b),axis=0)
# reshaped_stacked_ab=ht.reshape(stacked_ab,(stacked_ab.shape[0]*stacked_ab.shape[1],stacked_ab.shape[2]))
b = ht.array([[10, 20, 30, 40], [10, 20, 30, 40]])
stacked_ab = ht.row_stack((a, b))
# create random numbers between 0 and 1
random = ht.random.rand(y.shape[0])
# translate into a uniform probability distribution
random_prob = random / random.sum()
# find the indices for which the condition test1<test holds (is to be understood elementwise)
test = ht.array([0.3, 0.5, 0.8])
test1 = ht.array([0.2, 0.6, 0.4])
find_indices = ht.where(test1 < test)
# find the largest value in a vector
some_vector = np.array([1, 2, 4, 4])
some_vector_max = (
    some_vector.max()
)  # when dealing with ht.array one should add an .item() at the end, to ensure that the dndarray or torch tensor is transformed to a scalar
weights = torch.tensor(np.array([np.sum(some_vector == i) for i in range(0, some_vector.shape[0])]))
"""     # ensure that all processes have the same data
if ht.MPI_WORLD.rank == 0:
    weights=weights
else:
# tensor with zeros that has the same size as reclustered centroids, in order to to allocate memory (necessary for broadcast)
    weights = torch.zeros(
        (weights.shape[0], weights.shape[1]), dtype=x.dtype.torch_type(), device=centroids.device)
    ht.MPI_WORLD.Bcast(
        weights, root=0) """

from batchparallelclustering import _initialize_plus_plus, BatchParallelKMeans

""" X = torch.rand(100, 3)
W = torch.tensor(w.larray)
"""
""" n_clusters=3
BPK=BatchParallelKMeans(n_clusters) """
from heat.utils.data.spherical import create_spherical_dataset

""" data = create_spherical_dataset(
            num_samples_cluster=100, radius=1.0, offset=4.0, dtype=ht.float32, random_state=1
        )
data=ht.array(data,split=0) """

import matplotlib.pyplot as plt
import numpy as np

"""
def plot_clusters(data, labels, centers, title="Clustering Visualization"):
    # Visualizes clustered data in 2D or 3D.
    # Parameters:
    # - data (numpy.ndarray): Input data of shape (n_samples, n_features).
    # - labels (numpy.ndarray): Cluster labels for each point (optional).
    # - centers (numpy.ndarray): Coordinates of cluster centers (optional).
    # - title (str): Title of the plot.
    # Determine dimensionality
    dim = data.shape[1]
    if dim not in [2, 3]:
        raise ValueError("Data must be 2D or 3D for plotting.")
    # Set up plot
    fig = plt.figure(figsize=(8, 8))
    if dim == 2:
        ax = fig.add_subplot(111)
    else:
        ax = fig.add_subplot(111, projection="3d")
    unique_labels = np.unique(labels)
    # Loop through unique labels (clusters)
    for i in unique_labels:
        cluster_data = data[labels == i]  # Get all data points for the current label
        if dim == 2:
            ax.scatter(cluster_data[:, 0], cluster_data[:, 1], label=f"Cluster {i}")
        else:
            ax.scatter(
                cluster_data[:, 0], cluster_data[:, 1], cluster_data[:, 2], label=f"Cluster {i}"
            )
    # Plot cluster centers if provided
    if dim == 2:
        ax.scatter(centers[:, 0], centers[:, 1], c="red", marker="x", s=200, label="Centers")
    else:
        ax.scatter(
            centers[:, 0],
            centers[:, 1],
            centers[:, 2],
            c="red",
            marker="x",
            s=200,
            label="Centers",
        )
    # Add labels and legend
    ax.set_title(title)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    if dim == 3:
        ax.set_zlabel("Feature 3")
    ax.legend()
    plt.savefig("plot.pdf")
    plt.show() """


# Example usage:
# Assuming you have your data, labels, and centers in numpy arrays
print("Start plotting \n\n\n")
data = ht.utils.data.spherical.create_spherical_dataset(
    num_samples_cluster=20000000, radius=2.0, offset=10.0, dtype=ht.float32, random_state=1
)
# data = data[:, :-1]

start_time = time.time()
kmeans = ht.cluster.KMeans(n_clusters=4, init="kmeans++", max_iter=400)
# kmeans = ht.cluster.KMedians(n_clusters=5, init="kmedians++", max_iter=400)
kmeans.fit(data, oversampling=10, iter_multiplier=1)
end_time = time.time()

# Laufzeit berechnen
print(f"Runtime for clustering: {end_time - start_time:.4f} Sekunden")

labels = kmeans._labels
labels = ht.reshape(labels, labels.shape[0])
centers = kmeans._cluster_centers
# Convert data, labels, and centers from heat tensors to numpy arrays
data = data.numpy()
# data = data.resplit_(None).larray.cpu().numpy()
labels = labels.resplit_(None).larray.cpu().numpy()
centers = centers.resplit_(None).larray.cpu().numpy()
# print("centroids= ", centers)
# Call the plot function
# plot_clusters(data, labels, centers)
