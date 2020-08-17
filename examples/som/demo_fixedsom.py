import sys
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Fix python Path if run from terminal
curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(curdir, "../../")))

import heat as ht
from heat.som.som import FixedSOM
from heat.cluster.kmeans import KMeans

# Load Dataset from hdf5 file
X = ht.load_hdf5("../../heat/datasets/data/iris.h5", dataset="data", split=0)

# Generate Keys for the iris.h5 dataset
keys = []
for i in range(50):
    keys.append(0)
for i in range(50, 100):
    keys.append(1)
for i in range(100, 150):
    keys.append(2)
Y = ht.array(keys, split=0)

som = FixedSOM(
    10,
    10,
    4,
    initial_learning_rate=0.1,
    target_learning_rate=0.01,
    initial_radius=8,
    target_radius=2,
    max_epoch=400,
    batch_size=5,
    seed=1,
    data_split=0,
)
som.fit_batch(X, 1)

print(som.umatrix(), "batch")

dist = ht.spatial.cdist(X, som.network)
row_min = ht.argmin(dist, axis=1)
row_min_2d = som.get_2d_network(row_min)
print(row_min_2d, Y, "labels into batch")

som = FixedSOM(
    10,
    10,
    4,
    initial_learning_rate=0.1,
    target_learning_rate=0.01,
    initial_radius=8,
    target_radius=2,
    max_epoch=400,
    batch_size=5,
    seed=1,
    data_split=0,
)
som.fit_iterative(X)

print(som.umatrix(), "iterative")

dist = ht.spatial.cdist(X, som.network)
row_min = ht.argmin(dist, axis=1)
row_min_2d = som.get_2d_network(row_min)

print(row_min_2d, Y, "labels into batch")
