import sys
import os
import random

# Fix python Path if run from terminal
curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(curdir, "../../")))

import heat as ht
from heat.som.som import FixedSOM
from heat.classification.knn import KNN

X = ht.load_hdf5("../../heat/datasets/data/iris.h5", dataset="data", split=0)

keys = []
for i in range(50):
    keys.append(0)
for i in range(50, 100):
    keys.append(1)
for i in range(100, 150):
    keys.append(2)
Y = ht.array(keys, split=0)


def create_fold(dataset_x, dataset_y, size, seed=None):
    """
    Randomly splits the dataset into two parts for cross-validation.

    Parameters
    ----------
    dataset_x : ht.DNDarray
        data vectors, required
    dataset_y : ht.DNDarray
        labels for dataset_x, required
    size : int
        the size of the split to create
    seed: int, optional
        seed for the random generator, allows deterministic testing

    Returns
    ----------
    fold_x : ht.DNDarray
        DNDarray of shape (size,) containing data vectors from dataset_x
    fold_y : ht.DNDarray
        DNDarray of shape(size,) containing labels from dataset_y
    verification_x : ht.DNDarray
        DNDarray of shape(len(dataset_x - size),) containing all items from dataset_x not in fold_x
    verification_y : ht.DNDarray
        DNDarray of shape(len(dataset_y - size),) containing all items from dataset_y not in fold_y
    """
    assert len(dataset_y) == len(dataset_x)
    assert size < len(dataset_x)

    data_length = len(dataset_x)

    if seed:
        random.seed(seed)
    indices = [i for i in range(data_length)]
    random.shuffle(indices)
    data_indices = ht.array(indices[0:size], split=0)
    verification_indices = ht.array(indices[size:], split=0)

    fold_x = ht.array(dataset_x[data_indices], is_split=0)
    fold_y = ht.array(dataset_y[data_indices], is_split=0)
    verification_y = ht.array(dataset_y[verification_indices], is_split=0)
    verification_x = ht.array(dataset_x[verification_indices], is_split=0)

    # Balance arrays
    fold_x.balance_()
    fold_y.balance_()
    verification_y.balance_()
    verification_x.balance_()

    return fold_x, fold_y, verification_x, verification_y


def get_bmu(som, x):
    distances = ht.spatial.cdist(x, som.network)
    min_dist = ht.argmin(distances, axis=1)

    new_x = som.network_indices[min_dist.flatten()]
    new_x.balance_()

    return new_x


def test_net(som, x, y, split_number, split_size, seed=None):
    accuracies = []
    for split in range(split_number):
        fold_x, fold_y, verification_x, verification_y = create_fold(x, y, split_size, seed)

        new_x = get_bmu(som, fold_x)
        verification_x = get_bmu(som, verification_x)

        knn = KNN(new_x, fold_y, 5)
        result = knn.predict(verification_x)
        accuracies.append(
            (ht.sum(ht.where(result == verification_y, 1, 0)) / verification_y.shape[0]).item()
        )
    return accuracies


som = FixedSOM(
    10,
    10,
    4,
    initial_learning_rate=0.1,
    target_learning_rate=0.01,
    initial_radius=8,
    target_radius=1,
    max_epoch=200,
    batch_size=25,
    seed=1,
    data_split=0,
)

som.fit_batch(X, 1)

random.seed(2)
print(test_net(som, X, Y, 10, 30))

som = FixedSOM(
    10,
    10,
    4,
    initial_learning_rate=0.1,
    target_learning_rate=0.01,
    initial_radius=8,
    target_radius=1,
    max_epoch=200,
    batch_size=25,
    seed=1,
    data_split=0,
)

som.fit_iterative(X)

random.seed(2)
print(test_net(som, X, Y, 10, 30))
