import sys
import os

# Fix python Path if run from terminal
curdir = os.path.dirname(os.path.abspath(__file__)) 
sys.path.insert(0, os.path.abspath(os.path.join(curdir, '../../'))) 

import heat as ht
from heat.classification.knn import KNN

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


def calculate_accuracy(new_y, verification_y):
    """
    Calculates the accuracy of classification/clustering-Algorithms.
    Note this only works with integer/discrete classes. For Algorithms that give Approximations an error function is
    required.
    ----------
    new_y : ht.tensor of shape (n_samples,), required
        The new labels that are generated
    verification_y : ht.tensor of shape (n_samples,), required
        Known labels
    Returns
    ----------
    float
        the accuracy, number of properly labeled samples divided by amount of labels.
    """
    assert len(new_y) == len(verification_y)
    length = len(new_y)
    count = 0
    for index in range(length):
        if new_y[index] == verification_y[index].item():
            count += 1
    return count / length


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
        ht.random.seed(seed)
    indices = ht.random.randint(low=0, high=data_length - 1, size=(size,), split=0)
    indices = ht.unique(indices, sorted=True)
    while len(indices) < size:
        diff = size - len(indices)
        additional = ht.random.randint(low=0, high=data_length, size=(diff,), split=0)
        indices = ht.concatenate((indices, additional))
        indices = ht.unique(indices, sorted=True)

    indices = [ind.item() for ind in indices]
    all_indices = [index for index in range(data_length)]
    verification_indices = [index for index in all_indices if index not in indices]

    fold_x = dataset_x[indices]
    fold_y = dataset_y[indices]
    verification_y = dataset_y[verification_indices]
    verification_x = dataset_x[verification_indices]

    return fold_x, fold_y, verification_x, verification_y


def verify_algorithm(x, y, split_number, split_size, k, seed=None):
    """
    Parameters
    ----------
    x : ht.DNDarray
        array containing data vectors
    y : ht.DNDarray
        array containing the labels for x (must be in same order)
    split_number: int
        the number of test iterations
    split_size : int
        the number of vectors used by the KNN-Algorithm
    k : int
        The number of neighbours for KNN-Algorithm
    seed : int
        Seed for the random generator used in creating folds. Used for deterministic testing purposes.
    Returns
    -------
    accuracies : ht.DNDarray
        array of shape (split_number,) containing the accuracy per run
    """
    assert len(x) == len(y)
    assert split_size < len(x)
    assert k < len(x)

    accuracies = []

    for split_index in range(split_number):
        fold_x, fold_y, verification_x, verification_y = create_fold(x, y, split_size, seed)
        network = KNN(fold_x, fold_y, k)
        result_y = []
        for item in verification_x:
            label = network.assign_label(ht.copy(item))
            result_y.append(label)
        accuracies.append(calculate_accuracy(result_y, verification_y))
    return accuracies


print(verify_algorithm(X, Y, 5, 30, 10))
