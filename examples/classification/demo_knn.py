import sys
import os
import random

import heat as ht
from heat.classification.kneighborsclassifier import KNeighborsClassifier

import pkg_resources

# Load dataset from hdf5 file
iris_path = pkg_resources.resource_filename(
    pkg_resources.Requirement.parse("heat"), "heat/datasets/iris.h5"
)

X = ht.load_hdf5(iris_path, dataset="data", split=0)

# Generate keys for the iris.h5 dataset
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
    Calculates the accuracy of classification/clustering-algorithms.
    Note this only works with integer/discrete classes. For algorithms that give approximations an error function is
    required.

    Parameters
    ----------
    new_y : ht.tensor of shape (n_samples, n_features), required
        The new labels that where generated
    verification_y : ht.tensor of shape (n_samples, n_features), required
        Known labels

    Returns
    ----------
    float
        the accuracy, number of properly labeled samples divided by amount of labels.
    """

    if new_y.gshape != verification_y.gshape:
        raise ValueError(
            "Expecting results of same length, got {}, {}".format(
                new_y.gshape, verification_y.gshape
            )
        )

    count = ht.sum(ht.where(new_y == verification_y, 1, 0))

    return count / new_y.gshape[0]


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

    fold_x = dataset_x[data_indices]
    fold_y = dataset_y[data_indices]
    verification_y = dataset_y[verification_indices]
    verification_x = dataset_x[verification_indices]

    # Balance arrays
    fold_x.balance_()
    fold_y.balance_()
    verification_y.balance_()
    verification_x.balance_()

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
        the number of vectors used by the KNeighborsClassifier-Algorithm
    k : int
        The number of neighbours for KNeighborsClassifier-Algorithm
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
        classifier = KNeighborsClassifier(k)
        classifier.fit(fold_x, fold_y)
        result_y = classifier.predict(verification_x)
        accuracies.append(calculate_accuracy(result_y, verification_y).item())
    return accuracies


print("Accuracy: {}".format(verify_algorithm(X, Y, 1, 30, 5, 1)))
