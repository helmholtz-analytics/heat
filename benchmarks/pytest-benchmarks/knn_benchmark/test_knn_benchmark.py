from demo_knn import verify_algorithm
from datetime import datetime
import json

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

def test_knn(benchmark):
    benchmark(verify_algorithm,X, Y, 1, 30, 5, 1)

