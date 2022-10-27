from demo_knn import verify_algorithm
from datetime import datetime
import json
import timeit

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


def test_knn(results_bag):
    start = timeit.default_timer()
    # print("Accuracy: {}".format(verify_algorithm(X, Y, 1, 30, 5, 1)))
    result = "{}".format(verify_algorithm(X, Y, 1, 30, 5, 1))
    stop = timeit.default_timer()
    execution_time = stop - start
    results_bag.result = result
    results_bag.execution_time = execution_time


def test_synthesis(fixture_store):
    print("\n   Contents of `fixture_store['results_bag']`:")
    print(fixture_store["results_bag"])
    json_object = json.dumps(fixture_store["results_bag"], indent=4)
    print(json_object)
    with open("results.json", "w") as outfile:
        outfile.write(json_object)
