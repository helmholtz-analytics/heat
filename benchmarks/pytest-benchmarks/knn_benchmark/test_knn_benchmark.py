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

def test_knn(results_bag):
    print("Accuracy: {}".format(verify_algorithm(X, Y, 1, 30, 5, 1)))
    result = "Accuracy: {}".format(verify_algorithm(X, Y, 1, 30, 5, 1))
    results_bag.result = result
    results_bag.current_time = datetime.now().isoformat()

def test_synthesis(fixture_store):
    print("\n   Contents of `fixture_store['results_bag']`:")
    print(fixture_store['results_bag'])
    json_object = json.dumps(fixture_store['results_bag'], indent = 4) 
    print(json_object)
    with open("results.json", 'r+') as outfile:
        file_data = json.load(outfile)
        file_data.append(json_object)
        outfile.seek(0)
        json.dump(file_data, outfile, indent = 4)

