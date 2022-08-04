import demo_knn
import json
import timeit

def test_knn(results_bag):
    start = timeit.default_timer()
    #print("Accuracy: {}".format(verify_algorithm(X, Y, 1, 30, 5, 1)))
    result = "{}".format(demo_knn.verify_algorithm(demo_knn.X, demo_knn.Y, 1, 30, 5, 1))
    stop = timeit.default_timer()
    execution_time = stop - start
    results_bag.result = result
    results_bag.execution_time = execution_time

def test_synthesis(fixture_store):
    print("\n   Contents of `fixture_store['results_bag']`:")
    print(fixture_store['results_bag'])
    json_object = json.dumps(fixture_store['results_bag'], indent = 4) 
    print(json_object)
    with open("results.json", "w") as outfile:
        outfile.write(json_object)
