
import heat as ht

from numpy.testing import assert_equal

import os
current_path = os.path.dirname(os.path.abspath(__file__))


def test_kmeans():
    array = ht.tensor()
    array.load(os.path.join(current_path, "../../../datasets/data/iris.h5"), "data")

    kmeans = ht.ml.cluster.KMeans(n_clusters=3, max_iter=1000, tol=1e-4)
    mean, std, normalized_data = kmeans.standardize(array)
    centroids = kmeans.fit(normalized_data)
    # TODO: .all operator needed
