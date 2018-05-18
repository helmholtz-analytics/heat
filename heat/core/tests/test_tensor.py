
import heat as ht

from numpy.testing import assert_equal

import os
current_path = os.path.dirname(os.path.abspath(__file__))


def test_load_tensor():
    array = ht.tensor()
    array.load(os.path.join(current_path, "../../datasets/data/iris.h5"), "data")


def test_tensor_gshape():
    array = ht.tensor()
    array.load(os.path.join(current_path, "../../datasets/data/iris.h5"), "data")
    assert_equal(True, array.gshape == (150, 4))
