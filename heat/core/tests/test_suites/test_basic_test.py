import unittest
import heat as ht
import numpy as np

from heat.core.tests.test_suites.basic_test import BasicTest


class TestBasicTest(BasicTest):

    def test_compare_results(self):
        size = ht.MPI.COMM_WORLD.size
        rank = ht.MPI.COMM_WORLD.rank
        print('rank', rank)
        heat_array = ht.ones((size, 10, 10), dtype=ht.float32, split=0)
        np_array = np.ones((size, 10, 10), dtype=np.float32)
        self.compare_results(heat_array, np_array)
        self.fail()
