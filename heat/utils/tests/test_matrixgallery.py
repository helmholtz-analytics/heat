import unittest

import heat as ht

from heat.core.tests.test_suites.basic_test import TestCase


class TestMatrixgallery(TestCase):
    def __check_parter(self, parter):
        self.assertEqual(parter.shape, (20, 20))
        # TODO: check for singular values of the parter matrix

    def test_parter(self):
        parter = ht.utils.matrixgallery.parter(20)
        self.__check_parter(parter)

        parters0 = ht.utils.matrixgallery.parter(20, split=0, comm=ht.MPI_WORLD)
        self.__check_parter(parters0)

        parters1 = ht.utils.matrixgallery.parter(20, split=1, comm=ht.MPI_WORLD)
        self.__check_parter(parters1)

        with self.assertRaises(ValueError):
            ht.utils.matrixgallery.parter(20, split=2, comm=ht.MPI_WORLD)
