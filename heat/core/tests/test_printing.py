import math

import heat as ht
from .test_suites.basic_test import TestCase


class TestPrinting(TestCase):
    def setUp(self):
        # rmove to CPU only for the print testing
        ht.use_device("cpu")

    def tearDown(self):
        # reset the print options back to default after each test run
        ht.set_printoptions(profile="default")
        # reset the default device
        ht.use_device(self.device)

    def test_get_default_options(self):
        print_options = ht.get_printoptions()
        comparison = {
            "precision": 4,
            "threshold": 1000,
            "edgeitems": 3,
            "linewidth": 120,
            "sci_mode": None,
        }

        self.assertIsInstance(print_options, dict)
        for key, value in print_options.items():
            self.assertEqual(value, comparison[key])

    def test_set_get_short_options(self):
        ht.set_printoptions(profile="short")

        print_options = ht.get_printoptions()
        comparison = {
            "precision": 2,
            "threshold": 1000,
            "edgeitems": 2,
            "linewidth": 120,
            "sci_mode": None,
        }

        self.assertIsInstance(print_options, dict)
        for key, value in print_options.items():
            self.assertEqual(value, comparison[key])

    def test_set_get_full_options(self):
        ht.set_printoptions(profile="full")

        print_options = ht.get_printoptions()
        comparison = {
            "precision": 4,
            "threshold": math.inf,
            "edgeitems": 3,
            "linewidth": 120,
            "sci_mode": None,
        }

        self.assertIsInstance(print_options, dict)
        for key, value in print_options.items():
            self.assertEqual(value, comparison[key])

    def test_set_get_precision(self):
        ht.set_printoptions(precision=6)
        self.assertEqual(6, ht.get_printoptions()["precision"])

    def test_set_get_threshold(self):
        ht.set_printoptions(threshold=7)
        self.assertEqual(7, ht.get_printoptions()["threshold"])

    def test_set_get_edgeitems(self):
        ht.set_printoptions(edgeitems=8)
        self.assertEqual(8, ht.get_printoptions()["edgeitems"])

    def test_set_get_linewidth(self):
        ht.set_printoptions(linewidth=9)
        self.assertEqual(9, ht.get_printoptions()["linewidth"])

    def test_set_get_sci_mode(self):
        ht.set_printoptions(sci_mode=True)
        self.assertEqual(True, ht.get_printoptions()["sci_mode"])

    def test_empty(self):
        tensor = ht.array([], dtype=ht.int64)
        __repr = "DNDarray([], dtype=ht.int64, device=cpu:0, split=None)"

        self.assertEqual(repr(tensor), __repr)

    def test_scalar(self):
        self.assertTrue(False)

    def test_unsplit_below_threshold(self):
        tensor = ht.arange(2 * 3 * 4).reshape((2, 3, 4))
        __repr = (
            "DNDarray([[[ 0,  1,  2,  3],\n"
            "           [ 4,  5,  6,  7],\n"
            "           [ 8,  9, 10, 11]],\n"
            "\n"
            "          [[12, 13, 14, 15],\n"
            "           [16, 17, 18, 19],\n"
            "           [20, 21, 22, 23]]], dtype=ht.int32, device=cpu:0, split=None)"
        )

        self.assertEqual(repr(tensor), __repr)

    def test_unsplit_above_threshold(self):
        tensor = ht.arange(12 * 13 * 14).reshape((12, 13, 14))
        __repr = (
            "DNDarray([[[   0,    1,    2,  ...,   11,   12,   13],\n"
            "           [  14,   15,   16,  ...,   25,   26,   27],\n"
            "           [  28,   29,   30,  ...,   39,   40,   41],\n"
            "           ...,\n"
            "           [ 140,  141,  142,  ...,  151,  152,  153],\n"
            "           [ 154,  155,  156,  ...,  165,  166,  167],\n"
            "           [ 168,  169,  170,  ...,  179,  180,  181]],\n"
            "\n"
            "          [[ 182,  183,  184,  ...,  193,  194,  195],\n"
            "           [ 196,  197,  198,  ...,  207,  208,  209],\n"
            "           [ 210,  211,  212,  ...,  221,  222,  223],\n"
            "           ...,\n"
            "           [ 322,  323,  324,  ...,  333,  334,  335],\n"
            "           [ 336,  337,  338,  ...,  347,  348,  349],\n"
            "           [ 350,  351,  352,  ...,  361,  362,  363]],\n"
            "\n"
            "          [[ 364,  365,  366,  ...,  375,  376,  377],\n"
            "           [ 378,  379,  380,  ...,  389,  390,  391],\n"
            "           [ 392,  393,  394,  ...,  403,  404,  405],\n"
            "           ...,\n"
            "           [ 504,  505,  506,  ...,  515,  516,  517],\n"
            "           [ 518,  519,  520,  ...,  529,  530,  531],\n"
            "           [ 532,  533,  534,  ...,  543,  544,  545]],\n"
            "\n"
            "          ...,\n"
            "\n"
            "          [[1638, 1639, 1640,  ..., 1649, 1650, 1651],\n"
            "           [1652, 1653, 1654,  ..., 1663, 1664, 1665],\n"
            "           [1666, 1667, 1668,  ..., 1677, 1678, 1679],\n"
            "           ...,\n"
            "           [1778, 1779, 1780,  ..., 1789, 1790, 1791],\n"
            "           [1792, 1793, 1794,  ..., 1803, 1804, 1805],\n"
            "           [1806, 1807, 1808,  ..., 1817, 1818, 1819]],\n"
            "\n"
            "          [[1820, 1821, 1822,  ..., 1831, 1832, 1833],\n"
            "           [1834, 1835, 1836,  ..., 1845, 1846, 1847],\n"
            "           [1848, 1849, 1850,  ..., 1859, 1860, 1861],\n"
            "           ...,\n"
            "           [1960, 1961, 1962,  ..., 1971, 1972, 1973],\n"
            "           [1974, 1975, 1976,  ..., 1985, 1986, 1987],\n"
            "           [1988, 1989, 1990,  ..., 1999, 2000, 2001]],\n"
            "\n"
            "          [[2002, 2003, 2004,  ..., 2013, 2014, 2015],\n"
            "           [2016, 2017, 2018,  ..., 2027, 2028, 2029],\n"
            "           [2030, 2031, 2032,  ..., 2041, 2042, 2043],\n"
            "           ...,\n"
            "           [2142, 2143, 2144,  ..., 2153, 2154, 2155],\n"
            "           [2156, 2157, 2158,  ..., 2167, 2168, 2169],\n"
            "           [2170, 2171, 2172,  ..., 2181, 2182, 2183]]], dtype=ht.int32, device=cpu:0, split=None)"
        )

        self.assertEqual(repr(tensor), __repr)

    def test_split_0_below_threshold(self):
        ht.set_printoptions(precision=2)
        tensor = ht.arange(0.5, 2 * 3 * 4 + 0.5, split=0).reshape((2, 3, 4))
        string = (
            "DNDarray([[[ 0.50,  1.50,  2.50,  3.50],\n"
            "           [ 4.50,  5.50,  6.50,  7.50],\n"
            "           [ 8.50,  9.50, 10.50, 11.50]],\n"
            "\n"
            "          [[12.50, 13.50, 14.50, 15.50],\n"
            "           [16.50, 17.50, 18.50, 19.50],\n"
            "           [20.50, 21.50, 22.50, 23.50]]], dtype=ht.float32, device=cpu:0, split=0)"
        )

        self.assertEqual(repr(tensor), string)

    def test_split_0_above_threshold(self):
        self.assertTrue(False)

    def test_split_1_below_threshold(self):
        self.assertTrue(False)

    def test_split_1_above_threshold(self):
        self.assertTrue(False)

    def test_split_2_below_threshold(self):
        self.assertTrue(False)

    def test_split_2_above_threshold(self):
        self.assertTrue(False)
