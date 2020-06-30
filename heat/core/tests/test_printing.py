import math

import heat as ht
from .test_suites.basic_test import TestCase


class TestPrinting(TestCase):
    def setUp(self):
        # move to CPU only for the testing printing, otherwise the compare string will become messy
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
        comparison = "DNDarray([], dtype=ht.int64, device=cpu:0, split=None)"
        __str = str(tensor)

        if tensor.comm.rank == 0:
            self.assertEqual(comparison, __str)

    def test_scalar(self):
        tensor = ht.array(42)
        comparison = "DNDarray(42, dtype=ht.int64, device=cpu:0, split=None)"
        __str = str(tensor)

        if tensor.comm.rank == 0:
            self.assertEqual(comparison, __str)

    def test_unsplit_below_threshold(self):
        dndarray = ht.arange(2 * 3 * 4).reshape((2, 3, 4))
        comparison = (
            "DNDarray([[[ 0,  1,  2,  3],\n"
            "           [ 4,  5,  6,  7],\n"
            "           [ 8,  9, 10, 11]],\n"
            "\n"
            "          [[12, 13, 14, 15],\n"
            "           [16, 17, 18, 19],\n"
            "           [20, 21, 22, 23]]], dtype=ht.int32, device=cpu:0, split=None)"
        )
        __str = str(dndarray)

        if dndarray.comm.rank == 0:
            self.assertEqual(comparison, __str)

    def test_unsplit_above_threshold(self):
        dndarray = ht.arange(12 * 13 * 14).reshape((12, 13, 14))
        comparison = (
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
        __str = str(dndarray)

        if dndarray.comm.rank == 0:
            self.assertEqual(comparison, __str)

    def test_split_0_below_threshold(self):
        ht.set_printoptions(precision=2)
        dndarray = ht.arange(0.5, 2 * 3 * 4 + 0.5, split=0).reshape((2, 3, 4))
        comparison = (
            "DNDarray([[[ 0.50,  1.50,  2.50,  3.50],\n"
            "           [ 4.50,  5.50,  6.50,  7.50],\n"
            "           [ 8.50,  9.50, 10.50, 11.50]],\n"
            "\n"
            "          [[12.50, 13.50, 14.50, 15.50],\n"
            "           [16.50, 17.50, 18.50, 19.50],\n"
            "           [20.50, 21.50, 22.50, 23.50]]], dtype=ht.float32, device=cpu:0, split=0)"
        )
        __str = str(dndarray)

        if dndarray.comm.rank == 0:
            self.assertEqual(comparison, __str)

    def test_split_0_above_threshold(self):
        ht.set_printoptions(precision=1)
        dndarray = ht.arange(0.2, 10 * 11 * 12 + 0.2).reshape((10, 11, 12)).resplit_(0)
        self.maxDiff = None
        comparison = (
            "DNDarray([[[2.0e-01, 1.2e+00, 2.2e+00,  ..., 9.2e+00, 1.0e+01, 1.1e+01],\n"
            "           [1.2e+01, 1.3e+01, 1.4e+01,  ..., 2.1e+01, 2.2e+01, 2.3e+01],\n"
            "           [2.4e+01, 2.5e+01, 2.6e+01,  ..., 3.3e+01, 3.4e+01, 3.5e+01],\n"
            "           ...,\n"
            "           [9.6e+01, 9.7e+01, 9.8e+01,  ..., 1.1e+02, 1.1e+02, 1.1e+02],\n"
            "           [1.1e+02, 1.1e+02, 1.1e+02,  ..., 1.2e+02, 1.2e+02, 1.2e+02],\n"
            "           [1.2e+02, 1.2e+02, 1.2e+02,  ..., 1.3e+02, 1.3e+02, 1.3e+02]],\n"
            "\n"
            "          [[1.3e+02, 1.3e+02, 1.3e+02,  ..., 1.4e+02, 1.4e+02, 1.4e+02],\n"
            "           [1.4e+02, 1.5e+02, 1.5e+02,  ..., 1.5e+02, 1.5e+02, 1.6e+02],\n"
            "           [1.6e+02, 1.6e+02, 1.6e+02,  ..., 1.7e+02, 1.7e+02, 1.7e+02],\n"
            "           ...,\n"
            "           [2.3e+02, 2.3e+02, 2.3e+02,  ..., 2.4e+02, 2.4e+02, 2.4e+02],\n"
            "           [2.4e+02, 2.4e+02, 2.4e+02,  ..., 2.5e+02, 2.5e+02, 2.5e+02],\n"
            "           [2.5e+02, 2.5e+02, 2.5e+02,  ..., 2.6e+02, 2.6e+02, 2.6e+02]],\n"
            "\n"
            "          [[2.6e+02, 2.7e+02, 2.7e+02,  ..., 2.7e+02, 2.7e+02, 2.8e+02],\n"
            "           [2.8e+02, 2.8e+02, 2.8e+02,  ..., 2.9e+02, 2.9e+02, 2.9e+02],\n"
            "           [2.9e+02, 2.9e+02, 2.9e+02,  ..., 3.0e+02, 3.0e+02, 3.0e+02],\n"
            "           ...,\n"
            "           [3.6e+02, 3.6e+02, 3.6e+02,  ..., 3.7e+02, 3.7e+02, 3.7e+02],\n"
            "           [3.7e+02, 3.7e+02, 3.7e+02,  ..., 3.8e+02, 3.8e+02, 3.8e+02],\n"
            "           [3.8e+02, 3.9e+02, 3.9e+02,  ..., 3.9e+02, 3.9e+02, 4.0e+02]],\n"
            "\n"
            "          ...,\n"
            "\n"
            "          [[9.2e+02, 9.3e+02, 9.3e+02,  ..., 9.3e+02, 9.3e+02, 9.4e+02],\n"
            "           [9.4e+02, 9.4e+02, 9.4e+02,  ..., 9.5e+02, 9.5e+02, 9.5e+02],\n"
            "           [9.5e+02, 9.5e+02, 9.5e+02,  ..., 9.6e+02, 9.6e+02, 9.6e+02],\n"
            "           ...,\n"
            "           [1.0e+03, 1.0e+03, 1.0e+03,  ..., 1.0e+03, 1.0e+03, 1.0e+03],\n"
            "           [1.0e+03, 1.0e+03, 1.0e+03,  ..., 1.0e+03, 1.0e+03, 1.0e+03],\n"
            "           [1.0e+03, 1.0e+03, 1.0e+03,  ..., 1.1e+03, 1.1e+03, 1.1e+03]],\n"
            "\n"
            "          [[1.1e+03, 1.1e+03, 1.1e+03,  ..., 1.1e+03, 1.1e+03, 1.1e+03],\n"
            "           [1.1e+03, 1.1e+03, 1.1e+03,  ..., 1.1e+03, 1.1e+03, 1.1e+03],\n"
            "           [1.1e+03, 1.1e+03, 1.1e+03,  ..., 1.1e+03, 1.1e+03, 1.1e+03],\n"
            "           ...,\n"
            "           [1.2e+03, 1.2e+03, 1.2e+03,  ..., 1.2e+03, 1.2e+03, 1.2e+03],\n"
            "           [1.2e+03, 1.2e+03, 1.2e+03,  ..., 1.2e+03, 1.2e+03, 1.2e+03],\n"
            "           [1.2e+03, 1.2e+03, 1.2e+03,  ..., 1.2e+03, 1.2e+03, 1.2e+03]],\n"
            "\n"
            "          [[1.2e+03, 1.2e+03, 1.2e+03,  ..., 1.2e+03, 1.2e+03, 1.2e+03],\n"
            "           [1.2e+03, 1.2e+03, 1.2e+03,  ..., 1.2e+03, 1.2e+03, 1.2e+03],\n"
            "           [1.2e+03, 1.2e+03, 1.2e+03,  ..., 1.2e+03, 1.2e+03, 1.2e+03],\n"
            "           ...,\n"
            "           [1.3e+03, 1.3e+03, 1.3e+03,  ..., 1.3e+03, 1.3e+03, 1.3e+03],\n"
            "           [1.3e+03, 1.3e+03, 1.3e+03,  ..., 1.3e+03, 1.3e+03, 1.3e+03],\n"
            "           [1.3e+03, 1.3e+03, 1.3e+03,  ..., 1.3e+03, 1.3e+03, 1.3e+03]]], dtype=ht.float32, device=cpu:0, split=0)"
        )

        __str = str(dndarray)

        if dndarray.comm.rank == 0:
            self.assertEqual(comparison, __str)

    def test_split_1_below_threshold(self):
        ht.set_printoptions(sci_mode=True)
        dndarray = ht.arange(0.5, 4 * 5 * 6 + 0.5, dtype=ht.float64).reshape((4, 5, 6)).resplit_(1)
        comparison = (
            "DNDarray([[[5.0000e-01, 1.5000e+00, 2.5000e+00, 3.5000e+00, 4.5000e+00, 5.5000e+00],\n"
            "           [6.5000e+00, 7.5000e+00, 8.5000e+00, 9.5000e+00, 1.0500e+01, 1.1500e+01],\n"
            "           [1.2500e+01, 1.3500e+01, 1.4500e+01, 1.5500e+01, 1.6500e+01, 1.7500e+01],\n"
            "           [1.8500e+01, 1.9500e+01, 2.0500e+01, 2.1500e+01, 2.2500e+01, 2.3500e+01],\n"
            "           [2.4500e+01, 2.5500e+01, 2.6500e+01, 2.7500e+01, 2.8500e+01, 2.9500e+01]],\n"
            "\n"
            "          [[3.0500e+01, 3.1500e+01, 3.2500e+01, 3.3500e+01, 3.4500e+01, 3.5500e+01],\n"
            "           [3.6500e+01, 3.7500e+01, 3.8500e+01, 3.9500e+01, 4.0500e+01, 4.1500e+01],\n"
            "           [4.2500e+01, 4.3500e+01, 4.4500e+01, 4.5500e+01, 4.6500e+01, 4.7500e+01],\n"
            "           [4.8500e+01, 4.9500e+01, 5.0500e+01, 5.1500e+01, 5.2500e+01, 5.3500e+01],\n"
            "           [5.4500e+01, 5.5500e+01, 5.6500e+01, 5.7500e+01, 5.8500e+01, 5.9500e+01]],\n"
            "\n"
            "          [[6.0500e+01, 6.1500e+01, 6.2500e+01, 6.3500e+01, 6.4500e+01, 6.5500e+01],\n"
            "           [6.6500e+01, 6.7500e+01, 6.8500e+01, 6.9500e+01, 7.0500e+01, 7.1500e+01],\n"
            "           [7.2500e+01, 7.3500e+01, 7.4500e+01, 7.5500e+01, 7.6500e+01, 7.7500e+01],\n"
            "           [7.8500e+01, 7.9500e+01, 8.0500e+01, 8.1500e+01, 8.2500e+01, 8.3500e+01],\n"
            "           [8.4500e+01, 8.5500e+01, 8.6500e+01, 8.7500e+01, 8.8500e+01, 8.9500e+01]],\n"
            "\n"
            "          [[9.0500e+01, 9.1500e+01, 9.2500e+01, 9.3500e+01, 9.4500e+01, 9.5500e+01],\n"
            "           [9.6500e+01, 9.7500e+01, 9.8500e+01, 9.9500e+01, 1.0050e+02, 1.0150e+02],\n"
            "           [1.0250e+02, 1.0350e+02, 1.0450e+02, 1.0550e+02, 1.0650e+02, 1.0750e+02],\n"
            "           [1.0850e+02, 1.0950e+02, 1.1050e+02, 1.1150e+02, 1.1250e+02, 1.1350e+02],\n"
            "           [1.1450e+02, 1.1550e+02, 1.1650e+02, 1.1750e+02, 1.1850e+02, 1.1950e+02]]], dtype=ht.float64, device=cpu:0, split=1)"
        )
        __str = str(dndarray)

        if dndarray.comm.rank == 0:
            self.assertEqual(comparison, __str)

    def test_split_1_above_threshold(self):
        ht.set_printoptions(edgeitems=2)
        dndarray = ht.arange(10 * 11 * 12).reshape((10, 11, 12)).resplit_(1)
        comparison = (
            "DNDarray([[[   0,    1,  ...,   10,   11],\n"
            "           [  12,   13,  ...,   22,   23],\n"
            "           ...,\n"
            "           [ 108,  109,  ...,  118,  119],\n"
            "           [ 120,  121,  ...,  130,  131]],\n"
            "\n"
            "          [[ 132,  133,  ...,  142,  143],\n"
            "           [ 144,  145,  ...,  154,  155],\n"
            "           ...,\n"
            "           [ 240,  241,  ...,  250,  251],\n"
            "           [ 252,  253,  ...,  262,  263]],\n"
            "\n"
            "          ...,\n"
            "\n"
            "          [[1056, 1057,  ..., 1066, 1067],\n"
            "           [1068, 1069,  ..., 1078, 1079],\n"
            "           ...,\n"
            "           [1164, 1165,  ..., 1174, 1175],\n"
            "           [1176, 1177,  ..., 1186, 1187]],\n"
            "\n"
            "          [[1188, 1189,  ..., 1198, 1199],\n"
            "           [1200, 1201,  ..., 1210, 1211],\n"
            "           ...,\n"
            "           [1296, 1297,  ..., 1306, 1307],\n"
            "           [1308, 1309,  ..., 1318, 1319]]], dtype=ht.int32, device=cpu:0, split=1)"
        )
        __str = str(dndarray)

        if dndarray.comm.rank == 0:
            self.assertEqual(comparison, __str)

    def test_split_2_below_threshold(self):
        dndarray = ht.arange(4 * 5 * 6, dtype=ht.uint8).reshape((4, 5, 6)).resplit_(2)
        comparison = (
            "DNDarray([[[  0,   1,   2,   3,   4,   5],\n"
            "           [  6,   7,   8,   9,  10,  11],\n"
            "           [ 12,  13,  14,  15,  16,  17],\n"
            "           [ 18,  19,  20,  21,  22,  23],\n"
            "           [ 24,  25,  26,  27,  28,  29]],\n"
            "\n"
            "          [[ 30,  31,  32,  33,  34,  35],\n"
            "           [ 36,  37,  38,  39,  40,  41],\n"
            "           [ 42,  43,  44,  45,  46,  47],\n"
            "           [ 48,  49,  50,  51,  52,  53],\n"
            "           [ 54,  55,  56,  57,  58,  59]],\n"
            "\n"
            "          [[ 60,  61,  62,  63,  64,  65],\n"
            "           [ 66,  67,  68,  69,  70,  71],\n"
            "           [ 72,  73,  74,  75,  76,  77],\n"
            "           [ 78,  79,  80,  81,  82,  83],\n"
            "           [ 84,  85,  86,  87,  88,  89]],\n"
            "\n"
            "          [[ 90,  91,  92,  93,  94,  95],\n"
            "           [ 96,  97,  98,  99, 100, 101],\n"
            "           [102, 103, 104, 105, 106, 107],\n"
            "           [108, 109, 110, 111, 112, 113],\n"
            "           [114, 115, 116, 117, 118, 119]]], dtype=ht.uint8, device=cpu:0, split=2)"
        )
        __str = str(dndarray)

        if dndarray.comm.rank == 0:
            self.assertEqual(comparison, __str)

    def test_split_2_above_threshold(self):
        ht.set_printoptions(threshold=1)
        dndarray = ht.arange(3 * 10 * 12).reshape((3, 10, 12)).resplit_(2)
        comparison = (
            "DNDarray([[[  0,   1,   2,  ...,   9,  10,  11],\n"
            "           [ 12,  13,  14,  ...,  21,  22,  23],\n"
            "           [ 24,  25,  26,  ...,  33,  34,  35],\n"
            "           ...,\n"
            "           [ 84,  85,  86,  ...,  93,  94,  95],\n"
            "           [ 96,  97,  98,  ..., 105, 106, 107],\n"
            "           [108, 109, 110,  ..., 117, 118, 119]],\n"
            "\n"
            "          [[120, 121, 122,  ..., 129, 130, 131],\n"
            "           [132, 133, 134,  ..., 141, 142, 143],\n"
            "           [144, 145, 146,  ..., 153, 154, 155],\n"
            "           ...,\n"
            "           [204, 205, 206,  ..., 213, 214, 215],\n"
            "           [216, 217, 218,  ..., 225, 226, 227],\n"
            "           [228, 229, 230,  ..., 237, 238, 239]],\n"
            "\n"
            "          [[240, 241, 242,  ..., 249, 250, 251],\n"
            "           [252, 253, 254,  ..., 261, 262, 263],\n"
            "           [264, 265, 266,  ..., 273, 274, 275],\n"
            "           ...,\n"
            "           [324, 325, 326,  ..., 333, 334, 335],\n"
            "           [336, 337, 338,  ..., 345, 346, 347],\n"
            "           [348, 349, 350,  ..., 357, 358, 359]]], dtype=ht.int32, device=cpu:0, split=2)"
        )
        __str = str(dndarray)

        if dndarray.comm.rank == 0:
            self.assertEqual(comparison, __str)
