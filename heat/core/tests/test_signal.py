import numpy as np
import torch
import unittest
import heat as ht
from heat import manipulations
import scipy.signal as sig
from .test_suites.basic_test import TestCase


class TestSignal(TestCase):
    @classmethod
    def setUpClass(cls):
        super(TestSignal, cls).setUpClass()

    def test_convolve(self):
        full_odd = ht.array(
            [0, 1, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 29, 15]
        ).astype(ht.int)
        full_even = ht.array(
            [0, 1, 3, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 42, 29, 15]
        ).astype(ht.int)

        signal = ht.arange(0, 16, split=0).astype(ht.int)
        kernel_odd = ht.ones(3).astype(ht.int)
        kernel_even = [1, 1, 1, 1]

        with self.assertRaises(TypeError):
            signal_wrong_type = [0, 1, 2, "tre", 4, "five", 6, "ʻehiku", 8, 9, 10]
            ht.convolve(signal_wrong_type, kernel_odd, mode="full")
        with self.assertRaises(TypeError):
            filter_wrong_type = [1, 1, "pizza", "pineapple"]
            ht.convolve(signal, filter_wrong_type, mode="full")
        with self.assertRaises(ValueError):
            ht.convolve(signal, kernel_odd, mode="invalid")
        with self.assertRaises(ValueError):
            s = signal.reshape((2, -1))
            ht.convolve(s, kernel_odd)
        with self.assertRaises(ValueError):
            k = ht.eye(3)
            ht.convolve(signal, k)
        with self.assertRaises(ValueError):
            ht.convolve(kernel_even, full_even)
        with self.assertRaises(ValueError):
            ht.convolve(signal, kernel_even, mode="same")
        if self.comm.size > 1:
            with self.assertRaises(TypeError):
                k = ht.ones(4, split=0).astype(ht.int)
                ht.convolve(signal, k)
        if self.comm.size >= 5:
            with self.assertRaises(ValueError):
                ht.convolve(signal, kernel_even, mode="valid")

        # test modes, avoid kernel larger than signal chunk
        if self.comm.size <= 3:
            modes = ["full", "same", "valid"]
            for i, mode in enumerate(modes):
                # odd kernel size
                conv = ht.convolve(signal, kernel_odd, mode=mode)
                gathered = manipulations.resplit(conv, axis=None)
                self.assertTrue(ht.equal(full_odd[i : len(full_odd) - i], gathered))
                # different data types
                conv = ht.convolve(signal.astype(ht.float), kernel_odd)
                gathered = manipulations.resplit(conv, axis=None)
                self.assertTrue(ht.equal(full_odd.astype(ht.float), gathered))

                # even kernel size
                # skip mode 'same' for even kernels
                if mode != "same":
                    conv = ht.convolve(signal, kernel_even, mode=mode)
                    gathered = manipulations.resplit(conv, axis=None)

                    if mode == "full":
                        self.assertTrue(ht.equal(full_even, gathered))
                    else:
                        self.assertTrue(ht.equal(full_even[3:-3], gathered))

        # test edge cases
        # non-distributed signal, size-1 kernel
        signal = ht.arange(0, 16).astype(ht.int)
        alt_signal = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
        kernel = ht.ones(1).astype(ht.int)
        conv = ht.convolve(alt_signal, kernel)
        self.assertTrue(ht.equal(signal, conv))

    def test_convolve2d(self):

        full_odd = ht.array(
            [[   0,    0,    1,    4,    7,   10,   13,   16,   19,   22,   25,   28,   31,   34,   37,   40,   43,   30],
             [   0,   19,   59,   74,   89,  104,  119,  134,  149,  164,  179,  194,  209,  224,  239,  254,  221,  137],
             [  48,  153,  318,  354,  390,  426,  462,  498,  534,  570,  606,  642,  678,  714,  750,  786,  630,  369],
             [ 192,  489,  894,  930,  966, 1002, 1038, 1074, 1110, 1146, 1182, 1218, 1254, 1290, 1326, 1362, 1062,  609],
             [ 336,  825, 1470, 1506, 1542, 1578, 1614, 1650, 1686, 1722, 1758, 1794, 1830, 1866, 1902, 1938, 1494,  849],
             [ 480, 1161, 2046, 2082, 2118, 2154, 2190, 2226, 2262, 2298, 2334, 2370, 2406, 2442, 2478, 2514, 1926, 1089],
             [ 624, 1497, 2622, 2658, 2694, 2730, 2766, 2802, 2838, 2874, 2910, 2946, 2982, 3018, 3054, 3090, 2358, 1329],
             [ 768, 1833, 3198, 3234, 3270, 3306, 3342, 3378, 3414, 3450, 3486, 3522, 3558, 3594, 3630, 3666, 2790, 1569],
             [ 912, 2169, 3774, 3810, 3846, 3882, 3918, 3954, 3990, 4026, 4062, 4098, 4134, 4170, 4206, 4242, 3222, 1809],
             [1056, 2505, 4350, 4386, 4422, 4458, 4494, 4530, 4566, 4602, 4638, 4674, 4710, 4746, 4782, 4818, 3654, 2049],
             [1200, 2841, 4926, 4962, 4998, 5034, 5070, 5106, 5142, 5178, 5214, 5250, 5286, 5322, 5358, 5394, 4086, 2289],
             [1344, 3177, 5502, 5538, 5574, 5610, 5646, 5682, 5718, 5754, 5790, 5826, 5862, 5898, 5934, 5970, 4518, 2529],
             [1488, 3513, 6078, 6114, 6150, 6186, 6222, 6258, 6294, 6330, 6366, 6402, 6438, 6474, 6510, 6546, 4950, 2769],
             [1632, 3849, 6654, 6690, 6726, 6762, 6798, 6834, 6870, 6906, 6942, 6978, 7014, 7050, 7086, 7122, 5382, 3009],
             [1776, 4185, 7230, 7266, 7302, 7338, 7374, 7410, 7446, 7482, 7518, 7554, 7590, 7626, 7662, 7698, 5814, 3249],
             [1920, 4521, 7806, 7842, 7878, 7914, 7950, 7986, 8022, 8058, 8094, 8130, 8166, 8202, 8238, 8274, 6246, 3489],
             [2064, 4601, 7613, 7646, 7679, 7712, 7745, 7778, 7811, 7844, 7877, 7910, 7943, 7976, 8009, 8042, 5867, 3187],
             [1440, 3126, 5059, 5080, 5101, 5122, 5143, 5164, 5185, 5206, 5227, 5248, 5269, 5290, 5311, 5332, 3817, 2040]]
        ).astype(ht.int)
        full_even = ht.array(
            [[    0,     0,     1,     4,    10,    16,    22,    28,    34,    40,    46,    52,    58,    64,    70,
              76,    82,    72,    45],
            [    0,    20,    62,   128,   156,   184,   212,   240,   268,   296,   324,   352,   380,   408,   436,
                464,   428,   340,   198],
            [   64,   188,   375,   628,   694,   760,   826,   892,   958,  1024,  1090,  1156,  1222,  1288,  1354,
                1420,  1230,   932,   523],
            [  256,   632,  1132,  1760,  1880,  2000,  2120,  2240,  2360,  2480,  2600,  2720,  2840,  2960,  3080,
                3200,  2680,  1976,  1084],
            [  640,  1464,  2476,  3680,  3800,  3920,  4040,  4160,  4280,  4400,  4520,  4640,  4760,  4880,  5000,
                5120,  4216,  3064,  1660],
            [ 1024,  2296,  3820,  5600,  5720,  5840,  5960,  6080,  6200,  6320,  6440,  6560,  6680,  6800,  6920,
                7040,  5752,  4152,  2236],
            [ 1408,  3128,  5164,  7520,  7640,  7760,  7880,  8000,  8120,  8240,  8360,  8480,  8600,  8720,  8840,
                8960,  7288,  5240,  2812],
            [ 1792,  3960,  6508,  9440,  9560,  9680,  9800,  9920, 10040, 10160, 10280, 10400, 10520, 10640, 10760,
            10880,  8824,  6328,  3388],
            [ 2176,  4792,  7852, 11360, 11480, 11600, 11720, 11840, 11960, 12080, 12200, 12320, 12440, 12560, 12680,
            12800, 10360,  7416,  3964],
            [ 2560,  5624,  9196, 13280, 13400, 13520, 13640, 13760, 13880, 14000, 14120, 14240, 14360, 14480, 14600,
            14720, 11896,  8504,  4540],
            [ 2944,  6456, 10540, 15200, 15320, 15440, 15560, 15680, 15800, 15920, 16040, 16160, 16280, 16400, 16520,
            16640, 13432,  9592,  5116],
            [ 3328,  7288, 11884, 17120, 17240, 17360, 17480, 17600, 17720, 17840, 17960, 18080, 18200, 18320, 18440,
            18560, 14968, 10680,  5692],
            [ 3712,  8120, 13228, 19040, 19160, 19280, 19400, 19520, 19640, 19760, 19880, 20000, 20120, 20240, 20360,
            20480, 16504, 11768,  6268],
            [ 4096,  8952, 14572, 20960, 21080, 21200, 21320, 21440, 21560, 21680, 21800, 21920, 22040, 22160, 22280,
            22400, 18040, 12856,  6844],
            [ 4480,  9784, 15916, 22880, 23000, 23120, 23240, 23360, 23480, 23600, 23720, 23840, 23960, 24080, 24200,
            24320, 19576, 13944,  7420],
            [ 4864, 10616, 17260, 24800, 24920, 25040, 25160, 25280, 25400, 25520, 25640, 25760, 25880, 26000, 26120,
            26240, 21112, 15032,  7996],
            [ 5248, 11192, 17835, 25180, 25294, 25408, 25522, 25636, 25750, 25864, 25978, 26092, 26206, 26320, 26434,
            26548, 21030, 14768,  7759],
            [ 4608,  9700, 15278, 21344, 21436, 21528, 21620, 21712, 21804, 21896, 21988, 22080, 22172, 22264, 22356,
            22448, 17612, 12260,  6390],
            [ 2880,  6012,  9397, 13036, 13090, 13144, 13198, 13252, 13306, 13360, 13414, 13468, 13522, 13576, 13630,
            13684, 10666,  7380,  3825]]
        ).astype(ht.int)

        dis_signal = ht.arange(256, split=0).reshape((16, 16)).astype(ht.int)
        signal = ht.arange(256).reshape((16, 16)).astype(ht.int)

        kernel_odd = ht.arange(9).reshape((3, 3)).astype(ht.int)
        kernel_even = ht.arange(16).reshape((4, 4)).astype(ht.int)
        dis_kernel_odd = ht.arange(9, split=0).reshape((3, 3)).astype(ht.int)
        dis_kernel_even = ht.arange(16, split=0).reshape((4, 4)).astype(ht.int)

        with self.assertRaises(TypeError):
            signal_wrong_type = [[0, 1, 2, "tre", 4]]*5
            ht.convolve2d(signal_wrong_type, kernel_odd)
        with self.assertRaises(TypeError):
            filter_wrong_type = [[ 1, "pizza", "pineapple"]]*3
            ht.convolve2d(dis_signal, filter_wrong_type, mode="full")
        with self.assertRaises(ValueError):
            ht.convolve2d(dis_signal, kernel_odd, mode="invalid")
        with self.assertRaises(ValueError):
            s = dis_signal.reshape((2, 2, -1))
            ht.convolve2d(s, kernel_odd)
        with self.assertRaises(ValueError):
            k = ht.arange(3)
            ht.convolve2d(dis_signal, k)
        with self.assertRaises(ValueError):
            ht.convolve2d(dis_signal, kernel_even, mode="same")
        if self.comm.size > 2:
            with self.assertRaises(ValueError):
                ht.convolve2d(dis_signal, signal, mode="valid")

        # test modes, avoid kernel larger than signal chunk
        if self.comm.size <= 3:
            modes = ["full", "same", "valid"]
            for i, mode in enumerate(modes):
                # odd kernel size
                conv = ht.convolve2d(dis_signal, kernel_odd, mode=mode)
                gathered = manipulations.resplit(conv, axis=None)
                self.assertTrue(ht.equal(full_odd[i : len(full_odd) - i, i : len(full_odd) - i], gathered))

                conv = ht.convolve2d(dis_signal, dis_kernel_odd, mode=mode)
                gathered = manipulations.resplit(conv, axis=None)
                self.assertTrue(ht.equal(full_odd[i : len(full_odd) - i, i : len(full_odd) - i], gathered))

                conv = ht.convolve2d(signal, dis_kernel_odd, mode=mode)
                gathered = manipulations.resplit(conv, axis=None)
                self.assertTrue(ht.equal(full_odd[i : len(full_odd) - i, i : len(full_odd) - i], gathered))

                # different data types
                conv = ht.convolve2d(dis_signal.astype(ht.float), kernel_odd)
                gathered = manipulations.resplit(conv, axis=None)
                self.assertTrue(ht.equal(full_odd.astype(ht.float), gathered))

                conv = ht.convolve2d(dis_signal.astype(ht.float), dis_kernel_odd)
                gathered = manipulations.resplit(conv, axis=None)
                self.assertTrue(ht.equal(full_odd.astype(ht.float), gathered))

                conv = ht.convolve2d(signal.astype(ht.float), dis_kernel_odd)
                gathered = manipulations.resplit(conv, axis=None)
                self.assertTrue(ht.equal(full_odd.astype(ht.float), gathered))

                # even kernel size
                # skip mode 'same' for even kernels
                if mode != "same":
                    conv = ht.convolve2d(dis_signal, kernel_even, mode=mode)
                    dis_conv = ht.convolve2d(dis_signal, dis_kernel_even, mode=mode)
                    gathered = manipulations.resplit(conv, axis=None)
                    dis_gathered = manipulations.resplit(dis_conv, axis=None)

                    if mode == "full":
                        self.assertTrue(ht.equal(full_even, gathered))
                        self.assertTrue(ht.equal(full_even, dis_gathered))
                    else:
                        self.assertTrue(ht.equal(full_even[3:-3, 3:-3], gathered))
                        self.assertTrue(ht.equal(full_even[3:-3, 3:-3], dis_gathered))

                # distributed large signal and kernel
                np.random.seed(12)
                np_a = np.random.randint(1000, size = (140, 250))
                np_b = np.random.randint(1000, size = (39, 17))
                sc_conv = sig.convolve2d(np_a, np_b, mode=mode)

                a = ht.array(np_a, split=0, dtype=ht.int32)
                b = ht.array(np_b, split=0, dtype=ht.int32)
                conv = ht.convolve2d(a, b, mode=mode)
                self.assert_array_equal(conv, sc_conv)

                a = ht.array(np_a, split=1, dtype=ht.int32)
                b = ht.array(np_b, split=1, dtype=ht.int32)
                conv = ht.convolve2d(a, b, mode=mode)
                self.assert_array_equal(conv, sc_conv)

        # test edge cases
        # non-distributed signal, size-1 kernel
        signal = ht.arange(0, 16).reshape(4, 4).astype(ht.int)
        alt_signal = ht.arange(16).reshape(4, 4).astype(ht.int)
        kernel = ht.ones(1).reshape((1, 1)).astype(ht.int)
        conv = ht.convolve2d(alt_signal, kernel)
        self.assertTrue(ht.equal(signal, conv))
