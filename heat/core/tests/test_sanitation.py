import numpy as np
import torch

import heat as ht
from .test_suites.basic_test import TestCase


class TestSanitation(TestCase):
    def test_sanitize_in(self):
        torch_x = torch.arange(10)
        with self.assertRaises(TypeError):
            ht.sanitize_in(torch_x)
        np_x = np.arange(10)
        with self.assertRaises(TypeError):
            ht.sanitize_in(np_x)

    def test_sanitize_out(self):
        output_shape = (4, 5, 6)
        output_split = 1
        output_device = "cpu"
        out_wrong_type = torch.empty(output_shape)
        with self.assertRaises(TypeError):
            ht.sanitize_out(out_wrong_type, output_shape, output_split, output_device)
        out_wrong_shape = ht.empty((4, 7, 6), split=output_split, device=output_device)
        with self.assertRaises(ValueError):
            ht.sanitize_out(out_wrong_shape, output_shape, output_split, output_device)
        out_wrong_split = ht.empty(output_shape, split=2, device=output_device)
        with self.assertRaises(ValueError):
            ht.sanitize_out(out_wrong_split, output_shape, output_split, output_device)

    def test_sanitize_sequence(self):
        # test list seq
        seq = [1, 2, 3]
        seq = ht.sanitize_sequence(seq)
        self.assertTrue(isinstance(seq, list))
        # test tuple seq
        seq = (1, 2, 3)
        seq = ht.sanitize_sequence(seq)
        self.assertTrue(isinstance(seq, list))
        # test exceptions
        split_seq = ht.arange(10, dtype=ht.float32, split=0)
        with self.assertRaises(TypeError):
            ht.sanitize_sequence(split_seq)
        np_seq = np.arange(10)
        with self.assertRaises(TypeError):
            ht.sanitize_sequence(np_seq)

    def test_scalar_to_1d(self):
        ht_scalar = ht.array(8)
        ht_1d = ht.scalar_to_1d(ht_scalar)
        self.assertTrue(ht_1d.ndim == 1)
        self.assertTrue(ht_1d.shape == (1,))
