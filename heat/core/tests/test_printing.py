import math

import heat as ht
from .test_suites.basic_test import TestCase


class TestPrinting(TestCase):
    def test_get_default_options(self):
        print_options = ht.get_printoptions()
        comparison = {
            "precision": 4,
            "threshold": 1000,
            "edgeitems": 3,
            "linewidth": 80,
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
            "linewidth": 80,
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
            "linewidth": 80,
            "sci_mode": None,
        }

        self.assertIsInstance(print_options, dict)
        for key, value in print_options.items():
            self.assertEqual(value, comparison[key])

    def test_wrong_profile_exception(self):
        with self.assertRaises(ValueError):
            ht.set_printoptions(profile="foo")

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
        ht.set_printoptions(sci_mode="yes")
        self.assertEqual(True, ht.get_printoptions()["sci_mode"])

    def test_wrong_parameter_type(self):
        with self.assertRaises(ValueError):
            ht.set_printoptions(precision="wrong")

        with self.assertRaises(ValueError):
            ht.set_printoptions(threshold="wrong")

        with self.assertRaises(ValueError):
            ht.set_printoptions(edgeitems="wrong")

        with self.assertRaises(ValueError):
            ht.set_printoptions(linewidth="wrong")
