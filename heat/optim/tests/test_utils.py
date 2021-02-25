import heat as ht

import os
import torch

from heat.core.tests.test_suites.basic_test import TestCase


class TestUtils(TestCase):
    def test_DetectMetricPlateau(self):
        from heat.optim.utils import DetectMetricPlateau

        with self.assertRaises(ValueError):
            DetectMetricPlateau(mode="asdf")
        with self.assertRaises(ValueError):
            DetectMetricPlateau(threshold_mode="asdf")

        # tests needed: need to test min and max modes

        # min tests
        values = [1, 0.9, 0.8, 0.7, 0.8, 0.9, 1, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
        dmp = DetectMetricPlateau(mode="min", patience=2, threshold_mode="abs")
        for c, v in enumerate(values):
            t = dmp.test_if_improving(v)
            if c in {6, 9}:
                self.assertTrue(t)
            else:
                self.assertFalse(t)
        dmp = DetectMetricPlateau(mode="min", patience=2, threshold_mode="abs", cooldown=1)
        for c, v in enumerate(values):
            t = dmp.test_if_improving(v)
            if c in {6, 10}:
                self.assertTrue(t)
            else:
                self.assertFalse(t)
        dmp = DetectMetricPlateau(mode="max", patience=2, threshold_mode="abs")
        for c, v in enumerate(values):
            t = dmp.test_if_improving(v)
            if c in {3, 6, 10, 13}:
                self.assertTrue(t)
            else:
                self.assertFalse(t)
        dmp = DetectMetricPlateau(mode="max", patience=2, threshold_mode="rel")
        for c, v in enumerate(values):
            t = dmp.test_if_improving(v)
            if c in {3, 6, 10, 13}:
                self.assertTrue(t)
            else:
                self.assertFalse(t)
            # get and load the dict to make sure that this doesnt change anything during a loop
            if c == 5:
                dmp.load_dict(dmp.get_dict())
