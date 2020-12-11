import heat as ht

from heat.core.tests.test_suites.basic_test import TestCase


class TestOptim(TestCase):
    def test_optim_getattr(self):
        with self.assertRaises(AttributeError):
            ht.optim.asdf()
