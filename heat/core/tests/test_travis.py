import unittest
import heat as ht


class TestTravis(unittest.TestCase):
    def test_one_failed(self):
        size = ht.get_comm().size
        print("size", size)
        if size < 4:
            self.fail()
        else:
            self.assertTrue(True)
