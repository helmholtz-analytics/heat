import heat as ht
import unittest


class TestVisionTransforms(unittest.TestCase):
    def test_vision_transforms_getattr(self):
        ht.utils.vision_transforms.ToTensor()
        with self.assertRaises(AttributeError):
            ht.utils.vision_transforms.asdf()
