import heat as ht
import heat.ndimage.affine as affine

from heat.testing.basic_test import TestCase


class TestAffine(TestCase):

    def test_offset(self):
        # 2d
        rnd_image = ht.random.random((64, 64, 3), dtype=ht.float32)
        rnd_transform = ht.random.random((3, 3), dtype=ht.float32)
        rnd_offset = ht.random.random((3), dtype=ht.float32)
        combined = ht.hstack((rnd_transform, rnd_offset[:, None]))
        with_offset_result: ht.DNDarray = ht.ndimage.affine.affine_transform(
            rnd_image, rnd_transform, offset=rnd_offset
        )
        combined_result: ht.DNDarray = affine.affine_transform(rnd_image, combined)
        self.assertTrue(ht.equal(with_offset_result, combined_result))

        # 2d and bulk
        rnd_image = ht.random.random((4, 64, 64, 3), dtype=ht.float32)
        rnd_transform = ht.random.random((4, 3, 3), dtype=ht.float32)
        rnd_offset = ht.random.random((4, 3), dtype=ht.float32)
        combined = ht.concatenate((rnd_transform, rnd_offset[:, :, None]), 2)
        with_offset_result: ht.DNDarray = ht.ndimage.affine.affine_transform(
            rnd_image, rnd_transform, offset=rnd_offset
        )
        combined_result: ht.DNDarray = affine.affine_transform(rnd_image, combined)
        self.assertTrue(ht.equal(with_offset_result, combined_result))

        # 2d and bulk and distributed
        rnd_image = ht.random.random((4, 64, 64, 3), dtype=ht.float32, split=0)
        rnd_transform = ht.random.random((4, 3, 3), dtype=ht.float32, split=0)
        rnd_offset = ht.random.random((4, 3), dtype=ht.float32, split=0)
        combined = ht.concatenate((rnd_transform, rnd_offset[:, :, None]), 2)
        with_offset_result: ht.DNDarray = ht.ndimage.affine.affine_transform(
            rnd_image, rnd_transform, offset=rnd_offset
        )
        combined_result: ht.DNDarray = affine.affine_transform(rnd_image, combined)
        self.assertTrue(ht.equal(with_offset_result, combined_result))

        # 3d
        rnd_image = ht.random.random((32, 64, 64, 3), dtype=ht.float32)
        rnd_transform = ht.random.random((4, 4), dtype=ht.float32)
        rnd_offset = ht.random.random((4), dtype=ht.float32)
        combined = ht.hstack((rnd_transform, rnd_offset[:, None]))
        with_offset_result: ht.DNDarray = ht.ndimage.affine.affine_transform(
            rnd_image, rnd_transform, offset=rnd_offset
        )
        combined_result: ht.DNDarray = affine.affine_transform(rnd_image, combined)
        self.assertTrue(ht.equal(with_offset_result, combined_result))

        # 3d and bulk
        rnd_image = ht.random.random((4, 32, 64, 64, 3), dtype=ht.float32)
        rnd_transform = ht.random.random((4, 4, 4), dtype=ht.float32)
        rnd_offset = ht.random.random((4, 4), dtype=ht.float32)
        combined = ht.concatenate((rnd_transform, rnd_offset[:, :, None]), 2)
        with_offset_result: ht.DNDarray = ht.ndimage.affine.affine_transform(
            rnd_image, rnd_transform, offset=rnd_offset
        )
        combined_result: ht.DNDarray = affine.affine_transform(rnd_image, combined)
        self.assertTrue(ht.equal(with_offset_result, combined_result))

        # 3d and bulk and distributed
        rnd_image = ht.random.random((4, 32, 64, 64, 3), dtype=ht.float32, split=0)
        rnd_transform = ht.random.random((4, 4, 4), dtype=ht.float32, split=0)
        rnd_offset = ht.random.random((4, 4), dtype=ht.float32, split=0)
        combined = ht.concatenate((rnd_transform, rnd_offset[:, :, None]), 2)
        with_offset_result: ht.DNDarray = ht.ndimage.affine.affine_transform(
            rnd_image, rnd_transform, offset=rnd_offset
        )
        combined_result: ht.DNDarray = affine.affine_transform(rnd_image, combined)
        self.assertTrue(ht.equal(with_offset_result, combined_result))
