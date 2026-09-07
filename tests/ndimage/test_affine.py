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
        with_offset_result: ht.DNDarray = affine.affine_transform(
            rnd_image, rnd_transform, offset=rnd_offset
        )
        combined_result: ht.DNDarray = affine.affine_transform(rnd_image, combined)
        self.assertTrue(ht.equal(with_offset_result, combined_result))

        # 2d and bulk
        rnd_image = ht.random.random((4, 64, 64, 3), dtype=ht.float32)
        rnd_transform = ht.random.random((4, 3, 3), dtype=ht.float32)
        rnd_offset = ht.random.random((4, 3), dtype=ht.float32)
        combined = ht.concatenate((rnd_transform, rnd_offset[:, :, None]), 2)
        with_offset_result: ht.DNDarray = affine.affine_transform(
            rnd_image, rnd_transform, offset=rnd_offset
        )
        combined_result: ht.DNDarray = affine.affine_transform(rnd_image, combined)
        self.assertTrue(ht.equal(with_offset_result, combined_result))

        # 2d and bulk and distributed
        rnd_image = ht.random.random((4, 64, 64, 3), dtype=ht.float32, split=0)
        rnd_transform = ht.random.random((4, 3, 3), dtype=ht.float32, split=0)
        rnd_offset = ht.random.random((4, 3), dtype=ht.float32, split=0)
        combined = ht.concatenate((rnd_transform, rnd_offset[:, :, None]), 2)
        with_offset_result: ht.DNDarray = affine.affine_transform(
            rnd_image, rnd_transform, offset=rnd_offset
        )
        combined_result: ht.DNDarray = affine.affine_transform(rnd_image, combined)
        self.assertTrue(ht.equal(with_offset_result, combined_result))

        # 3d
        rnd_image = ht.random.random((32, 64, 64, 3), dtype=ht.float32)
        rnd_transform = ht.random.random((4, 4), dtype=ht.float32)
        rnd_offset = ht.random.random((4), dtype=ht.float32)
        combined = ht.hstack((rnd_transform, rnd_offset[:, None]))
        with_offset_result: ht.DNDarray = affine.affine_transform(
            rnd_image, rnd_transform, offset=rnd_offset
        )
        combined_result: ht.DNDarray = affine.affine_transform(rnd_image, combined)
        self.assertTrue(ht.equal(with_offset_result, combined_result))

        # 3d and bulk
        rnd_image = ht.random.random((4, 32, 64, 64, 3), dtype=ht.float32)
        rnd_transform = ht.random.random((4, 4, 4), dtype=ht.float32)
        rnd_offset = ht.random.random((4, 4), dtype=ht.float32)
        combined = ht.concatenate((rnd_transform, rnd_offset[:, :, None]), 2)
        with_offset_result: ht.DNDarray = affine.affine_transform(
            rnd_image, rnd_transform, offset=rnd_offset
        )
        combined_result: ht.DNDarray = affine.affine_transform(rnd_image, combined)
        self.assertTrue(ht.equal(with_offset_result, combined_result))

        # 3d and bulk and distributed
        rnd_image = ht.random.random((4, 32, 64, 64, 3), dtype=ht.float32, split=0)
        rnd_transform = ht.random.random((4, 4, 4), dtype=ht.float32, split=0)
        rnd_offset = ht.random.random((4, 4), dtype=ht.float32, split=0)
        combined = ht.concatenate((rnd_transform, rnd_offset[:, :, None]), 2)
        with_offset_result: ht.DNDarray = affine.affine_transform(
            rnd_image, rnd_transform, offset=rnd_offset
        )
        combined_result: ht.DNDarray = affine.affine_transform(rnd_image, combined)
        self.assertTrue(ht.equal(with_offset_result, combined_result))

    def test_untouched_axes(self):
        array = ht.array(([1, 0, 0], [0, 1, 1], [0, 0, 0.2]))
        untouched_axes = affine._untouched_axes(array)
        self.assertTrue(ht.equal(untouched_axes, ht.array(([True, False, False]))))
        array = ht.array(
            ([[1, 0, 0], [0, 1, 1], [0, 0, 0.2]], [[1, 0, 0], [0, 1, 1], [0, 0, 1]])
        )
        untouched_axes = affine._untouched_axes(array)
        print(untouched_axes)

    def test_non_bulk_split(self):
        matrix = ht.array(
            (
                [[1, 0, 0, 0], [0, 1, 1, 0], [0, 0, 0.2, 0]],
                [[1, 0, 0, 0], [0, 1, 1, 0], [0, 0, 1, 0]],
            )
        )
        # only split=1 or split=0 should not error because other axes get changed

        rnd_image = ht.random.random((2, 64, 64, 3), dtype=ht.float32, split=0)
        affine.affine_transform(rnd_image, matrix)

        rnd_image = ht.random.random((2, 64, 64, 3), dtype=ht.float32, split=1)
        affine.affine_transform(rnd_image, matrix)

        rnd_image = ht.random.random((2, 64, 64, 3), dtype=ht.float32, split=2)
        with self.assertRaises(RuntimeError):
            affine.affine_transform(rnd_image, matrix)

        rnd_image = ht.random.random((2, 64, 64, 3), dtype=ht.float32, split=3)
        with self.assertRaises(RuntimeError):
            affine.affine_transform(rnd_image, matrix)
