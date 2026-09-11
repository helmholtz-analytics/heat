"""
utility and convenience functions for working with image data using heat
"""

from matplotlib import pyplot as plt
from typing import Iterable
import scipy.ndimage as ndimg
import numpy as np
import heat as ht
from heat.ndimage.affine import affine_transform


def visual_compare_2d(heat_image, numpy_image):
    """
    Creates a plot to show an 2d DNDarray and a corresponding numpy array side by side.
    useful to compare two images visually for differences
    """
    _, axs = plt.subplots(1, 2, figsize=(10, 16))
    axs = axs.ravel()

    axs[0].imshow(heat_image.numpy().astype(np.uint8))
    axs[1].imshow(numpy_image.astype(np.uint8))
    axs[0].scatter(heat_image.shape[1] / 2, heat_image.shape[0] / 2)
    axs[1].scatter(numpy_image.shape[1] / 2, numpy_image.shape[0] / 2)

    plt.tight_layout()
    plt.show()


def affine_comparison(
    image: ht.DNDarray, matrix: ht.DNDarray, **kwargs
) -> tuple[ht.DNDarray, np.ndarray]:
    """
    Convenience function to pass the same arguments to heats and scipy version of affine transform
    """
    result = affine_transform(image, matrix, **kwargs)

    if "offset" in kwargs:
        offset = kwargs["offset"]
        kwargs["offset"] = offset.numpy()

    compare = ndimg.affine_transform(image.numpy(), matrix.numpy(), **kwargs)

    return result, compare


def center_transform(affine_matrix, image_dims: tuple) -> ht.DNDarray:
    """
    Take an [Bx]NxN transformation matrix and create an [Bx]xNxN+1 reduced affine matrix out of it,
    that centers the transformation in the image. This is convenient because in scipy convention the
    transformation orgigin is at the upper left corner.
    if the matrix has 2 dimensions, the first axis is interpreted as batch axis
    """
    if affine_matrix.ndim > 3:
        raise RuntimeError("only one batch dimension supported")

    if affine_matrix.ndim < 2:
        raise RuntimeError("matrix needs at least 2 dimensions")

    shape = affine_matrix.shape
    if shape[-1] != shape[-2]:
        raise RuntimeError("original matrix/matrices needs to be square")

    matrix = ht.array(affine_matrix)
    c: ht.DNDarray
    if matrix.ndim == 3:
        offsets = ht.array(image_dims[1:]) / 2
        offsets = offsets[None]  # new axis at position 0
        c = ht.repeat(offsets, image_dims[0], axis=0)
    else:
        c = ht.array(image_dims) / 2

    c = ht.expand_dims(
        c, c.ndim
    )  # necessary because [...,None] not working as inteded, maybe fixed after indexing pr?
    c.resplit_(matrix.split)
    b = c - matrix @ c
    return ht.concatenate([matrix, b], axis=(affine_matrix.ndim - 1)).astype(ht.float32)


def create_checker(
    shape: Iterable[int],
    checker_size: int,
    min_value: int = 0,
    max_value: int = 256,
    dtype: ht.dtype = ht.float32,
) -> np.ndarray:
    """
    Parameters
    ----------
    shape : Iterable[int]
        shape of the output excluding the color axis wich is always dimension 3 for Color information
    checker_size : int
        edge length of the checkers
    min_value: int
            smallest value the checker can contain. default is 0
    max_value: int
        highest value the checker can contain. default is 256 because images are often stored in
        8 bit integers
    dtype : numpy.dtype
        datatype of the values of the resulting array. currently only integer types are supported
    """
    axes = [ht.arange(0, axis_length, dtype=ht.int32) for axis_length in shape]
    indices: ht.DNDarray = ht.stack(ht.meshgrid(*axes, indexing="ij"))
    checker_index = indices // checker_size
    odd: ht.DNDarray = checker_index & 1
    mask: ht.DNDarray = (ht.sum(odd, axis=0) % 2).astype(bool)
    checker_index_sum = ht.sum(checker_index, axis=0)
    blue_channel = (((200 / checker_size) * (20 + checker_index_sum)) % max_value) + min_value
    blue_channel = blue_channel.astype(dtype)

    result = ht.full(shape + (3,), max_value - 1, dtype=dtype)
    result[..., 0][mask] = min_value
    result[..., 1][mask] = min_value
    result[..., 2][mask] = blue_channel[mask]
    return result
