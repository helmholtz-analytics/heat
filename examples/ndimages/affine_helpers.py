from heat.ndimage.affine import affine_transform
import heat as ht
import scipy.ndimage as ndimg


def affine_comparison(vol, M, **kwargs):

    result = affine_transform(vol, M, **kwargs)

    if "offset" in kwargs:
        offset = kwargs["offset"]
        kwargs["offset"] = offset.numpy()

    compare = ndimg.affine_transform(vol.numpy(), M.numpy(), **kwargs)

    return result, compare


def centered_linear(A: ht.DNDarray, dims: tuple) -> ht.DNDarray:
    """
    3×4 affine around volume center (z, y, x).
    if the matrix has more than 2 dimension, the first axis is interpreted as batch axis
    """
    matrix = ht.array(A)
    c: ht.DNDarray
    if matrix.ndim > 2:
        offsets = ht.array(dims[1:]) / 2
        offsets = offsets[None]  # new axis at position 0
        c = ht.repeat(offsets, dims[0], axis=0)
    else:
        c = ht.array(dims) / 2

    c = ht.expand_dims(c, c.ndim)  # necessary becasue [...,None] not working as inteded
    c.resplit_(matrix.split)
    b = c - matrix @ c
    return ht.concatenate([matrix, b], axis=(A.ndim - 1)).astype(np.float32)


def create_checker_volume(d: int, w: int, h: int, checker_size: int) -> ht.DNDarray:
    """creates a DND array for testing

    :param w: width
    :type w: int
    :param h: height
    :type h: int
    :param checker_size: size in pixel that one checker should have
    :type checker_size: int
    :return: heat array
    :rtype: heat.DNDarray
    """
    print("start creating test volume")
    array = ht.full([d, h, w, 3], 255, dtype=ht.float32)

    for k in range(0, d, checker_size):
        for i in range(0, h, checker_size):
            for j in range(0, w, checker_size):
                y = i // checker_size
                x = j // checker_size
                z = k // checker_size
                if (y & 1) ^ (x & 1) ^ (z & 1):
                    color_difference = 50 / checker_size
                    color = (0, 0, (color_difference * (20 + i + j + k)) % 256)
                    array[
                        k : checker_size + k, i : checker_size + i, j : checker_size + j
                    ] = color
    print("finish creating test volume")
    return array
