import itertools


def broadcast_shape(shapea, shapeb):
    """
    Inspired from stackoverflow answer to is_broadcastable
    """
    it = itertools.zip_longest(shapea[::-1], shapeb[::-1], fillvalue=1)
    res = max(len(shapea), len(shapeb)) * [None]
    for i, (a, b) in enumerate(it):
        if a == 1 or b == 1 or a == b:
            res[i] = max(a, b)
        else:
            raise ValueError("Operands could not be broadcasted with shapes {} {}".format(shapea, shapeb))

    return tuple(res[::-1])
