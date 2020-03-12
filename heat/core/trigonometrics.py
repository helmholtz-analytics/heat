import torch
from .constants import pi
from .operations import __local_op as local_op


__all__ = [
    "arccos",
    "arcsin",
    "arctan",
    "cos",
    "cosh",
    "deg2rad",
    "degrees",
    "rad2deg",
    "radians",
    "sin",
    "sinh",
    "tan",
    "tanh",
]


def arccos(x, out=None):
    """
    Return the trigonometric arccos, element-wise.

    Parameters
    ----------
    x : ht.DNDarray
        The value for which to compute the trigonometric cosine.
    out : ht.DNDarray or None, optional
        A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
        or set to None, a fresh tensor is allocated.

    Returns
    -------
    arccos : ht.DNDarray
        A tensor of the same shape as x, containing the trigonometric arccos of each element in this tensor.
        Input elements outside [-1., 1.] are returned as nan. If out was provided, arccos is a reference to it.

    Examples
    --------
    >>> ht.arccos(ht.array([-1.,-0., 0.83]))
    tensor([3.1416, 1.5708, 0.5917])
    """
    return local_op(torch.acos, x, out)


def arcsin(x, out=None):
    """
    Return the trigonometric arcsin, element-wise.

    Parameters
    ----------
    x : ht.DNDarray
        The value for which to compute the trigonometric cosine.
    out : ht.DNDarray or None, optional
        A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
        or set to None, a fresh tensor is allocated.

    Returns
    -------
    arcsin : ht.DNDarray
        A tensor of the same shape as x, containing the trigonometric arcsin of each element in this tensor.
        Input elements outside [-1., 1.] are returned as nan. If out was provided, arcsin is a reference to it.

    Examples
    --------
    >>> ht.arcsin(ht.array([-1.,-0., 0.83]))
    tensor([-1.5708,  0.0000,  0.9791])
    """
    return local_op(torch.asin, x, out)


def arctan(x, out=None):
    """
    Return the trigonometric arctan, element-wise.

    Parameters
    ----------
    x : ht.DNDarray
        The value for which to compute the trigonometric cosine.
    out : ht.DNDarray or None, optional
        A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
        or set to None, a fresh tensor is allocated.

    Returns
    -------
    arcstan : ht.DNDarray
        A tensor of the same shape as x, containing the trigonometric arctan of each element in this tensor.
        If out was provided, arctan is a reference to it.

    Examples
    --------
    >>> ht.arctan(ht.arange(-6, 7, 2))
    tensor([-1.4056, -1.3258, -1.1071,  0.0000,  1.1071,  1.3258,  1.4056],
       dtype=torch.float64)
    """
    return local_op(torch.atan, x, out)


def cos(x, out=None):
    """
    Return the trigonometric cosine, element-wise.

    Parameters
    ----------
    x : ht.DNDarray
        The value for which to compute the trigonometric cosine.
    out : ht.DNDarray or None, optional
        A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
        or set to None, a fresh tensor is allocated.

    Returns
    -------
    cosine : ht.DNDarray
        A tensor of the same shape as x, containing the trigonometric cosine of each element in this tensor.
        Negative input elements are returned as nan. If out was provided, square_roots is a reference to it.

    Examples
    --------
    >>> ht.cos(ht.arange(-6, 7, 2))
    tensor([0.96017029, -0.65364362, -0.41614684,  1., -0.41614684, -0.65364362,  0.96017029])
    """
    return local_op(torch.cos, x, out)


def cosh(x, out=None):
    """
    Return the hyperbolic cosine, element-wise.

    Parameters
    ----------
    x : ht.DNDarray
        The value for which to compute the hyperbolic cosine.
    out : ht.DNDarray or None, optional
        A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
        or set to None, a fresh tensor is allocated.

    Returns
    -------
    hyperbolic cosine : ht.DNDarray
        A tensor of the same shape as x, containing the hyperbolic cosine of each element in this tensor.
        Negative input elements are returned as nan. If out was provided, square_roots is a reference to it.

    Examples
    --------
    >>> ht.cosh(ht.arange(-6, 7, 2))
    tensor([201.7156,  27.3082,   3.7622,   1.0000,   3.7622,  27.3082, 201.7156])
    """
    return local_op(torch.cosh, x, out)


def deg2rad(x, out=None):
    """
    Convert angles from degrees to radians.

    Parameters
    ----------
    x : ht.DNDarray
        The value for which to compute the angles in radians.
    out : ht.DNDarray or None, optional
        A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
        or set to None, a fresh tensor is allocated.

    Returns
    -------
    y : ht.DNDarray
        The corresponding angle in radians

    Examples elements =
    --------
    >>> ht.deg2rad(ht.array([0.,20.,45.,78.,94.,120.,180., 270., 311.]))
    tensor([0.0000, 0.3491, 0.7854, 1.3614, 1.6406, 2.0944, 3.1416, 4.7124, 5.4280])
    """
    # deg2rad torch version
    def torch_deg2rad(torch_tensor):
        if not torch.is_tensor(torch_tensor):
            raise TypeError("Input is not a torch tensor but {}".format(type(torch_tensor)))
        return torch_tensor * pi / 180.0

    return local_op(torch_deg2rad, x, out)


def degrees(x, out=None):
    """
    Convert angles from radians to degrees.

    Parameters
    ----------
    x : ht.DNDarray
        The value for which to compute the angles in degrees.
    out : ht.DNDarray or None, optional
        A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
        or set to None, a fresh tensor is allocated.

    Returns
    -------
    y : ht.DNDarray
        The corresponding angle in degrees

    Examples
    --------
    >>> ht.degrees(ht.array([0.,0.2,0.6,0.9,1.2,2.7,3.14]))
    tensor([  0.0000,  11.4592,  34.3775,  51.5662,  68.7549, 154.6986, 179.9088])
    """
    return rad2deg(x, out=out)


def rad2deg(x, out=None):
    """
    Convert angles from radians to degrees.

    Parameters
    ----------
    x : ht.DNDarray
        The value for which to compute the angles in degrees.
    out : ht.DNDarray or None, optional
        A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
        or set to None, a fresh tensor is allocated.

    Returns
    -------
    y : ht.DNDarray
        The corresponding angle in degrees

    Examples
    --------
    >>> ht.rad2deg(ht.array([0.,0.2,0.6,0.9,1.2,2.7,3.14]))
    tensor([  0.0000,  11.4592,  34.3775,  51.5662,  68.7549, 154.6986, 179.9088])
    """
    # rad2deg torch version
    def torch_rad2deg(torch_tensor):
        if not torch.is_tensor(torch_tensor):
            raise TypeError("Input is not a torch tensor but {}".format(type(torch_tensor)))
        return 180.0 * torch_tensor / pi

    return local_op(torch_rad2deg, x, out=out)


def radians(x, out=None):
    """
    Convert angles from degrees to radians.

    Parameters
    ----------
    x : ht.DNDarray
        The value for which to compute the angles in radians.
    out : ht.DNDarray or None, optional
        A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
        or set to None, a fresh tensor is allocated.

    Returns
    -------
    y : ht.DNDarray
        The corresponding angle in radians

    Examples
    --------
    >>> ht.radians(ht.array([0., 20., 45., 78., 94., 120., 180., 270., 311.]))
    tensor([0.0000, 0.3491, 0.7854, 1.3614, 1.6406, 2.0944, 3.1416, 4.7124, 5.4280])
    """

    return deg2rad(x, out=None)


def sin(x, out=None):
    """
    Return the trigonometric sine, element-wise.

    Parameters
    ----------
    x : ht.DNDarray
        The value for which to compute the trigonometric tangent.
    out : ht.DNDarray or None, optional
        A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
        or set to None, a fresh tensor is allocated.

    Returns
    -------
    sine : ht.DNDarray
        A tensor of the same shape as x, containing the trigonometric sine of each element in this tensor.
        Negative input elements are returned as nan. If out was provided, square_roots is a reference to it.

    Examples
    --------
    >>> ht.sin(ht.arange(-6, 7, 2))
    tensor([ 0.2794,  0.7568, -0.9093,  0.0000,  0.9093, -0.7568, -0.2794])
    """
    return local_op(torch.sin, x, out)


def sinh(x, out=None):
    """
    Return the hyperbolic sine, element-wise.

    Parameters
    ----------
    x : ht.DNDarray
        The value for which to compute the hyperbolic sine.
    out : ht.DNDarray or None, optional
        A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
        or set to None, a fresh tensor is allocated.

    Returns
    -------
    hyperbolic sine : ht.DNDarray
        A tensor of the same shape as x, containing the trigonometric sine of each element in this tensor.
        Negative input elements are returned as nan. If out was provided, square_roots is a reference to it.

    Examples
    --------
    >>> ht.sinh(ht.arange(-6, 7, 2))
    tensor([[-201.7132,  -27.2899,   -3.6269,    0.0000,    3.6269,   27.2899,  201.7132])
    """
    return local_op(torch.sinh, x, out)


def tan(x, out=None):
    """
    Compute tangent element-wise.

    Equivalent to ht.sin(x) / ht.cos(x) element-wise.

    Parameters
    ----------
    x : ht.DNDarray
        The value for which to compute the trigonometric tangent.
    out : ht.DNDarray or None, optional
        A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
        or set to None, a fresh tensor is allocated.

    Returns
    -------
    tangent : ht.DNDarray
        A tensor of the same shape as x, containing the trigonometric tangent of each element in this tensor.

    Examples
    --------
    >>> ht.tan(ht.arange(-6, 7, 2))
    tensor([ 0.29100619, -1.15782128,  2.18503986,  0., -2.18503986, 1.15782128, -0.29100619])
    """
    return local_op(torch.tan, x, out)


def tanh(x, out=None):
    """
    Return the hyperbolic tangent, element-wise.

    Parameters
    ----------
    x : ht.DNDarray
        The value for which to compute the hyperbolic tangent.
    out : ht.DNDarray or None, optional
        A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
        or set to None, a fresh tensor is allocated.

    Returns
    -------
    hyperbolic tangent : ht.DNDarray
        A tensor of the same shape as x, containing the hyperbolic tangent of each element in this tensor.

    Examples
    --------
    >>> ht.tanh(ht.arange(-6, 7, 2))
    tensor([-1.0000, -0.9993, -0.9640,  0.0000,  0.9640,  0.9993,  1.0000])
    """
    return local_op(torch.tanh, x, out)
