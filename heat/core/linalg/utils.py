import torch
from typing import Tuple

__all__ = ["gen_house_mat", "gen_house_vec", "apply_house", "larftx2ce", "gbelr"]


# @torch.jit.script
def gen_house_mat(v, tau):
    # type: (torch.Tensor, torch.Tensor) -> torch.Tensor
    h = torch.eye(v.numel(), dtype=v.dtype, device=v.device)
    h -= tau.item() * torch.matmul(v.T, v)
    return h


# @torch.jit.script
def gen_house_vec(x, n=2):
    # type: (torch.Tensor, int) -> Tuple[torch.Tensor, torch.Tensor]
    """
    What is implemented now is only generating ONE reflector, for more would need to implement the following:

    Note, this overwrites x, todo: does it?
    Parameters
    ----------
    n : int
        order of the reflector
    alpha : float
        the value alpha
    x : torch.Tensor
        overwritten
    overwrite: bool
        if true: overwrite x with the transform vector
        default: True

    Returns
    -------
    tau : torch.Tensor
        tau value
    v : torch.Tensor
        output vector
    Notes
    -----
    https://www.netlib.org/lapack/explore-html/d8/d9b/group__double_o_t_h_e_rauxiliary_gaabb59655e820b3551af27781bd716143.html
    What is implemented now is only generating ONE reflector, for more would need to implement the following:
    http://icl.cs.utk.edu/plasma/docs/dlarft_8f_source.html

    """
    v = x.clone().reshape(-1, 1)
    # if isinstance(alpha, (float, int)):
    #     alpha = torch.tensor(alpha, device=x.device, dtype=x.dtype)
    if n <= 1:
        tau = torch.tensor(0, device=x.device, dtype=x.dtype)
        return v, tau
    # traditional householder generation
    v[0] = 1.0
    sig = v[1:].t() @ v[1:]
    # tau = 0.
    # if sig == 0 and x[0] >= 0:
    #     tau = 0.
    # elif sig == 0 and x[0] < 0:
    #     tau = -2.
    # else:
    mu = (x[0] ** 2 + sig).sqrt()
    if x[0] <= 0:
        v[0] = x[0] - mu
    else:
        v[0] = (-1 * sig / (x[0] + mu))[0]
    tau = 2 * v[0] ** 2 / (sig + v[0] ** 2)
    v /= v[0].clone()

    return v.reshape(1, -1), tau


# @torch.jit.script
def apply_house(side, v, tau, c):
    # type: (str, torch.Tensor, torch.Tensor, torch.Tensor) -> torch.Tensor
    """
    replacement for plasma core__dlarfx2
    applies the matrix H = I - tau * v * v.T to c1 and c2
    if side == left: return H @ c1 and H @ c2
    elif side == right: return c1 @ H and c2 @ H
    Parameters
    ----------
    side
    v
    tau
    c1
    c2

    Returns
    -------


    Notes
    -----
    http://icl.cs.utk.edu/plasma/docs/core__dlarfx__tbrd_8c.html#a80d7223148dcbf874885d5bb0707f231

    """
    if tau == 0:
        return c
    h = gen_house_mat(v, tau)
    if side == "left":
        r = torch.matmul(h, c)
    elif side == "right":
        r = torch.matmul(c, h)
    else:
        raise ValueError("side must be either 'left' or 'right', currently: {}".format(side))
    return r


# @torch.jit.script
def larftx2ce(uplo, vl, taul, c):
    # type: (str, torch.Tensor, torch.Tensor, torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    """
    replacement for dlarfx2ce, this is the special case that the input is 2x2,
    it applies H on both sides
    Parameters
    ----------
    uplo
    v
    tau
    c : the 2x2 matrix

    Returns
    -------

    """
    if taul == 0:
        return vl, taul, c, torch.zeros_like(vl), taul

    # generate H for first transform (the one passed)
    h = gen_house_mat(vl, taul)
    # h = torch.eye(2, dtype=vl.dtype, device=vl.device)
    # h -= tau * torch.dot(vl, vl.t())
    # apply to C
    # todo: not sure if the out keyword works here or not. found issues when combined with JIT
    c = torch.matmul(h, c)
    # generate Householder transforms to annihilate the created value
    # created value will be at (top right for LOWER, bottom left for UPPER)
    if uplo == "lower":
        vr, taur = gen_house_vec(n=2, x=c[0].flatten().clone())
    elif uplo == "upper":
        vr, taur = gen_house_vec(n=2, x=c[:, 0].flatten().clone())
    else:
        raise ValueError("uplo must be lower or upper, currently {}".format(uplo))
    # create and apply the new H
    h = gen_house_mat(vr, taur)
    # h = torch.eye(2, dtype=vr.dtype, device=vr.device)
    # h -= taur * torch.dot(vr, vr.t())
    # apply to C
    c = torch.matmul(c, h)
    return vl, taul, c, vr, taur


# @torch.jit.script
def gbelr(uplo, arr):
    # (str, int, torch.Tensor, int, int) -> Tuple[torch.Tensor, torch.Tensor]
    """
    partial function for bulge chasing, designed for the case that the matrix is upper block diagonal
    this function will start from the end of the block given to it. st and end give the global dimensions of the black,
    if the matrix is lower

    Parameters
    ----------
    n : int
        order of matrix arr
    arr : torch.Tensor
        tensor on which to do the work, will be overwritten
    st : starting index
    end : ending index

    Returns
    -------
    arr : torch.Tensor
        the same tile which was passed is returned, modified by function
    v : torch.Tensor
        the scalar elementary reflectors
    tau : torch.Tensor
        scalar factors of the elementary reflectors

    Notes
    -----
    https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6267821

    """
    # todo: need to make sure that the tile passed to this function has the full band width for it!
    # first thing is the lower case, it is the same as PLASMA implementation
    # | x . . . . . |
    # | x x . . . . |
    # | x x x . . . |
    # | x x x x . . |
    # | f x x x x * |
    # | f x x x x x |
    # get householder vector for the last two elements (f)
    v_left_dict = {}
    tau_left_dict = {}
    v_right_dict = {}
    tau_right_dict = {}
    if uplo == "lower":
        for i in range(arr.shape[0] - 1, 1, -1):  # this operates on the (i + 1)'th element
            # 1. generate house vec for the last to elements of the first column
            s, e = i - 1, i + 1
            vl, taul = gen_house_vec(x=arr[s:e, 0])
            v_left_dict[s] = vl
            tau_left_dict[s] = taul
            arr[s:e, :] = apply_house(side="left", v=vl, tau=taul, c=arr[s:e, :])
            vr, taur = gen_house_vec(x=arr[s, s:e])  # this should eliminate the temp element at *
            v_right_dict[s] = vr
            tau_right_dict[s] = taur
            # apply vr from right to the 2x2
            arr[s:e, s:e] = apply_house(side="right", v=vr, tau=taur, c=arr[s:e, s:e])
        for i in range(arr.shape[0] - 1, 1, -1):
            s, e = i - 1, arr.shape[0]
            if s + 2 < arr.shape[0]:
                arr[s + 2 :, i - 1 : i + 1] = apply_house(
                    side="right",
                    v=v_right_dict[s],
                    tau=tau_right_dict[s],
                    c=arr[s + 2 :, i - 1 : i + 1],
                )
    elif uplo == "upper":
        for i in range(arr.shape[1] - 1, 1, -1):  # this operates on the (i + 1)'th element
            # 1. generate house vec for the last to elements of the first column
            s, e = i - 1, i + 1
            vl, taul = gen_house_vec(x=arr[0, s:e])
            v_left_dict[s] = vl
            tau_left_dict[s] = taul
            arr[:, s:e] = apply_house(side="left", v=vl, tau=taul, c=arr[:, s:e].T).T
            vr, taur = gen_house_vec(x=arr[s:e, s])  # this should eliminate the temp element at *
            v_right_dict[s] = vr
            tau_right_dict[s] = taur
            # apply vr from right to the 2x2
            arr[s:e, s:e] = apply_house(side="right", v=vr, tau=taur, c=arr[s:e, s:e].T).T
        for i in range(arr.shape[1] - 1, 1, -1):
            s, e = i - 1, arr.shape[0]
            if s + 2 < arr.shape[1]:
                res = apply_house(
                    side="right",
                    v=v_right_dict[s],
                    tau=tau_right_dict[s],
                    c=arr[i - 1 : i + 1, s + 2 :].T,
                ).T
                arr[i - 1 : i + 1, s + 2 :] = res
    else:
        raise ValueError("")
    # print((arr * 100000).round())
    return arr, v_left_dict, tau_left_dict, v_right_dict, tau_right_dict
