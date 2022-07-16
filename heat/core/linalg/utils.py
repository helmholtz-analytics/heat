"""
Functions related to householder reflectors.
"""
import torch
from typing import Tuple
from ..communication import MPI
import heat as ht
from .. import arithmetics
from .. import complex_math   
from .. import constants
from .. import exponential
from ..dndarray import DNDarray
from .. import factories
from .. import manipulations
from .. import rounding
from .. import sanitation
from .. import statistics
from .. import stride_tricks
from .. import types


__all__ = [
    "gen_house_mat",
    "gen_house_vec",
    "apply_house",
    "gbelr",
    "full_H",
    "apply_house_left",
    "apply_house_right",
]


# @torch.jit.script
def gen_house_mat(v, tau):
    """
    Return Householder reflector. Similar to Full_H function.
    """
    # type: (torch.Tensor, torch.Tensor) -> torch.Tensor
    h = torch.eye(v.numel(), dtype=v.dtype, device=v.device)
    h -= tau.item() * torch.matmul(v.T, v)
    return h


# @torch.jit.script
def gen_house_vec(x):
    # type: (DNDarray) -> Tuple[DNDarray, int]
    """
    What is implemented now is only generating ONE reflector,

    Note, this overwrites x, todo: does it?
    Parameters
    ----------
    alpha : float
        the value alpha
    x : torch.Tensor
        overwritten
    overwrite: bool
        if true: overwrite x with the transform vector
        default: True

    Returns
    -------
    tau : int
          tau value
    v : DNDarray
        Output vector
    Notes
    -----
    https://www.netlib.org/lapack/explore-html/d8/d9b/group__double_o_t_h_e_rauxiliary_gaabb59655e820b3551af27781bd716143.html
    What is implemented now is only generating ONE reflector, for more would need to implement the following:
    http://icl.cs.utk.edu/plasma/docs/dlarft_8f_source.html

    """
    # x = ht.larray(x)
    v = ht.copy(x)
    # The input vector is copied into 'v'.
    sig = ht.dot((v[1:]), (v[1:].T))
    # sig = sum of squares of all the elements of v except the 1st one (v[0])

    v[0] = 1.0
    # if isinstance(alpha, (float, int)):
    # alpha = torch.tensor(alpha, device=x.device, dtype=x.dtype)
    # print(x.dtype)

    # info.eps gives the difference between 1.0 and the next smallest representable float larger than 1.0
    info = ht.finfo(ht.float32)
    if sig < info.eps:
        # which means all the elements in v excluding the 1st element are ~ 0
        tau = 0
        return v, tau

    else:
        # traditional householder generation

        # tau = 0.
        # if sig == 0 and x[0] >= 0:
        #     tau = 0.
        # elif sig == 0 and x[0] < 0:
        #     tau = -2.
        # else:
        mu = (x[0] ** 2 + sig).sqrt()
        # mu is esentially the norm of the vector input vector 'x'.

        if x[0] <= 0:
            v[0] = x[0] - mu
        else:
            v[0] = -1 * sig / (x[0] + mu)

        tau = 2 * v[0] ** 2 / (sig + v[0] ** 2)
        v /= v[0]

    v = ht.flatten(v)
    return v, tau


def full_H(n, i, v, tau):
    """
    Generate the full householder matrix H
    If we want to return U1,vt1 in the bi_diagonalize function in future,
    we can use this function.

    U1[:] = U1 @ full_H(total_rows, i, h_left, tau_left)
    vt1[:] = full_H(total_cols, i+1, h_right, tau_right) @ (vt1)
    
    Parameters
    ------------
    n : int - total rows/cols of the input matrix.
    i : int
    v : dndarray - h_left or h_right
    tau: int
    
    Returns
    --------
    H : dndarray
    """
    H = ht.eye(n)
    H[i:, i:] -= tau * ht.outer(v, v)
    return H


def apply_house_left(sub_arr, h_left, tau_left, U1, total_rows, i):
    """
    We are applying the left householder transform to the sub_arr = arr[i:,i:]
    after finding h.
    we mutate the sub_arr as, sub_arr[:] = h @ sub_arr
    This function makes all the elements below the diagonal in the ith column of the sub_arr = zero.
    
    Parameters
    ------------
    sub_arr : dndaarray - part of the input matrix.
    h_left : dndarray
    tau_left : int
    U1 : dndarray
    total_rows : int - total rows of the input matrix.
    i: int
    
    Returns
    --------
    Returns none but mutates the input sub_arr matrix.
    
    """
    m, n = sub_arr.shape
    H = ht.eye(total_rows)
    h = H[total_rows - m :, total_rows - m :]

    h -= tau_left * ht.outer(h_left, h_left)

    sub_arr[:] = h @ sub_arr
    # U1[:] = U1 @ (full_H(total_rows, i, h_left, tau_left))


def apply_house_right(sub_arr, h_right, tau_right, vt1, total_cols, i):
    """
    We are applying the right householder transform to the sub_arr = arr[i:,i+1:]
    after finding h.
    we mutate the sub_arr as, sub_arr[:] = sub_arr @ h
    This function makes all the elements to the right of diagonal and upper diagonal elements,
    in the ith row of the sub_arr = zero.
    
    Parameters
    ------------
    sub_arr : dndaarray - part of the input matrix.
    h_right : dndarray
    tau_right : int
    vt1 : dndarray
    total_cols : int - total cols of the input matrix.
    i: int
    
    Returns
    --------
    Returns none but mutates the input sub_arr matrix.
    
    """
    m, n = sub_arr.shape
    H = ht.eye(total_cols)
    h = H[total_cols - n :, total_cols - n :]

    h -= tau_right * ht.outer(h_right, h_right)

    sub_arr[:] = (sub_arr) @ (h)
    # vt1[:] = (full_H(total_cols, i+1, h_right, tau_right)) @ (vt1)


# @torch.jit.script
def apply_house(side, v, tau, c):
    # type: (str, torch.Tensor, torch.Tensor, torch.Tensor) -> torch.Tensor
    """
    Applies the matrix H = I - tau * v * v.T to c1 and c2
    if side == left: return H @ c1 and H @ c2
    elif side == right: return c1 @ H and c2 @ H
    Parameters
    ----------
    side
    v
    tau
    c

    Returns
    -------
    Depending on string side
    torch.matmul(h,c) or torch.matmul(c,h) : torch.tensor

    Notes
    -----
    http://icl.cs.utk.edu/plasma/docs/core__dlarfx__tbrd_8c.html#a80d7223148dcbf874885d5bb0707f231

    """
    if tau == 0:
        return c
    h = gen_house_mat(v, tau)
    if side == "left":
        return torch.matmul(h, c)
    elif side == "right":
        return torch.matmul(c, h)
    else:
        raise ValueError("side must be either 'left' or 'right', currently: {}".format(side))


# @torch.jit.script
def gbelr(uplo, arr):
    # (str, int, torch.Tensor, int, int) -> Tuple[torch.Tensor, torch.Tensor]
    """
    Partial function for bulge chasing, designed for the case that the matrix is upper block diagonal
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
    uplo: str
          lower or upper

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
            # eleminate the bludge created by above
            vr, taur = gen_house_vec(x=arr[s, s:e])
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
        raise ValueError("?")
    # print((arr * 100000).round())
    return arr, v_left_dict, tau_left_dict, v_right_dict, tau_right_dict


def gbrce(uplo, arr, st, v_left, tau_left, v_right, tau_right):
    # this is the same stuff as the gbelr without the initial generation
    # first step: apply house vectors from right (upper, left for lower)
    #   generate the householder vectors to eliminate the bulge
    #   apply these to the left
    # todo: st and end should be global indices, need to convert to local ones
    if uplo == "lower":
        pass
        # end = arr.shape[1]
        # # apply the vectors from the right (essentially this continues the application form the first last function)
        # #   (starting at st, should be the same as in the function which created the vectors)
        # # st, end = 0, arr.size[1]
        # # todo: only need to pass in the section of the band with elements:
        # #   starting column is 1 more than the start index of the previous function
        # for i in range(end - 1, -1, -1):
        #     # each vector is 2 elements and operates on two columns, i and i + 1
        #     s0, s1, e = 0, i, i + 1
        #     if s0 + 2 < arr.shape[0]:
        #         arr[s0:e, s1:e] = apply_house(
        #             side="right", v=v_right[st + i], tau=tau_right[st + i], c=arr[s0:e, s1:e]
        #         )
        #     # todo: move to next loop? same order i guess....
        #     # generate the vectors to eliminate the created bulges
        #     vl, taul = gen_house_vec(x=arr[i : e + 1, i])
        #     v_left[st + i] = vl
        #     tau_left[st + i] = taul
        # for i in range(end - 1, -1, -1):
        #     # after the house vectors are created, need to apply them from the left
        #     s0, s1, e = i, i, i + 1
        #     if s0 + 2 < arr.shape[0]:
        #         arr[s0:e, s1:] = apply_house(
        #             side="left", v=v_left[st + i], tau=taur, c=arr[s:e, s:e]
        #         )
        #     pass
