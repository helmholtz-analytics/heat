from copy import copy as _copy
import operator
import numpy as np

from .communicator import mpi, NoneCommunicator
from .stride_tricks import *
from . import types


class tensor:
    def __init__(self, array, gshape, split, comm):
        self.__array = array
        self.__gshape = gshape
        self.__split = split
        self.__comm = comm

    @property
    def dtype(self):
        return types.as_heat_type(self.__array.dtype)

    @property
    def gshape(self):
        return self.__gshape

    @property
    def lshape(self):
        return tuple(self.__array.shape)

    @property
    def shape(self):
        return self.__gshape

    @property
    def split(self):
        return self.__split

    def astype(self, dtype, copy=True):
        """
        Returns a casted version of this array.

        Parameters
        ----------
        dtype : ht.dtype
            HeAT type to which the array is cast
        copy : bool, optional
            By default the operation returns a copy of this array. If copy is set to false the cast is performed
            in-place and this tensor is returned

        Returns
        -------
        casted_tensor : ht.tensor
            casted_tensor is a new tensor of the same shape but with given type of this tensor. If copy is True, the
            same tensor is returned instead.
        """
        casted_array = self.__array.type(types.as_torch_type(dtype))
        if copy:
            return tensor(casted_array, self.shape, self.split, _copy(self.__comm))

        self.__array = casted_array
        return self

    def __reduce_op(self, partial, op, axis):
        # TODO: document me
        # TODO: test me
        # TODO: sanitize input
        # TODO: make me more numpy API complete
        if self.__comm.is_distributed() and (axis is None or axis == self.__split):
            mpi.all_reduce(partial, op, self.__comm.group)
            return tensor(partial, partial.shape, split=None, comm=NoneCommunicator())

        # TODO: verify if this works for negative split axis
        output_shape = self.gshape[:axis] + (1,) + self.gshape[axis + 1:]
        return tensor(partial, output_shape, self.split, comm=_copy(self.__comm))

    def argmin(self, axis):
        # TODO: document me
        # TODO: test me
        # TODO: sanitize input
        # TODO: make me more numpy API complete
        # TODO: Fix me, I am not reduce_op.MIN!
        _, argmin_axis = self.__array.min(dim=axis, keepdim=True)
        return self.__reduce_op(argmin_axis, mpi.reduce_op.MIN, axis)

    def mean(self, axis):
        # TODO: document me
        # TODO: test me
        # TODO: sanitize input
        # TODO: make me more numpy API complete
        return self.sum(axis) / self.gshape[axis]

    def sum(self, axis=None):
        # TODO: document me
        # TODO: test me
        # TODO: sanitize input
        # TODO: make me more numpy API complete
        # TODO: Return our own tensor
        if axis is not None:
            sum_axis = self.__array.sum(axis, keepdim=True)
        else:
            return self.__array.sum()

        return self.__reduce_op(sum_axis, mpi.reduce_op.SUM, axis)

    def clip(self, a_min, a_max):
        # TODO: test me
        # TODO: sanitize input
        # TODO: make me more numpy API complete
        return tensor(self.__array.clamp(a_min, a_max), self.shape, self.split, _copy(self.__comm))

    def copy(self):
        # TODO: document me
        # TODO: test me
        # TODO: sanitize input
        # TODO: make me more numpy API complete
        return tensor(self.__array.clone(), self.shape, self.split, _copy(self.__comm))

    def expand_dims(self, axis):
        # TODO: document me
        # TODO: test me
        # TODO: sanitize input
        # TODO: make me more numpy API complete
        # TODO: fix negative axis
        return tensor(
            self.__array.unsqueeze(dim=axis),
            self.shape[:axis] + (1,) + self.shape[axis:],
            self.split if self.split is None or self.split < axis else self.split + 1,
            _copy(self.__comm)
        )

    def __binop(self, op, other):
        # TODO: document me
        # TODO: test me
        # TODO: sanitize input
        # TODO: make me more numpy API complete
        # TODO: ... including the actual binops
        if np.isscalar(other):
            return tensor(op(self.__array, other), self.shape, self.split, _copy(self.__comm))

        elif isinstance(other, tensor):
            output_shape = broadcast_shape(self.shape, other.shape)

            # TODO: implement complex NUMPY rules
            if other.dtype != self.dtype:
                other = other.astype(self.dtype)

            if other.split is None or other.split == self.split:
                return tensor(op(self.__array, other.__array), output_shape, self.split, _copy(self.__comm))
            else:
                raise NotImplementedError('Not implemented for other splittings')
        else:
            raise NotImplementedError('Not implemented for non scalar')

    def __add__(self, other):
        return self.__binop(operator.add, other)

    def __sub__(self, other):
        return self.__binop(operator.sub, other)

    def __truediv__(self, other):
        return self.__binop(operator.truediv, other)

    def __mul__(self, other):
        return self.__binop(operator.mul, other)

    def __pow__(self, other):
        return self.__binop(operator.pow, other)

    def __eq__(self, other):
        return self.__binop(operator.eq, other)

    def __ne__(self, other):
        return self.__binop(operator.ne, other)

    def __lt__(self, other):
        return self.__binop(operator.lt, other)

    def __le__(self, other):
        return self.__binop(operator.le, other)

    def __gt__(self, other):
        return self.__binop(operator.gt, other)

    def __ge__(self, other):
        return self.__binop(operator.ge, other)

    def __str__(self, *args):
        # TODO: document me
        # TODO: generate none-PyTorch str
        return self.__array.__str__(*args)

    def __repr__(self, *args):
        # TODO: document me
        # TODO: generate none-PyTorch repr
        return self.__array.__repr__(*args)

    def __getitem__(self, key):
        # TODO: document me
        # TODO: test me
        # TODO: sanitize input
        # TODO: make me more numpy API complete
        return tensor(self.__array[key], self.shape, self.split, _copy(self.__comm))

    def __setitem__(self, key, value):
        # TODO: document me
        # TODO: test me
        # TODO: sanitize input
        # TODO: make me more numpy API complete
        if self.__split is not None:
            raise NotImplementedError('Slicing not supported for __split != None')

        if np.isscalar(value):
            self.__array.__setitem__(key, value)
        elif isinstance(value, tensor):
            self.__array.__setitem__(key, value.__array)
        else:
            raise NotImplementedError('Not implemented for {}'.format(value.__class__.__name__))
