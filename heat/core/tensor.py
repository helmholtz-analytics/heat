import torch
import torch.distributed as dist
import operator
import numpy as np

from .communicator import *
from .stride_tricks import *


class tensor:
    def __init__(self, Comm=MPICommunicator, *args):
        self.comm = Comm()
        self.splitaxis = None
        self.__gshape = None

    @classmethod
    def set_gseed(cls, seed):  # TODO: move this somewhere else
        torch.manual_seed(seed)

    @classmethod
    def uniform(cls, shape, lower, upper):
        ret = tensor(Comm=NoneCommunicator)
        ret.array = torch.Tensor(*shape).uniform_(lower, upper)
        ret._tensor__gshape = shape

        return ret

    def load(self, path, datasetname):
        import h5py

        with h5py.File(path, "r") as f:
            dataset = f[datasetname]
            self.__gshape = tuple(dataset.shape)
            size = len(dataset)
            if self.comm.is_distributed():
                chunksize = size // self.comm.comm_size
                rest = size % self.comm.comm_size
                if rest > self.comm.rank:
                    chunksize += 1
                    offset = self.comm.rank * chunksize
                else:
                    offset = self.comm.rank * chunksize + rest

                self.splitaxis = 0  # TODO: generalize
            else:
                chunksize = size
                offset = 0

            self.array = torch.tensor(dataset[offset:offset + chunksize])

    def mean(self, axis):
        return self.sum(axis) / float(self.gshape[axis])

    def sum(self, axis=None):
        if axis is not None:
            sum_axis = self.array.sum(axis, keepdim=True)
        else:
            return self.array.sum()  # TODO: Return our own tensor

        return self.reduce_op(sum_axis, dist.reduce_op.SUM, axis)

    def argmin(self, axis):
        _, argmin_axis = self.array.min(dim=axis, keepdim=True)
        # XXX: Fix me, I am not reduce_op.MIN!
        return self.reduce_op(argmin_axis, dist.reduce_op.MIN, axis)

    def reduce_op(self, partial, op, axis):
        if self.comm.is_distributed() and (axis is None or axis == self.splitaxis):  # TODO: support more splitting axes
            # if  and axis == self.splitaxis and :
            dist.all_reduce(partial, op, self.comm.group)

            ret = tensor(NoneCommunicator)
            ret.array = partial
            ret._tensor__gshape = ret.array.shape
            return ret

        # in theory this should be done by the calling function, e.g., sum etc.
        """
        if axis is None:
            return partial
        """
        ret = tensor(Comm=self.comm.__class__)
        ret.array = partial
        ret.splitaxis = self.splitaxis
        ret._tensor__gshape = self.gshape[:axis] + \
            (1,) + self.gshape[axis + 1:]

        return ret

    def copy(self):
        ret = tensor(self.comm.__class__)
        ret._tensor__gshape = self.gshape
        ret.splitaxis = self.splitaxis
        ret.array = self.array.clone()

        return ret

    def expand_dims(self, axis=0):
        # TODO: fix negative axis
        self.array = self.array.unsqueeze(dim=axis)
        self.__gshape = self.__gshape[:axis] + (1,) + self.__gshape[axis:]
        if self.splitaxis >= axis:
            self.splitaxis += 1

        return self

    def __add__(self, other):
        return self.binop(operator.add, other)

    def __sub__(self, other):
        return self.binop(operator.sub, other)

    def __truediv__(self, other):
        return self.binop(operator.truediv, other)

    def __mul__(self, other):
        return self.binop(operator.mul, other)

    def __pow__(self, other):
        return self.binop(operator.pow, other)

    def __eq__(self, other):
        return self.binop(operator.eq, other)

    def __ne__(self, other):
        return self.binop(operator.ne, other)

    def __lt__(self, other):
        return self.binop(operator.lt, other)

    def __le__(self, other):
        return self.binop(operator.le, other)

    def __gt__(self, other):
        return self.binop(operator.gt, other)

    def __ge__(self, other):
        return self.binop(operator.ge, other)

    def binop(self, op, other):
        if np.isscalar(other):
            ret = tensor(Comm=self.comm.__class__)
            ret.array = op(self.array, other)
            ret.splitaxis = self.splitaxis
            ret._tensor__gshape = self.gshape
            return ret
        elif isinstance(other, tensor):
            retshape = broadcast_shape(self.gshape, other.gshape)

            if other.dtype != self.dtype:  # TODO: implement complex NUMPY rules
                other = other.astype(self.dtype)

            if other.splitaxis is None or other.splitaxis == self.splitaxis:
                ret = tensor(Comm=self.comm.__class__)
                ret.array = op(self.array, other.array)
                ret.splitaxis = self.splitaxis
                ret._tensor__gshape = retshape
                return ret
            else:
                raise NotImplementedError("Not implemented for other splittings")
        else:
            raise NotImplementedError("Not implemented for non scalar")

    @property
    def dtype(self):
        # TODO: return our own dtype type
        return self.array.type()

    def astype(self, dtype):
        ret = tensor(Comm=self.comm.__class__)
        ret.array = self.array.type(dtype)
        ret.splitaxis = self.splitaxis
        ret._tensor__gshape = self.gshape
        return ret

    def clip(self, a_min, a_max):
        ret = tensor(Comm=self.comm.__class__)
        ret.array = self.array.clamp(a_min, a_max)
        ret.splitaxis = self.splitaxis
        ret._tensor__gshape = self.gshape
        return ret

    def __str__(self, *args):
        return self.array.__str__(*args)

    def __repr__(self, *args):
        return self.array.__repr__(*args)

    def __getitem__(self, key):
        ret = tensor(Comm=self.comm.__class__)
        ret.array = self.array[key]
        ret.splitaxis = self.splitaxis
        ret._tensor__gshape = self.gshape
        return ret

    def __setitem__(self, key, value):
        if self.splitaxis is not None:
            raise NotImplementedError("Slicing not supported for splitaxis != None")

        if np.isscalar(value):
            self.array.__setitem__(key, value)
        elif isinstance(value, tensor):
            self.array.__setitem__(key, value.array)
        else:
            raise NotImplementedError("Not implemented for {}".format(value.__class__.__name__))

    @property
    def gshape(self):
        return self.__gshape

    @property
    def lshape(self):
        return self.array.shape
