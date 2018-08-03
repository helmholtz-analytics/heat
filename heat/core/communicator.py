import abc
import torch.distributed as mpi

from .stride_tricks import sanitize_axis


def initialize_mpi():
    mpi.init_process_group('mpi')


# initialize the MPI stack
initialize_mpi()


__all__ = []


class Communicator(metaclass=abc.ABCMeta):
    @staticmethod
    @abc.abstractmethod
    def is_distributed():
        pass

    @abc.abstractmethod
    def __init__(self):
        pass

    @abc.abstractmethod
    def chunk(self, shape, split):
        """
        Calculates the chunk of data that will be assigned to this compute node given a global data shape and a split
        axis.

        Parameters
        ----------
        shape : tuple of ints
            the global shape of the data to be split
        split : int
            the axis along which to chunk the data

        Returns
        -------
        offset : int
            the offset in the split dimension
        local_shape : tuple of ints
            the resulting local shape if the global input shape is chunked on the split axis
        slices : tuple of slices
            the chunk slices with respect to the given shape
        """
        pass


class NoneCommunicator(Communicator):
    @staticmethod
    def is_distributed():
        return False

    def __init__(self):
        pass

    def chunk(self, shape, split):
        # ensure the split axis is valid, we actually do not need it
        _ = sanitize_axis(shape, split)
        return 0, shape, tuple(slice(0, end) for end in shape)


class MPICommunicator(Communicator):
    @staticmethod
    def is_distributed():
        return True

    def __init__(self, group=mpi.group.WORLD):
        self.group = group
        self.rank = mpi.get_rank()
        self.size = mpi.get_world_size()

    def chunk(self, shape, split):
        if split is None:
            return NoneCommunicator.chunk(shape, split)

        dims = len(shape)
        split = sanitize_axis(shape, split)
        size = shape[split]
        chunk = size // self.size
        remainder = size % self.size

        if remainder > self.rank:
            chunk += 1
            start = self.rank * chunk
        else:
            start = self.rank * chunk + remainder
        end = start + chunk

        return start, \
            tuple(shape[i] if i != split else end - start for i in range(dims)), \
            tuple(slice(0, shape[i]) if i != split else slice(start, end) for i in range(dims))
