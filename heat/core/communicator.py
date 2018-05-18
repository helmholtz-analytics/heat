import torch.distributed as dist


def init_dist():
    dist.init_process_group('mpi')


init_dist()


class NoneCommunicator:
    @staticmethod
    def is_distributed():
        return False


class MPICommunicator:
    def __init__(self, group=dist.group.WORLD):
        self.rank = dist.get_rank()
        self.comm_size = dist.get_world_size()
        self.group = group

    @staticmethod
    def is_distributed():
        return True
