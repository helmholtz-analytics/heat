import heat as ht
from heat.utils.data.datatools import DistributedDataset, DistributedDataLoader
import torch
import torch
from torch.utils.data import TensorDataset, DataLoader as TorchDataloader
import pytest

from heat.testing.basic_test import TestCase

class SeedEnviroment:
    """
    Class to be used in a `with` Environment.
    Changes the torch seed to the given and then resets it to the previous one when exiting.
    """

    def __init__(self, seed: int | None = None):
        self.seed = seed

    def __enter__(self):
        self.state = torch.random.get_rng_state()

        if self.seed is not None:
            torch.random.manual_seed(self.seed)

    def __exit__(self, *args, **kwargs):
        torch.random.set_rng_state(self.state)


class TestDatatools:
    @classmethod
    def setup_class(cls):
        TestCase.setUpClass()

    def test_distributed_dataset(self):
        ...

    def test_distributed_dataloader(self):
        ...