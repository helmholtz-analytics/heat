from typing import Optional
import heat as ht
from heat.utils.data.datatools import DistributedDataset, DistributedSampler
import torch
import unittest


class SeedEnviroment:
    """
    Class to be used in a `with` Enviroment.
    Changes the torch seed to the given and then resets it to the previous one when exiting.
    """

    def __init__(self, seed: Optional[int] = None):
        self.seed = seed

    def __enter__(self):
        self.state = torch.random.get_rng_state()

        if self.seed is not None:
            torch.random.manual_seed(self.seed)

    def __exit__(self, *args, **kwargs):
        torch.random.set_rng_state(self.state)


class TestDistbributedData(unittest.TestCase):
    def test_dataset_and_sampler(self) -> bool:

        reference = ht.arange(100, dtype=torch.int32).reshape(20, 5)

        heat_array = ht.copy(reference).resplit_(0)
        dset = DistributedDataset(heat_array)
        dsampler = DistributedSampler(dset, shuffle=True)
        dsampler._shuffle()

        # To test this, the resulting array should be balanced, have the same number of elements as the original one, and the sum of all the columns should be the same
        # And the elements should not be equal to each other.
        self.assertTrue(dset.dndarray.size == reference.size)
        self.assertTrue(dset.dndarray.shape == reference.shape)
        self.assertTrue(dset.dndarray.balanced)

        ref_col_sum = reference.sum(0)
        col_sum = dset.dndarray.sum(0)

        self.assertTrue(ht.equal(col_sum, ref_col_sum))
        self.assertFalse(ht.equal(reference, dset.dndarray))

    def test_batches(self) -> bool:
        reference = ht.array(
            [
                [10, 11, 12, 13, 14],
                [20, 21, 22, 23, 24],
                [15, 16, 17, 18, 19],
                [0, 1, 2, 3, 4],
                [5, 6, 7, 8, 9],
            ],
            split=0,
            dtype=ht.int32,
        )

        with SeedEnviroment():
            arr = ht.arange(25, dtype=ht.int32, split=0).reshape(5, 5)
            dset = DistributedDataset(arr)
            dsampler = DistributedSampler(dset, shuffle=True, seed=42)

            dataloader = torch.utils.data.DataLoader(
                dset, batch_size=1, shuffle=False, sampler=dsampler
            )

            for batch in dataloader:
                found = False
                for larray in reference.larray:
                    if not torch.isclose(batch, larray).all():
                        continue
                    found = True
                    break
                self.assertTrue(found)

    def test_dataset_exceptions(self) -> bool:
        with self.assertRaises(TypeError):
            DistributedDataset("")
        with self.assertRaises(ValueError):
            DistributedDataset(ht.zeros(2, split=1))

    def test_data_sampler_exceptions(self) -> bool:
        with self.assertRaises(TypeError):
            DistributedSampler(ht.zeros(10))
        with self.assertRaises(TypeError):
            DistributedSampler(DistributedDataset(ht.zeros(2, split=0)), shuffle="")
        with self.assertRaises(TypeError):
            DistributedSampler(DistributedDataset(ht.zeros(2, split=0)), shuffle=True, seed="")
