import heat as ht
import torch
import unittest


@unittest.skipIf(torch.cuda.is_available() and torch.version.hip, "not supported for HIP")
class TestPartialDataset(unittest.TestCase):
    @unittest.skipUnless(ht.supports_hdf5(), "Requires HDF5")
    def test_partial_h5_dataset(self):
        # load h5 data and get the total shape
        full_data = ht.load("heat/datasets/iris.h5", dataset="data", split=None)
        target_shape = full_data.shape

        class TestDataset(ht.utils.data.partial_dataset.PartialH5Dataset):
            def __init__(self, file, comm, load, load_len, use_gpus=False):
                super(TestDataset, self).__init__(
                    file, comm=comm, initial_load=load, load_length=load_len, use_gpu=use_gpus
                )

            def __getitem__(self, item):
                return self.data[item]

        partial_dset = TestDataset("heat/datasets/iris.h5", full_data.comm, 30, 20)
        dl = ht.utils.data.DataLoader(dataset=partial_dset, batch_size=7)
        first_epoch = None
        second_epoch = None
        for epoch in range(2):
            elems = 0
            last_batch = None
            for batch in dl:
                elems += batch.shape[0]
                if last_batch is not None:
                    self.assertFalse(torch.allclose(last_batch, batch))
                self.assertEqual(batch.shape, (7, 4))
                last_batch = batch
                if epoch == 0:
                    if first_epoch is None:
                        first_epoch = batch
                    else:
                        first_epoch = torch.cat((first_epoch, batch), dim=0)
                else:
                    if second_epoch is None:
                        second_epoch = batch
                    else:
                        second_epoch = torch.cat((second_epoch, batch), dim=0)
            self.assertTrue(elems >= (target_shape[0] - 7) // full_data.comm.size)
        self.assertFalse(torch.allclose(first_epoch, second_epoch))

        partial_dset = TestDataset("heat/datasets/iris.h5", full_data.comm, 30, 20, True)
        dl = ht.utils.data.DataLoader(
            dataset=partial_dset,
            batch_size=7,
            pin_memory=True if torch.cuda.is_available() else False,
        )
        first_epoch = None
        second_epoch = None
        for epoch in range(2):
            elems = 0
            last_batch = None
            for batch in dl:
                elems += batch.shape[0]
                if last_batch is not None:
                    self.assertFalse(torch.allclose(last_batch, batch))
                self.assertEqual(batch.shape, (7, 4))
                last_batch = batch
                if epoch == 0:
                    if first_epoch is None:
                        first_epoch = batch
                    else:
                        first_epoch = torch.cat((first_epoch, batch), dim=0)
                else:
                    if second_epoch is None:
                        second_epoch = batch
                    else:
                        second_epoch = torch.cat((second_epoch, batch), dim=0)
            self.assertTrue(elems >= (target_shape[0] - 7) // full_data.comm.size)
        self.assertFalse(torch.allclose(first_epoch, second_epoch))
