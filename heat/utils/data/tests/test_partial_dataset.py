import heat as ht
import torch
import unittest


class TestPartialDataset(unittest.TestCase):
    def test_partial_h5_dataset(self):
        # load h5 data and get the total shape
        full_data = ht.load("heat/datasets/iris.h5", dataset="data", split=None)
        target_shape = full_data.shape

        class TestDataset(ht.utils.data.partial_dataset.PartialH5Dataset):
            def __init__(self, file, comm, load, load_len):
                super(TestDataset, self).__init__(
                    file, comm=comm, initial_load=load, load_length=load_len
                )

            def __getitem__(self, item):
                return self.data[item]

        partial_dset = TestDataset("heat/datasets/iris.h5", full_data.comm, 30, 20)
        dl = ht.utils.data.DataLoader(dataset=partial_dset, batch_size=5)
        first_epoch = None
        second_epoch = None
        for epoch in range(2):
            elems = 0
            last_batch = None
            for batch in dl:
                elems += batch.shape[0]
                if last_batch is not None:
                    self.assertFalse(torch.allclose(last_batch, batch))
                self.assertEqual(batch.shape, (5, 4))
                last_batch = batch
                if epoch == 0:
                    if first_epoch is None:
                        first_epoch = batch
                    else:
                        if len(first_epoch.shape) == 3 and first_epoch.shape[0] > 1:
                            batch = batch.unsqueeze(0)
                        first_epoch = torch.cat((first_epoch, batch), dim=0)
                else:
                    if second_epoch is None:
                        second_epoch = batch
                    else:
                        if len(second_epoch.shape) == 3 and second_epoch.shape[0] > 1:
                            batch = batch.unsqueeze(0)
                        second_epoch = torch.cat((second_epoch, batch), dim=0)
            self.assertTrue(elems >= target_shape[0] / full_data.comm.size)
        self.assertFalse(torch.allclose(first_epoch, second_epoch))
