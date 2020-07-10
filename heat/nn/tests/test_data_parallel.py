import heat as ht
import torch
import unittest

from heat.core.tests.test_suites.basic_test import TestCase
import heat.nn.functional as F


class TestDataParallel(unittest.TestCase):
    def test_data_parallel(self):
        class TestModel(ht.nn.Module):
            def __init__(self):
                super(TestModel, self).__init__()
                # 1 input image channel, 6 output channels, 3x3 square convolution
                # kernel
                self.conv1 = ht.nn.Conv2d(1, 6, 3)
                self.conv2 = ht.nn.Conv2d(6, 16, 3)
                # an affine operation: y = Wx + b
                self.fc1 = ht.nn.Linear(16 * 6 * 6, 120)  # 6*6 from image dimension
                self.fc2 = ht.nn.Linear(120, 84)
                self.fc3 = ht.nn.Linear(84, 10)

            def forward(self, x):
                # Max pooling over a (2, 2) window
                x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
                # If the size is a square you can only specify a single number
                x = F.max_pool2d(F.relu(self.conv2(x)), 2)
                x = x.view(-1, self.num_flat_features(x))
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                x = self.fc3(x)
                return x

        # create model and move it to GPU with id rank
        model = TestModel()
        # ddp_model = DDP(model, device_ids=[rank])
        data = ht.random.rand(2 * ht.MPI_WORLD.size, 2 * ht.MPI_WORLD.size, split=0)
        # need dataset + dataloader
        dataset = ht.utils.data.datatools.Dataset(data)
        dataloader = ht.utils.data.datatools.DataLoader(lcl_dataset=dataset, batch_size=2)
        ht_model = ht.nn.DataParallel(model, data.comm)

        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(ht_model.parameters(), lr=0.001)

        for data in dataloader:
            self.assertEqual(data.shape[0], 2)
            optimizer.zero_grad()
            outputs = ht_model(data)
            labels = ht.random.randn(2 * ht.MPI_WORLD.size, 5, split=0)
            loss_fn(outputs, labels).backward()
            optimizer.step()
