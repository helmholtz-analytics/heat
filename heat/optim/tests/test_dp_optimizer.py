import heat as ht

import os
import torch

from heat.core.tests.test_suites.basic_test import TestCase


class TestDASO(TestCase):
    def test_daso(self):
        if ht.MPI_WORLD.size != 8:
            # only run these tests for 2 nodes, each of which has 4 GPUs
            return
        import heat.nn.functional as F
        import heat.optim as optim

        class Model(ht.nn.Module):
            def __init__(self):
                super(Model, self).__init__()
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
                x = self.conv1(x)
                x = F.max_pool2d(F.relu(x), (2, 2))
                # If the size is a square you can only specify a single number
                x = F.max_pool2d(F.relu(self.conv2(x)), 2)
                x = x.view(-1, self.num_flat_features(x))
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                x = self.fc3(x)
                return x

            @staticmethod
            def num_flat_features(x):
                size = x.size()[1:]  # all dimensions except the batch dimension
                num_features = 1
                for s in size:
                    num_features *= s
                return num_features

        class TestDataset(ht.utils.data.Dataset):
            def __init__(self, array, ishuffle):
                super(TestDataset, self).__init__(array, ishuffle=ishuffle)

            def __getitem__(self, item):
                return self.data[item]

            def Ishuffle(self):
                if not self.test_set:
                    ht.utils.data.dataset_ishuffle(self, attrs=[["data", None]])

            def Shuffle(self):
                if not self.test_set:
                    ht.utils.data.dataset_shuffle(self, attrs=[["data", None]])

        def train(model, device, optimizer, target, batches=20):
            model.train()
            optimizer.last_batch = batches - 1
            loss_fn = torch.nn.MSELoss()
            torch.random.manual_seed(10)
            data = torch.rand(batches, 2, 1, 32, 32, device=ht.get_device().torch_device)
            for b in range(batches):
                d, t = data[b].to(device), target[b].to(device)
                optimizer.zero_grad()
                output = model(d)
                loss = loss_fn(output, t)
                ret_loss = loss.clone().detach()
                loss.backward()
                optimizer.step()
            return ret_loss

        # Training settings
        # todo: break if there is no GPUs / CUDA
        torch.manual_seed(1)

        gpus = torch.cuda.device_count()
        loc_rank = ht.MPI_WORLD.rank % gpus
        device = "cuda:" + str(loc_rank)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"
        os.environ["NCCL_SOCKET_IFNAME"] = "ib"
        torch.distributed.init_process_group(backend="nccl", rank=loc_rank, world_size=gpus)
        torch.cuda.set_device(device)
        device = torch.device("cuda")

        model = Model().to(device)
        optimizer = optim.SGD(model.parameters(), lr=0.1)
        epochs = 20
        daso_optimizer = ht.optim.DASO(
            local_optimizer=optimizer,
            total_epochs=epochs,  # args["epochs"],
            max_global_skips=8,
            stability_level=0.9999,  # this should make it drop every time (hopefully)
            warmup_epochs=1,
            cooldown_epochs=1,
            # use_mpi_groups=False,
            verbose=True,
        )
        dp_model = ht.nn.DataParallelMultiGPU(model, daso_optimizer)

        # daso_optimizer.print0("finished inti")
        target = torch.rand((20, 2, 10), device=ht.get_device().torch_device)
        for epoch in range(epochs):
            ls = train(dp_model, device, daso_optimizer, target, batches=20)
            if epoch == 0:
                first_ls = ls
            daso_optimizer.epoch_loss_logic(ls)
            # daso_optimizer.print0(epoch, ls)
        # test that the loss decreases
        self.assertTrue(ls < first_ls)
        # test if the smaller split value also works

        daso_optimizer.reset()
        daso_optimizer.split_val = 10
        daso_optimizer.verbose = False
        for epoch in range(epochs):
            ls = train(dp_model, device, daso_optimizer, target, batches=20)
            if epoch == 0:
                first_ls = ls
            daso_optimizer.epoch_loss_logic(ls)
            # daso_optimizer.print0(epoch, ls)
        # test that the loss decreases
        self.assertTrue(ls < first_ls)
