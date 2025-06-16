import heat as ht

import os
import torch
import unittest

from heat.core.tests.test_suites.basic_test import TestCase


class TestDASO(TestCase):

    @unittest.skipUnless(
        len(TestCase.get_hostnames()) >= 2
        and torch.cuda.device_count() > 1
        and TestCase.device == "cuda",
        f"only supported for GPUs and at least two nodes, Nodes = {TestCase.get_hostnames()}, torch.cuda.device_count() = {torch.cuda.device_count()}, rank = {ht.MPI_WORLD.rank}",
    )
    def test_daso(self):
        import heat.nn.functional as F
        import heat.optim as optim

        print(
            f"rank = {ht.MPI_WORLD.rank}, host = {os.uname()[1]}, torch.cuda.device_count() = {torch.cuda.device_count()}, torch.cuda.current_device() = {torch.cuda.current_device()}, NNodes = {len(TestCase.get_hostnames())}"
        )

        class Model(ht.nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.conv1 = ht.nn.Conv2d(1, 6, 3)
                self.conv2 = ht.nn.Conv2d(6, 16, 3)
                self.fc1 = ht.nn.Linear(16 * 6 * 6, 120)
                self.fc2 = ht.nn.Linear(120, 84)
                self.fc3 = ht.nn.Linear(84, 10)

            def forward(self, x):
                x = self.conv1(x)
                x = F.max_pool2d(F.relu(x), (2, 2))
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

        def train(model, device, optimizer, target, batches=20, scaler=None):
            model.train()
            optimizer.last_batch = batches - 1
            loss_fn = torch.nn.MSELoss()
            torch.random.manual_seed(10)
            data = torch.rand(batches, 2, 1, 32, 32, device=ht.get_device().torch_device)
            for b in range(batches):
                d, t = data[b].to(device), target[b].to(device)
                optimizer.zero_grad()
                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        output = model(d)
                        loss = loss_fn(output, t)
                    ret_loss = loss.clone().detach()
                    scaler.scale(loss).backward()
                else:
                    output = model(d)
                    loss = loss_fn(output, t)
                    ret_loss = loss.clone().detach()
                    loss.backward()

                optimizer.step()
            return ret_loss

        model = Model()
        optimizer = optim.SGD(model.parameters(), lr=0.1)

        with self.assertRaises(TypeError):
            ht.optim.DASO(local_optimizer="asdf", total_epochs=1)
        with self.assertRaises(TypeError):
            ht.optim.DASO(local_optimizer=optimizer, total_epochs="aa")
        with self.assertRaises(TypeError):
            ht.optim.DASO(local_optimizer=optimizer, total_epochs=1, warmup_epochs="asdf")
        with self.assertRaises(TypeError):
            ht.optim.DASO(local_optimizer=optimizer, total_epochs=1, cooldown_epochs="asdf")
        with self.assertRaises(TypeError):
            ht.optim.DASO(local_optimizer=optimizer, total_epochs=1, scheduler="asdf")
        with self.assertRaises(TypeError):
            ht.optim.DASO(local_optimizer=optimizer, total_epochs=1, stability_level="asdf")
        with self.assertRaises(TypeError):
            ht.optim.DASO(local_optimizer=optimizer, total_epochs=1, max_global_skips="asdf")
        with self.assertRaises(TypeError):
            ht.optim.DASO(local_optimizer=optimizer, total_epochs=1, sending_chunk_size="asdf")
        with self.assertRaises(TypeError):
            ht.optim.DASO(local_optimizer=optimizer, total_epochs=1, verbose="asdf")
        with self.assertRaises(TypeError):
            ht.optim.DASO(local_optimizer=optimizer, total_epochs=1, use_mpi_groups="asdf")
        with self.assertRaises(TypeError):
            ht.optim.DASO(local_optimizer=optimizer, total_epochs=1, downcast_type="asdf")
        with self.assertRaises(TypeError):
            ht.optim.DASO(local_optimizer=optimizer, total_epochs=1, comm="asdf")
        with self.assertRaises(TypeError):
            ht.optim.DASO(local_optimizer=optimizer, total_epochs=1, local_skip_factor="asdf")
        with self.assertRaises(TypeError):
            ht.optim.DASO(local_optimizer=optimizer, total_epochs=1, skip_reduction_factor="asdf")
            # local_skip_factor
            # skip_reduction_factor
        with self.assertRaises(ValueError):
            ht.optim.DASO(local_optimizer=optimizer, total_epochs=1, downcast_type=torch.bool)
        with self.assertRaises(ValueError):
            ht.optim.DASO(local_optimizer=optimizer, total_epochs=1, warmup_epochs=-1)
        with self.assertRaises(ValueError):
            ht.optim.DASO(local_optimizer=optimizer, total_epochs=1, cooldown_epochs=-1)
        with self.assertRaises(ValueError):
            ht.optim.DASO(local_optimizer=optimizer, total_epochs=1, max_global_skips=-1)
        with self.assertRaises(ValueError):
            ht.optim.DASO(local_optimizer=optimizer, total_epochs=1, sending_chunk_size=-1)
        with self.assertRaises(ValueError):
            ht.optim.DASO(local_optimizer=optimizer, total_epochs=-1)
        with self.assertRaises(ValueError):
            ht.optim.DASO(local_optimizer=optimizer, total_epochs=1, local_skip_factor=-1)
        with self.assertRaises(ValueError):
            ht.optim.DASO(local_optimizer=optimizer, total_epochs=1, skip_reduction_factor=-1)

        # Training settings
        torch.manual_seed(1)

        gpus = torch.cuda.device_count()
        loc_rank = ht.MPI_WORLD.rank % gpus
        device = f"cuda:{str(loc_rank)}"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"
        os.environ["NCCL_SOCKET_IFNAME"] = "ib"
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl", rank=loc_rank, world_size=gpus)
        torch.cuda.set_device(device)
        device = torch.device("cuda")

        model = Model().to(device)
        optimizer = optim.SGD(model.parameters(), lr=0.1)

        epochs = 20

        daso_optimizer = ht.optim.DASO(
            local_optimizer=optimizer,
            total_epochs=epochs,
            max_global_skips=8,
            stability_level=0.9999,
            warmup_epochs=1,
            cooldown_epochs=1,
            verbose=True,
        )
        dp_model = ht.nn.DataParallelMultiGPU(model, daso_optimizer)

        target = torch.rand((20, 2, 10), device=ht.get_device().torch_device)
        for epoch in range(epochs):
            ls = train(dp_model, device, daso_optimizer, target, batches=20)
            if epoch == 0:
                first_ls = ls
            daso_optimizer.epoch_loss_logic(ls)
        # test that the loss decreases
        self.assertTrue(ls < first_ls)
        # test if the smaller split value also works

        daso_optimizer.reset()
        epochs = 4
        daso_optimizer = ht.optim.DASO(
            local_optimizer=optimizer,
            total_epochs=epochs,
            max_global_skips=8,
            stability_level=0.9999,
            warmup_epochs=2,
            cooldown_epochs=1,
            use_mpi_groups=False,
            verbose=False,
            downcast_type=torch.half,
            sending_chunk_size=61194,
        )
        dp_model = ht.nn.DataParallelMultiGPU(model, daso_optimizer)
        scaler = torch.cuda.amp.GradScaler()
        daso_optimizer.add_scaler(scaler)
        for epoch in range(epochs):
            ls = train(dp_model, device, daso_optimizer, target, batches=20, scaler=scaler)
            if epoch == 0:
                first_ls = ls
            daso_optimizer.epoch_loss_logic(ls, loss_globally_averaged=True)
        # test that the loss decreases
        self.assertTrue(ls < first_ls)
        with self.assertRaises(ValueError):
            daso_optimizer._prev_params = [1, 2]
            daso_optimizer._gs_rcv_update_params_last_batch(current_ranks=[0, 4])
        with self.assertRaises(ValueError):
            daso_optimizer.last_batch = None
            daso_optimizer.step()
