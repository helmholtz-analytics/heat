import heat as ht
import heat.nn.functional as F
import heat.optim as optim
from heat.optim.lr_scheduler import StepLR
from heat.utils import vision_transforms
from heat.utils.data.mnist import MNISTDataset

import os
import torch

from heat.core.tests.test_suites.basic_test import TestCase


class TestDASO(TestCase):
    def test_daso(self):
        class Net(ht.nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.conv1 = ht.nn.Conv2d(1, 32, 3, 1)
                self.conv2 = ht.nn.Conv2d(32, 64, 3, 1)
                self.dropout1 = ht.nn.Dropout2d(0.25)
                self.dropout2 = ht.nn.Dropout2d(0.5)
                self.fc1 = ht.nn.Linear(9216, 128)
                self.fc2 = ht.nn.Linear(128, 10)

            def forward(self, x):
                x = self.conv1(x)
                x = F.relu(x)
                x = self.conv2(x)
                x = F.relu(x)
                x = F.max_pool2d(x, 2)
                x = self.dropout1(x)
                x = torch.flatten(x, 1)
                x = self.fc1(x)
                x = F.relu(x)
                x = self.dropout2(x)
                x = self.fc2(x)
                output = F.log_softmax(x, dim=1)
                return output

        def train(model, device, train_loader, optimizer):
            model.train()
            optimizer.last_batch = 20
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = F.nll_loss(output, target)
                ret_loss = loss.clone().detatch()
                loss.backward()
                optimizer.step()
                if batch_idx == 20:
                    break
            return ret_loss

        def MNIST_train():
            # Training settings
            args = {"epochs": 14, "batch_size": 64}
            # todo: break if there is no GPUs / CUDA
            if not torch.cuda.is_available():
                return
            torch.manual_seed(1)

            args.gpus = torch.cuda.device_count()
            loc_rank = ht.MPI_WORLD.rank % args.gpus
            args.loc_rank = loc_rank
            device = "cuda:" + str(loc_rank)
            port = str(29500)  # + (args.world_size % args.gpus))
            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["MASTER_PORT"] = port  # "29500"
            os.environ["NCCL_SOCKET_IFNAME"] = "ib"

            torch.distributed.init_process_group(
                backend="nccl", rank=loc_rank, world_size=args.gpus
            )
            torch.cuda.set_device(device)
            args.gpu = loc_rank
            device = torch.device("cuda")
            kwargs = {"batch_size": 64, "num_workers": 1, "pin_memory": True}
            transform = ht.utils.vision_transforms.Compose(
                [vision_transforms.ToTensor(), vision_transforms.Normalize((0.1307,), (0.3081,))]
            )
            dataset1 = MNISTDataset(
                "../../heat/datasets", train=True, transform=transform, ishuffle=False
            )

            train_loader = ht.utils.data.datatools.DataLoader(dataset=dataset1, **kwargs)
            model = Net().to(device)
            optimizer = optim.SGD(model.parameters(), lr=1.0)
            daso_optimizer = ht.optim.DASO(
                local_optimizer=optimizer,
                total_epochs=args["epochs"],
                max_global_skips=4,
                stability_level=0.9,  # this should make it drop every time (hopefully)
                warmup_epochs=1,
                cooldown_epochs=1,
            )
            scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
            dp_model = ht.nn.DataParallelMultiGPU(model, daso_optimizer)

            for epoch in range(1, 14):
                ls = train(dp_model, device, train_loader, daso_optimizer)
                daso_optimizer.epoch_loss_logic(ls)
                scheduler.step()
                if epoch + 1 == args.epochs:
                    train_loader.last_epoch = True

        MNIST_train()
