"""
Simple example demonstrating distributed SimCLR
"""
from __future__ import print_function
import argparse
import sys
import time
import torch

sys.path.append("../../")
import heat as ht
import heat.nn.functional as F
import heat.optim as optim
from heat.optim.lr_scheduler import StepLR
from heat.utils import vision_transforms
from heat.utils.data.cifar10ssl import CIFAR10SSLDataset
from heat.core.communication import MPIGather, backward, get_comm

comm = get_comm()


def simclr_loss(out_1, out_2, batch_size, temperature, npes):
    # simclr loss according to https://arxiv.org/abs/2002.05709
    out = torch.cat([out_1, out_2], dim=0)
    sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
    mask = (
        torch.ones_like(sim_matrix) - torch.eye(2 * npes * batch_size, device=sim_matrix.device)
    ).bool()
    sim_matrix = sim_matrix.masked_select(mask).view(2 * npes * batch_size, -1)
    pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
    pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
    loss = (-torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
    return loss


class Net(ht.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = ht.nn.Conv2d(3, 6, 5)
        self.conv2 = ht.nn.Conv2d(6, 16, 5)
        self.fc1 = ht.nn.Linear(16 * 5 * 5, 120)
        self.fc2 = ht.nn.Linear(120, 84)
        self.fc3 = ht.nn.Linear(84, 50)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out


def main(batch_size=32, temperature=0.5, num_iter=10, lr=1e-2):
    torch.manual_seed(0)

    transform = ht.utils.vision_transforms.Compose(
        [
            vision_transforms.RandomHorizontalFlip(p=0.5),
            vision_transforms.RandomApply(
                [vision_transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8
            ),
            vision_transforms.RandomGrayscale(p=0.2),
            vision_transforms.ToTensor(),
            vision_transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    dataset1 = CIFAR10SSLDataset(
        "../../heat/datasets",
        download=True,
        train=True,
        transform=transform,
        ishuffle=False,
    )
    dataset2 = CIFAR10SSLDataset(
        "../../heat/datasets",
        download=True,
        train=False,
        transform=transform,
        ishuffle=False,
        test_set=True,
    )

    kwargs = {"batch_size": 32}
    train_loader = ht.utils.data.datatools.DataLoader(dataset=dataset1, **kwargs)
    test_loader = ht.utils.data.datatools.DataLoader(dataset=dataset2, **kwargs)

    torch.manual_seed(0)
    model = Net()
    optimizer = optim.Adadelta(model.parameters(), lr=0.1)
    blocking = True
    dp_optim = ht.optim.DataParallelOptimizer(optimizer, blocking=blocking)
    net = ht.nn.DataParallel(
        model,
        comm=dataset1.comm,
        optimizer=dp_optim,
        blocking_parameter_updates=blocking,
        scale_gradient_average=comm.size,
    )
    torch.manual_seed(0)
    net.train()
    torch.manual_seed(0)
    model.train()
    torch.manual_seed(0)

    for xiter, (x1, x2, _) in enumerate(train_loader):
        dp_optim.zero_grad()
        x1 = net(x1)
        x2 = net(x2)

        output1 = MPIGather(x1)
        output2 = MPIGather(x2)

        if comm.rank == 0:
            # apply single node SimCLR loss function
            loss = simclr_loss(output1, output2, batch_size, temperature, comm.size)
            # envoce backward pass
            loss.backward()
            print("loss: ", xiter, loss)
        if comm.rank > 0:
            # envoce backward pass on dummy variables
            backward(output1 + output2)

        dp_optim.step()


main()
