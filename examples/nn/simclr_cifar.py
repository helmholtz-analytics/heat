"""
Example demonstrating distributed SimCLR on Cifar-10
"""
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


def simclr_loss(
    output1: torch.Tensor,
    output2: torch.Tensor,
    batch_size: int,
    temperature: float,
    num_processes: int,
) -> torch.Tensor:
    """
    Computes the SimCLR loss.

    Parameters
    ----------
    output1 : torch.Tensor
        Output tensor of the first set of augmented images. Shape (batch_size, feature_dim).
    output2 : torch.Tensor
        Output tensor of the second set of augmented images. Shape (batch_size, feature_dim).
    batch_size : int
        Number of samples in a batch.
    temperature : float
        Temperature parameter used for the softmax function.
    num_processes : int
        Number of processing elements used for parallel computing.

    Returns
    -------
    torch.Tensor
        SimCLR loss tensor of shape (1,).
    """
    # Concatenate the output tensors.
    output = torch.cat([output1, output2], dim=0)

    # Compute the similarity matrix.
    similarity_matrix = torch.exp(torch.mm(output, output.t().contiguous()) / temperature)

    # Create a mask to exclude the diagonal entries of the similarity matrix.
    mask = (
        torch.ones_like(similarity_matrix)
        - torch.eye(2 * num_processes * batch_size, device=similarity_matrix.device)
    ).bool()

    # Mask the similarity matrix and reshape it.
    similarity_matrix = similarity_matrix.masked_select(mask).view(
        2 * num_processes * batch_size, -1
    )

    # Compute the positive similarities and concatenate them.
    positive_similarity = torch.exp(torch.sum(output1 * output2, dim=-1) / temperature)
    positive_similarity = torch.cat([positive_similarity, positive_similarity], dim=0)

    # Compute the loss.
    loss = (-torch.log(positive_similarity / similarity_matrix.sum(dim=-1))).mean()

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

    train_transform = ht.utils.vision_transforms.Compose(
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

    test_transform = ht.utils.vision_transforms.Compose(
        [
            vision_transforms.ToTensor(),
            vision_transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
        ]
    )

    dataset1 = CIFAR10SSLDataset(
        "../../heat/datasets",
        download=True,
        train=True,
        transform=train_transform,
        ishuffle=False,
    )
    dataset2 = CIFAR10SSLDataset(
        "../../heat/datasets",
        download=True,
        train=False,
        transform=test_transform,
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
