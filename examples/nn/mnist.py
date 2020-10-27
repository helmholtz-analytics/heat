from __future__ import print_function
import argparse
import torch
import time
import sys

sys.path.append("../../")
import heat as ht
import heat.nn.functional as F
import heat.optim as optim
from heat.optim.lr_scheduler import StepLR
from heat.utils import vision_transforms
from heat.utils.data.mnist import MNISTDataset

"""
This file is an example script for how to use the HeAT DataParallel class to train a network on the MNIST dataset.
To run this file execute the following in the examples/nn/ directory:
    mpirun -np N python -u mnist.py
where N is the number of processes.
"""


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


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    t_list = []
    for batch_idx, (data, target) in enumerate(train_loader):
        t = time.perf_counter()
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        # print("end forward")
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print(
                f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} "
                f"({100.0 * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}"
            )
            if args.dry_run:
                break
        t_list.append(time.perf_counter() - t)
    print("average time", sum(t_list) / len(t_list))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print(
        f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)}"
        f" ({100.0 * correct / len(test_loader.dataset):.0f}%)\n"
    )


def main():
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=14,
        metavar="N",
        help="number of epochs to train (default: 14)",
    )
    parser.add_argument(
        "--lr", type=float, default=1.0, metavar="LR", help="learning rate (default: 1.0)"
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.7,
        metavar="M",
        help="Learning rate step gamma (default: 0.7)",
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument(
        "--dry-run", action="store_true", default=False, help="quickly check a single pass"
    )
    parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--save-model", action="store_true", default=False, help="For Saving the current Model"
    )
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {"batch_size": args.batch_size}
    if use_cuda:
        kwargs.update({"num_workers": 1, "pin_memory": True, "shuffle": True})
    transform = ht.utils.vision_transforms.Compose(
        [vision_transforms.ToTensor(), vision_transforms.Normalize((0.1307,), (0.3081,))]
    )
    dataset1 = MNISTDataset("../../heat/datasets", train=True, transform=transform, ishuffle=False)
    dataset2 = MNISTDataset(
        "../../heat/datasets", train=False, transform=transform, ishuffle=False, test_set=True
    )

    train_loader = ht.utils.data.datatools.DataLoader(dataset=dataset1, **kwargs)
    test_loader = ht.utils.data.datatools.DataLoader(dataset=dataset2, **kwargs)
    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    blocking = False
    # torch.nn.parallel.DistributedDataParallel
    dp_optim = ht.optim.DataParallelOptimizer(optimizer, blocking=blocking)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    dp_model = ht.nn.DataParallel(
        model, comm=dataset1.comm, optimizer=dp_optim, blocking_parameter_updates=blocking
    )

    for epoch in range(1, args.epochs + 1):
        train(args, dp_model, device, train_loader, dp_optim, epoch)
        test(dp_model, device, test_loader)
        scheduler.step()
        if epoch + 1 == args.epochs:
            train_loader.last_epoch = True
            test_loader.last_epoch = True

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == "__main__":
    main()
