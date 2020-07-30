from __future__ import print_function
import argparse
from mpi4py import MPI
import numpy as np
import torch
import torchvision.models as models
import sys

sys.path.append("../../")
import heat as ht
import heat.nn.functional as F
import heat.optim as optim
from heat.utils.data.datatools import Dataset

import timeit


class ExtDataset(Dataset):
    def __init__(self, data, targets):
        super().__init__(data)
        self.targets = targets

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])

        return img, target


def timeit_wrapper(func, *args, **kwargs):
    def wrapped():
        return func(*args, **kwargs)

    return wrapped


def generate_synthetic_data(mu, sample_cnt, img_size):
    # creates synthetic dataset with len(mu) classes
    # dataset has shape sample_cnt x 3 x img_size x img_size
    data = ht.zeros((len(mu) * sample_cnt, 3, img_size, img_size), dtype=ht.float32)
    target = ht.zeros((len(mu) * sample_cnt, 1), dtype=ht.float64)
    for i in range(len(mu)):
        data[sample_cnt * i : sample_cnt * (i + 1), :] = (
            ht.clip(ht.random.randn(sample_cnt, 3, img_size, img_size) + mu[i], 0, 255) / 255
        )
        target[sample_cnt * i : sample_cnt * (i + 1), :] = ht.ones((sample_cnt, 1)) * i
    permutation = np.random.permutation(data.shape[0])
    return ht.array(data[permutation], ndmin=4, split=0), ht.array(target[permutation])


def train_epoch(args, model, device, train_loader, optimizer):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        if args.dry_run:
            return False
    return True


def train(args, model, device, train_loader, optimizer):
    for epoch in range(1, args.epochs + 1):
        continue_training = train_epoch(args, model, device, train_loader, optimizer)
        print("Epoch ", epoch, " of ", args.epochs, " finished.")
        if not continue_training:
            break

    # call needed to finalize wait handles in case of non-blocking
    model.eval()


def main():
    comm = MPI.COMM_WORLD
    node_cnt = comm.Get_size()
    rank = comm.Get_rank()

    # Training settings
    parser = argparse.ArgumentParser(
        description="PyTorch Distributed Gradient Computation Benchmark"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        metavar="N",
        help="input batch size for training (default: 128)",
    )
    parser.add_argument(
        "--classes",
        type=int,
        default=50,
        metavar="N",
        help="number of different classes for synthetic data (default: 50)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        metavar="N",
        help="number of epochs to train (default: 50)",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=224,
        metavar="N",
        help="height and width of image data (default: 224)",
    )
    parser.add_argument(
        "--nn-id",
        type=int,
        default=1,
        metavar="N",
        help="network architecture: 1 - AlexNet, 2 - ResNet-101, 3 - VGG_16 (default: 1)",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=10,
        metavar="N",
        help="number of execution repetitions for training (default: 10)",
    )
    parser.add_argument(
        "--lr", type=float, default=1.0, metavar="LR", help="learning rate (default: 1.0)"
    )
    parser.add_argument(
        "--blocking",
        action="store_true",
        default=False,
        help="enables blocking gradient communication",
    )
    parser.add_argument("--cuda", action="store_true", default=False, help="enables CUDA training")
    parser.add_argument(
        "--dry-run", action="store_true", default=False, help="quickly check a single pass"
    )

    args = parser.parse_args()
    use_cuda = not args.cuda and torch.cuda.is_available()
    nn_id = args.nn_id
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {"batch_size": args.batch_size}

    mu = np.arange(args.classes) * 25.5
    data, targets = generate_synthetic_data(mu, args.batch_size * node_cnt, args.img_size)
    dataset = ExtDataset(data, targets)

    if rank == 0:
        print("Synthetic data has been generated.")

    if nn_id == 1:
        tmodel = models.AlexNet()
    elif nn_id == 2:
        tmodel = models.resnet101()
    elif nn_id == 3:
        tmodel = models.vgg16()
    else:
        print("Invalid NN id.")
        return

    optimizer = optim.Adadelta(tmodel.parameters(), lr=args.lr)
    dp_optim = ht.optim.DataParallelOptimizer(optimizer)
    model = ht.nn.DataParallel(
        tmodel, data.comm, dp_optim, blocking_parameter_updates=args.blocking
    )

    train_loader = ht.utils.data.datatools.DataLoader(dataset.data, lcl_dataset=dataset, **kwargs)
    timed_training = timeit_wrapper(train, args, model, device, train_loader, optimizer)

    # drop first run from measure
    timed_training()

    # measure total local runtime
    loc_duration = timeit.timeit(timed_training, number=args.repetitions)

    # average local runtime
    loc_duration /= args.repetitions

    # get maximum time across nodes
    buf = ht.array([loc_duration])
    buf.comm.Allreduce(MPI.IN_PLACE, buf, MPI.MAX)
    glo_duration = buf.item()

    if rank == 0:
        print("Benchmark has been finished.")
        print("Time: ", glo_duration)


if __name__ == "__main__":
    main()
