from __future__ import print_function
import argparse
import csv
from mpi4py import MPI
import numpy as np
import timeit
import torch
import torchvision.models as models
import sys

sys.path.append("../../")
import heat as ht
import heat.nn.functional as F
import heat.optim as optim
from heat.utils.data.datatools import Dataset


# file path for results csv
CSV_PATH = "./grad_perf_eval.csv"

# nn id mapping
NN_ID_MAPPING = {
    1: ("AlexNet", models.AlexNet),
    2: ("ResNet-101", models.resnet101),
    3: ("VGG-16", models.vgg16),
}


# extension of the Dataset class to provide labeled data
class LabeledDataset(Dataset):
    def __init__(self, data, targets):
        super().__init__(data)
        self.targets = targets

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])

        return img, target


# writes measure results to csv file
def save_results(
    nn_name,
    batch_size,
    node_cnt,
    blocking,
    strong_scaling,
    repetitions,
    img_size,
    epochs,
    min_duration,
    raw_durations,
):
    # create file if not exists
    with open(CSV_PATH, mode="a+") as csv_file:
        # go to beginning of the file
        csv_file.seek(0)

        # get number of rows
        reader = csv.reader(csv_file, delimiter=",")
        data = list(reader)
        row_count = len(data)

        writer = csv.writer(csv_file, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
        fieldnames = [
            "id",
            "nn_name",
            "batch_size",
            "node_cnt",
            "blocking",
            "strong_scaling",
            "repetitions",
            "img_size",
            "epochs",
            "min_duration",
            "raw_durations",
        ]

        # write field names row, if file has just been created
        if row_count == 0:
            writer.writerow(fieldnames)
            row_count += 1

        # write content row for given conditions and result
        writer.writerow(
            [
                row_count,
                nn_name,
                batch_size,
                node_cnt,
                blocking,
                strong_scaling,
                repetitions,
                img_size,
                epochs,
                min_duration,
                raw_durations,
            ]
        )


# wrapper function to pass parameterized function to timeit
def timeit_wrapper(func, *args, **kwargs):
    def wrapped():
        return func(*args, **kwargs)

    return wrapped


# creates synthetic dataset with len(mu) classes
# dataset has shape len(mu)*sample_cnt x 3 x img_size x img_size
def generate_synthetic_data(mu, sample_cnt, img_size, node_cnt, strong_scaling):
    if not strong_scaling:
        sample_cnt *= node_cnt

    data = ht.zeros((len(mu) * sample_cnt, 3, img_size, img_size), dtype=ht.float32)
    target = ht.zeros((len(mu) * sample_cnt, 1), dtype=ht.float32)

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
        if not continue_training:
            break

    # call needed to finalize wait handles in case of non-blocking
    model.eval()


def main():
    comm = MPI.COMM_WORLD
    node_cnt = comm.Get_size()
    rank = comm.Get_rank()

    if rank == 0:
        print("HeAT Distributed Gradient Computation Benchmark")

    # Benchmark settings
    parser = argparse.ArgumentParser(description="HeAT Distributed Gradient Computation Benchmark")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        metavar="N",
        help="input batch size for training (default: 32)",
    )
    parser.add_argument(
        "--classes",
        type=int,
        default=4,
        metavar="N",
        help="number of different classes for synthetic data, should be divisor of batch size (default: 4)",
    )
    parser.add_argument(
        "--epochs", type=int, default=1, metavar="N", help="number of epochs to train (default: 1)"
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
        help="network architecture: 1 - AlexNet, 2 - ResNet-101, 3 - VGG-16 (default: 1)",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=10,
        metavar="N",
        help="number of execution repetitions for training (default: 10)",
    )
    parser.add_argument(
        "--strong-scaling",
        action="store_true",
        default=False,
        help="enables strong scaling (constant global batch size)",
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
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {"batch_size": args.batch_size}

    if rank == 0:
        print(
            "Parameters:",
            "NN =",
            NN_ID_MAPPING[args.nn_id][0],
            ", Batch_size =",
            args.batch_size,
            ", Node_count =",
            node_cnt,
            ", Blocking =",
            args.blocking,
            ", Strong_scaling =",
            args.strong_scaling,
            ", Repetitions =",
            args.repetitions,
            ", Image_size =",
            args.img_size,
            ", Epochs =",
            args.epochs,
        )

    # create synthetic data
    mu = np.arange(args.classes) * 25.5
    data, targets = generate_synthetic_data(
        mu, args.batch_size // len(mu), args.img_size, node_cnt, args.strong_scaling
    )
    dataset = LabeledDataset(data, targets)
    if rank == 0:
        print("Synthetic data has been generated. Benchmark will begin in a few moments...")

    # setup local nn
    if args.nn_id in NN_ID_MAPPING:
        tmodel = NN_ID_MAPPING[args.nn_id][1]()
    else:
        print("Invalid NN id.")
        return

    # setup local optimizer
    optimizer = optim.Adadelta(tmodel.parameters(), lr=args.lr)

    # setup dp_nn and dp_optimizer
    dp_optim = ht.optim.DataParallelOptimizer(optimizer)
    model = ht.nn.DataParallel(
        tmodel, data.comm, dp_optim, blocking_parameter_updates=args.blocking
    )

    # setup data loader
    train_loader = ht.utils.data.datatools.DataLoader(dataset.data, lcl_dataset=dataset, **kwargs)

    # create wrapper function for timeit
    timed_training = timeit_wrapper(train, args, model, device, train_loader, optimizer)

    # drop first run from measure
    timed_training()

    # measure total local runtime
    loc_durations = timeit.repeat(timed_training, repeat=args.repetitions, number=1)

    # get maximum time across nodes
    buf = ht.array(loc_durations)
    buf.comm.Allreduce(MPI.IN_PLACE, buf, MPI.MAX)
    glo_durations = buf.numpy()
    min_glo_duration = min(glo_durations)

    if rank == 0:
        print("Benchmark has been finished.")
        print("Time: ", min_glo_duration)

        # write result to csv
        save_results(
            NN_ID_MAPPING[args.nn_id][0],
            args.batch_size,
            node_cnt,
            int(args.blocking),
            int(args.strong_scaling),
            args.repetitions,
            args.img_size,
            args.epochs,
            min_glo_duration,
            glo_durations,
        )


if __name__ == "__main__":
    main()
