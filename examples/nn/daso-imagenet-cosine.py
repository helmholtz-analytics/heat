import argparse
import os
import shutil
import time
import math
from mpi4py import MPI

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import pickle
import sys

import pandas as pd

sys.path.append("../../")
import heat as ht


def print0(*args, **kwargs):
    if ht.MPI_WORLD.rank == 0:
        print(*args, **kwargs)


import torch.utils.data.distributed
from torchvision import datasets, transforms, models

try:
    from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy
    from nvidia.dali.pipeline import pipeline_def
    import nvidia.dali.types as types
    import nvidia.dali.fn as fn
except ImportError:
    raise ImportError(
        "Please install DALI from https://www.github.com/NVIDIA/DALI to run this example."
    )


def parse():
    model_names = sorted(
        name
        for name in models.__dict__
        if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
    )
    # torch args
    parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")
    parser.add_argument(
        "--train",
        metavar="DIR",
        default="/p/project/haf/data/imagenet/train/",
        nargs="*",
        help="path(s) to training dataset (TFRecords)",
    )
    parser.add_argument(
        "--validate",
        metavar="DIR",
        default="/p/project/haf/data/imagenet/val/",
        nargs="*",
        help="path(s) to validation datasets (TFRecords)",
    )
    parser.add_argument(
        "--train_indexes",
        metavar="DIR",
        default="/p/project/haf/data/imagenet/train-idx/",
        nargs="*",
        help="path(s) to training indexes dataset (see ht.utils.data._utils.tfrecords2idx)",
    )
    parser.add_argument(
        "--validate_indexes",
        metavar="DIR",
        default="/p/project/haf/data/imagenet/val-idx/",
        nargs="*",
        help="path(s) to validation indexes dataset (see ht.utils.data._utils.tfrecords2idx)",
    )
    parser.add_argument(
        "--arch",
        "-a",
        metavar="ARCH",
        default="resnet50",
        choices=model_names,
        help="model architecture: " + " | ".join(model_names) + " (default: resnet50)",
    )
    parser.add_argument(
        "-j",
        "--workers",
        default=4,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 4)",
    )
    parser.add_argument(
        "--epochs", default=90, type=int, metavar="N", help="number of total epochs to run"
    )
    parser.add_argument(
        "--start-epoch",
        default=0,
        type=int,
        metavar="N",
        help="manual epoch number (useful on restarts)",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        default=256,
        type=int,
        metavar="N",
        help="mini-batch size per process (default: 256)",
    )
    parser.add_argument(
        "-s",
        "--batch-skip",
        default=2,
        type=int,
        metavar="N",
        help="number of batches between global parameter synchronizations",
    )
    parser.add_argument(
        "-L",
        "--local-batch-skip",
        default=1,
        type=int,
        metavar="N",
        help="number of batches between local parameter synchronizations",
    )
    parser.add_argument(
        "--gs",
        "--global-skip-decay",
        default=1,
        type=int,
        metavar="GS",
        help="number of batches after parameters are sent when global parameters are received for the global update",
    )
    parser.add_argument(
        "--lr",
        "--learning-rate",
        default=0.1,  # og: 0.0125
        type=float,
        metavar="LR",
        help="Initial learning rate.  Will be scaled by <global batch size>/256: args.lr = args.lr*float(args.batch_size*args.world_size)/256.  A warmup schedule will also be applied over the first 5 epochs.",
    )
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    parser.add_argument(
        "--weight-decay",
        "--wd",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
    )
    parser.add_argument(
        "--print-freq",
        "-p",
        default=10,
        type=int,
        metavar="N",
        help="print frequency (default: 10)",
    )
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        metavar="PATH",
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument(
        "-e",
        "--evaluate",
        dest="evaluate",
        action="store_true",
        help="evaluate model on validation set",
    )
    parser.add_argument(
        "--pretrained", dest="pretrained", action="store_true", help="use pre-trained model"
    )
    # dali args
    parser.add_argument(
        "--dali_cpu", action="store_true", help="Runs CPU based version of DALI pipeline."
    )
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--local-rank", default=0, type=int)
    parser.add_argument("--loss-scale", type=str, default=None)
    parser.add_argument(
        "-t", "--test", action="store_true", help="Launch test mode with preset arguments"
    )
    parser.add_argument(
        "--local-comms",
        default="nccl",
        type=str,
        help="communications backend for local comms (default: nccl), if NCCL isnt there, fallback is MPI",
    )
    parser.add_argument(
        "--benchmarking",
        default=False,
        type=bool,
        help="save the results to a benchmarking csv with the node count",
    )
    parser.add_argument(
        "--manual_dist",
        action="store_true",
        help="manually override the local distribution attributes, must also set the number of local GPUs",
    )
    args = parser.parse_args()
    return args


@pipeline_def
def create_dali_pipeline(
    data_dir,
    crop,
    size,
    shard_id=ht.MPI_WORLD.rank,
    num_shards=ht.MPI_WORLD.size,
    dali_cpu=False,
    is_training=True,
):
    shard_id = ht.MPI_WORLD.rank
    num_shards = ht.MPI_WORLD.size
    images, labels = fn.readers.file(
        file_root=data_dir,
        shard_id=shard_id,
        num_shards=num_shards,
        random_shuffle=is_training,
        pad_last_batch=True,
        name="Reader",
    )
    dali_device = "cpu" if dali_cpu else "gpu"
    decoder_device = "cpu" if dali_cpu else "mixed"
    device_memory_padding = 211025920 if decoder_device == "mixed" else 0
    host_memory_padding = 140544512 if decoder_device == "mixed" else 0
    if is_training:
        images = fn.decoders.image_random_crop(
            images,
            device=decoder_device,
            output_type=types.RGB,
            device_memory_padding=device_memory_padding,
            host_memory_padding=host_memory_padding,
            random_aspect_ratio=[0.8, 1.25],
            random_area=[0.1, 1.0],
            num_attempts=100,
        )
        images = fn.resize(
            images,
            device=dali_device,
            resize_x=crop,
            resize_y=crop,
            interp_type=types.INTERP_TRIANGULAR,
        )
        mirror = fn.random.coin_flip(probability=0.5)
    else:
        images = fn.decoders.image(images, device=decoder_device, output_type=types.RGB)
        images = fn.resize(
            images,
            device=dali_device,
            size=size,
            mode="not_smaller",
            interp_type=types.INTERP_TRIANGULAR,
        )
        mirror = False

    images = fn.crop_mirror_normalize(
        images.gpu(),
        dtype=types.FLOAT,
        output_layout="CHW",
        crop=(crop, crop),
        mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
        std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
        mirror=mirror,
    )
    labels = labels.gpu()
    return images, labels


# item() is a recent addition, so this helps with backward compatibility. (from DALI)
def to_python_float(t):
    if hasattr(t, "item"):
        return t.item()
    else:
        return t[0]


def save_obj(obj, name):
    with open(name + ".pkl", "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def main():
    global best_prec1, args
    best_prec1 = 0
    args = parse()

    # todo: remove??
    # test mode, use default args for sanity test
    if args.test:
        args.epochs = 1
        args.start_epoch = 0
        args.arch = "resnet50"
        args.batch_size = 64
        args.data = []
        print0("Test mode - no DDP, no apex, RN50, 10 iterations")

    args.distributed = True  # TODO: DDDP: if ht.MPI_WORLD.size > 1 else False
    print0("loss_scale = {}".format(args.loss_scale), type(args.loss_scale))
    print0("\nCUDNN VERSION: {}\n".format(torch.backends.cudnn.version()))

    cudnn.benchmark = True
    best_prec1 = 0
    # todo: remove?
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.manual_seed(ht.MPI_WORLD.rank)
        torch.set_printoptions(precision=10)
        print0("deterministic==True, seed set to global rank")
    else:
        torch.manual_seed(999999999)

    args.gpu = 0
    args.world_size = ht.MPI_WORLD.size
    args.rank = ht.MPI_WORLD.rank
    rank = args.rank
    device = torch.device("cpu")
    if torch.cuda.device_count() > 1:
        args.gpus = torch.cuda.device_count()
        loc_rank = rank % args.gpus
        args.loc_rank = loc_rank
        device = "cuda:" + str(loc_rank)
        port = str(29500)  # + (args.world_size % args.gpus))
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = port  # "29500"
        if args.local_comms == "nccl":
            os.environ["NCCL_SOCKET_IFNAME"] = "ib"
        torch.distributed.init_process_group(
            backend=args.local_comms, rank=loc_rank, world_size=args.gpus
        )
        torch.cuda.set_device(device)
        args.gpu = loc_rank
        args.local_rank = loc_rank
    elif args.gpus == 1:
        args.gpus = torch.cuda.device_count()
        args.distributed = False
        device = "cuda:0"
        args.local_rank = 0
        torch.cuda.set_device(device)
    torch.cuda.empty_cache()

    args.total_batch_size = args.world_size * args.batch_size
    assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."

    # create model
    if args.pretrained:
        print0("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print0("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()

    if (
        not args.distributed
        and hasattr(torch, "channels_last")
        and hasattr(torch, "contiguous_format")
    ):
        if args.channels_last:
            memory_format = torch.channels_last
        else:
            memory_format = torch.contiguous_format
        model = model.to(device, memory_format=memory_format)
    else:
        model = model.to(device)
    # model = tDDP(model) -> done in the ht model initialization
    # Scale learning rate based on global batch size
    # todo: change the learning rate adjustments to be reduce on plateau
    args.lr = (
        0.0125
    )  # (1. / args.world_size * (5 * (args.world_size - 1) / 6.)) * 0.0125 * args.world_size
    # args.lr = (1. / args.world_size * (5 * (args.world_size - 1) / 6.)) * 0.0125 * args.world_size
    optimizer = torch.optim.SGD(
        model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )

    # create DP optimizer and model:
    daso_optimizer = ht.optim.DASO2(
        local_optimizer=optimizer,
        total_epochs=args.epochs,
        max_global_skips=4,
        stability_level=0.05,
        module=model,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=5, threshold=0.05, min_lr=1e-4
    )
    # htmodel = ht.nn.DataParallelMultiGPU(model, daso_optimizer)
    htmodel = model

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(device)

    # Optionally resume from a checkpoint
    if args.resume:
        # Use a local scope to avoid dangling references
        def resume():
            if os.path.isfile(args.resume):
                print0("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(
                    args.resume, map_location=lambda storage, loc: storage.cuda(args.gpu)
                )
                args.start_epoch = checkpoint["epoch"]
                # best_prec1 = checkpoint["best_prec1"]
                htmodel.load_state_dict(checkpoint["state_dict"])
                optimizer.load_state_dict(checkpoint["optimizer"])

                ce = checkpoint["epoch"]
                print0(f"=> loaded checkpoint '{args.resume}' (epoch {ce})")
            else:
                try:
                    resfile = "imgnet-checkpoint-" + str(args.world_size) + ".pth.tar"
                    print0("=> loading checkpoint '{}'".format(resfile))
                    checkpoint = torch.load(
                        resfile, map_location=lambda storage, loc: storage.cuda(args.gpu)
                    )
                    args.start_epoch = checkpoint["epoch"]
                    # best_prec1 = checkpoint["best_prec1"]
                    htmodel.load_state_dict(checkpoint["state_dict"])
                    optimizer.load_state_dict(checkpoint["optimizer"])

                    ce = checkpoint["epoch"]
                    print0(f"=> loaded checkpoint '{resfile}' (epoch {ce})")
                except FileNotFoundError:
                    print0(f"=> no checkpoint found at '{args.resume}'")

        resume()
    # if args.benchmarking:
    # import pandas as pd
    nodes = str(int(daso_optimizer.comm.size / torch.cuda.device_count()))
    cwd = os.getcwd()
    fname = cwd + "/" + nodes + "imagenet-benchmark"
    if args.resume and rank == 0 and os.path.isfile(fname + ".pkl"):
        with open(fname + ".pkl", "rb") as f:
            out_dict = pickle.load(f)
        nodes2 = str(daso_optimizer.comm.size / torch.cuda.device_count())
        old_keys = [
            nodes2 + "-avg-batch-time",
            nodes2 + "-total-train-time",
            nodes2 + "-train-top1",
            nodes2 + "-train-top5",
            nodes2 + "-train-loss",
            nodes2 + "-val-acc1",
            nodes2 + "-val-acc5",
        ]
        new_keys = [
            nodes + "-avg-batch-time",
            nodes + "-total-train-time",
            nodes + "-train-top1",
            nodes + "-train-top5",
            nodes + "-train-loss",
            nodes + "-val-acc1",
            nodes + "-val-acc5",
        ]
        for k in range(len(old_keys)):
            if old_keys[k] in out_dict.keys():
                out_dict[new_keys[k]] = out_dict[old_keys[k]]
                del out_dict[old_keys[k]]
    else:
        out_dict = {
            "epochs": [],
            nodes + "-avg-batch-time": [],
            nodes + "-total-train-time": [],
            nodes + "-train-top1": [],
            nodes + "-train-top5": [],
            nodes + "-train-loss": [],
            nodes + "-val-acc1": [],
            nodes + "-val-acc5": [],
        }
        print0("Output dict:", fname)

    if args.arch == "inception_v3":
        raise RuntimeError("Currently, inception_v3 is not supported by this example.")
        # crop_size = 299
        # val_size = 320 # I chose this value arbitrarily, we can adjust.
    else:
        crop_size = 224  # should this be 256?
        val_size = 256

        data_dir = "/hkfs/work/workspace/scratch/qv2382-heat/imagenet-raw/ILSVRC/Data/CLS-LOC/"

    # todo: change this to be the old DALI loader to work on the booster
    pipe = create_dali_pipeline(
        batch_size=args.batch_size,
        num_threads=args.workers,
        device_id=args.local_rank,
        seed=12 + args.local_rank,
        data_dir=data_dir + "train/",
        crop=crop_size,
        size=val_size,
        dali_cpu=args.dali_cpu,
        # shard_id=args.local_rank,
        # num_shards=args.world_size,
        is_training=True,
    )
    pipe.build()
    train_loader = DALIClassificationIterator(
        pipe, reader_name="Reader", last_batch_policy=LastBatchPolicy.PARTIAL
    )

    pipe = create_dali_pipeline(
        batch_size=args.batch_size,
        num_threads=args.workers,
        device_id=args.local_rank,
        seed=12 + args.local_rank,
        data_dir=data_dir + "val/",
        crop=crop_size,
        size=val_size,
        dali_cpu=args.dali_cpu,
        # shard_id=args.local_rank,
        # num_shards=args.world_size,
        is_training=False,
    )
    pipe.build()
    val_loader = DALIClassificationIterator(
        pipe, reader_name="Reader", last_batch_policy=LastBatchPolicy.PARTIAL
    )

    if args.evaluate:
        validate(device, val_loader, htmodel, criterion)
        return

    model.epochs = args.start_epoch
    args.factor = 0
    total_time = AverageMeter()

    daso_optimizer.stop_local_sync()
    daso_optimizer.save_master_model_state()

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        avg_train_time, tacc1, tacc5, ls, train_time = train(
            device, train_loader, htmodel, criterion, daso_optimizer, epoch
        )
        total_time.update(avg_train_time)
        if args.test:
            break
        # evaluate on validation set
        [prec1, prec5] = validate(device, val_loader, htmodel, criterion)

        # epoch loss logic to adjust learning rate based on loss
        # daso_optimizer.epoch_loss_logic(ls)
        # avg_loss.append(ls)
        print0(
            "scheduler stuff",
            ls,
            scheduler.best * (1.0 - scheduler.threshold),
            scheduler.num_bad_epochs,
        )
        scheduler.step(ls)
        print0("next lr:", daso_optimizer.local_optimizer.param_groups[0]["lr"])

        # remember best prec@1 and save checkpoint
        if args.rank == 0:
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            # if epoch in [30, 60, 80]:
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "arch": args.arch,
                    "state_dict": htmodel.state_dict(),
                    "best_prec1": best_prec1,
                    "optimizer": optimizer.state_dict(),
                },
                is_best=is_best,
            )
            if epoch == args.epochs - 1:
                print0(
                    "##Top-1 {0}\n"
                    "##Top-5 {1}\n"
                    "##Perf  {2}".format(prec1, prec5, args.total_batch_size / total_time.avg)
                )

            out_dict["epochs"].append(epoch)
            out_dict[nodes + "-avg-batch-time"].append(avg_train_time)
            out_dict[nodes + "-total-train-time"].append(train_time)
            out_dict[nodes + "-train-top1"].append(tacc1)
            out_dict[nodes + "-train-top5"].append(tacc5)
            out_dict[nodes + "-train-loss"].append(ls)
            out_dict[nodes + "-val-acc1"].append(prec1)
            out_dict[nodes + "-val-acc5"].append(prec5)

            # save the dict to pick up after the checkpoint
            save_obj(out_dict, fname)

        train_loader.reset()
        val_loader.reset()

    if args.rank == 0:
        print("\nRESULTS\n")
        df = pd.DataFrame.from_dict(out_dict)
        with pd.option_context("display.max_rows", None, "display.max_columns", None):
            # more options can be specified also
            print(df)
        if args.benchmarking:
            try:
                fulldf = pd.read_csv(cwd + "/bench-results.csv")
                fulldf = pd.concat([df, fulldf], axis=1)
            except FileNotFoundError:
                fulldf = df
            fulldf.to_csv(cwd + "/bench-results.csv")


def train(dev, train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    total_train_time = time.perf_counter()

    # switch to train mode
    model.train()
    end = time.time()
    train_loader_len = int(math.ceil(train_loader._size / args.batch_size))
    # must set last batch for the model to work properly
    # TODO: how to handle this?
    optimizer.last_batch = train_loader_len - 1
    for i, data in enumerate(train_loader):
        input = data[0]["data"].cuda(dev)
        target = data[0]["label"].squeeze().cuda(dev).long()

        lr_warmup(optimizer, epoch, i, train_loader_len)

        # if args.test:
        if i > 20:
            break

        output = model(input)

        loss = criterion(output, target)

        # compute gradient and do SGD step
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        if i % args.print_freq == 0 or i == train_loader_len - 1:
            # Every print_freq iterations, check the loss, accuracy, and speed.
            # For best performance, it doesn't make sense to print these metrics every
            # iteration, since they incur an allreduce and some host<->device syncs.

            # Measure accuracy
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

            # Average loss and accuracy across processes for logging
            reduced_loss = loss.data

            # to_python_float incurs a host<->device sync
            losses.update(to_python_float(reduced_loss), input.size(0))
            top1.update(to_python_float(prec1), input.size(0))
            top5.update(to_python_float(prec5), input.size(0))

            batch_time.update((time.time() - end) / args.print_freq)
            end = time.time()

            print0(
                "Epoch: [{0}][{1}/{2}]\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Speed {3:.3f} ({4:.3f})\t"
                "Loss {loss.val:.10f} ({loss.avg:.4f})\t"
                "Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t"
                "Prec@5 {top5.val:.3f} ({top5.avg:.3f})".format(
                    epoch,
                    i,
                    train_loader_len,
                    args.world_size * args.batch_size / batch_time.val,
                    args.world_size * args.batch_size / batch_time.avg,
                    batch_time=batch_time,
                    loss=losses,
                    top1=top1,
                    top5=top5,
                )
            )

    # todo average loss, and top1 and top5
    total_train_time = time.perf_counter() - total_train_time
    top1.avg = reduce_tensor(torch.tensor(top1.avg), comm=model.comm)
    top5.avg = reduce_tensor(torch.tensor(top5.avg), comm=model.comm)
    batch_time.avg = reduce_tensor(torch.tensor(batch_time.avg), comm=model.comm)
    losses.avg = reduce_tensor(torch.tensor(losses.avg), comm=model.comm)
    return batch_time.avg, top1.avg, top5.avg, losses.avg, total_train_time


def validate(dev, val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()

    for i, data in enumerate(val_loader):
        input = data[0]["data"].cuda(dev)
        target = data[0]["label"].squeeze().cuda(dev).long()
        val_loader_len = int(val_loader._size / args.batch_size)

        # compute output
        with torch.no_grad():
            output = model(input)
            loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

        reduced_loss = loss.data

        losses.update(to_python_float(reduced_loss), input.size(0))
        top1.update(to_python_float(prec1), input.size(0))
        top5.update(to_python_float(prec5), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # TODO:  Change timings to mirror train().
        if i % args.print_freq == 0:
            # if i % args.print_freq == 0:
            print0(
                "Test: [{0}/{1}]\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Speed {2:.3f} ({3:.3f})\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t"
                "Prec@5 {top5.val:.3f} ({top5.avg:.3f})".format(
                    i,
                    val_loader_len,
                    args.world_size * args.batch_size / batch_time.val,
                    args.world_size * args.batch_size / batch_time.avg,
                    batch_time=batch_time,
                    loss=losses,
                    top1=top1,
                    top5=top5,
                )
            )
    top1.avg = reduce_tensor(torch.tensor(top1.avg), comm=model.comm)
    top5.avg = reduce_tensor(torch.tensor(top5.avg), comm=model.comm)
    losses.avg = reduce_tensor(torch.tensor(losses.avg), comm=model.comm)
    if args.local_rank == 0:
        print0(f"\n * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} loss: {losses.avg:.3f}\n")

    return [top1.avg, top5.avg]


def save_checkpoint(state, is_best):
    sz = ht.MPI_WORLD.size
    filename = "imgnet-checkpoint-" + str(sz) + ".pth.tar"
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, "model_best.pth.tar")


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def lr_warmup(optimizer, epoch, bn, len_epoch):
    """
    Using a high learning rate at the very beginning of training leads to a worse final
    accuracy. During the first 5 epochs the learning rate is increased in the way presenting
    in https://arxiv.org/abs/1706.02677. After this point, this function is not called.
    """
    if epoch < 5 and bn is not None:
        sz = ht.MPI_WORLD.size
        epoch += float(bn + 1) / len_epoch
        lr_adj = 1.0 / sz * (epoch * (sz - 1) / 6.0)
    else:
        return

    for param_group in optimizer.local_optimizer.param_groups:
        param_group["lr"] = args.lr * ht.MPI_WORLD.size * lr_adj


def accuracy(output, target, topk=(1,)):
    """
    Computes the precision@k for the specified values of k
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def reduce_tensor(tensor, comm):
    rt = tensor / float(comm.size)
    comm.Allreduce(MPI.IN_PLACE, rt, MPI.SUM)
    return rt


if __name__ == "__main__":
    total_time = time.perf_counter()
    main()
    print0("\n\ntotal time", time.perf_counter() - total_time)
