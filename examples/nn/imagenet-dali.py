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
import torchvision.models as models

import sys

sys.path.append("../../")
import heat as ht

try:
    from nvidia.dali.plugin.pytorch import DALIClassificationIterator
    from nvidia.dali.pipeline import Pipeline
    import nvidia.dali as dali
    import nvidia.dali.ops as ops
    import nvidia.dali.tfrecord as tfrec
except ImportError:
    raise ImportError(
        "Please install DALI from https://www.github.com/NVIDIA/DALI to run this example."
    )


def print0(*args, **kwargs):
    if ht.MPI_WORLD.rank == 0:
        print(*args, **kwargs)


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
        default=0.1,
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
    parser.add_argument(
        "--prof", default=-1, type=int, help="Only run 10 iterations for profiling."
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
    args = parser.parse_args()
    return args


# item() is a recent addition, so this helps with backward compatibility. (from DALI)
def to_python_float(t):
    if hasattr(t, "item"):
        return t.item()
    else:
        return t[0]


class HybridPipe(Pipeline):
    def __init__(
        self,
        batch_size,
        num_threads,
        device_id,
        data_dir,
        label_dir,
        crop,
        dali_cpu=False,
        training=True,
    ):
        shard_id = ht.MPI_WORLD.rank
        num_shards = ht.MPI_WORLD.size
        super(HybridPipe, self).__init__(batch_size, num_threads, device_id, seed=68 + shard_id)

        data_dir_list = [data_dir + d for d in os.listdir(data_dir)]
        label_dir_list = [label_dir + d for d in os.listdir(label_dir)]

        self.input = dali.ops.TFRecordReader(
            path=data_dir_list,
            index_path=label_dir_list,
            random_shuffle=True if training else False,
            shard_id=shard_id,
            num_shards=num_shards,
            initial_fill=10000,
            features={
                "image/encoded": dali.tfrecord.FixedLenFeature((), dali.tfrecord.string, ""),
                "image/class/label": dali.tfrecord.FixedLenFeature([1], dali.tfrecord.int64, -1),
                "image/class/text": dali.tfrecord.FixedLenFeature([], dali.tfrecord.string, ""),
                "image/object/bbox/xmin": dali.tfrecord.VarLenFeature(dali.tfrecord.float32, 0.0),
                "image/object/bbox/ymin": dali.tfrecord.VarLenFeature(dali.tfrecord.float32, 0.0),
                "image/object/bbox/xmax": dali.tfrecord.VarLenFeature(dali.tfrecord.float32, 0.0),
                "image/object/bbox/ymax": dali.tfrecord.VarLenFeature(dali.tfrecord.float32, 0.0),
            },
        )
        # let user decide which pipeline works him bets for RN version he runs
        dali_device = "cpu" if dali_cpu else "gpu"
        decoder_device = "cpu" if dali_cpu else "mixed"
        # This padding sets the size of the internal nvJPEG buffers to be able to
        # handle all images from full-sized ImageNet without additional reallocations
        # leaving the padding in for now to allow for the case for loading to GPUs
        # todo: move info to GPUs
        device_memory_padding = 211025920 if decoder_device == "mixed" else 0
        host_memory_padding = 140544512 if decoder_device == "mixed" else 0
        if training:
            self.decode = ops.ImageDecoderRandomCrop(
                device="cpu",  # decoder_device,
                output_type=dali.types.RGB,
                device_memory_padding=device_memory_padding,
                host_memory_padding=host_memory_padding,
                random_aspect_ratio=[0.75, 1.33],
                random_area=[0.05, 1.0],
                num_attempts=100,
            )
            self.resize = ops.Resize(
                device="cpu",  # dali_device,
                resize_x=crop,
                resize_y=crop,
                interp_type=dali.types.INTERP_TRIANGULAR,
            )
        else:
            self.decode = dali.ops.ImageDecoder(device="cpu", output_type=dali.types.RGB)
            self.resize = ops.Resize(
                device="cpu", resize_shorter=crop, interp_type=dali.types.INTERP_TRIANGULAR
            )
        # should this be CPU or GPU? -> if prefetching then do it on CPU before sending
        self.normalize = ops.CropMirrorNormalize(
            device="cpu",  # need to make this work with the define graph
            # dtype=dali.types.FLOAT,  # todo: not implemented on test system (old version of DALI)
            output_layout=dali.types.NCHW,
            crop=(crop, crop),
            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
            std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
        )
        self.coin = ops.CoinFlip(probability=0.5)
        self.training = training
        print0(f"Completed init of DALI Dataset on '{dali_device}', is training set? -> {training}")

    def define_graph(self):
        inputs = self.input(name="Reader")
        images = inputs["image/encoded"]
        labels = inputs["image/class/label"] - 1
        images = self.decode(images)
        images = self.resize(images)
        if self.training:
            images = self.normalize(images, mirror=self.coin())
        else:
            images = self.normalize(images)
        return images, labels


def main():
    global best_prec1, args
    best_prec1 = 0
    args = parse()

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
    # print("here")
    # if torch.cuda.is_available():
    #     dev_id = ht.MPI_WORLD.rank % torch.cuda.device_count()
    #     # todo: change for DDDP
    #     torch.cuda.set_device(dev_id)
    # else:
    #     dev_id = None

    cudnn.benchmark = True
    best_prec1 = 0
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.manual_seed(ht.MPI_WORLD.rank)
        torch.set_printoptions(precision=10)
        print0("deterministic==True, seed set to global rank")
    else:
        torch.manual_seed(999999999)
        # torch.manual_seed(ht.MPI_WORLD.rank)

    args.gpu = 0
    args.world_size = ht.MPI_WORLD.size
    args.rank = ht.MPI_WORLD.rank
    rank = args.rank
    args.gpus = torch.cuda.device_count()
    device = torch.device("cpu")
    loc_dist = True if args.gpus > 1 else False
    loc_rank = rank % args.gpus
    args.gpu = loc_rank
    args.local_rank = loc_rank
    if args.distributed and loc_dist:
        device = "cuda:" + str(loc_rank)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"
        if args.local_comms == "nccl":
            os.environ["NCCL_SOCKET_IFNAME"] = "ib"
        torch.distributed.init_process_group(
            backend=args.local_comms, rank=loc_rank, world_size=args.gpus
        )
        torch.cuda.set_device(device)
    elif args.gpus == 1:
        args.gpus = torch.cuda.device_count()
        args.distributed = False
        device = "cuda:0"
        args.local_rank = 0
        torch.cuda.set_device(device)

    args.total_batch_size = args.world_size * args.batch_size
    assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."

    # create model
    if args.pretrained:
        print0("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print0("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()

    # todo: set the model cuda stuff later
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
    args.lr = args.lr * float(args.batch_size * ht.MPI_WORLD.size) / 256.0
    optimizer = torch.optim.SGD(
        model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )

    # create DP optimizer and model:
    skip_batches = args.batch_skip
    local_skip = args.local_batch_skip
    dp_optimizer = ht.optim.SkipBatches(
        local_optimizer=optimizer,
        skip_batches=skip_batches,
        local_skip=local_skip,
        loss_floor=1.0,
        global_skip_delay=args.gs,
    )
    htmodel = ht.nn.DataParallelMultiGPU(model, ht.MPI_WORLD, dp_optimizer)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(device)

    # Optionally resume from a checkpoint
    if args.resume:
        # print(args.resume)
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
                print0(f"=> no checkpoint found at '{args.resume}'")

        resume()

    if args.arch == "inception_v3":
        raise RuntimeError("Currently, inception_v3 is not supported by this example.")
        # crop_size = 299
        # val_size = 320 # I chose this value arbitrarily, we can adjust.
    else:
        crop_size = 224  # should this be 256?
        val_size = 256

    pipe = HybridPipe(
        batch_size=args.batch_size,
        num_threads=args.workers,
        device_id=loc_rank,
        data_dir=args.train,
        label_dir=args.train_indexes,
        crop=crop_size,
        dali_cpu=args.dali_cpu,
        training=True,
    )
    pipe.build()

    train_loader = DALIClassificationIterator(pipe, reader_name="Reader", fill_last_batch=False)

    pipe = HybridPipe(
        batch_size=args.batch_size,
        num_threads=args.workers,
        device_id=loc_rank,
        data_dir=args.validate,
        label_dir=args.validate_indexes,
        crop=val_size,
        dali_cpu=args.dali_cpu,
        training=False,
    )
    pipe.build()
    val_loader = DALIClassificationIterator(pipe, reader_name="Reader", fill_last_batch=False)

    if args.evaluate:
        validate(device, val_loader, htmodel, criterion)
        return

    model.epochs = args.start_epoch
    args.factor = 0
    total_time = AverageMeter()
    batch_time_avg, train_acc1, train_acc5, avg_loss = [], [], [], []
    val_acc1, val_acc5 = [], []
    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        avg_train_time, tacc1, tacc5, ls = train(
            device, train_loader, htmodel, criterion, dp_optimizer, epoch
        )
        total_time.update(avg_train_time)
        if args.test:
            break

        # evaluate on validation set
        [prec1, prec5] = validate(device, val_loader, htmodel, criterion)

        # epoch loss logic to adjust learning rate based on loss
        # dp_optimizer.epoch_loss_logic(ls)
        dp_optimizer.new_loss_logic(ls)
        adjust_learning_rate(dp_optimizer, ls)

        # remember best prec@1 and save checkpoint
        if args.rank == 0:
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            if epoch in [30, 60, 80]:
                save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        "arch": args.arch,
                        "state_dict": htmodel.state_dict(),
                        "best_prec1": best_prec1,
                        "optimizer": optimizer.state_dict(),
                    },
                    epoch,
                    is_best,
                )
            if epoch == args.epochs - 1:
                print0(
                    "##Top-1 {0}\n"
                    "##Top-5 {1}\n"
                    "##Perf  {2}".format(prec1, prec5, args.total_batch_size / total_time.avg)
                )
            val_acc1.append(prec1)
            val_acc5.append(prec5)
            batch_time_avg.append(total_time.avg)
            train_acc1.append(tacc1)
            train_acc5.append(tacc5)
            avg_loss.append(ls)
        train_loader.reset()
        val_loader.reset()
    if args.rank == 0:
        print("\nRESULTS\n")
        print("Epoch\tAvg Batch Time\tTrain Top1\tTrain Top5\tTrain Loss\tVal Top1\tVal Top5")
        for c in range(args.start_epoch, args.epochs):
            cp = c - args.start_epoch
            print(
                f"{c}\t{batch_time_avg[cp]}\t{train_acc1[cp]}\t{train_acc5[cp]}\t"
                f"{avg_loss[cp]}\t{val_acc1[cp]}\t{val_acc5[cp]}"
            )


def train(dev, train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()
    train_loader_len = int(math.ceil(train_loader._size / args.batch_size))
    # must set last batch for the model to work properly
    optimizer.last_batch = train_loader_len
    for i, data in enumerate(train_loader):
        # print(i)
        # tt = time.perf_counter()
        input = data[0]["data"].cuda(dev)
        target = data[0]["label"].squeeze().cuda(dev).long()

        if 0 <= args.prof == i:
            print("Profiling begun at iteration {}".format(i))
            torch.cuda.cudart().cudaProfilerStart()

        if args.prof >= 0:
            torch.cuda.nvtx.range_push("Body of iteration {}".format(i))

        lr_warmup(optimizer, epoch, i, train_loader_len)
        if args.test:
            if i > 10:
                break

        # compute output
        if args.prof >= 0:
            torch.cuda.nvtx.range_push("forward")
        # t3 = time.perf_counter()
        output = model(input)
        # print("forward", time.perf_counter() - t3)
        if args.prof >= 0:
            torch.cuda.nvtx.range_pop()
        loss = criterion(output, target)

        # compute gradient and do SGD step
        optimizer.zero_grad()

        if args.prof >= 0:
            torch.cuda.nvtx.range_push("backward")
        # t2 = time.perf_counter()
        loss.backward()
        # print("backwards time", time.perf_counter() - t2)
        if args.prof >= 0:
            torch.cuda.nvtx.range_pop()

        if args.prof >= 0:
            torch.cuda.nvtx.range_push("optimizer.step()")
        optimizer.step()
        if args.prof >= 0:
            torch.cuda.nvtx.range_pop()

        if i % args.print_freq == 0 or i == train_loader_len:
            # Every print_freq iterations, check the loss, accuracy, and speed.
            # For best performance, it doesn't make sense to print these metrics every
            # iteration, since they incur an allreduce and some host<->device syncs.

            # Measure accuracy
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

            # Average loss and accuracy across processes for logging
            # if args.distributed:
            #    reduced_loss = reduce_tensor(loss.data, comm=model.comm)
            #    prec1 = reduce_tensor(prec1, comm=model.comm)
            #    prec5 = reduce_tensor(prec5, comm=model.comm)
            # else:
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

        # Pop range "Body of iteration {}".format(i)
        if args.prof >= 0:
            torch.cuda.nvtx.range_pop()

        if args.prof >= 0 and i == args.prof + 10:
            print0("Profiling ended at iteration {}".format(i))
            torch.cuda.cudart().cudaProfilerStop()
            quit()
    # todo average loss, and top1 and top5
    top1.avg = reduce_tensor(torch.tensor(top1.avg), comm=model.comm)
    top5.avg = reduce_tensor(torch.tensor(top5.avg), comm=model.comm)
    batch_time.avg = reduce_tensor(torch.tensor(batch_time.avg), comm=model.comm)
    losses.avg = reduce_tensor(torch.tensor(losses.avg), comm=model.comm)
    return batch_time.avg, top1.avg, top5.avg, losses.avg


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

        # if args.distributed:
        #    reduced_loss = reduce_tensor(loss.data, comm=model.comm)
        #    prec1 = reduce_tensor(prec1, comm=model.comm)
        #    prec5 = reduce_tensor(prec5, comm=model.comm)
        # else:
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


def save_checkpoint(state, is_best, epoch, filename="checkpoint.pth.tar"):
    filename = "checkpoint-epoch" + str(epoch.item) + ".pth.tar"
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


def adjust_learning_rate(optimizer, loss):
    """LR schedule that should yield 76% converged accuracy with batch size 256"""
    if loss <= 1.05:
        factor = 3
    elif loss <= 1.45:
        factor = 2
    elif loss <= 2.25:
        factor = 1
    else:
        factor = 0
    lr = args.lr * (0.1 ** factor)
    print0(f"LR: {lr}, Factor: {factor}")

    for param_group in optimizer.lcl_optimizer.param_groups:
        param_group["lr"] = lr


def lr_warmup(optimizer, epoch, step, len_epoch):
    if epoch < 5 and step is not None:
        lr = args.lr * float(1 + step + epoch * len_epoch) / (5.0 * len_epoch)

    for param_group in optimizer.lcl_optimizer.param_groups:
        param_group["lr"] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
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
