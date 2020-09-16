import argparse
import os
import shutil
import time
import math

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed as tdist
import torch.multiprocessing as tmp
from torch.nn.parallel import DistributedDataParallel as tDDP
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
        default="resnet18",
        choices=model_names,
        help="model architecture: " + " | ".join(model_names) + " (default: resnet18)",
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
        # get the rank and size to work with
        shard_id = ht.MPI_WORLD.rank
        num_shards = ht.MPI_WORLD.size
        super(HybridPipe, self).__init__(batch_size, num_threads, device_id, shard_id)

        data_dir_list = [data_dir + d for d in os.listdir(data_dir)]
        label_dir_list = [label_dir + d for d in os.listdir(label_dir)]

        self.input = dali.ops.TFRecordReader(
            path=data_dir_list,
            index_path=label_dir_list,
            random_shuffle=True if training else False,
            shard_id=shard_id,  # todo: multi GPU/node
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
        # print("end of training if/else")
        # should this be CPU or GPU? -> if prefetching then do it on CPU before sending
        self.normalize = ops.CropMirrorNormalize(
            device="cpu",  # need to make this work with the define graph
            # dtype=dali.types.FLOAT,  # todo: not implemented on test system
            output_layout=dali.types.NCHW,
            crop=(crop, crop),
            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
            std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
        )
        self.coin = ops.CoinFlip(probability=0.5)
        self.training = training
        print(f"DALI '{dali_device}' variant, training set: {training}")

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
        print("Test mode - no DDP, no apex, RN50, 10 iterations")

    args.distributed = True  # TODO: DDDP: if ht.MPI_WORLD.size > 1 else False
    print("loss_scale = {}".format(args.loss_scale), type(args.loss_scale))
    print("\nCUDNN VERSION: {}\n".format(torch.backends.cudnn.version()))

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
        torch.manual_seed(args.local_rank)
        torch.set_printoptions(precision=10)

    args.gpu = 0
    args.world_size = ht.MPI_WORLD.size
    loc_gpus = torch.cuda.device_count()
    device = torch.device("cpu")
    loc_dist = True if loc_gpus > 1 else False
    rank = ht.MPI_WORLD.rank
    loc_rank = rank % loc_gpus
    # reduced_comm, base_loc_ranks = None, None
    # rank = 0
    twice_dist = False
    if args.distributed and loc_dist:  # todo: DDDP
        args.gpus = torch.cuda.device_count()
        twice_dist = True
        # indv_node_group = ht.MPI_WORLD.group.Excl(base_loc_ranks)
        # indv_node_comm = ht.MPI_WORLD.Create_group(indv_node_group)

        # rank = torch.distributed.get_rank()
        # args.local_rank = torch.distributed.get_rank() + (
        #     ht.MPI_WORLD.rank * torch.cuda.device_count()
        # )
        # todo: get device name from torch rank?
        device = "cuda:" + str(loc_rank)
        torch.cuda.set_device(device=device)
    elif loc_gpus == 1:
        args.gpus = torch.cuda.device_count()
        args.distributed = False
        device = "cuda:0"
        args.local_rank = 0
        torch.cuda.set_device(device)

    args.total_batch_size = args.world_size * args.batch_size
    assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()

    # todo: set the model cuda stuff later
    if (
        not args.distributed
        and hasattr(torch, "channels_last")
        and hasattr(torch, "contiguous_format")
    ):
        # if args.channels_last:
        #    memory_format = torch.channels_last
        # else:
        memory_format = torch.contiguous_format
        model = model.cuda(device).to(memory_format=memory_format)
        # model = model.to(device, memory_format=memory_format)
    else:
        model = model.cuda(device)
        # model = model.to(device)
    # model = tDDP(model)
    # Scale learning rate based on global batch size
    args.lr = args.lr * float(args.batch_size * ht.MPI_WORLD.size) / 256.0
    optimizer = torch.optim.SGD(
        model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )

    # torch_init_file = "file:///p/home/jusers/coquelin1/hdfml/heat/heat/examples/nn/distributed_test"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    os.environ["NCCL_SOCKET_IFNAME"] = "ib"
    local_rank = rank % loc_gpus
    # init_method_file = torch_init_file if not None else "file:///"
    # if init_method_file[:7] != "file://":
    #     init_method_file = "file://" + init_method_file
    torch.distributed.init_process_group(
        backend="nccl", rank=local_rank, world_size=loc_gpus  # init_method=init_method_file,
    )

    # make sure that gradients are allocated lazily, so that they are not shared here
    # model.share_memory()

    # create DP optimizer and model:
    blocking = False  # choose blocking or non-blocking parameter updates
    dp_optimizer = ht.optim.dp_optimizer.DataParallelOptimizer(optimizer, blocking)
    htmodel = ht.nn.DataParallelMultiGPU(
        model, ht.MPI_WORLD, dp_optimizer, overlap=True, distributed_twice=twice_dist
    )

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(device)

    # Optionally resume from a checkpoint
    # if args.resume:
    #     # Use a local scope to avoid dangling references
    #     def resume():
    #         if os.path.isfile(args.resume):
    #             print("=> loading checkpoint '{}'".format(args.resume))
    #             checkpoint = torch.load(
    #                 args.resume, map_location=lambda storage, loc: storage.cuda(args.gpu)
    #             )
    #             args.start_epoch = checkpoint["epoch"]
    #             # best_prec1 = checkpoint["best_prec1"]
    #             htmodel.load_state_dict(checkpoint["state_dict"])
    #             optimizer.load_state_dict(checkpoint["optimizer"])
    #             print(
    #                 "=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint["epoch"])
    #             )
    #         else:
    #             print("=> no checkpoint found at '{}'".format(args.resume))
    #
    #     resume()

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
    # print('end of first pip')
    train_loader = DALIClassificationIterator(pipe, reader_name="Reader", fill_last_batch=False)

    pipe = HybridPipe(
        batch_size=args.batch_size,
        num_threads=args.workers,
        device_id=loc_rank,  # device,
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

    total_time = AverageMeter()
    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        avg_train_time = train(device, train_loader, htmodel, criterion, dp_optimizer, epoch)
        total_time.update(avg_train_time)
        if args.test:
            break

        # evaluate on validation set
        [prec1, prec5] = validate(device, val_loader, htmodel, criterion)

        # remember best prec@1 and save checkpoint
        if args.local_rank == 0:
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "arch": args.arch,
                    "state_dict": htmodel.state_dict(),
                    "best_prec1": best_prec1,
                    "optimizer": optimizer.state_dict(),
                },
                is_best,
            )
            if epoch == args.epochs - 1:
                print(
                    "##Top-1 {0}\n"
                    "##Top-5 {1}\n"
                    "##Perf  {2}".format(prec1, prec5, args.total_batch_size / total_time.avg)
                )

        train_loader.reset()
        val_loader.reset()


def train(dev, train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()
    train_loader_len = int(math.ceil(train_loader._size / args.batch_size))
    # TODO: must set last batch!
    model.last_batch = train_loader_len
    for i, data in enumerate(train_loader):
        tt = time.perf_counter()
        input = data[0]["data"].cuda(dev)
        target = data[0]["label"].squeeze().cuda(dev).long()

        if args.prof >= 0 and i == args.prof:
            print("Profiling begun at iteration {}".format(i))
            torch.cuda.cudart().cudaProfilerStart()

        if args.prof >= 0:
            torch.cuda.nvtx.range_push("Body of iteration {}".format(i))

        adjust_learning_rate(optimizer, epoch, i, train_loader_len)
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

        if i % args.print_freq == 0:
            # Every print_freq iterations, check the loss, accuracy, and speed.
            # For best performance, it doesn't make sense to print these metrics every
            # iteration, since they incur an allreduce and some host<->device syncs.

            # Measure accuracy
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

            # Average loss and accuracy across processes for logging
            # if args.distributed:
            #     reduced_loss = reduce_tensor(loss.data)
            #     prec1 = reduce_tensor(prec1)
            #     prec5 = reduce_tensor(prec5)
            # else:
            reduced_loss = loss.data

            # to_python_float incurs a host<->device sync
            losses.update(to_python_float(reduced_loss), input.size(0))
            top1.update(to_python_float(prec1), input.size(0))
            top5.update(to_python_float(prec5), input.size(0))

            # todo: is this needed?
            # torch.cuda.synchronize()
            batch_time.update((time.time() - end) / args.print_freq)
            end = time.time()

            if args.local_rank == 0:
                print(
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
            print("Profiling ended at iteration {}".format(i))
            torch.cuda.cudart().cudaProfilerStop()
            quit()
        if ht.MPI_WORLD.rank == 0:
            print("batch", i, "time", time.perf_counter() - tt)
    #    if i == 20:
    #        break
    # for name, param in model.named_parameters():
    #    # print(model.comm.allreduce(param.clone(), ht.MPI.SUM) / ht.MPI_WORLD.size)
    #    print(param.flatten())
    #    break
    return batch_time.avg


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
        #     reduced_loss = reduce_tensor(loss.data)
        #     prec1 = reduce_tensor(prec1)
        #     prec5 = reduce_tensor(prec5)
        # else:
        reduced_loss = loss.data

        losses.update(to_python_float(reduced_loss), input.size(0))
        top1.update(to_python_float(prec1), input.size(0))
        top5.update(to_python_float(prec5), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # TODO:  Change timings to mirror train().
        if args.local_rank == 0 and i % args.print_freq == 0:
            print(
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

    print(" * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}".format(top1=top1, top5=top5))

    return [top1.avg, top5.avg]


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
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


def adjust_learning_rate(optimizer, epoch, step, len_epoch):
    """LR schedule that should yield 76% converged accuracy with batch size 256"""
    factor = epoch // 30

    if epoch >= 80:
        factor = factor + 1

    lr = args.lr * (0.1 ** factor)

    """Warmup"""
    if epoch < 5:
        lr = lr * float(1 + step + epoch * len_epoch) / (5.0 * len_epoch)

    for param_group in optimizer.torch_optimizer.param_groups:
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


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= args.world_size
    return rt


if __name__ == "__main__":
    # print('here')
    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '12355'
    # tmp.spawn(main, nprocs=torch.cuda.device_count())
    main()
