import argparse
import base64
import random
import shutil
import sys
import time
import warnings

import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.models as models

sys.path.append("../../")
import heat as ht

model_names = sorted(
    name
    for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)

parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")
parser.add_argument(
    "-a",
    "--arch",
    metavar="ARCH",
    default="resnet50",
    choices=model_names,
    help="model architecture: " + " | ".join(model_names) + " (default: resnet50)",
)
parser.add_argument(
    "--epochs", default=90, type=int, metavar="N", help="number of total epochs to run"
)
parser.add_argument(
    "--start-epoch",
    default=0,
    type=int,
    metavar="N",
    help="manual epoch start number (useful on restarts)",
)
parser.add_argument(
    "-b",
    "--batch-size",
    default=256,
    type=int,
    metavar="N",
    help="mini-batch size (default: 256) for each network",
)
parser.add_argument(
    "--lr",
    "--learning-rate",
    default=0.1,
    type=float,
    metavar="LR",
    help="initial learning rate",
    dest="lr",
)
parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
parser.add_argument(
    "--wd",
    "--weight-decay",
    default=1e-4,
    type=float,
    metavar="W",
    help="weight decay (default: 1e-4)",
    dest="weight_decay",
)
parser.add_argument(
    "-p", "--print-freq", default=10, type=int, metavar="N", help="print frequency (default: 10)"
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
parser.add_argument("--seed", default=None, type=int, help="seed for initializing training. ")
# todo: what to rename the following??
parser.add_argument(
    "--checkpointing",
    action="store_true",
    help="Use multi-processing distributed training to launch "
    "N processes per node, which has N GPUs. This is the "
    "fastest way to use PyTorch for either single node or "
    "multi node data parallel training",
)

best_acc1 = 0


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        warnings.warn(
            "You have chosen to seed training. "
            "This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! "
            "You may see unexpected behavior when restarting "
            "from checkpoints."
        )

    return main_worker(args)


class ImagenetDataset(ht.utils.data.partial_dataset.PartialH5Dataset):
    def __init__(self, file, transforms=None, validate_set=False):
        names = ["images", "metadata"]
        """
        file notes
        * "images" : encoded ASCII string of the decoded RGB JPEG image.
                    - to decode: `torch.as_tensor(bytearray(base64.binascii.a2b_base64(string_repr.encode('ascii'))), dtype=torch.uint8)`
                    - note: the images must be reshaped using: `.reshape(file["metadata"]["image/height"], file["metadata"]["image/height"], 3)`
                            (3 is the number of channels, all images are RGB)
            * "metadata" : the metadata for each image quotes are the titles for each column
                    0. "image/height"
                    1. "image/width"
                    2. "image/channels"
                    3. "image/class/label"
                    4. "image/object/bbox/xmin"
                    5. "image/object/bbox/xmax"
                    6. "image/object/bbox/ymin"
                    7. "image/object/bbox/ymax"
                    8. "image/object/bbox/label"
            * "file_info" : string information related to each image
                    0. "image/format"
                    1. "image/filename"
                    2. "image/class/synset"
                    3. "image/class/text"
        """
        super(ImagenetDataset, self).__init__(
            file, dataset_names=names, transforms=transforms, validate_set=validate_set
        )

    def __getitem__(self, index):
        shape = (int(self.metadata[index][0].item()), int(self.metadata[index][1].item()), 3)
        str_repr = base64.binascii.a2b_base64(self.images[index])
        img = np.frombuffer(str_repr, dtype=np.uint8).reshape(shape)
        target = torch.as_tensor(
            self.metadata[index][3], dtype=torch.long, device=torch.device("cpu")
        )
        return img, target


def main_worker(args):
    global best_acc1

    # create model:
    if args.pretrained:
        # use pretrained?
        print(f"=> using pre-trained model '{args.arch}'")
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print(f"=> creating model '{args.arch}'")
        model = models.__dict__[args.arch]()

    if not torch.cuda.is_available():
        print("using CPU, this will be slow")
        criterion = torch.nn.CrossEntropyLoss()
    else:
        # if cuda is available, then use the GPUs which are there
        dev_id = ht.MPI_WORLD.rank % torch.cuda.device_count()
        torch.cuda.set_device(ht.MPI_WORLD.rank % torch.cuda.device_count())
        model.cuda(device=torch.device("cuda:" + str(dev_id)))
        criterion = torch.nn.CrossEntropyLoss().cuda(device=torch.device("cuda:" + str(dev_id)))

    optimizer = torch.optim.SGD(
        model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )

    # create DP optimizer and model:
    blocking = False  # choose blocking or non-blocking parameter updates
    dp_optimizer = ht.optim.dp_optimizer.DataParallelOptimizer(optimizer, blocking)
    model = ht.nn.DataParallel(
        model, ht.MPI_WORLD, dp_optimizer, blocking_parameter_updates=blocking
    )

    # Data loading code
    train_file = "/p/project/haf/data/imagenet_merged.h5"
    val_file = "/p/project/haf/data/imagenet_merged_validation.h5"
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_img_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    train_dataset = ImagenetDataset(train_file, transforms=[train_img_transform, None])
    train_loader = ht.utils.data.datatools.DataLoader(
        lcl_dataset=train_dataset, batch_size=args.batch_size, pin_memory=False
    )

    val_img_transforms = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    )
    val_dataset = ImagenetDataset(
        val_file, transforms=[val_img_transforms, None], validate_set=True
    )
    val_loader = ht.utils.data.datatools.DataLoader(
        lcl_dataset=val_dataset, batch_size=args.batch_size, pin_memory=True
    )

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(dp_optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, dp_optimizer, epoch, args)
        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if args.checkpointing:
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "arch": args.arch,
                    "state_dict": model.state_dict(),
                    "best_acc1": best_acc1,
                    "optimizer": dp_optimizer.torch_optimizer.state_dict(),
                },
                is_best,
            )
    return (len(train_dataset) + len(val_dataset)) * (args.epochs - args.start_epoch)


def train(train_loader, model, criterion, dp_optimizer, epoch, args):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        len(train_loader), [batch_time, data_time, losses, top1, top5], prefix=f"Epoch: [{epoch}]"
    )

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        dp_optimizer.zero_grad()
        loss.backward()
        dp_optimizer.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(len(val_loader), [batch_time, losses, top1, top5], prefix="Test: ")

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(" * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}".format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, "model_best.pth.tar")


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def adjust_learning_rate(dp_optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in dp_optimizer.torch_optimizer.param_groups:
        param_group["lr"] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
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


if __name__ == "__main__":
    time0 = time.perf_counter()
    total_images = main()
    total_time = time.perf_counter() - time0
    ipspr = total_images / total_time
    ipst = ht.MPI_WORLD.allreduce(ipspr, ht.MPI.SUM)
    print(f"Total time {total_time}, images/sec/process: {ipspr}, Total images/sec: {ipst}")
