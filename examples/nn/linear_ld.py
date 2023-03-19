import argparse

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from thop import profile, clever_format
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.datasets import CIFAR10
from tqdm import tqdm

import utils
from model import Model
from datetime import datetime

from timeit import default_timer as timer

import numpy as np
from torchvision import datasets
from torchvision import transforms

from PIL import Image, ImageOps, ImageFilter

from torchinfo import summary


class Net(nn.Module):
    def __init__(self, num_class, pretrained_path):
        super(Net, self).__init__()

        # encoder
        self.f = Model().f
        # classifier
        self.fc = nn.Linear(2048, num_class, bias=True)
        ckpt = torch.load(pretrained_path, map_location='cpu')
        self.load_state_dict(ckpt['model_state_dict'], strict=False)
        # self.load_state_dict(torch.load(pretrained_path, map_location='cpu'), strict=False)

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.fc(feature)
        return out


# train or test for one epoch
def train_val(net, data_loader, train_optimizer):
    is_train = train_optimizer is not None
    net.train() if is_train else net.eval()

    total_loss, total_correct_1, total_correct_5, total_num, data_bar = 0.0, 0.0, 0.0, 0, tqdm(data_loader)
    with (torch.enable_grad() if is_train else torch.no_grad()):
        for data, target in data_bar:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            out = net(data)
            loss = loss_criterion(out, target)

            if is_train:
                train_optimizer.zero_grad()
                loss.backward()
                train_optimizer.step()

            total_num += data.size(0)
            total_loss += loss.item() * data.size(0)
            prediction = torch.argsort(out, dim=-1, descending=True)
            total_correct_1 += torch.sum((prediction[:, 0:1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_correct_5 += torch.sum((prediction[:, 0:5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()

            data_bar.set_description('{} Epoch: [{}/{}] Loss: {:.4f} ACC@1: {:.2f}% ACC@5: {:.2f}%'
                                     .format('Train' if is_train else 'Test', epoch, epochs, total_loss / total_num,
                                             total_correct_1 / total_num * 100, total_correct_5 / total_num * 100))

    return total_loss / total_num, total_correct_1 / total_num * 100, total_correct_5 / total_num * 100


def get_cifar10(train_labeled_idxs, root):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

    train_labeled_dataset = CIFAR10SSL_Linear(
        root, train_labeled_idxs, train=True,
        transform=train_transform)

    return train_labeled_dataset


class CIFAR10SSL_Linear(datasets.CIFAR10):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            pos_1 = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return pos_1, target


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Linear Evaluation')
    parser.add_argument('--model_path', type=str, default='results/semisup/128_0.5_512_400_1.0_1.0_cpt.pth',
                        help='The pretrained model path')
    parser.add_argument('--batch_size', type=int, default=512, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', type=int, default=100, help='Number of sweeps over the dataset to train')
    parser.add_argument('--trainwhole', type=int, default=0, help='0: only train FC. 1: train whole model')


    args = parser.parse_args()

    start_time = timer()

    model_path, batch_size, epochs = args.model_path, args.batch_size, args.epochs
    ckpt = torch.load(model_path, map_location='cpu')
    print(f"Distribution of used labeled examples in pretraining: {ckpt['dist_labeled']}")
    labeled_idxs = ckpt['labeled_idxs']
    print(f"Labeled idxs: {labeled_idxs}")

    train_data = get_cifar10(labeled_idxs, 'data')
    print(f"Length train dataset: {len(train_data)}")
    train_loader = DataLoader(train_data, batch_size=args.batch_size, num_workers=1, pin_memory=True)

    test_data = CIFAR10(root='data', train=False, transform=utils.test_transform, download=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)

    model = Net(num_class=len(train_data.classes), pretrained_path=model_path).cuda()
    if args.trainwhole == 0:
        print("Training only FC layer...")
        for param in model.f.parameters():
            param.requires_grad = False
    else:
        print("Training whole model...")

    input_size = (args.batch_size, 3, 32, 32)
    summary(model=model,
           input_size= input_size,  # make sure this is "input_size", not "input_shape"
           # col_names=["input_size"], # uncomment for smaller output
           col_names=["input_size", "output_size", "num_params", "trainable"])

    flops, params = profile(model, inputs=(torch.randn(1, 3, 32, 32).cuda(),))
    flops, params = clever_format([flops, params])
    print('# Model Params: {} FLOPs: {}'.format(params, flops))
    optimizer = optim.Adam(model.fc.parameters(), lr=1e-3, weight_decay=1e-6)
    loss_criterion = nn.CrossEntropyLoss()
    results = {'train_loss': [], 'train_acc@1': [], 'train_acc@5': [],
               'test_loss': [], 'test_acc@1': [], 'test_acc@5': []}

    title1 = args.model_path.split('/')[2]
    if args.trainwhole == 0:
        trainwhole = "FC"
    else:
        trainwhole = "whole"
    #timestamp = datetime.now().strftime("%m-%d")  # returns current date in YYYY-MM-DD format

    save_name_pre = '{}_{}'.format(trainwhole, title1)
    print(f"Saving to file: {save_name_pre}_linear.csv")

    best_acc = 0.0
    for epoch in range(1, epochs + 1):

        train_loss, train_acc_1, train_acc_5 = train_val(model, train_loader, optimizer)
        results['train_loss'].append(train_loss)
        results['train_acc@1'].append(train_acc_1)
        results['train_acc@5'].append(train_acc_5)
        test_loss, test_acc_1, test_acc_5 = train_val(model, test_loader, None)
        print(test_loss, test_acc_1, test_acc_5)
        results['test_loss'].append(test_loss)
        results['test_acc@1'].append(test_acc_1)
        results['test_acc@5'].append(test_acc_5)
        # save statistics
        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        data_frame.to_csv('results/noloop/linear/{}_linear.csv'.format(save_name_pre), index_label='epoch')
        if test_acc_1 > best_acc:
            best_acc = test_acc_1
            torch.save(model.state_dict(), 'results/noloop/linear/{}_linmodel.pth'.format(save_name_pre))

    end_time = timer()
    total_time = end_time - start_time
    total_time_mins = total_time / 60
    print(f"Train time on: {total_time_mins: .3f} minutes) \n")
