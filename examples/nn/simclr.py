import argparse
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from thop import profile, clever_format
import torch.nn.functional as F
from tqdm import tqdm

from timeit import default_timer as timer

import utils
from torchvision.models.resnet import resnet50

# new
from kekmodelcifar import DATASET_GETTERS
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

import numpy as np
import sys

sys.path.append("../../")
import heat as ht
import heat.nn.functional as F
import heat.optim as optim
from heat.optim.lr_scheduler import StepLR
from heat.utils import vision_transforms
from heat.utils.data.cifar10ssl import CIFAR10SSLDataset
from heat.core.communication import AdjointGather, backward, get_comm


comm = get_comm()


class Model_Sup(nn.Module):
    def __init__(self, feature_dim=128, num_class=10):
        super(Model_Sup, self).__init__()

        self.f = []
        for name, module in resnet50().named_children():
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                self.f.append(module)
        # encoder
        self.f = nn.Sequential(*self.f)
        # projection head
        self.g = nn.Sequential(nn.Linear(2048, 512, bias=False), nn.BatchNorm1d(512),
                               nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))

        # classification head
        self.c = nn.Linear(2048, num_class, bias=True)

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)
        c = self.c(feature)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1), c


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
        Output tensor of the first set of augmented images. Shape (batch_size, feature_dim)
    output2 : torch.Tensor
        Output tensor of the second set of augmented images. Shape (batch_size, feature_dim)
    batch_size : int
        Number of samples in a batch
    temperature : float
        Temperature parameter used for the softmax function
    num_processes : int
        Number of processing elements used for parallel computing

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



def train(net, data_loader, train_optimizer, criterion, unsup_weight, sup_weight):
    net.train()
    total_loss, total_num = 0.0, 0
    total_sim_loss, total_sup_loss = 0.0, 0.0
    sup_acc = 0.0


 
    for pos_1, pos_2, target in data_loader:

        pos_1, pos_2 = pos_1.to(device, non_blocking=True), pos_2.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        _, out_1, _ = net(pos_1)
        _, out_2, _ = net(pos_2)

        out_1 = AdjointGather(out_1)
        out_2 = AdjointGather(out_2)

       
        train_optimizer.zero_grad()

        if comm.rank == 0:

            # apply single node SimCLR loss function
            loss = simclr_loss(out_1, out_2, batch_size, temperature, comm.size)

            # envoce backward pass
            loss.backward()

            print(loss)

          
        if comm.rank > 0:
            # envoce backward pass on dummy variables
            backward(out_1 + out_2)

        train_optimizer.step()

       

    return total_loss / total_num, sup_acc, total_sim_loss, total_sup_loss


# test for one epoch, use weighted knn to find the most similar images' label to assign the test image
def test(net, memory_data_loader, test_data_loader, criterion):
    net.eval()
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    sup_acc = 0.0
    total_test_loss = 0.0

    with torch.no_grad():
        # generate feature bank
        for data, _, target in tqdm(memory_data_loader, desc='Feature extracting'):
            feature, out, cfied = net(data.cuda(non_blocking=True))
            feature_bank.append(feature)
        # # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # # [N]
        feature_labels = torch.tensor(memory_data_loader.dataset.targets, device=feature_bank.device)
        # loop test data to predict the label by weighted knn search
        test_bar = tqdm(test_data_loader)
        for data, _, target in test_bar:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            feature, out, c1 = net(data)

            # test loss (SUPERVISED)
            test_loss = criterion(c1, target)
            total_test_loss += test_loss.item()

            # accuracy supervised
            acc1 = accuracy_fn(c1, target)

            total_num += data.size(0)
            # # compute cos similarity between each feature vector and feature bank ---> [B, N]
            sim_matrix = torch.mm(feature, feature_bank)
            # # [B, K]
            sim_weight, sim_indices = sim_matrix.topk(k=k, dim=-1)
            # # [B, K]
            sim_labels = torch.gather(feature_labels.expand(data.size(0), -1), dim=-1, index=sim_indices)
            sim_weight = (sim_weight / temperature).exp()

            # # counts for each class
            one_hot_label = torch.zeros(data.size(0) * k, c, device=sim_labels.device)
            # # [B*K, C]
            one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
            # # weighted score ---> [B, C]
            pred_scores = torch.sum(one_hot_label.view(data.size(0), -1, c) * sim_weight.unsqueeze(dim=-1), dim=1)
            #
            pred_labels = pred_scores.argsort(dim=-1, descending=True)
            total_top1 += torch.sum((pred_labels[:, :1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_top5 += torch.sum((pred_labels[:, :5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()

            sup_acc += acc1

            # test_bar.set_description('Test Epoch: [{}/{}] Current Acc:{:.2f}% %'
            #                          .format(epoch, epochs, acc1))

            test_bar.set_description('Test Epoch: [{}/{}] Acc@1:{:.2f}% Acc@5:{:.2f}% Current Acc:{:.2f}%'
                                     .format(epoch, epochs, total_top1 / total_num * 100, total_top5 / total_num * 100,
                                             acc1))

    sup_acc /= len(test_data_loader)
    total_test_loss /= len(test_data_loader)

    return sup_acc, total_top1 / total_num * 100, total_top5 / total_num * 100, total_test_loss



def accuracy_fn(z1, label):
    pred_probs = torch.softmax(z1, dim=0)
    z1_labels = pred_probs.argmax(dim=1)
    correct = torch.eq(label, z1_labels).sum().item()
    acc_z1 = (correct / len(z1_labels)) * 100
    return acc_z1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SimCLR')
    parser.add_argument('--feature_dim', default=128, type=int, help='Feature dim for latent vector')
    parser.add_argument('--temperature', default=0.5, type=float, help='Temperature used in softmax')
    parser.add_argument('--k', default=200, type=int, help='Top k most similar images used to predict the label')
    parser.add_argument('--batch_size', default=32, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', default=5, type=int, help='Number of sweeps over the dataset to train')
    parser.add_argument('--unsup_weight', type=float, default=0.5, help='weight for the UNsupervised loss')
    parser.add_argument('--sup_weight', type=float, default=0.5, help='weight for the supervised loss')
    parser.add_argument('--num_labeled', type=int, default=50000)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--checkpoint', type=str, default='', help='continue training')
    parser.add_argument("--seed", type=int, default=0, metavar="S", help="random seed (default: 0)")
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    start_time = timer()

    # args parse
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    feature_dim, temperature, k = args.feature_dim, args.temperature, args.k
    batch_size, epochs = args.batch_size, args.epochs

    # data prepare

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")


    memory_data = utils.CIFAR10Pair(root='data', train=True, transform=utils.test_transform, download=True)
    memory_loader = DataLoader(memory_data, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)
    test_dataset = utils.CIFAR10Pair(root='data', train=False, transform=utils.test_transform, download=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)

    

    train_data = CIFAR10SSLDataset(
        "data",
        download=True,
        train=True,
        transform=utils.train_transform,
        ishuffle=False,
    )
        
    kwargs = {"batch_size": args.batch_size}
    train_loader = ht.utils.data.datatools.DataLoader(dataset=train_data, **kwargs)


    #train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True,
    #                          drop_last=True)
    dist_labeled = {'airplane': 5000, 'automobile': 5000, 'bird': 5000, 'cat': 5000, 'deer': 5000, 'dog': 5000,
                    'frog': 5000, 'horse': 5000, 'ship': 5000, 'truck': 5000}
    labeled_idxs = []


    # model setup and optimizer config
    model = Model_Sup(feature_dim).to(device)

    
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    blocking = True
    optimizer = ht.optim.DataParallelOptimizer(optimizer, blocking=blocking)
    net = ht.nn.DataParallel(
        model,
        comm=train_data.comm,
        optimizer=optimizer,
        blocking_parameter_updates=blocking,
        scale_gradient_average=comm.size,
    )

   
    c = len(memory_data.classes)
    criterion = nn.CrossEntropyLoss()

    if args.checkpoint:
        print(f"Checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint)
        start_epoch = ckpt['epoch'] + 1
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

    else:
        start_epoch = 0

    end_epoch = start_epoch + int(args.epochs)

    results = {'epoch': [], 'train_loss': [], 'sim_loss': [], 'sup_loss': [],
               'train_sup_accuracy': [],
               'test_sup_loss': [], 'test_sup_acc': [],
               'test_knn_acc@1': [], 'test_knn_acc@5': [], }

    if args.checkpoint:
        save_name_pre = '{}labeled_{}_{}epochs_{}_{}'.format(args.num_labeled, start_epoch, end_epoch,
                                                                   args.unsup_weight, args.sup_weight)
    else:
        save_name_pre = '{}labeled_{}epochs_{}_{}'.format(args.num_labeled, end_epoch, args.unsup_weight,
                                                                args.sup_weight)
    print(f"Saving to: {save_name_pre}")

    if not os.path.exists('results/noloop'):
        os.mkdir('results/noloop')

    best_acc = 0.0

    args_sup_weight = args.sup_weight

    if args.num_labeled == 50000:
        for epoch in range(start_epoch, end_epoch):
            train_loss, sup_acc, sim_loss, sup_loss = train(model, train_loader, optimizer, criterion,
                                                            args.unsup_weight, args.sup_weight)

            if comm.rank > 0:
                results['epoch'].append(epoch)
                results['train_loss'].append(train_loss)
                results['sim_loss'].append(sim_loss)
                results['sup_loss'].append(sup_loss)
                results['train_sup_accuracy'].append(sup_acc)

            
                data_frame = pd.DataFrame(data=results)
                data_frame = data_frame.set_index('epoch')
                data_frame.to_csv('results/noloop/{}.csv'.format(save_name_pre))  # , index_label='epoch')

         
        if comm.rank > 0:

            # save checkpoint so that you can train further
            torch.save({
                    'dist_labeled': dist_labeled,
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
            }, 'results/noloop/{}_cpt.pth'.format(save_name_pre))

              

   
