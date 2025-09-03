import os
import copy
import argparse

import torch
import torchvision
import random

import numpy as np
import torchvision.transforms as transforms

from utils.set_dataset import separate_data, save_dataset

parser = argparse.ArgumentParser()

parser.add_argument('--clients', default=50, type=int)
parser.add_argument('--split', default='iid', type=str)
parser.add_argument('--alpha', default=0.5, type=float)
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--seed', default=999, type=int)

args, _ = parser.parse_known_args()

seed = args.seed
np.random.seed(seed)
random.seed(seed)
num_clients = args.clients
dist = args.split
dataset = args.dataset
alpha = args.alpha
if dist = 'dirichlet':
    dir_path = f'./data/generated_dataset/{dataset}/{dist}{str(alpha).replace('.', '')}_{num_clients}'
else:
    dir_path = f'./data/generated_dataset/{dataset}/{dist}_{num_clients}'

os.makedirs(dir_path, exist_ok=True)

transform = transforms.Compose(
    [transforms.ToTensor(),
])

if dataset == 'cifar10':
    trainset = torchvision.datasets.CIFAR10(
        root='./data/cifar10', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(
        root='./data/cifar10', train=False, download=True, transform=transform)
elif dataset == 'cifar100':
    trainset = torchvision.datasets.CIFAR100(
        root='./data/cifar100', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR100(
        root='./data/cifar100', train=False, download=True, transform=transform)
else:
    raise NotImplementedError('Check dataset name')

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=len(trainset.data), shuffle=False)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=len(testset.data), shuffle=False)

for _, train_data in enumerate(trainloader, 0):
    trainset.data, trainset.targets = train_data
for _, test_data in enumerate(testloader, 0):
    testset.data, testset.targets = test_data

dataset_image = []
dataset_label = []

dataset_image.extend(trainset.data.cpu().detach().numpy())
dataset_label.extend(trainset.targets.cpu().detach().numpy())
dataset_image = np.array(dataset_image)
dataset_label = np.array(dataset_label)

num_classes = len(set(dataset_label))

testset_image = []
testset_label = []
testset_image.extend(testset.data.cpu().detach().numpy())
testset_label.extend(testset.targets.cpu().detach().numpy())

test_data = {'x': testset_image, 'y': testset_label}

with open(os.path.join(dir_path, 'test.npz'), 'wb') as f:
    np.savez_compressed(f, data=test_data)

separate_dataset, statistic = separate_data(dataset_image, dataset_label, num_classes, num_clients, dist, alpha)
save_dataset(dir_path, separate_dataset, statistic, num_classes, num_clients)

