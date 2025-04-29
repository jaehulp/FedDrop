import os
import copy

import torch
import torchvision
import random

import numpy as np
import torchvision.transforms as transforms

from utils.set_dataset import separate_data, save_dataset

seed = 999
np.random.seed(seed)
random.seed(seed)
num_clients = 50
dist = 'shard'
dir_path = './data/generated_dataset/cifar10/shard_50'
os.makedirs(dir_path, exist_ok=True)

transform = transforms.Compose(
    [transforms.ToTensor(),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data/cifar10', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(
    root='./data/cifar10', train=False, download=True, transform=transform)
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

separate_dataset, statistic = separate_data(dataset_image, dataset_label, num_classes, num_clients, dist)
save_dataset(dir_path, separate_dataset, statistic, num_classes, num_clients)

