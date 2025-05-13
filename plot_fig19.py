import os
import sys
import torch
import logging
import matplotlib.pyplot as plt
import numpy as np

from utils.disagreement import *
from utils.model import load_model
from utils.ops import client_num_samples

from dotmap import DotMap


model_dir = './output/cifar10/resnet/dirichlet/client50/default'
data_dir = './data/generated_dataset/cifar10/dirichlet_50'
dataset_name = 'cifar10'

num_client = 50
num_join_client = 10
num_classes = 10

model_dict = {
    'model_name': 'ResNet',
    'block': 'BasicBlock',
    'num_classes': num_classes,
    'num_blocks': [2,2,2,2]
}
from utils.dataset import read_client_data
client_data = read_client_data(data_dir, '0')
clientloader = torch.utils.data.DataLoader(client_data, 64, shuffle=True)

if dataset_name == 'cifar10':
    trainset = torchvision.datasets.CIFAR10(
            root='./data/cifar10', train=True, download=True, transform=torchvision.transforms.ToTensor()
    )
    trainloader = torch.utils.data.DataLoader(trainset, 1000, shuffle=True)
    testset = torchvision.datasets.CIFAR10(
            root='./data/cifar10', train=False, download=True, transform=torchvision.transforms.ToTensor()
    )
    testloader = torch.utils.data.DataLoader(testset, 1000, shuffle=False)

model_dict = DotMap(model_dict)
model = load_model(model_dict)

model_state_dict = torch.load('./output/cifar10/resnet/dirichlet/client50/default/client/0/epoch_263.pt')
#model_state_dict = torch.load('./output/cifar10/resnet/dirichlet/client50/default/epoch_262.pt')

model.load_state_dict(model_state_dict)
model.cuda()
for param in model.parameters():
    param.requires_grad = True
for param in model.fc.parameters():
    param.requires_grad = True

optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.01, momentum=0.9)
criterion = torch.nn.CrossEntropyLoss()

from utils.ops import accuracy, AverageMeter
for j in range(10):
    acc1_meter = AverageMeter()
    model.eval()
    for i, (xs, ys) in enumerate(testloader):
        xs = xs.cuda()
        ys = ys.cuda()
        logits = model(xs)
        acc1, acc5 = accuracy(logits, ys, topk=(1,5))
        acc1_meter.update(acc1)
    print(f'Test acc: {acc1_meter.result():.4f}')
    model.train()
    acc1_meter = AverageMeter()
    for i, (xs, ys) in enumerate(trainloader):
        optimizer.zero_grad()
        xs = xs.cuda()
        ys = ys.cuda()
        logits = model(xs)
        acc1, acc5 = accuracy(logits, ys, topk=(1,5))
        loss = criterion(logits, ys)
        loss.backward()
        optimizer.step()
        acc1_meter.update(acc1)
    print(f'Train acc: {acc1_meter.result():.4f}')
"""
for param in model.parameters():
    param.requires_grad = False
for param in model.fc.parameters():
    param.requires_grad = True
optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.01, momentum=0.9)

for j in range(10):
    acc1_meter = AverageMeter()
    model.eval()
    for i, (xs, ys) in enumerate(testloader):
        xs = xs.cuda()
        ys = ys.cuda()
        logits = model(xs)
        acc1, acc5 = accuracy(logits, ys, topk=(1,5))
        acc1_meter.update(acc1)
    print(f'Test acc: {acc1_meter.result():.4f}')
    model.train()
    acc1_meter = AverageMeter()
    for i, (xs, ys) in enumerate(clientloader):
        optimizer.zero_grad()
        xs = xs.cuda()
        ys = ys.cuda()
        logits = model(xs)
        acc1, acc5 = accuracy(logits, ys, topk=(1,5))
        loss = criterion(logits, ys)
        loss.backward()
        optimizer.step()
        acc1_meter.update(acc1)
    print(f'Train acc: {acc1_meter.result():.4f}')
"""