import os
import sys
import torch
import logging
import matplotlib.pyplot as plt
import numpy as np

from utils.disagreement import *
from utils.model import load_model
from utils.ops import client_num_samples, generate_confusion_matrix

from dotmap import DotMap

model_dir = './output/cifar10/resnet/shard/client50/default'
data_dir = './data/generated_dataset/cifar10/shard_50'
dataset_name = 'cifar10'

epoch = 300
num_client = 50
num_join_client = 10
num_classes = 10

model_dict = {
    'model_name': 'ResNet',
    'block': 'BasicBlock',
    'num_classes': num_classes,
    'num_blocks': [2,2,2,2]
}

if dataset_name == 'cifar10':
    testset = torchvision.datasets.CIFAR10(
            root='./data/cifar10', train=False, download=True, transform=torchvision.transforms.ToTensor()
    )
    testloader = torch.utils.data.DataLoader(testset, 1000, shuffle=False)

    trainset = torchvision.datasets.CIFAR10(
            root='./data/cifar10', train=True, download=True, transform=torchvision.transforms.ToTensor()
    )
    trainloader = torch.utils.data.DataLoader(trainset, 1024, shuffle=True)


log_dir = './plot_log'

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.FileHandler(os.path.join(log_dir, 'confusion_matrix.txt'), mode='w'))
logger.addHandler(logging.StreamHandler(sys.stdout))

model_dict = DotMap(model_dict)
model = load_model(model_dict)

num_samples = client_num_samples(data_dir, num_client, num_classes)

"""
for i in range(1, epoch+1):

    state_dict = torch.load(os.path.join(model_dir, f'epoch_{i}.pt'))
    model.load_state_dict(state_dict)

    confusion = generate_confusion_matrix(model, testloader, num_classes)
    logger.info(f'Epoch {i} confusion \n{confusion}')
"""
i = 299
state_dict = torch.load(os.path.join(model_dir, f'epoch_{i}.pt'))
model.load_state_dict(state_dict)

for param in model.parameters():
    param.requires_grad = False

for param in model.fc.parameters():
    param.requires_grad = True

optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=0.0005, momentum=0.9)
criterion = torch.nn.CrossEntropyLoss()

model.cuda()
model.train()
for k in range(10):
    for j, (xs, ys) in enumerate(trainloader):
        optimizer.zero_grad()
        xs = xs.cuda()
        ys = ys.cuda()
        logits = model(xs)
        loss = criterion(logits, ys)
        loss.backward()
        optimizer.step()

confusion = generate_confusion_matrix(model, testloader, num_classes)
logger.info(f'Epoch {i} confusion \n{confusion}')

from utils.ops import AverageMeter, accuracy

acc1_meter = AverageMeter()
model.eval()
with torch.no_grad():
    for step, (xs, ys) in enumerate(testloader):

        xs = xs.cuda()
        ys = ys.cuda()
        logits = model(xs)

        acc1, acc5 = accuracy(logits, ys, topk=(1,5))
        loss = criterion(logits, ys)
        acc1_meter.update(acc1)

logger.info(f'Acc 1 {acc1_meter.result():.4f}')