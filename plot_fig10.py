import os
import re
import yaml

import torch
import numpy as np

import matplotlib.pyplot as plt 

log_path = './plot_log/dirichlet_log.txt'

with open(log_path, 'r') as file:
    log_data = file.readlines()

acc1_default = []
acc1_drop_test_acc = []
acc1_drop_cov_mean = []
acc1_drop_svd_mean = []
acc1_drop_topk_svd = []
acc1_avg_cov_mean = []
acc1_avg_svd_mean = []
acc1_avg_topk_svd = []

acc1_client_testset = []

cov_mean_list = []
svd_mean_list = []
topk_mean_list = []

cov_std_list = []
svd_std_list = []

test_cov_mean_list = []
test_svd_mean_list = []
test_topk_mean_list = []

disagreement_list = []

context = False

disagg = -1

for line in log_data:
    line = line.strip()

    if line.startswith('client dataset'):
        context = 'clientset'
    elif line.startswith('test dataset'):
        context = 'testset'

    if line.startswith("disagreement tensor"):
        matches = re.findall(r"[-+]?\d*\.\d+|\d+", line)
        numbers = [float(m) for m in matches]
        disagreement_list.append(torch.tensor(numbers))
        disagg = 1
    

    if context == 'clientset':
        if line.startswith("cov_mean_list"):
            matches = re.findall(r"[-+]?\d*\.\d+|\d+", line)
            numbers = [float(m) for m in matches]
            cov_mean_list.append(torch.tensor(numbers))

    if context == 'testset':
        if line.startswith("client_acc_list"):
            matches = re.findall(r"[-+]?\d*\.\d+|\d+", line)
            numbers = [float(m) for m in matches]
            acc1_client_testset.append(torch.tensor(numbers))
        if line.startswith("cov_mean_list"):
            matches = re.findall(r"[-+]?\d*\.\d+|\d+", line)
            numbers = [float(m) for m in matches]
            test_cov_mean_list.append(torch.tensor(numbers))

cov_mean_list = torch.stack(cov_mean_list)

import sys
import logging

from dotmap import DotMap
from utils.model import load_model
from utils.ops import client_num_samples
from utils.disagreement import *

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

model_dict = DotMap(model_dict)
model = load_model(model_dict)

testset = torchvision.datasets.CIFAR10(
        root='./data/cifar10', train=False, download=True, transform=torchvision.transforms.ToTensor()
)
testloader = torch.utils.data.DataLoader(testset, 1000, shuffle=False)

log_dir = './plot_log'

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.FileHandler(os.path.join(log_dir, 'acc_drop89.txt'), mode='w'))
logger.addHandler(logging.StreamHandler(sys.stdout))

num_samples = client_num_samples(data_dir, num_client, num_classes)

for cov, i in zip(cov_mean_list, range(2, 301)):
    epoch = i
    model_list, idx_list = return_client_model_list(epoch, num_client, num_join_client, model_dir)

    idx_list = [int(i) for i in idx_list]
    weight = num_samples[idx_list]

    idx = np.argsort(cov)
    idx = torch.cat((idx[0:1], idx[3:]))
    norm_weight = weight[idx]
    norm_weight = norm_weight / sum(norm_weight)
    acc1 = simple_model_aggregate(model, [model_list[i] for i in idx], norm_weight, testloader)
    logger.info(f'Acc1 drop cov index 2,3 {acc1}')
