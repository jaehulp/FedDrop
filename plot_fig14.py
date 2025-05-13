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

if dataset_name == 'cifar10':
    testset = torchvision.datasets.CIFAR10(
            root='./data/cifar10', train=False, download=True, transform=torchvision.transforms.ToTensor()
    )
    testloader = torch.utils.data.DataLoader(testset, 1000, shuffle=False)

log_dir = './plot_log'

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.FileHandler(os.path.join(log_dir, 'raw_penult_output.txt'), mode='w'))
logger.addHandler(logging.StreamHandler(sys.stdout))

model_dict = DotMap(model_dict)
model = load_model(model_dict)

num_samples = client_num_samples(data_dir, num_client, num_classes)

target_list = [9]
for target_client in target_list:
    svd_list = []
    epochs = []
    e = [266]
    for i in e:
        epoch = i
        logger.info(f'Epoch {i}')
        model_list, idx_list = return_client_model_list(epoch, num_client, num_join_client, model_dir)
        if str(target_client) not in idx_list:
            continue
        epochs.append(str(i))
        idx = idx_list.index(str(target_client))
        target_model_list = [model_list[idx]]
        #testset_client_acc_list, testset_cov_mean_list, testset_svd_cov_mean_list, testset_topk_svd_mean_list, dis_list, testset_cov_std_list, testset_svd_std_list = measure_client_testset(epoch, model_dir, data_dir, testloader, model_dict, model_list, idx_list)
        #print(testset_client_acc_list)
        print(idx_list)

        idx_list = [int(i) for i in idx_list]
        weight = num_samples[idx_list]

        idx = list(range(10))
        norm_weight = weight[idx]
        norm_weight = norm_weight / sum(norm_weight)
        acc1 = simple_model_aggregate(model, [model_list[i] for i in idx], norm_weight, testloader)
        logger.info(f'Acc1 default test_acc {acc1}')

        simple_weight = [1/len(idx) for i in idx]
        acc1 = simple_model_aggregate(model, [model_list[i] for i in idx], simple_weight, testloader)
        logger.info(f'Acc1 simple avg test_acc {acc1}')

        idx = [0, 1, 3, 4, 5, 6, 7, 8, 9]
        norm_weight = weight[idx]
        norm_weight = norm_weight / sum(norm_weight)
        acc1 = simple_model_aggregate(model, [model_list[i] for i in idx], norm_weight, testloader)
        logger.info(f'Acc1 drop 9 {acc1}')

        idx = [0, 1, 3, 4, 5, 6, 7, 8, 9]
        simple_weight = [1/len(idx) for i in idx]
        acc1 = simple_model_aggregate(model, [model_list[i] for i in idx], simple_weight, testloader)
        logger.info(f'Acc1 drop 9 simple avg test_acc {acc1}')

        idx = [0, 1, 2, 3, 5, 6, 7, 8, 9]
        norm_weight = weight[idx]
        norm_weight = norm_weight / sum(norm_weight)
        acc1 = simple_model_aggregate(model, [model_list[i] for i in idx], norm_weight, testloader)
        logger.info(f'Acc1 drop 28 {acc1}')

        idx = [0, 1, 2, 3, 5, 6, 7, 8, 9]
        simple_weight = [1/len(idx) for i in idx]
        acc1 = simple_model_aggregate(model, [model_list[i] for i in idx], simple_weight, testloader)
        logger.info(f'Acc1 drop 28 simple avg test_acc {acc1}')
