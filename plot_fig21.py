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

e = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]


plt.figure()

for i, ep in enumerate(e):
    epoch = ep
    logger.info(f'Epoch {ep}')
    model_list, idx_list = return_client_model_list(epoch, num_client, num_join_client, model_dir)
    svd_list, server_singular, client_singular, singular_diff = measure_client_raw_svd(epoch, model_dir, data_dir, model_dict, model_list, idx_list)

    idx_list = [int(i) for i in idx_list]
    weight = num_samples[idx_list]

    idx = list(range(10))
    norm_weight = weight[idx]
    norm_weight = norm_weight / sum(norm_weight)
    default_acc1 = simple_model_aggregate(model, [model_list[i] for i in idx], norm_weight, testloader)
    logger.info(f'Acc1 default test_acc {default_acc1}')

    drop_acc1 = []

    for j in range(len(idx_list)):
        idx = list(range(0,j)) + list(range(j+1,10))
        norm_weight = weight[idx]
        norm_weight = norm_weight / sum(norm_weight)
        acc1 = simple_model_aggregate(model, [model_list[j] for j in idx], norm_weight, testloader)
        drop_acc1.append(acc1)

    svd_list = torch.tensor(svd_list)
    client_singular = torch.tensor(client_singular)
    server_singular = torch.tensor(server_singular)
    norm_client_singular = client_singular[:50] / torch.sum(client_singular[:50])
    norm_server_singular = server_singular[:50] / torch.sum(server_singular[:50])
    server_weightsum = torch.einsum('ij, ij -> i', svd_list, norm_server_singular)
    client_weightsum = torch.einsum('ij, ij -> i', svd_list, norm_client_singular)

    default_acc1 = default_acc1.item()
    drop_acc1 = torch.tensor(drop_acc1) - default_acc1
    server_index = torch.argsort(server_weightsum)
    client_index = torch.argsort(client_weightsum)

    Ssorted_drop_acc1 = torch.gather(drop_acc1, dim=0, index=server_index)
    Csorted_drop_acc1 = torch.gather(drop_acc1, dim=0, index=client_index)
    
    plt.scatter(client_weightsum, drop_acc1)
    plt.xlabel('WeightSum SVD Corr Rank')
    plt.ylabel('Drop Acc1 diff')
    plt.grid(True)
    plt.title(f'SvdCorr vs Drop acc1 epoch{epoch}(Client singular Val)')
    plt.savefig(f'./graph_img2/DropAcc_WeightSumSVD_client_{epoch}_ResNet18.png', dpi=300)
    plt.clf()


"""
plt.scatter(range(1, 11), Csorted_drop_acc1)
plt.xlabel('WeightSum SVD Corr Rank')
plt.ylabel('Drop Acc1 diff')
plt.title('SvdCorr Rank vs Drop acc1 (Client singular Val)')
plt.grid(True)
plt.savefig(f'./graph_img/DropAccRank_WeightSumSVD_client_epoch{epoch}_ResNet18.png', dpi=300)
plt.clf()
"""