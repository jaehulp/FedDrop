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

model_dict = {
    'model_name': 'ResNet',
    'block': 'BasicBlock',
    'num_classes': 10,
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
logger.addHandler(logging.FileHandler(os.path.join(log_dir, 'dirichlet_testtemp.txt'), mode='w'))
logger.addHandler(logging.StreamHandler(sys.stdout))

model_dict = DotMap(model_dict)
model = load_model(model_dict)

num_samples = client_num_samples(data_dir, num_client, num_join_client)

for i in range(2, 301):
    epoch = i
    logger.info(f'Epoch {i}')
    model_list, idx_list = return_client_model_list(epoch, num_client, num_join_client, model_dir)
    
    client_acc_list, cov_mean_list, svd_cov_mean_list, topk_svd_mean_list = measure_client_cov(epoch, model_dir, data_dir, model_dict, model_list, idx_list)
    logger.info(f'client dataset')
    logger.info(f'cov_mean_list {cov_mean_list}')
    logger.info(f'svd_cov_mean_list {svd_cov_mean_list}')
    logger.info(f'topk_svd_mean_list {topk_svd_mean_list}')

    idx_list = [int(i) for i in idx_list]
    weight = num_samples[idx_list]

    idx = list(range(10))
    norm_weight = weight[idx]
    norm_weight = norm_weight / sum(norm_weight)
    new_list = [model_list[i] for i in idx]
    acc1 = simple_model_aggregate(model, [model_list[i] for i in idx], norm_weight, testloader)
    logger.info(f'Acc1 default test_acc {acc1}')

    temp = 1

    weight = torch.nn.functional.softmax(torch.tensor(cov_mean_list) * temp).tolist()
    acc1 = simple_model_aggregate(model, model_list, weight, testloader)
    logger.info(f'Temp1 Acc1 avg cov {acc1}')

    weight = torch.nn.functional.softmax(torch.tensor(svd_cov_mean_list) * temp).tolist()
    acc1 = simple_model_aggregate(model, model_list, weight, testloader)
    logger.info(f'Temp1 Acc1 avg svd_cov {acc1}')

    weight = torch.nn.functional.softmax(torch.tensor(topk_svd_mean_list) * temp).tolist()
    acc1 = simple_model_aggregate(model, model_list, weight, testloader)
    logger.info(f'Temp1 Acc1 avg topk_svd_cov {acc1}')

    temp = 3

    weight = torch.nn.functional.softmax(torch.tensor(cov_mean_list) * temp).tolist()
    acc1 = simple_model_aggregate(model, model_list, weight, testloader)
    logger.info(f'Temp3 Acc1 avg cov {acc1}')

    weight = torch.nn.functional.softmax(torch.tensor(svd_cov_mean_list) * temp).tolist()
    acc1 = simple_model_aggregate(model, model_list, weight, testloader)
    logger.info(f'Temp3 Acc1 avg svd_cov {acc1}')

    weight = torch.nn.functional.softmax(torch.tensor(topk_svd_mean_list) * temp).tolist()
    acc1 = simple_model_aggregate(model, model_list, weight, testloader)
    logger.info(f'Temp3 Acc1 avg topk_svd_cov {acc1}')

    temp = 5

    weight = torch.nn.functional.softmax(torch.tensor(cov_mean_list) * temp).tolist()
    acc1 = simple_model_aggregate(model, model_list, weight, testloader)
    logger.info(f'Temp5 Acc1 avg cov {acc1}')

    weight = torch.nn.functional.softmax(torch.tensor(svd_cov_mean_list) * temp).tolist()
    acc1 = simple_model_aggregate(model, model_list, weight, testloader)
    logger.info(f'Temp5 Acc1 avg svd_cov {acc1}')

    weight = torch.nn.functional.softmax(torch.tensor(topk_svd_mean_list) * temp).tolist()
    acc1 = simple_model_aggregate(model, model_list, weight, testloader)
    logger.info(f'Temp5 Acc1 avg topk_svd_cov {acc1}')

    temp = 7

    weight = torch.nn.functional.softmax(torch.tensor(cov_mean_list) * temp).tolist()
    acc1 = simple_model_aggregate(model, model_list, weight, testloader)
    logger.info(f'Temp7 Acc1 avg cov {acc1}')

    weight = torch.nn.functional.softmax(torch.tensor(svd_cov_mean_list) * temp).tolist()
    acc1 = simple_model_aggregate(model, model_list, weight, testloader)
    logger.info(f'Temp7 Acc1 avg svd_cov {acc1}')

    weight = torch.nn.functional.softmax(torch.tensor(topk_svd_mean_list) * temp).tolist()
    acc1 = simple_model_aggregate(model, model_list, weight, testloader)
    logger.info(f'Temp7 Acc1 avg topk_svd_cov {acc1}')

    temp = 9

    weight = torch.nn.functional.softmax(torch.tensor(cov_mean_list) * temp).tolist()
    acc1 = simple_model_aggregate(model, model_list, weight, testloader)
    logger.info(f'Temp9 Acc1 avg cov {acc1}')

    weight = torch.nn.functional.softmax(torch.tensor(svd_cov_mean_list) * temp).tolist()
    acc1 = simple_model_aggregate(model, model_list, weight, testloader)
    logger.info(f'Temp9 Acc1 avg svd_cov {acc1}')

    weight = torch.nn.functional.softmax(torch.tensor(topk_svd_mean_list) * temp).tolist()
    acc1 = simple_model_aggregate(model, model_list, weight, testloader)
    logger.info(f'Temp9 Acc1 avg topk_svd_cov {acc1}')


"""

print(client_acc_list)
print(cov_mean_list)
print(svd_cov_mean_list)
fig, ax1 = plt.subplots()
ax1.set_xlabel('Client')
ax1.set_ylabel('Acc')
#ax1.scatter(range(0, 10), server_acc_list, label='server acc')
ax1.scatter(range(0, 10), acc_list, label='client acc')
plt.legend()

ax2 = ax1.twinx()

ax2.set_ylabel('Cov')
ax2.scatter(range(0, 10), cov_mean_list, label='Cov mean', color='tab:red')
ax2.scatter(range(0, 10), svd_cov_mean_list, label='Cov mean(svd)', color='tab:green')
fig.tight_layout()
plt.legend()
plt.grid(True)
plt.savefig('./graph_img/cnn2_dirichlet_epoch50_top1svd_clientset.png', dpi=300)
plt.clf()

server_acc_list, client_acc_list, cov_mean_list, svd_cov_mean_list = measure_client_testset(epoch, model_dir, data_dir, model_dict, model_list, idx_list)

print(client_acc_list)
print(cov_mean_list)
print(svd_cov_mean_list)

fig, ax1 = plt.subplots()
ax1.set_xlabel('Client')
ax1.set_ylabel('Acc')
#ax1.scatter(range(0, 10), server_acc_list, label='server acc')
ax1.scatter(range(0, 10), acc_list, label='client acc')
plt.legend()

ax2 = ax1.twinx()

ax2.set_ylabel('Cov')
ax2.scatter(range(0, 10), cov_mean_list, label='Cov mean', color='tab:red')
ax2.scatter(range(0, 10), svd_cov_mean_list, label='Cov mean(svd)', color='tab:green')
fig.tight_layout()
plt.legend()
plt.grid(True)
plt.savefig('./graph_img/cnn2_dirichlet_epoch50_top1svd_testset.png', dpi=300)
"""