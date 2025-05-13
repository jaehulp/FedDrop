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

target_list = [33]
for target_client in target_list:
    svd_list = []
    epochs = []
    for i in range(2,301):
        epoch = i
        logger.info(f'Epoch {i}')
        model_list, idx_list = return_client_model_list(epoch, num_client, num_join_client, model_dir)
        if str(target_client) not in idx_list:
            continue
        epochs.append(str(i))
        idx = idx_list.index(str(target_client))
        model_list = [model_list[idx]]

        svd = measure_client_raw_svd(epoch, model_dir, data_dir, model_dict, model_list, [str(target_client)])
        svd_list.append(svd[0])

    svd_list = np.array(svd_list)
    plt.figure(figsize=(12, 12))
    plt.imshow(svd_list, aspect='auto', cmap = 'inferno', vmin=0, vmax=1)

    plt.xlabel('Model feature')
    plt.ylabel('Epoch')
    plt.yticks(range(len(epochs)), epochs)
    plt.colorbar()
    plt.title(f'Correlation of client{target_client} SVD feature')
    plt.tight_layout()
    plt.savefig(f'./graph_img/CorrSVDFeature_model{target_client}_ResNet18.png', dpi=300)
    plt.clf()
