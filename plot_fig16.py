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

e = [10, 50, 100, 150, 200, 250]

negatives = []
positives = []
negatives_singular = []
positives_singular = []
negatives_singular_diff = []
positives_singular_diff = []
for i in e:
    epoch = i
    logger.info(f'Epoch {i}')
    model_list, idx_list = return_client_model_list(epoch, num_client, num_join_client, model_dir)
    svd_list, server_singular, client_singular, singular_diff = measure_client_raw_svd(epoch, model_dir, data_dir, model_dict, model_list, idx_list)

    idx_list = [int(i) for i in idx_list]
    weight = num_samples[idx_list]

    idx = list(range(10))
    norm_weight = weight[idx]
    norm_weight = norm_weight / sum(norm_weight)
    default_acc1 = simple_model_aggregate(model, [model_list[i] for i in idx], norm_weight, testloader)
    logger.info(f'Acc1 default test_acc {default_acc1}')

    simple_weight = [1/len(idx) for i in idx]
    simple_acc1 = simple_model_aggregate(model, [model_list[i] for i in idx], simple_weight, testloader)
    logger.info(f'Acc1 simple avg test_acc {simple_acc1}')

    for i in range(len(idx_list)):
        idx = list(range(0,i)) + list(range(i+1,10))
        norm_weight = weight[idx]
        norm_weight = norm_weight / sum(norm_weight)
        acc1 = simple_model_aggregate(model, [model_list[i] for i in idx], norm_weight, testloader)
        logger.info(f'Acc1 drop model{idx_list[i]} {acc1-default_acc1}')
        if acc1-default_acc1 > 0.1:
            positives.append(svd_list[i])
            positives_singular.append(client_singular[i])
            positives_singular_diff.append(singular_diff[i])
        elif acc1-default_acc1 < -0.1:
            negatives.append(svd_list[i])
            negatives_singular.append(client_singular[i])
            negatives_singular_diff.append(singular_diff[i])

        simple_weight = [1/len(idx) for i in idx]
        acc1 = simple_model_aggregate(model, [model_list[i] for i in idx], simple_weight, testloader)
        logger.info(f'Acc1 drop model{idx_list[i]} simple avg test_acc {acc1-simple_acc1}')

negatives = np.array(negatives)
positives = np.array(positives)
negatives_singular = np.array(negatives_singular)
positives_singular = np.array(positives_singular)
negatives_singular_diff = np.array(negatives_singular_diff)
positives_singular_diff = np.array(positives_singular_diff)

plt.figure(figsize=(12, 8))
plt.imshow(negatives, aspect='auto', cmap = 'inferno', vmin=0, vmax=1)

plt.xlabel('Model feature')
plt.ylabel('Client')
plt.yticks(range(len(idx_list)), idx_list)
plt.colorbar()
plt.title('SVD Correlation of each model feature')
plt.tight_layout()
plt.savefig(f'./graph_img/CorrSVDFeature_stacknegative_ResNet18.png', dpi=300)
plt.clf()


plt.imshow(positives, aspect='auto', cmap = 'inferno', vmin=0, vmax=1)

plt.xlabel('Model feature')
plt.ylabel('Client')
plt.yticks(range(len(idx_list)), idx_list)
plt.colorbar()
plt.title('SVD Correlation of each model feature')
plt.tight_layout()
plt.savefig(f'./graph_img/CorrSVDFeature_stackpositive_ResNet18.png', dpi=300)
plt.clf()

plt.figure(figsize=(12, 8))
plt.imshow(negatives_singular, aspect='auto', cmap = 'inferno')

plt.xlabel('Model feature')
plt.ylabel('Client')
plt.yticks(range(len(idx_list)), idx_list)
plt.colorbar()
plt.title('SVD Correlation of each model feature')
plt.tight_layout()
plt.savefig(f'./graph_img/CorrSVDFeature_stacknegative_singular_ResNet18.png', dpi=300)
plt.clf()


plt.imshow(positives_singular, aspect='auto', cmap = 'inferno')

plt.xlabel('Model feature')
plt.ylabel('Client')
plt.yticks(range(len(idx_list)), idx_list)
plt.colorbar()
plt.title('SVD Correlation of each model feature')
plt.tight_layout()
plt.savefig(f'./graph_img/CorrSVDFeature_stackpositive_singular_ResNet18.png', dpi=300)
plt.clf()

plt.figure(figsize=(12, 8))
plt.imshow(negatives_singular_diff, aspect='auto', cmap = 'inferno')

plt.xlabel('Model feature')
plt.ylabel('Client')
plt.yticks(range(len(idx_list)), idx_list)
plt.colorbar()
plt.title('SVD Correlation of each model feature')
plt.tight_layout()
plt.savefig(f'./graph_img/CorrSVDFeature_stacknegative_singulardiff_ResNet18.png', dpi=300)
plt.clf()


plt.imshow(positives_singular_diff, aspect='auto', cmap = 'inferno')

plt.xlabel('Model feature')
plt.ylabel('Client')
plt.yticks(range(len(idx_list)), idx_list)
plt.colorbar()
plt.title('SVD Correlation of each model feature')
plt.tight_layout()
plt.savefig(f'./graph_img/CorrSVDFeature_stackpositive_singulardiff_ResNet18.png', dpi=300)
plt.clf()