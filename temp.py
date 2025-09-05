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
from utils.ops import layer_wise_norms

model_dir = './output/cifar10/resnet/dirichlet/client50/mrl'
data_dir = './data/generated_dataset/cifar10/dirichlet_50'
dataset_name = 'cifar10'

num_client = 50
num_join_client = 10
num_classes = 100

model_dict = {
    'model_name': 'ResNet',
    'block': 'BasicBlock',
    'num_classes': num_classes,
    'num_blocks': [2,2,2,2]
}
model_dict = DotMap(model_dict)

from utils.disagreement import return_client_model_list

epoch = 100

model_list, idx_list = return_client_model_list(epoch, num_client, num_join_client, model_dir)

model = load_model(model_dict)

server_path = os.path.join(model_dir, 'server')
client_path = os.path.join(model_dir, 'client')
server_model = torch.load(os.path.join(server_path, f'epoch_{epoch}.pt'))

model.load_state_dict(server_model)

def print_norm(model):
    norms = layer_wise_norms(model)

    for layer, norm in norms.items():
        print(f"{layer}: {norm:.4f}")

for model_state_dict, idx in zip(model_list, idx_list):
    model.load_state_dict(model_state_dict)
    print_norm(model)
