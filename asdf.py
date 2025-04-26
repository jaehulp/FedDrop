import os
import torchvision
import torch
import json
import numpy as np

from utils.dataset import read_client_data

path = './data/generated_dataset/cifar10/dirichlet_50/config.json'
with open(path) as jsonfile:
    data = json.load(jsonfile)


datalist = data['Size of samples for labels in clients']
arr = np.zeros(shape=(50, 10))
for i, c in enumerate(datalist):
    for e in c:
        arr[i, e[0]] = e[1]

print(np.sum(arr,axis=1))

dataset = read_client_data('./data/generated_dataset/cifar10/dirichlet_50', 2, transform=None)
print(len(dataset))