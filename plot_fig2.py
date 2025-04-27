import os
import re
import yaml

import torch
import numpy as np

import matplotlib.pyplot as plt 

log_path = './plot_log/dirichlet_log.txt'

with open(log_path, 'r') as file:
    log_data = file.readlines()

cov_mean_list = []
svd_mean_list = []
topk_mean_list = []

test_cov_mean_list = []
test_svd_mean_list = []
test_topk_mean_list = []

context = False
for line in log_data:
    line = line.strip()

    if line.startswith('client dataset'):
        context = 'clientset'
    elif line.startswith('test dataset'):
        context = 'testset'

    if context == 'clientset':
        if line.startswith("cov_mean_list"):
            matches = re.findall(r"[-+]?\d*\.\d+|\d+", line)
            numbers = [float(m) for m in matches]
            cov_mean_list.append(torch.tensor(numbers))
        if line.startswith("svd_cov_mean_list"):
            matches = re.findall(r"[-+]?\d*\.\d+|\d+", line)
            numbers = [float(m) for m in matches]
            svd_mean_list.append(torch.tensor(numbers))
        if line.startswith("topk_svd_mean_list"):
            matches = re.findall(r"[-+]?\d*\.\d+|\d+", line)
            numbers = [float(m) for m in matches]
            topk_mean_list.append(torch.tensor(numbers))

    if context == 'testset':
        if line.startswith("cov_mean_list"):
            matches = re.findall(r"[-+]?\d*\.\d+|\d+", line)
            numbers = [float(m) for m in matches]
            test_cov_mean_list.append(torch.tensor(numbers))
        if line.startswith("svd_cov_mean_list"):
            matches = re.findall(r"[-+]?\d*\.\d+|\d+", line)
            numbers = [float(m) for m in matches]
            test_svd_mean_list.append(torch.tensor(numbers))
        if line.startswith("topk_svd_mean_list"):
            matches = re.findall(r"[-+]?\d*\.\d+|\d+", line)
            numbers = [float(m) for m in matches]
            test_topk_mean_list.append(torch.tensor(numbers))

cov_mean_list = torch.stack(cov_mean_list)
svd_mean_list = torch.stack(svd_mean_list)
topk_mean_list = torch.stack(topk_mean_list)

test_cov_mean_list = torch.stack(test_cov_mean_list)
test_svd_mean_list = torch.stack(test_svd_mean_list)
test_topk_mean_list = torch.stack(test_topk_mean_list)


cov_mean_list = torch.mean(cov_mean_list, dim=1)
svd_mean_list = torch.mean(svd_mean_list, dim=1)
topk_mean_list = torch.mean(topk_mean_list, dim=1)

test_cov_mean_list = torch.mean(test_cov_mean_list, dim=1)
test_svd_mean_list = torch.mean(test_svd_mean_list, dim=1)
test_topk_mean_list = torch.mean(test_topk_mean_list, dim=1)

plt.figure()
plt.xlabel('Epoch')
plt.ylabel('Cov mean')
plt.title('Change of covariance')
plt.plot(range(len(cov_mean_list)), cov_mean_list, linestyle='-', label='client set cov')
plt.plot(range(len(test_svd_mean_list)), test_svd_mean_list, linestyle='-', label='test set cov')
plt.plot(range(len(cov_mean_list)), cov_mean_list - test_svd_mean_list, linestyle='-', label='cov diff')
plt.grid(True)
plt.legend()
plt.savefig('./graph_img/Covmean_ResNet18.png', dpi=300)

plt.figure()
plt.xlabel('Epoch')
plt.ylabel('Cov mean')
plt.title('Change of covariance')
plt.plot(range(len(cov_mean_list)), svd_mean_list, linestyle='-', label='client set svd_cov')
plt.plot(range(len(test_svd_mean_list)), test_svd_mean_list, linestyle='-', label='test set svd_cov')
plt.plot(range(len(cov_mean_list)), svd_mean_list - test_svd_mean_list, linestyle='-', label='svd_cov diff')
plt.grid(True)
plt.legend()
plt.savefig('./graph_img/Svdmean_ResNet18.png', dpi=300)

plt.figure()
plt.xlabel('Epoch')
plt.ylabel('Cov mean')
plt.title('Change of covariance')
plt.plot(range(len(cov_mean_list)), topk_mean_list, linestyle='-', label='client set cov')
plt.plot(range(len(test_topk_mean_list)), test_topk_mean_list, linestyle='-', label='test set cov')
plt.plot(range(len(cov_mean_list)), topk_mean_list - test_topk_mean_list, linestyle='-', label='cov diff')
plt.grid(True)
plt.legend()
plt.savefig('./graph_img/Topkmean_ResNet18.png', dpi=300)