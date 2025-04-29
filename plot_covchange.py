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
topk_std_list = []

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

    if line.startswith("Acc1 default test_acc"):
        acc = float(re.search(r'Acc1 default test_acc (\d+\.\d+)', line).group(1))
        acc1_default.append(acc)
    if line.startswith("Acc1 drop least test_acc"):
        acc = float(re.search(r'Acc1 drop least test_acc (\d+\.\d+)', line).group(1))
        acc1_drop_test_acc.append(acc)
    if line.startswith("Acc1 drop least cov_mean"):
        acc = float(re.search(r'Acc1 drop least cov_mean (\d+\.\d+)', line).group(1))
        acc1_drop_cov_mean.append(acc)
    if line.startswith("Acc1 drop least svd_cov_mean"):
        acc = float(re.search(r'Acc1 drop least svd_cov_mean (\d+\.\d+)', line).group(1))
        acc1_drop_svd_mean.append(acc)
    if line.startswith("Acc1 drop least topk_svd_mean"):
        acc = float(re.search(r'Acc1 drop least topk_svd_mean (\d+\.\d+)', line).group(1))
        acc1_drop_topk_svd.append(acc)
    if line.startswith("Acc1 avg cov"):
        acc = float(re.search(r'Acc1 avg cov (\d+\.\d+)', line).group(1))
        acc1_avg_cov_mean.append(acc)
    if line.startswith("Acc1 avg svd_cov"):
        acc = float(re.search(r'Acc1 avg svd_cov (\d+\.\d+)', line).group(1))
        acc1_avg_svd_mean.append(acc)
    if line.startswith("Acc1 avg topk_svd_cov"):
        acc = float(re.search(r'Acc1 avg topk_svd_cov (\d+\.\d+)', line).group(1))
        acc1_avg_topk_svd.append(acc)
    if disagg == 1:
        matches = re.findall(r"[-+]?\d*\.\d+|\d+", line)
        numbers = [float(m) for m in matches]
        disagreement_list[-1] = torch.cat([disagreement_list[-1], torch.tensor([numbers[0]])])
        disagg = -1

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
        if line.startswith("svd_cov_mean_list"):
            matches = re.findall(r"[-+]?\d*\.\d+|\d+", line)
            numbers = [float(m) for m in matches]
            svd_mean_list.append(torch.tensor(numbers))
        if line.startswith("topk_svd_mean_list"):
            matches = re.findall(r"[-+]?\d*\.\d+|\d+", line)
            numbers = [float(m) for m in matches]
            topk_mean_list.append(torch.tensor(numbers))
        if line.startswith("cov_std_list"):
            matches = re.findall(r"[-+]?\d*\.\d+|\d+", line)
            numbers = [float(m) for m in matches]
            cov_std_list.append(torch.tensor(numbers))

    if context == 'testset':
        if line.startswith("client_acc_list"):
            matches = re.findall(r"[-+]?\d*\.\d+|\d+", line)
            numbers = [float(m) for m in matches]
            acc1_client_testset.append(torch.tensor(numbers))
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

cov_mean = torch.mean(cov_mean_list, dim=1)
svd_mean = torch.mean(svd_mean_list, dim=1)
topk_mean = torch.mean(topk_mean_list, dim=1)

test_cov_mean = torch.mean(test_cov_mean_list, dim=1)
test_svd_mean = torch.mean(test_svd_mean_list, dim=1)
test_topk_mean = torch.mean(test_topk_mean_list, dim=1)

cov_index = torch.argsort(cov_mean_list, dim=1)
sorted_cov_mean_list = torch.gather(cov_mean_list, dim=1, index=cov_index)

plt.figure()
plt.xlabel('Epoch')
plt.ylabel('Mean Covariance')

for i in [0, 9]:
    plt.plot(range(len(sorted_cov_mean_list[:,0])), sorted_cov_mean_list[:, i], linestyle='-', color='#1f77b4', linewidth=0.7, alpha=0.7)

plt.plot(range(len(cov_mean)), cov_mean, linestyle='-', label='mean cov')
std = torch.std(cov_mean_list, dim=1)
plt.fill_between(range(len(cov_mean)), cov_mean - std, cov_mean + std, alpha=0.3)

plt.grid(True)
plt.legend()
plt.savefig('./graph_img/CovChange_ResNet18.png', dpi=300)
plt.clf()

cov_std_list = torch.stack(cov_std_list)
cov_std_mean = torch.mean(cov_std_list, dim=1)
plt.xlabel('Epoch')
plt.ylabel('Std Cov')
plt.plot(range(len(cov_std_mean)), cov_std_mean, linestyle='-')
plt.grid(True)
plt.savefig('./graph_img/CovStd_ResNet18.png', dpi=300)