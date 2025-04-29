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

acc1_default = torch.tensor(acc1_default)
acc1_client_testset = torch.stack(acc1_client_testset)

cov_mean_rank = torch.argsort(cov_mean_list, dim=1, descending=True)
acc1_client_rank = torch.argsort(acc1_client_testset, dim=1, descending=True)

cov_mean_rank = cov_mean_rank * 10
rank = cov_mean_rank + acc1_client_rank
bincount = torch.bincount(rank.flatten())
assert len(bincount) == 100

bincount = bincount.reshape(10,10)
bincount = bincount / 2990
plt.imshow(bincount, cmap = 'inferno', vmin=0, vmax= 0.03)
plt.colorbar()
plt.xlabel('Test acc1 rank')
plt.ylabel('Covariance rank')
plt.title('Client Test acc1 vs Covariance')
plt.savefig('./graph_img/CovAccRank_heatmap_ResNet18.png', dpi=300)
plt.clf()

cov_mean_rank = torch.argsort(cov_mean_list[:50], dim=1, descending=True)
acc1_client_rank = torch.argsort(acc1_client_testset[:50], dim=1, descending=True)

cov_mean_rank = cov_mean_rank * 10
rank = cov_mean_rank + acc1_client_rank
bincount = torch.bincount(rank.flatten())
assert len(bincount) == 100

bincount = bincount.reshape(10,10)
bincount = bincount / 500
plt.imshow(bincount, cmap = 'inferno', vmin=0, vmax= 0.03)
plt.colorbar()
plt.xlabel('Test acc1 rank')
plt.ylabel('Covariance rank')
plt.title('Client Test acc1 vs Covariance')
plt.savefig('./graph_img/CovAccRank_heatmap_earlystage_ResNet18.png', dpi=300)
plt.clf()

cov_mean_rank = torch.argsort(cov_mean_list[50:], dim=1, descending=True)
acc1_client_rank = torch.argsort(acc1_client_testset[50:], dim=1, descending=True)

cov_mean_rank = cov_mean_rank * 10
rank = cov_mean_rank + acc1_client_rank
bincount = torch.bincount(rank.flatten())
assert len(bincount) == 100

bincount = bincount.reshape(10,10)
bincount = bincount / 2490
plt.imshow(bincount, cmap = 'inferno', vmin=0, vmax= 0.03)
plt.colorbar()
plt.xlabel('Test acc1 rank')
plt.ylabel('Covariance rank')
plt.title('Client Test acc1 vs Covariance')
plt.savefig('./graph_img/CovAccRank_heatmap_poststage_ResNet18.png', dpi=300)

