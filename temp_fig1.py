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

context = False
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


def minus_avg(list1, list2):
    a = torch.tensor(list1)
    b = torch.tensor(list2)
    c = a-b
    d = torch.mean(c)
    return c.tolist(), d.item()

plt.figure()
plt.xlabel('Epoch')
plt.ylabel('Accuracy difference')
plt.title(f'Acc diff between default merge & method')

cov_mean = torch.mean(cov_mean_list, dim=1)

fig, ax1 = plt.subplots()


ax1.plot(range(len(acc1_default)), acc1_default, linewidth=1, linestyle='-', label='acc1_default', color='b')
ax1.tick_params(axis='y', labelcolor='b')

ax2 = ax1.twinx()
ax2.plot(range(len(cov_mean)), cov_mean, linestyle='-', label='client set cov', color='r')
ax2.tick_params(axis='y', labelcolor='r')

plt.grid(True)
plt.legend()
plt.savefig('./graph_img/temp_img.png', dpi=300)

t = torch.tensor([acc1_default[150:], cov_mean[150:]])
coff = torch.corrcoef(t)
print(coff)

t = torch.tensor([acc1_default, svd_mean])
coff = torch.corrcoef(t)
print(coff)

t = torch.tensor([acc1_default, topk_mean])
coff = torch.corrcoef(t)
print(coff)

