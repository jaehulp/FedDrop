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

disagg = -1

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

acc1_client_testset = torch.stack(acc1_client_testset)
acc1_client_testset = acc1_client_testset - torch.mean(acc1_client_testset, dim=1, keepdim=True)

cov_mean_index = torch.argsort(cov_mean_list, dim=1)
sorted_acc1_client_testset= torch.gather(acc1_client_testset, dim=1, index=cov_mean_index)

plt.figure()
plt.xlabel('Epoch')
plt.ylabel('Centered Acc1')

for i in [0, 5, 9]:
    plt.plot(range(len(sorted_acc1_client_testset[:, 0])), sorted_acc1_client_testset[:, i], linestyle='-', label=f'sorted {i}({torch.mean(sorted_acc1_client_testset[:,i]):.4f})')

min_list = torch.amin(acc1_client_testset, dim=1)
plt.plot(range(len(sorted_acc1_client_testset[:, 0])), min_list, linestyle='-', label=f'Min Acc1({torch.mean(min_list):.4f})')
max_list = torch.amax(acc1_client_testset, dim=1)
plt.plot(range(len(sorted_acc1_client_testset[:, 0])), max_list, linestyle='-', label=f'Max Acc1({torch.mean(max_list):.4f})')

plt.grid(True)
plt.legend()
plt.title('Argsort Client Test Acc with testset covariance')
plt.savefig('./graph_img/CovArgsort_ClientTestsetAcc_ResNet18.png', dpi=300)
plt.clf()

plt.xlabel('Epoch')
plt.ylabel('Centered Disagreement')

disagreement_mat = torch.stack(disagreement_list)
disagreement = torch.mean(disagreement_mat, dim=1, keepdim=True)
disagreement_mat = disagreement_mat - disagreement
sorted_disagreement_mat = torch.gather(disagreement_mat, dim=1, index=cov_mean_index)

for i in [0, 5, 9]:
    plt.plot(range(len(sorted_disagreement_mat[:, 0])), sorted_disagreement_mat[:, i], linestyle='-', label=f'sorted {i}({torch.mean(sorted_disagreement_mat[:,i]):.4f})')

min_list = torch.amin(disagreement_mat, dim=1)
plt.plot(range(len(sorted_acc1_client_testset[:, 0])), min_list, linestyle='-', label=f'Min disagg({torch.mean(min_list):.4f})')
max_list = torch.amax(disagreement_mat, dim=1)
plt.plot(range(len(sorted_acc1_client_testset[:, 0])), max_list, linestyle='-', label=f'Max disagg({torch.mean(max_list):.4f})')

plt.grid(True)
plt.legend()
plt.title('Argsort Client Disagreement with testset SVD cov')
plt.savefig('./graph_img/CovArgsort_ClientDisagreement_ResNet18.png', dpi=300)
plt.clf()


svd_mean_index = torch.argsort(svd_mean_list, dim=1)
sorted_acc1_client_testset= torch.gather(acc1_client_testset, dim=1, index=svd_mean_index)
sorted_disagreement_mat = torch.gather(disagreement_mat, dim=1, index=svd_mean_index)


plt.xlabel('Epoch')
plt.ylabel('Centered Acc1')

for i in [0, 5, 9]:
    plt.plot(range(len(sorted_acc1_client_testset[:, 0])), sorted_acc1_client_testset[:, i], linestyle='-', label=f'sorted {i}({torch.mean(sorted_acc1_client_testset[:,i]):.4f})')

min_list = torch.amin(acc1_client_testset, dim=1)
plt.plot(range(len(sorted_acc1_client_testset[:, 0])), min_list, linestyle='-', label=f'Min Acc1({torch.mean(min_list):.4f})')
max_list = torch.amax(acc1_client_testset, dim=1)
plt.plot(range(len(sorted_acc1_client_testset[:, 0])), max_list, linestyle='-', label=f'Max Acc1({torch.mean(max_list):.4f})')

plt.grid(True)
plt.legend()
plt.title('Argsort Client Test Acc with testset covariance')
plt.savefig('./graph_img/SvdArgsort_ClientTestsetAcc_ResNet18.png', dpi=300)
plt.clf()

plt.xlabel('Epoch')
plt.ylabel('Centered Disagreement')

for i in [0, 5, 9]:
    plt.plot(range(len(sorted_disagreement_mat[:, 0])), sorted_disagreement_mat[:, i], linestyle='-', label=f'sorted {i}({torch.mean(sorted_disagreement_mat[:,i]):.4f})')

min_list = torch.amin(disagreement_mat, dim=1)
plt.plot(range(len(sorted_acc1_client_testset[:, 0])), min_list, linestyle='-', label=f'Min disagg({torch.mean(min_list):.4f})')
max_list = torch.amax(disagreement_mat, dim=1)
plt.plot(range(len(sorted_acc1_client_testset[:, 0])), max_list, linestyle='-', label=f'Max disagg({torch.mean(max_list):.4f})')

plt.grid(True)
plt.legend()
plt.title('Argsort Client Disagreement with testset SVD cov')
plt.savefig('./graph_img/SvdArgsort_ClientDisagreement_ResNet18.png', dpi=300)