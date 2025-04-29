import os
import re
import yaml

import torch
import numpy as np

import matplotlib.pyplot as plt 


log_path = './plot_log_copy/dirichlet_log.txt'

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

for line in log_data:
    line = line.strip()

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

diff, mean = minus_avg(acc1_drop_cov_mean, acc1_default)
plt.plot(range(len(diff)), diff, linewidth=1, linestyle='-', label=f'drop_cov({mean:.4f})')

plt.grid(True)
plt.legend()
plt.savefig(f'./graph_img/AccDiff_dropcov_ResNet18', dpi =300)
plt.clf()

plt.xlabel('Epoch')
plt.ylabel('Accuracy difference')
plt.title(f'Acc diff between default merge & method')

diff, mean = minus_avg(acc1_drop_svd_mean, acc1_default)
plt.plot(range(len(diff)), diff, linewidth=1, linestyle='-', label=f'drop_svd({mean:.4f})')

plt.grid(True)
plt.legend()
plt.savefig(f'./graph_img/AccDiff_dropsvd_ResNet18', dpi =300)
plt.clf()


plt.xlabel('Epoch')
plt.ylabel('Accuracy difference')
plt.title(f'Acc diff between default merge & method')
diff, mean = minus_avg(acc1_drop_topk_svd, acc1_default)
plt.plot(range(len(diff)), diff, linewidth=1, linestyle='-', label=f'drop_topk_svd({mean:.4f})')
plt.grid(True)
plt.legend()
plt.savefig(f'./graph_img/AccDiff_droptopk_ResNet18', dpi =300)
plt.clf()

plt.xlabel('Epoch')
plt.ylabel('Accuracy difference')
plt.title(f'Acc diff between default merge & method')
diff, mean = minus_avg(acc1_drop_test_acc, acc1_default)
plt.plot(range(len(diff)), diff, linewidth=1, linestyle='-', label=f'test_Acc({mean:.4f})')
plt.grid(True)
plt.legend()
plt.savefig(f'./graph_img/AccDiff_dropTestAcc_ResNet18', dpi =300)
plt.clf()

plt.xlabel('Epoch')
plt.ylabel('Accuracy difference')
plt.title(f'Acc diff between default merge & method')
diff, mean =minus_avg(acc1_avg_cov_mean, acc1_default)
plt.plot(range(len(diff)), diff, linewidth=1, linestyle='-', label=f'avg_cov({mean:.4f})')
plt.grid(True)
plt.legend()
plt.savefig(f'./graph_img/AccDiff_avgcov_ResNet18', dpi =300)
plt.clf()

plt.xlabel('Epoch')
plt.ylabel('Accuracy difference')
plt.title(f'Acc diff between default merge & method')
diff, mean =minus_avg(acc1_avg_svd_mean, acc1_default)
plt.plot(range(len(diff)), diff, linewidth=1, linestyle='-', label=f'avg_svd({mean:.4f})')
plt.grid(True)
plt.legend()
plt.savefig(f'./graph_img/AccDiff_avgsvd_ResNet18', dpi =300)
plt.clf()

plt.xlabel('Epoch')
plt.ylabel('Accuracy difference')
plt.title(f'Acc diff between default merge & method')
diff, mean =minus_avg(acc1_avg_topk_svd, acc1_default)
plt.plot(range(len(diff)), diff, linewidth=1, linestyle='-', label=f'avg_topk({mean:.4f})')
plt.grid(True)
plt.legend()
plt.savefig(f'./graph_img/AccDiff_avgtopk_ResNet18', dpi =300)
plt.clf()


