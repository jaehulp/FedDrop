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

for line in log_data:
    line = line.strip()

    if line.startswith("Acc1 default test_acc"):
        acc = float(re.search(r'Acc1 default test_acc (\d+\.\d+)', line).group(1))
        acc1_default.append(acc)

log_path = './plot_log/acc_drop89.txt'
with open(log_path, 'r') as file:
    log_data = file.readlines()

acc1_drop89 = []
for line in log_data:
    line = line.strip()
    if line.startswith("Acc1 drop cov index 2,3"):
        acc = float(re.search(r'Acc1 drop cov index 2,3 (\d+\.\d+)', line).group(1))
        acc1_drop89.append(acc)


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

diff, mean = minus_avg(acc1_drop89, acc1_default)
plt.plot(range(len(diff)), diff, linewidth=1, linestyle='-', label=f'drop_cov({mean:.4f})')

plt.grid(True)
plt.legend()
plt.savefig(f'./graph_img/AccDiff_drop89_ResNet18', dpi =300)
plt.clf()
