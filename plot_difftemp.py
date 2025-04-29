import os
import re
import yaml

import torch
import numpy as np

import matplotlib.pyplot as plt 


log_path = './plot_log/dirichlet_testtemp.txt'

with open(log_path, 'r') as file:
    log_data = file.readlines()

acc1_default = []
acc1_avg_cov_temp1 = []
acc1_avg_svd_temp1 = []
acc1_avg_topk_temp1 = []
acc1_avg_cov_temp3 = []
acc1_avg_svd_temp3 = []
acc1_avg_topk_temp3 = []
acc1_avg_cov_temp5 = []
acc1_avg_svd_temp5 = []
acc1_avg_topk_temp5 = []
acc1_avg_cov_temp7 = []
acc1_avg_svd_temp7 = []
acc1_avg_topk_temp7 = []
acc1_avg_cov_temp9 = []
acc1_avg_svd_temp9 = []
acc1_avg_topk_temp9 = []

for line in log_data:
    line = line.strip()
    if line.startswith("Acc1 default test_acc"):
        acc = float(re.search(r'Acc1 default test_acc (\d+\.\d+)', line).group(1))
        acc1_default.append(acc)
    if line.startswith("Temp1 Acc1 avg cov"):
        acc = float(re.search(r'Temp1 Acc1 avg cov (\d+\.\d+)', line).group(1))
        acc1_avg_cov_temp1.append(acc)
    if line.startswith("Temp1 Acc1 avg svd_cov"):
        acc = float(re.search(r'Temp1 Acc1 avg svd_cov (\d+\.\d+)', line).group(1))
        acc1_avg_svd_temp1.append(acc)
    if line.startswith("Temp1 Acc1 avg topk_svd_cov"):
        acc = float(re.search(r'Temp1 Acc1 avg topk_svd_cov (\d+\.\d+)', line).group(1))
        acc1_avg_topk_temp1.append(acc)
    if line.startswith("Temp3 Acc1 avg cov"):
        acc = float(re.search(r'Temp3 Acc1 avg cov (\d+\.\d+)', line).group(1))
        acc1_avg_cov_temp3.append(acc)
    if line.startswith("Temp3 Acc1 avg svd_cov"):
        acc = float(re.search(r'Temp3 Acc1 avg svd_cov (\d+\.\d+)', line).group(1))
        acc1_avg_svd_temp3.append(acc)
    if line.startswith("Temp3 Acc1 avg topk_svd_cov"):
        acc = float(re.search(r'Temp3 Acc1 avg topk_svd_cov (\d+\.\d+)', line).group(1))
        acc1_avg_topk_temp3.append(acc)
    if line.startswith("Temp5 Acc1 avg cov"):
        acc = float(re.search(r'Temp5 Acc1 avg cov (\d+\.\d+)', line).group(1))
        acc1_avg_cov_temp5.append(acc)
    if line.startswith("Temp5 Acc1 avg svd_cov"):
        acc = float(re.search(r'Temp5 Acc1 avg svd_cov (\d+\.\d+)', line).group(1))
        acc1_avg_svd_temp5.append(acc)
    if line.startswith("Temp5 Acc1 avg topk_svd_cov"):
        acc = float(re.search(r'Temp5 Acc1 avg topk_svd_cov (\d+\.\d+)', line).group(1))
        acc1_avg_topk_temp5.append(acc)
    if line.startswith("Temp7 Acc1 avg cov"):
        acc = float(re.search(r'Temp7 Acc1 avg cov (\d+\.\d+)', line).group(1))
        acc1_avg_cov_temp7.append(acc)
    if line.startswith("Temp7 Acc1 avg svd_cov"):
        acc = float(re.search(r'Temp7 Acc1 avg svd_cov (\d+\.\d+)', line).group(1))
        acc1_avg_svd_temp7.append(acc)
    if line.startswith("Temp7 Acc1 avg topk_svd_cov"):
        acc = float(re.search(r'Temp7 Acc1 avg topk_svd_cov (\d+\.\d+)', line).group(1))
        acc1_avg_topk_temp7.append(acc)
    if line.startswith("Temp9 Acc1 avg cov"):
        acc = float(re.search(r'Temp9 Acc1 avg cov (\d+\.\d+)', line).group(1))
        acc1_avg_cov_temp9.append(acc)
    if line.startswith("Temp9 Acc1 avg svd_cov"):
        acc = float(re.search(r'Temp9 Acc1 avg svd_cov (\d+\.\d+)', line).group(1))
        acc1_avg_svd_temp9.append(acc)
    if line.startswith("Temp9 Acc1 avg topk_svd_cov"):
        acc = float(re.search(r'Temp9 Acc1 avg topk_svd_cov (\d+\.\d+)', line).group(1))
        acc1_avg_topk_temp9.append(acc)

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

diff, mean = minus_avg(acc1_avg_cov_temp1, acc1_default)
plt.plot(range(len(diff)), diff, linewidth=1, linestyle='-', label=f'Avgcov_temp1({mean:.4f})')

plt.grid(True)
plt.legend()
plt.savefig(f'./graph_img/AccDiff_avgcov_temp1_ResNet18', dpi =300)
plt.clf()


plt.figure()
plt.xlabel('Epoch')
plt.ylabel('Accuracy difference')
plt.title(f'Acc diff between default merge & method')

diff, mean = minus_avg(acc1_avg_cov_temp3, acc1_default)
plt.plot(range(len(diff)), diff, linewidth=1, linestyle='-', label=f'Avgcov_temp3({mean:.4f})')

plt.grid(True)
plt.legend()
plt.savefig(f'./graph_img/AccDiff_avgcov_temp3_ResNet18', dpi =300)
plt.clf()
plt.figure()
plt.xlabel('Epoch')
plt.ylabel('Accuracy difference')
plt.title(f'Acc diff between default merge & method')

diff, mean = minus_avg(acc1_avg_cov_temp5, acc1_default)
plt.plot(range(len(diff)), diff, linewidth=1, linestyle='-', label=f'Avgcov_temp5({mean:.4f})')

plt.grid(True)
plt.legend()
plt.savefig(f'./graph_img/AccDiff_avgcov_temp5_ResNet18', dpi =300)
plt.clf()
plt.figure()
plt.xlabel('Epoch')
plt.ylabel('Accuracy difference')
plt.title(f'Acc diff between default merge & method')

diff, mean = minus_avg(acc1_avg_cov_temp7, acc1_default)
plt.plot(range(len(diff)), diff, linewidth=1, linestyle='-', label=f'Avgcov_temp7({mean:.4f})')

plt.grid(True)
plt.legend()
plt.savefig(f'./graph_img/AccDiff_avgcov_temp7_ResNet18', dpi =300)
plt.clf()
plt.figure()
plt.xlabel('Epoch')
plt.ylabel('Accuracy difference')
plt.title(f'Acc diff between default merge & method')

diff, mean = minus_avg(acc1_avg_cov_temp9, acc1_default)
plt.plot(range(len(diff)), diff, linewidth=1, linestyle='-', label=f'Avgcov_temp9({mean:.4f})')

plt.grid(True)
plt.legend()
plt.savefig(f'./graph_img/AccDiff_avgcov_temp9_ResNet18', dpi =300)
plt.clf()
