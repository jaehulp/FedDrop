import os
import re
import yaml

import torch
import numpy as np

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
    c = torch.mean(a-b)
    print(c)

minus_avg(acc1_drop_test_acc, acc1_default)
minus_avg(acc1_drop_cov_mean, acc1_default)
minus_avg(acc1_drop_svd_mean, acc1_default)
minus_avg(acc1_drop_topk_svd, acc1_default)
minus_avg(acc1_avg_cov_mean, acc1_default)
minus_avg(acc1_avg_svd_mean, acc1_default)
minus_avg(acc1_avg_topk_svd, acc1_default)
