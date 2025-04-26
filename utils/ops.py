import os
import math
import torch
import numpy as np

class features:
    pass

def hook(self, input, output):
    features.value = input[0].clone()

def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()

    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:k].reshape(-1).float().sum(0)*100. / batch_size for k in topk]

class AverageMeter(object):

    def __init__(self, fmt=".4f"):
        self.reset()
        self.fmt = fmt
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0
        self.sqsum = 0.0
        self.std = 0.0

    def reset(self):
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0
        self.sqsum = 0.0
        self.std = 0.0

    def update(self, xs, n=1):
        eps = 1e-8

        self.val = xs
        self.sum += xs
        self.sqsum += xs ** 2
        self.count += n

        self.avg = self.sum / (self.count + eps)
        self.std = self.sqsum / (self.count + eps) - self.avg ** 2
        self.std = math.sqrt(self.std) if self.std > 0.0 else 0.0

    def result(self):
        eps = 1e-8

        self.avg = self.sum / (self.count + eps)
        self.std = self.sqsum / (self.count + eps) - self.avg ** 2
        self.std = math.sqrt(self.std) if self.std > 0.0 else 0.0

        return self.avg

def compute_cov(proj_matrix):

    mean = torch.mean(proj_matrix, dim=1, keepdim=True)
    std = torch.std(proj_matrix, dim=1, keepdim=True)

    proj_normalized = (proj_matrix - mean) / (std + 1e-16)
    proj_reshape = proj_normalized.permute(0, 2, 1)  # Shape (M, K, N)
    cov_matrix = torch.einsum('mkn,pqn->mpkq', proj_reshape, proj_reshape) / proj_matrix.shape[1]  # Shape (M, M, K, K)
    cov_val = cov_matrix[0, 1, :, :]
    cov_val = torch.diagonal(cov_val)
    return cov_val

import json

def client_num_samples(data_dir, num_client, num_join_client):
    json_dir = os.path.join(data_dir, 'config.json')
    with open(json_dir) as file:
        data = json.load(file)
    
    samples_data = data['Size of samples for labels in clients']
    arr = np.zeros(shape=(num_client, num_join_client))
    for i, c in enumerate(samples_data):
        for e in c:
            arr[i, e[0]] = e[1]
    
    return np.sum(arr, axis=1)
    

