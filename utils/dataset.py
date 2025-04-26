import os
import torch

import numpy as np

from torchvision import transforms


def get_client_transform(args):
    t = []

    if 'flip' in args.transforms:
        t.append(transforms.RandomHorizontalFlip())
    if 'crop' in args.transforms:
        t.append(transforms.RandomCrop(32, padding=4))
    transform = transforms.Compose(t)
    return transform


def read_client_data(train_data_dir, idx, transform=None):

    train_file = os.path.join(train_data_dir, 'train_'+str(idx)+'.npz') 

    with open(train_file, 'rb') as f:
        train_data = np.load(f, allow_pickle=True)['data'].tolist()

    X = torch.Tensor(train_data['x']).type(torch.float32)
    Y = torch.Tensor(train_data['y']).type(torch.int64)

    if transform != None:
        X = torch.stack([transform(x) for x in X])

    train_data = [(xs, ys) for xs, ys in zip(X, Y)]

    return train_data


def read_server_data(test_data_dir):

    test_file = os.path.join(test_data_dir, 'test.npz') 

    with open(test_file, 'rb') as f:
        test_data = np.load(f, allow_pickle=True)['data'].tolist()

    X = torch.Tensor(test_data['x']).type(torch.float32)
    Y = torch.Tensor(test_data['y']).type(torch.int64)

    test_data = [(xs, ys) for xs, ys in zip(X, Y)]

    return test_data
