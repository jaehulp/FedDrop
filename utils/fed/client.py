import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.dataset import read_client_data, get_client_transform
from utils.loss import get_loss
from utils.ops import accuracy, AverageMeter
from utils.optimizer import load_optimizer

class Client(object):

    def __init__(self, args, idx: int, output_dir, datapath):

        self.args = args
        self.idx = idx
        self.batch_size = args.batch_size
        self.local_epochs = args.local_epochs
        self.loss = get_loss(args.loss)
        self.dataset = datapath
        self.train_samples = 0
        self.output_dir = os.path.join(output_dir, str(idx))
        self.global_epoch = -1
        self.save_epoch = -1
        if args.save_client == True:
            os.makedirs(self.output_dir, exist_ok=True)

    def set_device(self, gpu_id):
        self.device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

    def load_train_data(self):
        if self.args.transform.use_transform == True:
            transform = get_client_transform(self.args.transform)
        else:
            transform = None
        
        train_data = read_client_data(self.dataset, self.idx, transform=transform)
        return torch.utils.data.DataLoader(train_data, self.batch_size, shuffle=True, drop_last=True)

    def load_train_data_nonshuffle(self):
        
        train_data = read_client_data(self.dataset, self.idx)
        return torch.utils.data.DataLoader(train_data, self.batch_size, shuffle=False)

    def set_parameters(self, global_model):

        self.model = copy.deepcopy(global_model)
        self.optimizer = load_optimizer(self.args.optimizer, self.model)

    def save_client_model(self):

        path = os.path.join(self.output_dir, f'epoch_{self.global_epoch}.pt')
        torch.save(self.model.state_dict(), path)

from utils.fed.ClientSet.ClientAvg import ClientAvg
from utils.fed.ClientSet.ClientSAM import ClientSAM
from utils.fed.ClientSet.ClientMRL import ClientMRL

def set_client(base):

    if base == 'avg':
        ClientBase = ClientAvg
    elif base == 'sam': 
        ClientBase = ClientSAM
    elif base == 'mrl':
        ClientBase = ClientMRL
    return ClientBase
