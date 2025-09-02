import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.dataset import read_client_data, get_client_transform
from utils.loss import get_loss
from utils.ops import accuracy, AverageMeter
from utils.optimizer import load_optimizer
from utils.sam import ASAM, SAM
from utils.mrl import TopKOutputWrapper, PrefixTopKOutputWrapper

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


class ClientAvg(Client):

    def __init__(self, args, idx: int, output_dir, datapath):
        super().__init__(args, idx, output_dir, datapath)
    
    def train(self):

        acc1_meter, acc5_meter, loss_meter = AverageMeter(), AverageMeter(), AverageMeter()

        train_loader = self.load_train_data()
        self.model.to(self.device)
        self.train_samples = len(train_loader)
        if self.args.mrl.use_mrl == True:
            if self.args.mrl.prefix == True:
                self.model = PrefixTopKOutputWrapper(self.model, 'fc', k=self.args.mrl.topk, alpha=self.args.mrl.alpha)
            else:
                self.model = TopKOutputWrapper(self.model, 'fc', k=self.args.mrl.topk, alpha=self.args.mrl.alpha)
            self.model.set_device(self.device)
            if self.args.mrl.prefix == True:
                self.model.set_topk_indices(train_loader)
        self.model.train()
        self.model.to(self.device)

        if self.args.sam.minimizer == 'sam':
            minimizer = SAM(self.optimizer, self.model, self.args.sam.rho, self.args.sam.eta)
        elif self.args.sam.minimizer == 'asam':
            minimizer = ASAM(self.optimizer, self.model, self.args.sam.rho, self.args.sam.eta)

        for epoch in range(1, self.local_epochs+1):
            for i, (xs, ys) in enumerate(train_loader):
                self.optimizer.zero_grad()
                xs = xs.to(self.device)
                ys = ys.to(self.device)
                logits = self.model(xs)
                acc1, acc5 = accuracy(logits, ys, topk=(1,5))
                loss = self.loss(logits, ys)
                loss.backward()

                if (self.args.sam.minimizer == 'sam') or (self.args.sam.minimizer == 'asam'):
                    minimizer.ascent_step()
                    loss.backward()
                    minimizer.descent_step()
                else:
                    self.optimizer.step()

                acc1_meter.update(acc1), acc5_meter.update(acc5), loss_meter.update(loss)

        self.train_result = [acc1_meter.result().detach().cpu().numpy(), acc5_meter.result().detach().cpu().numpy(), loss_meter.result().detach().cpu().numpy()]

        del acc1_meter, acc5_meter, loss_meter

        if self.args.mrl.use_mrl == True:
            self.model = self.model.unwrap()

        if self.args.save_client:
            if (self.global_epoch % self.save_epoch == (self.save_epoch - 1)) or (self.global_epoch % self.save_epoch == 0):
                self.save_client_model()

