import os
import copy
import time
import torch
import torchinfo
import random

import numpy as np

from dotmap import DotMap

from utils.dataset import read_client_data, read_server_data
from utils.model import load_model
from utils.ops import accuracy, AverageMeter
from utils.loss import get_loss
from utils.fed.client import set_client

class Server(object):
    def __init__(self, args, **kwargs):

        self.args = args
        self.kwargs = DotMap(kwargs)
        self.args_client = args.client
        self.args_server = args.server

        self.output_dir = args.output_dir
        self.datapath = args.datapath
        self.global_model = load_model(args.model)

        self.num_gpus = torch.cuda.device_count()
        self.num_clients = self.args_server.num_clients
        self.num_join_clients = self.args_server.num_join_clients
        self.join_clients = []
        self.client_base = set_client(self.args_client.base)

        self.epochs = self.args_server.epochs
        self.loss = get_loss(self.args_client.loss)
        self.logger = self.kwargs.logger
        self.writer = self.kwargs.writer

        test_data = read_server_data(self.datapath)
        self.test_loader = torch.utils.data.DataLoader(test_data, batch_size=self.args_server.batch_size, drop_last=False, shuffle=False)
        self.logger.info(f"{torchinfo.summary(self.global_model, (self.args_client.batch_size, 3, 32, 32))}")

        model_save_path = os.path.join(self.output_dir, 'server')
        os.makedirs(model_save_path, exist_ok=True)

    def set_client(self, ClientBase, idx):
        train_data = read_client_data(self.args.datapath, idx)
        client = ClientBase(self.args_client, 
                            idx,
                            os.path.join(self.output_dir, 'client'),
                            self.datapath,
                            )
        client.save_epoch = self.args.save_interval
        return client

    def select_clients(self, epoch):
        join_clients = list(np.random.choice(self.num_clients, size=self.num_join_clients, replace=False))
        index = []
        client_list = []
        for c in join_clients:
            client = self.set_client(self.client_base, c)
            index.append(c)
            client.global_epoch = epoch
            client_list.append(client)
        self.logger.info(f'Client {index} joined')
        self.join_clients = client_list
        self.send_model()

    def send_model(self):

        for client in self.join_clients:
            client.set_parameters(self.global_model)

    def receive_models(self):

        self.uploaded_weights = []
        self.uploaded_models = []
        total_sample = 0

        for client in self.join_clients:
            total_sample += client.train_samples
            self.uploaded_weights.append(client.train_samples)
            self.uploaded_models.append(client.model)
        
        self.uploaded_weights = [weight / total_sample for weight in self.uploaded_weights]

    def aggregate_parameters(self):
        assert (len(self.uploaded_models) > 0)
        self.global_model = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_model.parameters():
            param.data.zero_()

        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            self.add_parameters(w, client_model)

    def add_parameters(self, w, client_model):
        for server_param, client_param in zip(self.global_model.to(0).parameters(), client_model.to(0).parameters()):
            server_param.data += client_param.data.clone() * w

    def save_global_model(self, epoch):
        model_path = os.path.join(self.output_dir, 'server')
        model_path = os.path.join(model_path, f'epoch_{epoch}.pt')
        torch.save(self.global_model.state_dict(), model_path)

    def save_local_model(self, epoch):

        for i, client_model in enumerate(self.uploaded_models):
            model_path = os.path.join(self.output_dir, i)
            model_path = os.path.join(model_path, f'epoch_{epoch}.pt')
            torch.save(client_model.state_dict(), model_path)

    def set_client_batch(self):

        client_batches = [[] for _ in range(self.num_gpus)]

        for i, client in enumerate(self.join_clients):
            client_batches[i % self.num_gpus].append(client)

        self.client_batches = client_batches

    def train_clients_on_gpu(self, clients, gpu_id):

        for client in clients:
            client.set_device(gpu_id)
            client.train()

    def del_client_model(self):

        for client in self.join_clients:
            del client
        self.join_clients = []

    def evaluate(self, epoch):
        
        self.global_model.eval()
        self.global_model.cuda()

        acc1_meter, acc5_meter, loss_meter, batch_time = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()

        with torch.no_grad():
            for step, (xs, ys) in enumerate(self.test_loader):

                start_time = time.time()
                xs = xs.cuda()
                ys = ys.cuda()
                logits = self.global_model(xs)
                start_time = time.time() - start_time

                acc1, acc5 = accuracy(logits, ys, topk=(1,5))
                loss = self.loss(logits, ys)
                acc1_meter.update(acc1), acc5_meter.update(acc5), loss_meter.update(loss), batch_time.update(start_time)

        self.logger.info(
            f'Test [{epoch}/{self.epochs}]  Test time: {batch_time.sum:.4f}\n'
            f'acc1: (avg) {acc1_meter.result():.4f}\t acc5: (avg) {acc5_meter.result():.4f}\t'
            f'loss: (avg) {loss_meter.result():.4f}\n'
        )

    def receive_results(self):

        result_array = np.zeros((self.num_join_clients, 3))
        for i, client in enumerate(self.join_clients):
            result_array[i] = client.train_result

        results = np.dot(self.uploaded_weights, result_array)
        return(results)


    def save_train_result(self, epoch):
        results = self.receive_results()

        self.logger.info(
            f'Train [{epoch}/{self.epochs}]\n'
            f'acc1: (avg) {results[0]:.4f}\t acc5: (avg) {results[1]:.4f}\t'
            f'loss: (avg) {results[2]:.4f}\n'
        )

from utils.fed.ServerSet.ServerAvg import FedAvg
from utils.fed.ServerSet.ServerCovAvg import FedCovAvg


def set_server(base):

    if base == 'avg':
        ServerBase = FedAvg
    elif base == 'covavg':
        ServerBase = FedCovAvg
    return ServerBase