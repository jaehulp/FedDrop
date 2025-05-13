import os
import copy
import time
import torch
import torchinfo
import random

import numpy as np

from dotmap import DotMap
from concurrent.futures import ThreadPoolExecutor

from utils.client import ClientAvg
from utils.dataset import read_client_data, read_server_data
from utils.model import load_model
from utils.ops import accuracy, AverageMeter, features, hook, compute_cov
from utils.loss import get_loss

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
        self.clients = []
        self.join_clients = []

        self.epochs = self.args_server.epochs
        self.loss = get_loss(self.args_client.loss)
        self.logger = self.kwargs.logger
        self.writer = self.kwargs.writer

        test_data = read_server_data(self.datapath)
        self.test_loader = torch.utils.data.DataLoader(test_data, batch_size=self.args_server.batch_size, drop_last=False, shuffle=False)

        if args.LC.use_LC == True:
            self.set_LC()

        self.logger.info(f"{torchinfo.summary(self.global_model, (self.args_client.batch_size, 3, 32, 32))}")

    def set_clients(self, ClientBase):

        for idx in range(self.num_clients):     

            train_data = read_client_data(self.args.datapath, idx)
            client = ClientBase(self.args_client, 
                                self.global_model,
                                idx,
                                os.path.join(self.output_dir, 'client'),
                                self.datapath,
                                )
            client.save_epoch = self.args.save_interval
            self.clients.append(client)

    def select_clients(self, epoch):

        join_clients = list(np.random.choice(self.clients, size=self.num_join_clients, replace=False))
        index = []
        for c in join_clients:
            index.append(c.idx)
            c.global_epoch = epoch
        self.logger.info(f'Client {index} joined')
        self.join_clients = join_clients
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

    def aggregate_cov_parameters(self):
        assert (len(self.uploaded_models) > 0)

        global_handle = self.global_model.fc.register_forward_hook(hook)
        self.global_model.cuda()
        cov_mean_list = []
        for c in self.join_clients:
            
            loader = c.load_train_data_nonshuffle()
            client_handle = c.model.fc.register_forward_hook(hook)
            c.model.cuda()

            global_output = []
            for i, (xs, ys) in enumerate(loader):
                xs = xs.cuda()
                logits = self.global_model(xs)
                hook_data = features.value.data.view(xs.shape[0], -1).cuda()
                global_output.append(hook_data)
                
            client_output = []
            for i, (xs, ys) in enumerate(loader):
                xs = xs.cuda()
                logits = c.model(xs)
                hook_data = features.value.data.view(xs.shape[0], -1).cuda()
                client_output.append(hook_data)
            client_handle.remove()

            global_output = torch.cat(global_output)
            global_mean = torch.mean(global_output, dim=1, keepdim=True)
            global_output = global_output - global_mean

            client_output = torch.cat(client_output)
            client_mean = torch.mean(client_output, dim=1, keepdim=True)
            client_output = client_output - client_mean

            if self.args_server.svd_proj == True:
                _, _, Vt = torch.linalg.svd(global_output)
                global_output = (Vt[:50] @ global_output.permute(1,0)).T
                _, S, Vt = torch.linalg.svd(client_output)
                client_output = (Vt[:50] @ client_output.permute(1,0)).T

            output_mat = torch.stack([global_output, client_output])
            cov_val = compute_cov(output_mat)

            if self.args_server.svd_proj == True:
                cov_val = abs(cov_val)
            mean = torch.mean(cov_val)
            std = torch.std(cov_val)

            self.logger.info(f'{c.idx} model mean {mean} std {std}')
            cov_mean_list.append(mean.cpu().item())
        global_handle.remove()

        self.logger.info(f'Covariance mean : {np.mean(cov_mean_list)}')

        self.global_model = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_model.parameters():
            param.data.zero_()
        
        if self.args_server.aggregate_cov == 'avg':
            temp = self.args_server.avg_temp
            cov_mean_list = torch.nn.functional.softmax(torch.tensor(cov_mean_list) * temp).tolist()
            self.logger.info(f'Avg weight : {cov_mean_list}')
            for w, client_model in zip(cov_mean_list, self.uploaded_models):
                self.add_parameters(w, client_model)
        
        if self.args_server.aggregate_cov == 'drop':
            num_drop = self.args_server.num_drop
            index = np.argsort(cov_mean_list)
            weights = np.array(self.uploaded_weights)
            weights[index[:num_drop]] = 0
            weights = weights / weights.sum()
            for idx in index[num_drop:]:
                self.add_parameters(weights[idx], self.uploaded_models[idx])

    def add_parameters(self, w, client_model):
        for server_param, client_param in zip(self.global_model.to(0).parameters(), client_model.to(0).parameters()):
            server_param.data += client_param.data.clone() * w

    def save_global_model(self, epoch):

        model_path = os.path.join(self.output_dir, f'epoch_{epoch}.pt')
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
            del client.model

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
        
        self.writer.add_scalar('Acc1/test', acc1_meter.result(), epoch)
        self.writer.add_scalar('Acc5/test', acc5_meter.result(), epoch)
        self.writer.add_scalar('Loss/test', loss_meter.result(), epoch)

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

        self.writer.add_scalar('Acc1/train', results[0], epoch)
        self.writer.add_scalar('Acc5/train', results[1], epoch)
        self.writer.add_scalar('Loss/train', results[2], epoch)


class FedAvg(Server):
    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)
        self.set_clients(ClientAvg)

    def train(self):

        for epoch in range(1, self.args_server.epochs+1):
            self.select_clients(epoch)
            self.set_client_batch()

            with ThreadPoolExecutor(max_workers=self.num_gpus) as executor:

                processes = []
                for gpu_id, clients in enumerate(self.client_batches):

                    process = executor.submit(
                        self.train_clients_on_gpu,
                        clients, gpu_id
                    )
                    processes.append(process)

                for process in processes:
                    try:
                        process.result()  # Ensure execution and catch runtime exceptions
                    except Exception as e:
                        print(f"Thread execution failed: {e}")
               
            self.receive_models()

            if self.args_server.aggregate_cov != False:
                self.aggregate_cov_parameters()
            else:
                self.aggregate_parameters()

            self.save_train_result(epoch)

            self.evaluate(epoch)
            if (epoch % self.args.save_interval == 0) or (epoch % self.args.save_interval == (self.args.save_interval - 1)):
                self.save_global_model(epoch)
            


